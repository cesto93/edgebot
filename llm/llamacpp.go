package llm

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"edgebot/config"
	"edgebot/stats"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

type LlamacppCaller struct {
	Model    string
	Executor ToolExecutor
	// MaxTokens caps generated tokens; <=0 means llamacppDefaultMaxTokens.
	MaxTokens int
	// CtxSize overrides the inference context size; <=0 means
	// llamacppDefaultCtxSize. OCR needs a larger window for dense pages.
	CtxSize int
	// MMProj is an explicit vision projector filename/path. Empty means
	// resolve it from the model name (then a unique projector).
	MMProj string
	// NoThink pre-fills an empty <think> block so reasoning models answer
	// directly instead of thinking out loud (for short-form tasks).
	NoThink bool
	// ThinkEffort is the unified reasoning-effort level ("": provider
	// default). Only "none" changes llamacpp behavior (equivalent to
	// NoThink); graded levels keep thinking enabled since llama.cpp exposes
	// an on/off thinking switch, not graded effort.
	ThinkEffort ThinkEffort

	once      sync.Once
	initErr   error
	lctx      llama.Context
	llm       llama.Model
	vocab     llama.Vocab
	smplr     llama.Sampler
	template  string
	libDir    string
	modelDir  string
	mtmdCtx   mtmd.Context
	hasVision bool
}

const llamacppDefaultCtxSize = 4096
const llamacppDefaultMaxTokens = 2048

// Repetition penalty for the default sampler chain: pure greedy decoding
// degenerates into repetition loops ("point point point...").
const llamacppPenaltyLastN = 64
const llamacppRepeatPenalty = 1.1

func NewLlamacppCaller(model string, executor ToolExecutor) *LlamacppCaller {
	return &LlamacppCaller{
		Model:    model,
		Executor: executor,
	}
}

// maxTokens returns the generation cap for this caller.
func (l *LlamacppCaller) maxTokens() int32 {
	if l.MaxTokens > 0 {
		return int32(l.MaxTokens)
	}
	return llamacppDefaultMaxTokens
}

func (l *LlamacppCaller) Call(ctx context.Context, systemPrompt string, messages []Message, tools []any) ([]Message, error) {
	if err := l.ensureInit(); err != nil {
		return nil, fmt.Errorf("llamacpp: %w", err)
	}
	return l.chat(ctx, systemPrompt, messages, l.smplr)
}

// CallStructured generates output constrained by a GBNF grammar derived from
// the JSON schema in the OpenAI-style response_format. The grammar sampler
// must come first in the chain: it masks out non-conforming logits so the
// greedy sampler that follows can only pick valid tokens. Adding it after
// would let greedy pick an invalid token (e.g. a <think> prefix from a
// reasoning model), which llama.cpp reports by throwing a C++ exception that
// aborts the process (SIGABRT).
func (l *LlamacppCaller) CallStructured(ctx context.Context, systemPrompt string, messages []Message, tools []any, responseFormat any) ([]Message, error) {
	if err := l.ensureInit(); err != nil {
		return nil, fmt.Errorf("llamacpp: %w", err)
	}
	grammar, err := grammarFromResponseFormat(responseFormat)
	if err != nil {
		return nil, fmt.Errorf("llamacpp: %w", err)
	}
	slog.Debug("llamacpp: structured output grammar", "grammar", grammar)

	chain := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	grammarSampler := llama.SamplerInitGrammar(l.vocab, grammar, "root")
	if grammarSampler == 0 {
		return nil, fmt.Errorf("llamacpp: grammar sampler failed to initialize (unsupported or invalid grammar)")
	}
	llama.SamplerChainAdd(chain, grammarSampler)
	llama.SamplerChainAdd(chain, llama.SamplerInitGreedy())
	defer llama.SamplerFree(chain)

	return l.chat(ctx, systemPrompt, messages, chain)
}

// chat routes text-only prompts through buildPrompt + generate and multimodal
// prompts (image attachments) through the mtmd vision path.
func (l *LlamacppCaller) chat(ctx context.Context, systemPrompt string, messages []Message, smplr llama.Sampler) ([]Message, error) {
	if imagesPresent(messages) {
		if l.mtmdCtx == 0 || !l.hasVision {
			return nil, fmt.Errorf("llamacpp: image input requires a vision projector. Place a *mmproj*.gguf matching your model in %s", l.modelDir)
		}
		inp, err := l.buildVisionPrompt(systemPrompt, messages)
		if err != nil {
			return nil, err
		}
		return l.generateVision(ctx, inp, smplr)
	}
	prompt, err := l.buildPrompt(systemPrompt, messages)
	if err != nil {
		return nil, err
	}
	return l.generate(ctx, prompt, smplr)
}

func (l *LlamacppCaller) ensureInit() error {
	l.once.Do(func() {
		l.initErr = l.initialize()
	})
	return l.initErr
}

// buildChatMessages assembles the chat-template messages from an optional
// system prompt plus the conversation. Tool messages get a "Tool result"
// prefix, and format converts a message's content into template text (the
// vision path also uses it to collect decoded image payloads in marker order).
func buildChatMessages(systemPrompt string, messages []Message, format func(any) (string, [][]byte)) []llama.ChatMessage {
	chatMsgs := make([]llama.ChatMessage, 0, 1+len(messages))
	if systemPrompt != "" {
		chatMsgs = append(chatMsgs, llama.NewChatMessage("system", systemPrompt))
	}
	for _, msg := range messages {
		content, _ := format(msg.Content)
		if msg.Role == "tool" {
			content = "Tool result (" + msg.ToolCallID + "): " + content
		}
		chatMsgs = append(chatMsgs, llama.NewChatMessage(msg.Role, content))
	}
	return chatMsgs
}

func (l *LlamacppCaller) buildPrompt(systemPrompt string, messages []Message) (string, error) {
	chatMsgs := buildChatMessages(systemPrompt, messages, func(content any) (string, [][]byte) {
		return formatContent(content), nil
	})
	prompt := l.applyChatTemplate(chatMsgs, true)
	if prompt == "" {
		return "", fmt.Errorf("chat template returned empty prompt")
	}
	return prompt, nil
}

// visionInput holds the chat-template rendered prompt plus the decoded image
// payloads in the order their markers appear in the prompt.
type visionInput struct {
	prompt string
	images [][]byte
}

// buildVisionPrompt renders a multimodal conversation for the chat template,
// inserting the mtmd media marker at each image's position and collecting the
// decoded image bytes.
func (l *LlamacppCaller) buildVisionPrompt(systemPrompt string, messages []Message) (*visionInput, error) {
	marker := mtmd.DefaultMarker()
	var images [][]byte
	chatMsgs := buildChatMessages(systemPrompt, messages, func(content any) (string, [][]byte) {
		text, imgs := formatVisionContent(content, marker)
		images = append(images, imgs...)
		return text, imgs
	})

	prompt := l.applyChatTemplate(chatMsgs, true)
	if prompt == "" {
		return nil, fmt.Errorf("chat template returned empty prompt")
	}
	return &visionInput{prompt: prompt, images: images}, nil
}

// formatVisionContent renders a message's content as template text, emitting
// the media marker once per image and returning the decoded image bytes.
func formatVisionContent(content any, marker string) (string, [][]byte) {
	switch v := content.(type) {
	case string:
		return v, nil
	case []ContentPart:
		var sb strings.Builder
		var images [][]byte
		for _, p := range v {
			switch p.Type {
			case "text":
				sb.WriteString(p.Text)
			case "image_url":
				if p.ImageURL == nil {
					continue
				}
				data, err := decodeDataURL(p.ImageURL.URL)
				if err != nil {
					slog.Warn("llamacpp: skipping invalid image data URL", "err", err)
					continue
				}
				sb.WriteString(marker)
				images = append(images, data)
			case "input_audio":
				slog.Warn("llamacpp: audio input is not supported")
			}
		}
		return sb.String(), images
	case nil:
		return "", nil
	default:
		return fmt.Sprintf("%v", v), nil
	}
}

// imagesPresent reports whether any message carries an image_url content part.
func imagesPresent(messages []Message) bool {
	for _, msg := range messages {
		if parts, ok := msg.Content.([]ContentPart); ok {
			for _, p := range parts {
				if p.Type == "image_url" && p.ImageURL != nil {
					return true
				}
			}
		}
	}
	return false
}

// ctxSize returns the inference context size for this caller.
func (l *LlamacppCaller) ctxSize() int {
	if l.CtxSize > 0 {
		return l.CtxSize
	}
	return llamacppDefaultCtxSize
}

func (l *LlamacppCaller) generate(ctx context.Context, prompt string, smplr llama.Sampler) ([]Message, error) {
	tokens := llama.Tokenize(l.vocab, l.maybeNoThink(prompt), true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenization failed")
	}
	if len(tokens) > l.ctxSize() {
		return nil, fmt.Errorf("llamacpp: prompt too long (%d tokens > context size %d), shorten the input",
			len(tokens), l.ctxSize())
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	if err := l.decodePrompt(ctx, tokens); err != nil {
		return nil, err
	}
	return l.sample(ctx, smplr, len(tokens))
}

// decodePrompt feeds the tokenized prompt to the context in chunks of at most
// n_batch tokens with explicit positions. Decoding the whole prompt as one
// batch aborts the process when it exceeds n_batch
// (GGML_ASSERT(n_tokens_all <= cparams.n_batch)). Only the final token needs
// logits, since sampling starts from it.
func (l *LlamacppCaller) decodePrompt(ctx context.Context, tokens []llama.Token) error {
	batch := llama.BatchInit(int32(l.ctxSize()), 0, 1)
	defer func() { _ = llama.BatchFree(batch) }()

	for offset := 0; offset < len(tokens); {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if err := batch.Clear(); err != nil {
			return fmt.Errorf("llamacpp decode: %w", err)
		}
		n := min(l.ctxSize(), len(tokens)-offset)
		for i := range n {
			last := offset+i == len(tokens)-1
			if err := batch.Add(tokens[offset+i], llama.Pos(offset+i), []llama.SeqId{0}, last); err != nil {
				return fmt.Errorf("llamacpp decode: %w", err)
			}
		}
		offset += n
		if _, err := llama.Decode(l.lctx, batch); err != nil {
			return fmt.Errorf("llamacpp decode: %w", err)
		}
	}
	return nil
}

// generateVision evaluates a multimodal prompt (text + images) through the
// mtmd context, then samples the completion starting from the resulting
// context position.
func (l *LlamacppCaller) generateVision(ctx context.Context, inp *visionInput, smplr llama.Sampler) ([]Message, error) {
	if len(inp.images) == 0 {
		return l.generate(ctx, inp.prompt, smplr)
	}
	if l.mtmdCtx == 0 {
		return nil, fmt.Errorf("llamacpp: multimodal context not initialized")
	}
	inp.prompt = l.maybeNoThink(inp.prompt)

	chunks := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(chunks)

	bitmaps := make([]mtmd.Bitmap, 0, len(inp.images))
	for _, data := range inp.images {
		if len(data) == 0 {
			continue
		}
		b := mtmd.BitmapInitFromBuf(l.mtmdCtx, &data[0], uint64(len(data)), false, mtmd.InitOptDefault())
		if b.Bitmap == 0 {
			return nil, fmt.Errorf("llamacpp: failed to decode image")
		}
		bitmaps = append(bitmaps, b.Bitmap)
	}
	defer func() {
		for _, b := range bitmaps {
			mtmd.BitmapFree(b)
		}
	}()
	if len(bitmaps) == 0 {
		return l.generate(ctx, inp.prompt, smplr)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	input := mtmd.NewInputText(inp.prompt, true, true)
	if res := mtmd.Tokenize(l.mtmdCtx, chunks, input, bitmaps); res != 0 {
		return nil, fmt.Errorf("llamacpp: mtmd tokenize failed: %d", res)
	}

	var nPast llama.Pos
	if res := mtmd.HelperEvalChunks(l.mtmdCtx, l.lctx, chunks, 0, 0, int32(l.ctxSize()), true, &nPast); res != 0 {
		return nil, fmt.Errorf("llamacpp: mtmd eval chunks failed: %d", res)
	}

	// nPast is the position after all prompt chunks, including image tokens.
	return l.sample(ctx, smplr, int(nPast))
}

// sample runs the token generation loop from the current context state (the
// logits of the last decoded token).
func (l *LlamacppCaller) sample(ctx context.Context, smplr llama.Sampler, promptTokens int) ([]Message, error) {
	var response strings.Builder
	completionTokens := 0
	limit := l.maxTokens()
	if remaining := l.ctxSize() - promptTokens; remaining < int(limit) {
		if remaining <= 0 {
			return nil, fmt.Errorf("llamacpp: prompt uses the whole context (%d/%d tokens)", promptTokens, l.ctxSize())
		}
		limit = int32(remaining)
		slog.Warn("llamacpp: generation limited by context window",
			"prompt_tokens", promptTokens, "ctx_size", l.ctxSize(), "max_new", limit)
	}
	for pos := int32(0); pos < limit; pos++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		token := llama.SamplerSample(smplr, l.lctx, -1)
		if llama.VocabIsEOG(l.vocab, token) {
			break
		}
		completionTokens++

		buf := make([]byte, 256)
		n := llama.TokenToPiece(l.vocab, token, buf, 0, false)
		if n > 0 {
			response.Write(buf[:n])
		}

		if _, err := llama.Decode(l.lctx, llama.BatchGetOne([]llama.Token{token})); err != nil {
			return nil, fmt.Errorf("llamacpp decode: %w", err)
		}
	}

	text := strings.TrimSpace(response.String())
	stats.RecordUsage("llamacpp", l.Model, stats.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	})
	return []Message{{Role: "assistant", Content: text}}, nil
}

func formatContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []ContentPart:
		var sb strings.Builder
		for _, p := range v {
			if p.Type == "text" {
				sb.WriteString(p.Text)
			}
		}
		return sb.String()
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", v)
	}
}

// emptyThinkBlock is a pre-filled empty reasoning block. Appending it to the
// rendered prompt disables thinking for reasoning models, mirroring what
// enable_thinking=false does in llama.cpp's chat template renderer: the model
// continues after the closed block instead of opening a <think> section.
const emptyThinkBlock = "<think>\n\n</think>\n\n"

// maybeNoThink appends emptyThinkBlock to prompt when thinking is disabled
// (NoThink set or ThinkEffort "none") and the model's chat template is a
// reasoning template (contains a <think> marker). Plain-text templates are
// returned unchanged, as are prompts that already end with the block
// (idempotent).
func (l *LlamacppCaller) maybeNoThink(prompt string) string {
	if !l.NoThink && l.ThinkEffort != ThinkEffortNone {
		return prompt
	}
	if !strings.Contains(l.template, "<think>") {
		return prompt
	}
	if strings.HasSuffix(prompt, emptyThinkBlock) {
		return prompt
	}
	return prompt + emptyThinkBlock
}

// llamacppFallbackTemplate is the builtin template used when the model's
// own chat template fails to render (e.g. it uses Jinja features newer than
// the bundled llama.cpp, as seen with Spark-X2.5). Output formatting may
// differ from the native template, hence the warning on fallback.
const llamacppFallbackTemplate = "chatml"

func (l *LlamacppCaller) applyChatTemplate(chatMsgs []llama.ChatMessage, addAssistant bool) string {
	if prompt := renderChatTemplate(l.template, chatMsgs, addAssistant); prompt != "" {
		return prompt
	}
	if l.template == llamacppFallbackTemplate {
		return ""
	}
	slog.Warn("llamacpp: model chat template failed to render, "+
		"falling back to builtin template",
		"model", l.Model, "fallback", llamacppFallbackTemplate)
	return renderChatTemplate(llamacppFallbackTemplate, chatMsgs, addAssistant)
}

func renderChatTemplate(template string, chatMsgs []llama.ChatMessage, addAssistant bool) string {
	buf := make([]byte, 131072)
	n := llama.ChatApplyTemplate(template, chatMsgs, addAssistant, buf)
	if n <= 0 {
		return ""
	}
	if n >= int32(len(buf)) {
		buf = make([]byte, n+1)
		n = llama.ChatApplyTemplate(template, chatMsgs, addAssistant, buf)
		if n <= 0 {
			return ""
		}
	}
	return string(buf[:n])
}

func (l *LlamacppCaller) initialize() error {
	libDir, err := config.LibDir()
	if err != nil {
		return fmt.Errorf("cannot determine lib dir: %w", err)
	}
	modelDir, err := config.ModelsDir("llamacpp")
	if err != nil {
		return fmt.Errorf("cannot determine models dir: %w", err)
	}
	l.libDir = libDir
	l.modelDir = modelDir

	slog.Debug("llamacpp: loading library", "dir", l.libDir)
	if err := llama.Load(l.libDir); err != nil {
		return fmt.Errorf("failed to load llama library from %s: %w", l.libDir, err)
	}

	llama.LogSet(llama.LogSilent())
	llama.Init()

	modelPath := filepath.Join(l.modelDir, l.Model)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join(l.modelDir, l.Model+".gguf")
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			return fmt.Errorf("model file not found: tried %s and %s",
				filepath.Join(l.modelDir, l.Model), modelPath)
		}
	}

	slog.Debug("llamacpp: loading model", "path", modelPath)
	llm, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("failed to load model from %s: %w", modelPath, err)
	}
	if llm == 0 {
		return fmt.Errorf("failed to load model from %s: returned null", modelPath)
	}
	l.llm = llm

	l.vocab = llama.ModelGetVocab(llm)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(l.ctxSize())
	ctxParams.NBatch = uint32(l.ctxSize())

	slog.Debug("llamacpp: creating context")
	lctx, err := llama.InitFromModel(llm, ctxParams)
	if err != nil {
		llama.ModelFree(llm)
		return fmt.Errorf("failed to create context: %w", err)
	}
	l.lctx = lctx

	chain := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(chain, llama.SamplerInitPenalties(
		llama.VocabNTokens(l.vocab), llamacppPenaltyLastN, llamacppRepeatPenalty, 0, 0))
	llama.SamplerChainAdd(chain, llama.SamplerInitGreedy())
	l.smplr = chain

	l.template = llama.ModelChatTemplate(llm, "")
	if l.template == "" {
		l.template = "chatml"
	}

	l.setupVision(llm)

	slog.Debug("llamacpp: initialized successfully",
		"model", filepath.Base(modelPath),
		"template", l.template,
		"vision", l.hasVision)

	return nil
}

// resolveMMProj returns the vision projector to load: the explicit MMProj
// when set, otherwise the projector resolved from the model name (with a
// unique-projector fallback).
func (l *LlamacppCaller) resolveMMProj() (string, error) {
	if l.MMProj == "" {
		return config.FindLlamacppMMProjForModel(l.Model)
	}
	path := l.MMProj
	if !filepath.IsAbs(path) {
		path = filepath.Join(l.modelDir, path)
	}
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if filepath.Ext(path) == "" {
		if _, err := os.Stat(path + ".gguf"); err == nil {
			return path + ".gguf", nil
		}
	}
	return "", fmt.Errorf("vision projector not found: %s", l.MMProj)
}

// setupVision initializes the mtmd (multimodal) context when a vision
// projector (mmproj) is available. Failures degrade gracefully: text-only
// inference keeps working and image requests surface a clear error.
func (l *LlamacppCaller) setupVision(llm llama.Model) {
	mmproj, err := l.resolveMMProj()
	if err != nil {
		slog.Warn("llamacpp: vision projector resolution failed, image input disabled",
			"model", l.Model, "err", err)
		return
	}
	if mmproj == "" {
		slog.Debug("llamacpp: no mmproj found, image input disabled")
		return
	}
	if err := mtmd.Load(l.libDir); err != nil {
		slog.Debug("llamacpp: llama library has no mtmd support, image input disabled", "err", err)
		return
	}
	mtmd.LogSet(llama.LogSilent())

	mctx, err := mtmd.InitFromFile(mmproj, llm, mtmd.ContextParamsDefault())
	if err != nil || mctx == 0 {
		slog.Warn("llamacpp: failed to initialize multimodal context, image input disabled", "mmproj", mmproj, "err", err)
		return
	}
	l.mtmdCtx = mctx
	l.hasVision = mtmd.SupportVision(mctx)
	slog.Debug("llamacpp: multimodal context initialized", "mmproj", mmproj, "vision", l.hasVision)
}
