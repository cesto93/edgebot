package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"edgebot/config"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const (
	litertlmDefaultBackend = "cpu"
	litertlmMaxToolHops    = 5
)

// litertlmClientCache shares one Engine per (lib dir, model, backend) so
// the model is loaded at most once per process.
var (
	litertlmClientMu    sync.Mutex
	litertlmClientCache = map[string]*litertlm.Client{}
)

// LitertLMCaller runs LiteRT-LM natively in-process through the
// litertlm-go binding. It expects the LiteRT-LM shared libraries
// (LITERTLM_LIB, else ~/.edgebot/lib) and a .litertlm model file
// (default ~/.edgebot/models/litertlm/).
type LitertLMCaller struct {
	Model    string
	Executor ToolExecutor
	Backend  string
	// ThinkEffort is the unified reasoning-effort level. The litertlm-go
	// binding exposes no thinking control, so it is accepted and ignored
	// (logged at debug) to keep the param usable across all providers.
	ThinkEffort ThinkEffort
}

func NewLitertLMCaller(model string, executor ToolExecutor) *LitertLMCaller {
	return &LitertLMCaller{Model: model, Executor: executor}
}

func (l *LitertLMCaller) Call(ctx context.Context, systemPrompt string, messages []Message, tools []any) ([]Message, error) {
	return l.call(ctx, systemPrompt, messages, tools)
}

// CallStructured prompt-engines the JSON schema into the system prompt,
// since the binding's native structured-output path (GenerateData[T])
// needs a compile-time struct while the extract command uses arbitrary
// schemas.
func (l *LitertLMCaller) CallStructured(ctx context.Context, systemPrompt string, messages []Message, tools []any, responseFormat any) ([]Message, error) {
	schemaPrompt := systemPrompt
	if s := schemaFromResponseFormat(responseFormat); s != "" {
		schemaPrompt = systemPrompt +
			"\n\nReturn only valid JSON that conforms to the following schema, with no additional text:\n" + s
	}
	return l.call(ctx, schemaPrompt, messages, tools)
}

func (l *LitertLMCaller) call(ctx context.Context, systemPrompt string, messages []Message, tools []any) ([]Message, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("litertlm: no messages to send")
	}
	if l.ThinkEffort != "" {
		slog.Debug("litertlm: think effort not supported, ignoring", "effort", l.ThinkEffort)
	}
	client, err := l.client(ctx)
	if err != nil {
		return nil, fmt.Errorf("litertlm: %w", err)
	}

	chatOpts := []litertlm.ChatOption{litertlm.WithSystemPrompt(systemPrompt)}
	if defs := toLitertLMTools(tools); len(defs) > 0 {
		chatOpts = append(chatOpts, litertlm.WithTool(defs...))
	}
	if history := messages[:len(messages)-1]; len(history) > 0 {
		chatOpts = append(chatOpts, litertlm.WithInitialMessages(toLitertLMMessages(history)))
	}

	chat, err := client.NewChat(ctx, chatOpts...)
	if err != nil {
		return nil, fmt.Errorf("litertlm: new chat: %w", err)
	}
	defer chat.Close()

	reply, err := l.sendUserTurn(ctx, chat, messages[len(messages)-1])
	if err != nil {
		return nil, err
	}

	text, err := l.dispatch(ctx, chat, reply)
	if err != nil {
		return nil, err
	}
	return []Message{{Role: "assistant", Content: text}}, nil
}

// sendUserTurn sends the user's latest message, routing multimodal parts
// (images/audio) through SendMulti and plain text through Send.
func (l *LitertLMCaller) sendUserTurn(ctx context.Context, chat *litertlm.Chat, msg Message) (*litertlm.Reply, error) {
	switch content := msg.Content.(type) {
	case string:
		return chat.Send(ctx, content)
	case []ContentPart:
		var text strings.Builder
		var parts []litertlm.Part
		hasMedia := false
		for _, p := range content {
			switch p.Type {
			case "text":
				text.WriteString(p.Text)
				parts = append(parts, litertlm.Text(p.Text))
			case "image_url":
				if p.ImageURL == nil {
					continue
				}
				data, err := decodeDataURL(p.ImageURL.URL)
				if err != nil {
					slog.Warn("litertlm: skipping invalid image data URL", "err", err)
					continue
				}
				parts = append(parts, litertlm.Image(data))
				hasMedia = true
			case "input_audio":
				if p.InputAudio == nil {
					continue
				}
				data, err := base64.StdEncoding.DecodeString(p.InputAudio.Data)
				if err != nil {
					slog.Warn("litertlm: skipping invalid audio data", "err", err)
					continue
				}
				parts = append(parts, litertlm.Audio(data))
				hasMedia = true
			}
		}
		if hasMedia {
			return chat.SendMulti(ctx, parts)
		}
		return chat.Send(ctx, text.String())
	default:
		return chat.Send(ctx, formatContent(msg.Content))
	}
}

// dispatch runs the tool-call loop: each model-requested tool is executed
// via the ToolExecutor and the result is fed back through SendToolResult
// until the model answers with text only.
func (l *LitertLMCaller) dispatch(ctx context.Context, chat *litertlm.Chat, reply *litertlm.Reply) (string, error) {
	for hops := 0; reply.HasToolCalls() && hops < litertlmMaxToolHops; hops++ {
		calls := reply.ToolCalls()
		for _, call := range calls {
			result, toolErr := l.Executor.ExecuteTool(ToolCall{
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			})
			if toolErr != nil {
				result = fmt.Sprintf("Error: %v", toolErr)
				if result == "" {
					result = toolErr.Error()
				} else if output := strings.TrimSpace(result); output != "" {
					// keep combined error+output already
				}
			}
			slog.Debug("litertlm: tool result", "tool", call.Function.Name)
			var execErr error
			reply, execErr = chat.SendToolResult(ctx, call.Function.Name, map[string]any{"result": result})
			if execErr != nil {
				return "", fmt.Errorf("litertlm: send tool result: %w", execErr)
			}
			if reply.HasToolCalls() {
				// New tool calls arrived; break inner loop to handle next hop correctly
				break
			}
		}
	}
	if reply.HasToolCalls() {
		return "", fmt.Errorf("tool call hop limit exceeded (%d)", litertlmMaxToolHops)
	}
	return strings.TrimSpace(reply.Text()), nil
}

// client resolves the shared library dir, model file, and backend, and
// returns a cached Client so the engine is constructed only once.
func (l *LitertLMCaller) client(ctx context.Context) (*litertlm.Client, error) {
	libDir := os.Getenv("LITERTLM_LIB")
	if libDir == "" {
		var err error
		libDir, err = config.LibDir()
		if err != nil {
			return nil, fmt.Errorf("cannot determine lib dir: %w", err)
		}
	}

	modelDir, err := config.ModelsDir("litertlm")
	if err != nil {
		return nil, fmt.Errorf("cannot determine models dir: %w", err)
	}
	modelPath := filepath.Join(modelDir, l.Model)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = filepath.Join(modelDir, l.Model+".litertlm")
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("model file not found: tried %s and %s",
				filepath.Join(modelDir, l.Model), modelPath)
		}
	}

	backend := l.Backend
	if backend == "" {
		backend = os.Getenv("LITERTLM_BACKEND")
	}
	if backend == "" {
		backend = litertlmDefaultBackend
	}

	// Pre-flight check so a missing runtime produces an actionable error
	// instead of a cryptic dlopen failure (e.g. a stale ~/.edgebot/lib
	// from an interrupted download inside docker).
	mainLib := "liblitertlm_c_cpu.so"
	if backend == "gpu" {
		mainLib = "liblitertlm_c.so"
	}
	for _, name := range []string{mainLib, "libGemmaModelConstraintProvider.so"} {
		p := filepath.Join(libDir, name)
		st, err := os.Stat(p)
		if err != nil || st.Size() == 0 {
			return nil, fmt.Errorf("required library %s is missing or empty: run `make install-litertlm` (or restart the container so docker-entrypoint.sh re-downloads it)", p)
		}
	}

	key := libDir + "\x00" + modelPath + "\x00" + backend

	litertlmClientMu.Lock()
	if c, ok := litertlmClientCache[key]; ok {
		litertlmClientMu.Unlock()
		return c, nil
	}
	litertlmClientMu.Unlock()

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	slog.Debug("litertlm: initializing engine", "lib", libDir, "model", modelPath, "backend", backend)
	// Use background context so cancellation of the caller doesn't tear down engine init
	c, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend(backend),
	)
	if err != nil {
		// The runtime links the Vulkan loader even for the cpu backend,
		// so a minimal image without libvulkan1 surfaces as a cryptic
		// dlopen error (seen on arm64 docker: "libvulkan.so.1: cannot
		// open shared object file"). Point at the fix instead.
		if strings.Contains(err.Error(), "libvulkan.so") {
			return nil, fmt.Errorf("%w (missing system Vulkan loader: install libvulkan1 / rebuild the docker image)", err)
		}
		return nil, err
	}
	litertlmClientMu.Lock()
	litertlmClientCache[key] = c
	litertlmClientMu.Unlock()
	slog.Debug("litertlm: engine initialized")
	return c, nil
}

// toLitertLMTools converts OpenAI-shaped tool definitions into litertlm
// RawTools (manual dispatch via Reply.ToolCalls + SendToolResult).
func toLitertLMTools(tools []any) []litertlm.ToolDefinition {
	var defs []litertlm.ToolDefinition
	for _, t := range tools {
		toolMap, ok := t.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := toolMap["function"].(map[string]any)
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		desc, _ := fn["description"].(string)
		params, _ := fn["parameters"].(map[string]any)
		defs = append(defs, litertlm.NewRawTool(name, desc, params))
	}
	return defs
}

// toLitertLMMessages flattens the conversation history into litertlm
// messages. Assistant and tool turns are folded into user turns since the
// high-level initial-message encoder only supports content parts safely.
func toLitertLMMessages(messages []Message) []litertlm.Message {
	var out []litertlm.Message
	for _, m := range messages {
		switch m.Role {
		case "assistant":
			text := formatContent(m.Content)
			if text == "" {
				continue
			}
			out = append(out, litertlm.Message{Role: "user", Parts: []litertlm.Part{litertlm.Text(text)}})
		case "user", "tool":
			prefix := ""
			if m.Role == "tool" {
				prefix = "Tool result: "
			}
			out = append(out, litertlm.Message{Role: "user", Parts: []litertlm.Part{litertlm.Text(prefix + formatContent(m.Content))}})
		}
	}
	return out
}

// schemaFromResponseFormat extracts the JSON schema object from an
// OpenAI-style response_format value.
func schemaFromResponseFormat(responseFormat any) string {
	rf, ok := responseFormat.(map[string]any)
	if !ok {
		return ""
	}
	js, ok := rf["json_schema"].(map[string]any)
	if !ok {
		return ""
	}
	schema, ok := js["schema"]
	if !ok {
		return ""
	}
	b, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return ""
	}
	return string(b)
}
