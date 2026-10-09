package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/service"
	"edgebot/stats"

	"github.com/hybridgroup/yzma/exp/decide"
	"github.com/spf13/cobra"
)

var execCommand = exec.Command

// commitCallLLM runs the commit prompt through the running service when
// available, falling back to a local LLM call otherwise.
func commitCallLLM(cfg *config.Config, systemPrompt string, messages []llm.Message) ([]llm.Message, error) {
	req := service.ChatRequest{
		Messages:        messages,
		SystemPrompt:    systemPrompt,
		Model:           cfg.LLM.Model,
		Provider:        cfg.LLM.Provider,
		Backend:         cfg.LitertLM.Backend,
		Confirm:         cfg.Shell.Confirm,
		AllowedCommands: append([]string(nil), cfg.Shell.AllowedCommands...),
	}
	return chatWithServiceFallback(req, nil, func() ([]llm.Message, error) {
		caller := llm.NewProviderCaller(cfg.LLM.Provider, cfg.LLM.Model, llm.NoopExecutor{})
		if lc, ok := caller.(*llm.LitertLMCaller); ok {
			lc.Backend = cfg.LitertLM.Backend
		}
		if lc, ok := caller.(*llm.LlamacppCaller); ok {
			lc.MaxTokens = commitMaxTokens
			lc.NoThink = true
		}
		return caller.Call(context.Background(), systemPrompt, messages, nil)
	})
}

var commitAll bool
var dryRun bool
var commitDecisionModel string
var commitNoFilter bool

// commitOpenDecider opens a System One decision model; swappable for tests.
var commitOpenDecider = llm.OpenDecisionModel

// commitMaxFileDiffChars caps the per-file diff used as decision-model state
// so a single huge file doesn't blow the decision model's context.
const commitMaxFileDiffChars = 2000

// commitMaxTokens bounds commit-message generation on local models:
// reasoning models otherwise spend minutes thinking out loud.
const commitMaxTokens = 512

// commitMaxDiffChars caps the staged diff sent to the model so prompts fit
// small local contexts (~8k chars ≈ 2-3k tokens). Truncation is marked so the
// model knows the diff is incomplete.
const commitMaxDiffChars = 8000

// truncateDiff cuts s to maxChars runes, appending a marker when truncated.
func truncateDiff(s string, maxChars int) string {
	r := []rune(s)
	if len(r) <= maxChars {
		return s
	}
	return string(r[:maxChars]) + "\n\n[... diff truncated to fit the model context ...]"
}

var commitCmd = &cobra.Command{
	Use:   "commit",
	Short: "Generate a commit message for staged changes using AI",
	RunE: func(cmd *cobra.Command, args []string) error {
		return runCommit()
	},
}

func init() {
	commitCmd.Flags().BoolVarP(&commitAll, "all", "A", false, "stage all changes before committing")
	commitCmd.Flags().BoolVarP(&dryRun, "dry-run", "d", false, "print the commit message without creating a commit")
	commitCmd.Flags().StringVar(&commitDecisionModel, "decision-model", "", "decision model for pre-filtering files (default: the configured decision model)")
	commitCmd.Flags().BoolVar(&commitNoFilter, "no-filter", false, "skip decision-model pre-filtering and send the full diff to the LLM")
}

func runCommit() (rErr error) {
	var staged bool
	if commitAll {
		if out, err := execCommand("git", "add", "-A").CombinedOutput(); err != nil {
			return fmt.Errorf("git add -A failed: %w\n%s", err, out)
		}
		staged = true
		defer func() {
			if staged && (dryRun || (rErr != nil && !dryRun)) {
				if err := execCommand("git", "reset").Run(); err != nil {
					slog.Warn("git reset failed", "err", err)
				}
			}
		}()
	}

	diffOutput, err := execCommand("git", "diff", "--cached").Output()
	if err != nil {
		return fmt.Errorf("failed to get staged diff: %w", err)
	}

	if strings.TrimSpace(string(diffOutput)) == "" {
		return fmt.Errorf("no staged changes to commit (use git add to stage files)")
	}

	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	// Local models have small contexts (llamacpp: 4k tokens); truncate huge
	// diffs so the prompt fits instead of failing. Cloud providers ignore the
	// marker and just see a shorter diff.
	diff := truncateDiff(strings.TrimSpace(string(diffOutput)), commitMaxDiffChars)

	// When a decision model is available, ask it (noul: include/exclude) per
	// staged file whether that file's changes should be given to the LLM for
	// commit-message generation. Generated files, lockfiles and other noise
	// are filtered out so the message describes the meaningful change.
	if filtered, ok := filterDiffWithDecisionModel(cfg, diff); ok {
		diff = filtered
	}

	systemPrompt := "You are a helpful assistant that writes concise git commit messages."
	userPrompt := fmt.Sprintf(`Generate a concise git commit message for the following staged changes.

Staged changes (git diff --cached):
%s

Write a commit message using conventional commits format (e.g., feat:, fix:, chore:, docs:, refactor:, test:, style:).
The first line should be a concise summary under 72 characters.
If more detail is needed, add a blank line followed by bullet points or a short body.
Use the imperative mood ("add" not "added").
Only output the commit message, nothing else.`,
		diff,
	)

	messages := []llm.Message{
		{Role: "user", Content: userPrompt},
	}

	slog.Debug("provider", "name", cfg.LLM.Provider, "model", cfg.LLM.Model)
	slog.Debug("system prompt", "prompt", systemPrompt)
	slog.Debug("user prompt", "prompt", userPrompt)

	llmStart := time.Now()
	resultMessages, err := commitCallLLM(cfg, systemPrompt, messages)
	llmDuration := time.Since(llmStart)
	if err != nil {
		return fmt.Errorf("LLM call failed: %w", err)
	}

	slog.Debug("timing", "llm", llmDuration)

	if len(resultMessages) == 0 {
		return fmt.Errorf("no response from LLM")
	}

	lastMsg := resultMessages[len(resultMessages)-1]
	content, ok := lastMsg.Content.(string)
	if !ok || strings.TrimSpace(content) == "" {
		return fmt.Errorf("empty response from LLM")
	}

	msg := strings.TrimSpace(content)
	msg = stripThinkBlock(msg)
	msg = stripCodeFences(msg)

	if msg == "" {
		return fmt.Errorf("empty commit message after cleanup")
	}

	fmt.Printf("\n%s\n\n", msg)

	if dryRun {
		return nil
	}

	tmpFile, err := os.CreateTemp("", "commit-msg-*.txt")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(msg); err != nil {
		tmpFile.Close()
		return fmt.Errorf("failed to write commit message: %w", err)
	}
	tmpFile.Close()

	gitCommit := execCommand("git", "commit", "-F", tmpFile.Name())
	gitCommit.Stdout = os.Stdout
	gitCommit.Stderr = os.Stderr

	if err := gitCommit.Run(); err != nil {
		return fmt.Errorf("git commit failed: %w", err)
	}

	return nil
}

var fenceRe = regexp.MustCompile("(?m)^```[a-zA-Z]*\\s*\\n?|\\n?```\\s*$")

// resolveCommitDecisionModel picks the decision model used to pre-filter
// staged files: --decision-model wins, then the configured decision default,
// then (for configs predating the decision default) the chat model when it
// points at a decision GGUF. Empty means no decision model is available.
func resolveCommitDecisionModel(cfg *config.Config) string {
	if strings.TrimSpace(commitDecisionModel) != "" {
		return strings.TrimSpace(commitDecisionModel)
	}
	if strings.TrimSpace(cfg.LLM.DecisionModel) != "" {
		return cfg.LLM.DecisionModel
	}
	if cfg.LLM.Provider == "llamacpp" && config.IsDecisionModel(cfg.LLM.Model) {
		return cfg.LLM.Model
	}
	return ""
}

// filterDiffWithDecisionModel asks the decision model (noul per staged file)
// whether that file should be given to the LLM, returning the diff rebuilt
// from included files only. It returns ok=false when filtering is disabled,
// no decision model is available, or filtering fails — callers then use the
// full diff unchanged.
func filterDiffWithDecisionModel(cfg *config.Config, fullDiff string) (filtered string, ok bool) {
	if commitNoFilter {
		return "", false
	}
	model := resolveCommitDecisionModel(cfg)
	if model == "" {
		return "", false
	}

	files := stagedFileList()
	if len(files) == 0 {
		return "", false
	}
	// A single staged file still goes through the decision model so pure
	// noise commits (e.g. lockfile-only) can be reported, but the diff is
	// never emptied — see the empty-filter fallback below.

	modelDir, err := config.ModelsDir("llamacpp")
	if err != nil {
		slog.Warn("decision pre-filter skipped: cannot determine models dir", "err", err)
		return "", false
	}
	modelPath, err := llm.ResolveDecisionModelPath(modelDir, model)
	if err != nil {
		slog.Warn("decision pre-filter skipped: model file not found", "model", model, "err", err)
		return "", false
	}
	libDir, err := config.LibDir()
	if err != nil {
		slog.Warn("decision pre-filter skipped: cannot determine lib dir", "err", err)
		return "", false
	}

	d, err := commitOpenDecider(libDir, modelPath, decide.Options{})
	if err != nil {
		slog.Warn("decision pre-filter skipped: cannot open decision model", "model", model, "err", err)
		return "", false
	}
	defer d.Close()

	var included, excluded []string
	diffsByFile := make(map[string]string, len(files))
	for _, f := range files {
		fileDiff, err := stagedFileDiff(f)
		if err != nil {
			slog.Warn("decision pre-filter skipped: cannot read file diff", "file", f, "err", err)
			return "", false
		}
		if strings.TrimSpace(fileDiff) == "" {
			slog.Warn("decision pre-filter skipped: empty file diff", "file", f)
			return "", false
		}
		diffsByFile[f] = fileDiff
		keep, err := askIncludeFile(d, model, f, fileDiff)
		if err != nil {
			slog.Warn("decision pre-filter: question failed, keeping file", "file", f, "err", err)
			included = append(included, f)
			continue
		}
		if keep {
			included = append(included, f)
		} else {
			excluded = append(excluded, f)
		}
	}

	if len(excluded) == 0 {
		return "", false
	}
	if len(included) == 0 {
		slog.Warn("decision pre-filter excluded every file; using the full diff")
		return "", false
	}

	fmt.Printf("Decision model %s: including %s", model, strings.Join(included, ", "))
	if len(excluded) > 0 {
		fmt.Printf(" (excluding %s)", strings.Join(excluded, ", "))
	}
	fmt.Println()

	var b strings.Builder
	for i, f := range included {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(strings.TrimSpace(diffsByFile[f]))
		b.WriteString("\n")
	}
	rebuilt := strings.TrimSpace(b.String())
	if rebuilt == "" {
		slog.Warn("decision pre-filter produced an empty diff; using the full diff")
		return "", false
	}
	return truncateDiff(rebuilt, commitMaxDiffChars), true
}

// stagedFileList returns the names of files with staged changes.
func stagedFileList() []string {
	out, err := execCommand("git", "diff", "--cached", "--name-only", "-z").Output()
	if err != nil {
		slog.Warn("decision pre-filter: cannot list staged files", "err", err)
		return nil
	}
	// -z separates names with NUL so spaces/newlines in paths survive.
	raw := strings.Split(string(out), "\x00")
	var files []string
	for _, f := range raw {
		if strings.TrimSpace(f) != "" {
			files = append(files, f)
		}
	}
	return files
}

// stagedFileDiff returns the staged diff for a single file.
func stagedFileDiff(file string) (string, error) {
	out, err := execCommand("git", "diff", "--cached", "--", file).Output()
	if err != nil {
		return "", fmt.Errorf("git diff --cached -- %s failed: %w", file, err)
	}
	return string(out), nil
}

// askIncludeFile asks the decision model the noul question "should this file
// be given to the LLM for generating the commit message?" for one staged
// file. The state is the file name plus its (truncated) diff.
func askIncludeFile(d *decide.Decider, model, file, fileDiff string) (bool, error) {
	state := fmt.Sprintf("File: %s\n\nStaged diff:\n%s",
		file, truncateDiff(strings.TrimSpace(fileDiff), commitMaxFileDiffChars))
	q := decide.Noul(
		"Should this file's changes be included when generating the git commit message? Include meaningful source, docs and config changes. Exclude generated files, lockfiles, build artifacts, vendored code and other noise unless they are the only meaningful change.",
		"exclude the file from commit message generation",
		"include the file in commit message generation",
	)
	resp, err := d.Answer(&decide.Request{
		State:     llm.ParseDecisionState(state),
		Questions: []decide.NamedQuestion{{ID: "include", Question: q}},
	})
	if err != nil {
		return false, err
	}
	stats.RecordUsage("llamacpp", model, stats.Usage{
		PromptTokens: resp.Usage.InputTokens,
		TotalTokens:  resp.Usage.InputTokens,
	})
	if len(resp.Answers) == 0 || resp.Answers[0].Result == nil {
		return false, fmt.Errorf("decide: empty answer")
	}
	r := resp.Answers[0].Result
	slog.Debug("decision pre-filter", "file", file, "answer", r.Answer,
		"p_true", r.Probability("true"))
	return r.Answer == "true", nil
}

// thinkRe matches a reasoning model's <think>...</think> section, which must
// not leak into the commit message.
var thinkRe = regexp.MustCompile(`(?s)<think>.*?</think>`)

// gemmaChannelRe matches a Gemma 4 thinking channel block
// (<|channel>thought\n...<channel|>), which Gemma models hosted on the
// Gemini API can leave in the output. The answer follows the closing tag.
// The trailing/leading pipes are optional to tolerate variant spellings.
var gemmaChannelRe = regexp.MustCompile(`(?s)<\|channel\|?>.*?\|?<channel\|>`)

// gemmaSentinelRe matches leftover Gemma/chat-template sentinel tokens that
// must never appear in a commit message (prompt markers, turn markers,
// sequence markers).
var gemmaSentinelRe = regexp.MustCompile(`<\|think\|?>|<\|channel\|?>|\|?<channel\|>|<\|turn\|?>\w*|\|?<turn\|>|<bos>|<eos>`)

// stripThinkBlock removes <think>...</think> sections (and a dangling
// unclosed tag with everything after it, e.g. when generation was capped
// mid-thought) from model output. It also removes Gemma thinking channel
// blocks (<|channel>...<channel|>) plus leftover sentinel tokens, so Gemma
// models on the Gemini API don't leak their thought process into the
// commit message.
func stripThinkBlock(s string) string {
	s = thinkRe.ReplaceAllString(s, "")
	s = gemmaChannelRe.ReplaceAllString(s, "")
	if i := strings.Index(s, "<think>"); i >= 0 {
		s = s[:i]
	}
	// A Gemma channel start without a closing tag means generation was
	// capped mid-thought (or a ghost channel with no answer yet): drop it
	// and everything after it, mirroring the unclosed <think> handling.
	// Only cut when no closing tag remains (closed blocks were removed
	// above); a start tag after the answer with no close is trailing noise.
	if !strings.Contains(s, "<channel|>") {
		if i := strings.Index(s, "<|channel>"); i >= 0 {
			s = s[:i]
		}
	}
	s = strings.ReplaceAll(s, "</think>", "")
	s = gemmaSentinelRe.ReplaceAllString(s, "")
	// Some Gemma outputs carry a bare "thought" channel label even without
	// thinking delimiters; strip it when it prefixes the whole output.
	if strings.HasPrefix(s, "thought\n") {
		s = strings.TrimPrefix(s, "thought\n")
	}
	return strings.TrimSpace(s)
}

func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)
	// Remove leading ```<lang> and trailing ```
	s = fenceRe.ReplaceAllString(s, "")
	s = strings.TrimSpace(s)
	// Handle case of single surrounding fences via simple trim
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}
