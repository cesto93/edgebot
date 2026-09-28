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
