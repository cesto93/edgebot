package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/hybridgroup/yzma/exp/decide"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// ResolveDecisionModelPath resolves a decision model name to its GGUF file,
// mirroring LlamacppCaller.initialize: the name is tried as-is, then with a
// .gguf suffix appended. modelDir is ~/.edgebot/models/llamacpp.
func ResolveDecisionModelPath(modelDir, model string) (string, error) {
	candidate := filepath.Join(modelDir, model)
	if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
		return candidate, nil
	}
	withExt := candidate + ".gguf"
	if info, err := os.Stat(withExt); err == nil && !info.IsDir() {
		return withExt, nil
	}
	return "", fmt.Errorf("model file not found: tried %s and %s", candidate, withExt)
}

// OpenDecisionModel loads a self-describing System One decision model
// (openjev, lev, laya, kev, ...) via yzma's exp/decide package. The llama
// library is loaded from libDir first, as decide.Open requires Load+Init.
func OpenDecisionModel(libDir, modelPath string, opts decide.Options) (*decide.Decider, error) {
	if err := llama.Load(libDir); err != nil {
		return nil, fmt.Errorf("decide: failed to load llama library from %s: %w", libDir, err)
	}
	llama.LogSet(llama.LogSilent())
	llama.Init()

	d, err := decide.Open(modelPath, opts)
	if err != nil {
		if strings.Contains(err.Error(), "has no decision type") {
			return nil, fmt.Errorf("decide: %s is not a decision model (no decision metadata in GGUF): %w", modelPath, err)
		}
		return nil, fmt.Errorf("decide: %w", err)
	}
	return d, nil
}

// ParseDecisionState interprets raw CLI state input. JSON objects and arrays
// are passed through as json.RawMessage so key order is preserved (Go maps
// sort keys); anything else is used as a plain string.
func ParseDecisionState(s string) any {
	trimmed := bytes.TrimSpace([]byte(s))
	if len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[') {
		var raw json.RawMessage
		if err := json.Unmarshal(trimmed, &raw); err == nil {
			return raw
		}
	}
	return s
}

// splitDecisionSpec splits a pipe-separated question spec ("instructions|..."),
// trimming whitespace around each segment. Empty segments are kept so that
// positional options (score levels, noul descriptions) keep their meaning.
func splitDecisionSpec(spec string) []string {
	parts := strings.Split(spec, "|")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

// BuildChoiceQuestion builds a choice question from
// "instructions|name|name=description|...".
func BuildChoiceQuestion(spec string) (decide.Question, error) {
	parts := splitDecisionSpec(spec)
	if len(parts) < 2 || strings.TrimSpace(parts[0]) == "" {
		return decide.Question{}, fmt.Errorf("decide: --choice needs instructions and at least one option: \"instructions|name|name=description|...\"")
	}
	opts := make([]decide.Option, 0, len(parts)-1)
	for _, p := range parts[1:] {
		name, desc, _ := strings.Cut(p, "=")
		name = strings.TrimSpace(name)
		if name == "" {
			return decide.Question{}, fmt.Errorf("decide: --choice option %q has no name", p)
		}
		opts = append(opts, decide.Option{Name: name, Description: strings.TrimSpace(desc)})
	}
	return decide.ChoiceDesc(parts[0], opts...), nil
}

// BuildScoreQuestion builds a score question from
// "instructions|level 0|level 1|...".
func BuildScoreQuestion(spec string) (decide.Question, error) {
	parts := splitDecisionSpec(spec)
	if len(parts) < 3 || strings.TrimSpace(parts[0]) == "" {
		return decide.Question{}, fmt.Errorf("decide: --score needs instructions and 2 to 10 levels: \"instructions|level 0|level 1|...\"")
	}
	return decide.Score(parts[0], parts[1:]...), nil
}

// BuildNoulQuestion builds a true/false question from
// "instructions|false description|true description".
func BuildNoulQuestion(spec string) (decide.Question, error) {
	parts := splitDecisionSpec(spec)
	if len(parts) != 3 || strings.TrimSpace(parts[0]) == "" {
		return decide.Question{}, fmt.Errorf("decide: --noul needs instructions plus the false and true descriptions: \"instructions|false|true\"")
	}
	return decide.Noul(parts[0], parts[1], parts[2]), nil
}

// FormatDecisionResponse renders a decision response as indented JSON in the
// TypeSafe /v1/systemone shape (answers keyed by question id plus usage).
func FormatDecisionResponse(resp *decide.Response) string {
	raw, err := json.Marshal(resp)
	if err != nil {
		return fmt.Sprintf("%v", resp)
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, raw, "", "  "); err != nil {
		return string(raw)
	}
	return buf.String()
}
