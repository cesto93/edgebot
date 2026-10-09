package llm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hybridgroup/yzma/exp/decide"
)

func testDecisionResponse() *decide.Response {
	return &decide.Response{
		Answers: []decide.Answer{
			{
				ID: "question",
				Result: &decide.Result{
					Answer:         "billing",
					Options:        []string{"billing", "technical"},
					Probabilities:  []float64{0.8, 0.2},
					Temperature:    1,
					TopProbability: 0.8,
					InputTokens:    42,
				},
			},
		},
	}
}

func TestResolveDecisionModelPath(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "jev.gguf"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "plain"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "adir.gguf"), 0755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		model   string
		want    string
		wantErr bool
	}{
		{"with extension", "jev.gguf", filepath.Join(dir, "jev.gguf"), false},
		{"without extension", "jev", filepath.Join(dir, "jev.gguf"), false},
		{"no extension on disk", "plain", filepath.Join(dir, "plain"), false},
		{"missing", "nope", "", true},
		{"directory is not a model", "adir", "", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveDecisionModelPath(dir, tc.model)
			if tc.wantErr != (err != nil) {
				t.Fatalf("ResolveDecisionModelPath(%q) err = %v, wantErr %v", tc.model, err, tc.wantErr)
			}
			if err == nil && got != tc.want {
				t.Errorf("ResolveDecisionModelPath(%q) = %q, want %q", tc.model, got, tc.want)
			}
		})
	}
}

func TestParseDecisionState(t *testing.T) {
	t.Run("json object keeps raw message", func(t *testing.T) {
		got := ParseDecisionState(`{"b": 1, "a": 2}`)
		raw, ok := got.(json.RawMessage)
		if !ok {
			t.Fatalf("got %T, want json.RawMessage", got)
		}
		if string(raw) != `{"b": 1, "a": 2}` {
			t.Errorf("key order not preserved: %s", raw)
		}
	})
	t.Run("json array keeps raw message", func(t *testing.T) {
		if _, ok := ParseDecisionState(`[1, 2]`).(json.RawMessage); !ok {
			t.Errorf("got %T, want json.RawMessage", ParseDecisionState(`[1, 2]`))
		}
	})
	t.Run("plain string stays a string", func(t *testing.T) {
		got := ParseDecisionState("the film was excellent.")
		if s, ok := got.(string); !ok || s != "the film was excellent." {
			t.Errorf("got %#v", got)
		}
	})
	t.Run("broken json falls back to string", func(t *testing.T) {
		got := ParseDecisionState(`{broken`)
		if s, ok := got.(string); !ok || s != `{broken` {
			t.Errorf("got %#v", got)
		}
	})
}

func TestBuildChoiceQuestion(t *testing.T) {
	q, err := BuildChoiceQuestion("Who handles this?|billing=payments, invoices|technical")
	if err != nil {
		t.Fatal(err)
	}
	if q.Type != "choice" || q.Text != "Who handles this?" {
		t.Errorf("unexpected question: %+v", q)
	}
	if len(q.Options) != 2 {
		t.Fatalf("options = %+v", q.Options)
	}
	if q.Options[0].Name != "billing" || q.Options[0].Description != "payments, invoices" {
		t.Errorf("option 0 = %+v", q.Options[0])
	}
	if q.Options[1].Name != "technical" || q.Options[1].Description != "" {
		t.Errorf("option 1 = %+v", q.Options[1])
	}
	if _, err := q.Names(); err != nil {
		t.Errorf("Names() = %v", err)
	}

	for _, spec := range []string{
		"",
		"only instructions, no options",
		"|billing",
		"q|=nodesc",
	} {
		t.Run("invalid "+spec, func(t *testing.T) {
			if _, err := BuildChoiceQuestion(spec); err == nil {
				t.Errorf("BuildChoiceQuestion(%q) succeeded, want error", spec)
			}
		})
	}
}

func TestBuildScoreQuestion(t *testing.T) {
	q, err := BuildScoreQuestion("How frustrated?|calm|annoyed|angry")
	if err != nil {
		t.Fatal(err)
	}
	if q.Type != "score" || len(q.Options) != 3 {
		t.Errorf("unexpected question: %+v", q)
	}
	if names, err := q.Names(); err != nil || len(names) != 3 {
		t.Errorf("Names() = %v, %v", names, err)
	}

	for _, spec := range []string{
		"",
		"instructions only",
		"q|just one level",
	} {
		t.Run("invalid "+spec, func(t *testing.T) {
			if _, err := BuildScoreQuestion(spec); err == nil {
				t.Errorf("BuildScoreQuestion(%q) succeeded, want error", spec)
			}
		})
	}
}

func TestBuildNoulQuestion(t *testing.T) {
	q, err := BuildNoulQuestion("Refund requested?|no money back asked|money back asked")
	if err != nil {
		t.Fatal(err)
	}
	if q.Type != "noul" {
		t.Errorf("type = %q", q.Type)
	}
	if names, err := q.Names(); err != nil || len(names) != 2 {
		t.Errorf("Names() = %v, %v", names, err)
	}

	for _, spec := range []string{
		"",
		"q|only false",
		"q|false|true|extra",
	} {
		t.Run("invalid "+spec, func(t *testing.T) {
			if _, err := BuildNoulQuestion(spec); err == nil {
				t.Errorf("BuildNoulQuestion(%q) succeeded, want error", spec)
			}
		})
	}
}

func TestFormatDecisionResponse(t *testing.T) {
	resp := testDecisionResponse()
	out := FormatDecisionResponse(resp)
	if !strings.Contains(out, `"question"`) || !strings.Contains(out, `"answers"`) {
		t.Errorf("response missing answers object:\n%s", out)
	}
	var raw any
	if err := json.Unmarshal([]byte(out), &raw); err != nil {
		t.Errorf("response is not valid JSON: %v\n%s", err, out)
	}
	if !strings.Contains(out, "\n  ") {
		t.Errorf("response is not indented:\n%s", out)
	}
}
