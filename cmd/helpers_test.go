package cmd

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"edgebot/config"
)

func TestSplitTelegramMessage(t *testing.T) {
	tests := []struct {
		name  string
		text  string
		limit int
		check func(t *testing.T, chunks []string)
	}{
		{
			name:  "short text is one chunk",
			text:  "hello",
			limit: 10,
			check: func(t *testing.T, chunks []string) {
				if len(chunks) != 1 || chunks[0] != "hello" {
					t.Fatalf("chunks = %q, want [hello]", chunks)
				}
			},
		},
		{
			name:  "splits on newline when possible",
			text:  "first line\nsecond line",
			limit: 12,
			check: func(t *testing.T, chunks []string) {
				if len(chunks) != 2 || chunks[0] != "first line\n" {
					t.Fatalf("chunks = %q, want a cut after the first line", chunks)
				}
				if strings.Join(chunks, "") != "first line\nsecond line" {
					t.Fatalf("chunks lost content: %q", chunks)
				}
			},
		},
		{
			name:  "hard splits without newlines",
			text:  strings.Repeat("a", 25),
			limit: 10,
			check: func(t *testing.T, chunks []string) {
				if len(chunks) != 3 {
					t.Fatalf("chunks = %d, want 3", len(chunks))
				}
				for _, c := range chunks {
					if len([]rune(c)) > 10 {
						t.Errorf("chunk %q exceeds limit", c)
					}
				}
			},
		},
		{
			name:  "does not split runes",
			text:  strings.Repeat("è", 7),
			limit: 3,
			check: func(t *testing.T, chunks []string) {
				if strings.Join(chunks, "") != strings.Repeat("è", 7) {
					t.Fatalf("chunks = %q, runes were corrupted", chunks)
				}
				for _, c := range chunks {
					if len([]rune(c)) > 3 {
						t.Errorf("chunk %q exceeds limit", c)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, splitTelegramMessage(tt.text, tt.limit))
		})
	}
}

func TestParseAllowList(t *testing.T) {
	tests := []struct {
		name      string
		raw       string
		wantIDs   []int64
		wantUsers []string
	}{
		{name: "empty", raw: ""},
		{name: "ids and users", raw: "123, @Alice,456,bob", wantIDs: []int64{123, 456}, wantUsers: []string{"alice", "bob"}},
		{name: "only blanks", raw: " , "},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			orig := botAllowFrom
			botAllowFrom = tt.raw
			t.Cleanup(func() { botAllowFrom = orig })

			ids, users := parseAllowList()
			if len(tt.wantIDs) == 0 && len(tt.wantUsers) == 0 {
				if ids != nil || users != nil {
					t.Fatalf("ids=%v users=%v, want both nil", ids, users)
				}
				return
			}
			for _, id := range tt.wantIDs {
				if !ids[id] {
					t.Errorf("id %d missing from %v", id, ids)
				}
			}
			for _, u := range tt.wantUsers {
				if !users[u] {
					t.Errorf("user %q missing from %v", u, users)
				}
			}
		})
	}
}

func TestIsAllowedMatchesIDOrUsername(t *testing.T) {
	ids := map[int64]bool{1: true}
	users := map[string]bool{"alice": true}

	if !isAllowed(1, "", ids, users) {
		t.Error("allowed chat id was rejected")
	}
	if !isAllowed(2, "Alice", ids, users) {
		t.Error("allowed username was rejected (case-insensitive)")
	}
	if isAllowed(2, "bob", ids, users) {
		t.Error("unknown chat was allowed")
	}
	if !isAllowed(99, "", nil, nil) {
		t.Error("empty allowlist should allow everyone")
	}
}

func TestStripCodeFences(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"plain", "plain"},
		{"```json\n{\"a\":1}\n```", "{\"a\":1}"},
		{"```\nline\n```", "line"},
		{"  ```go\nfmt.Println()\n```\n", "fmt.Println()"},
	}
	for _, tt := range tests {
		if got := stripCodeFences(tt.in); got != tt.want {
			t.Errorf("stripCodeFences(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestLoadAndSaveHistory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history")

	if got := loadHistory(path); len(got) != 0 {
		t.Fatalf("missing file history = %v, want empty", got)
	}

	want := []string{"first", "second"}
	if err := saveHistory(path, want); err != nil {
		t.Fatalf("saveHistory() = %v", err)
	}
	if got := loadHistory(path); !slices.Equal(got, want) {
		t.Errorf("loadHistory() = %v, want %v", got, want)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	if !strings.HasSuffix(string(data), "\n") {
		t.Errorf("history file lacks a trailing newline: %q", data)
	}

	if err := saveHistory(path, nil); err != nil {
		t.Fatalf("saveHistory(empty) = %v", err)
	}
	if got := loadHistory(path); len(got) != 0 {
		t.Errorf("history after clearing = %v, want empty", got)
	}
}

func TestEstimateTokens(t *testing.T) {
	if got := estimateTokens(""); got != 0 {
		t.Errorf("estimateTokens(\"\") = %d, want 0", got)
	}
	if got := estimateTokens("12345"); got != 2 {
		t.Errorf("estimateTokens(5 chars) = %d, want 2", got)
	}
}

func TestCompleteFiles(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"main.go", "notes.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}

	m := &ShellModel{}
	got := m.completeFiles(dir, "ma")
	if !slices.Equal(got, []string{filepath.Join(dir, "main.go")}) {
		t.Errorf("completeFiles(ma) = %v, want [main.go]", got)
	}

	got = m.completeFiles(dir, "ne")
	if len(got) != 1 || !strings.HasSuffix(got[0], "nested/") {
		t.Errorf("completeFiles(ne) = %v, want a directory entry with a trailing slash", got)
	}

	if got := m.completeFiles(filepath.Join(dir, "missing"), "x"); len(got) != 0 {
		t.Errorf("completeFiles(missing dir) = %v, want empty", got)
	}
}

func TestIsKnownTool(t *testing.T) {
	for _, name := range []string{"RunCommand", "WriteFile", "ReadFile", "KVSet", "KVGet", "KVList"} {
		if !isKnownTool(name) {
			t.Errorf("isKnownTool(%q) = false, want true", name)
		}
	}
	if isKnownTool("Nope") || isKnownTool("") {
		t.Error("isKnownTool accepted an unknown tool")
	}
}

func TestResolveCommitDecisionModel(t *testing.T) {
	orig := commitDecisionModel
	t.Cleanup(func() { commitDecisionModel = orig })

	commitDecisionModel = "flag-model"
	if got := resolveCommitDecisionModel(&config.Config{}); got != "flag-model" {
		t.Errorf("flag override = %q, want flag-model", got)
	}

	commitDecisionModel = ""
	cfg := &config.Config{}
	cfg.LLM.DecisionModel = "cfg-model"
	if got := resolveCommitDecisionModel(cfg); got != "cfg-model" {
		t.Errorf("config default = %q, want cfg-model", got)
	}

	if got := resolveCommitDecisionModel(&config.Config{}); got != "" {
		t.Errorf("no decision model = %q, want empty", got)
	}
}
