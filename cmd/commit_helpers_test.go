package cmd

import (
	"strings"
	"testing"
)

func TestTruncateDiff(t *testing.T) {
	short := "small diff"
	if got := truncateDiff(short, 8000); got != short {
		t.Errorf("truncateDiff(short) = %q, want unchanged", got)
	}

	exact := strings.Repeat("a", 100)
	if got := truncateDiff(exact, 100); got != exact {
		t.Errorf("truncateDiff(exact) changed input of exactly maxChars")
	}

	long := strings.Repeat("b", 200)
	got := truncateDiff(long, 100)
	if len([]rune(got)) <= 100 {
		t.Errorf("truncateDiff(long) too short: %d runes", len([]rune(got)))
	}
	if !strings.HasPrefix(got, strings.Repeat("b", 100)) {
		t.Errorf("truncateDiff(long) does not keep the head of the diff")
	}
	if !strings.Contains(got, "truncated") {
		t.Errorf("truncateDiff(long) = %q, want truncation marker", got)
	}

	multibyte := strings.Repeat("é", 200)
	got = truncateDiff(multibyte, 100)
	if n := len([]rune(got)); n != 100+len([]rune("\n\n[... diff truncated to fit the model context ...]")) {
		t.Errorf("truncateDiff(multibyte) = %d runes, want rune-safe cut", n)
	}
}

func TestStripThinkBlock(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"no think", "fix: something", "fix: something"},
		{
			"leading think",
			"<think>\nreasoning here\n</think>\n\nfix: something",
			"fix: something",
		},
		{
			"multiline think",
			"a\n<think>line1\nline2</think>\nfix: something",
			"a\n\nfix: something",
		},
		{
			"unclosed think at start drops everything",
			"<think>\nfix: something",
			"",
		},
		{
			"unclosed think drops the thought",
			"fix: preamble\n<think>\nendless reasoning without end",
			"fix: preamble",
		},
		{
			"gemma channel block",
			"<|channel>thought\nreasoning here\n<channel|>fix: something",
			"fix: something",
		},
		{
			"gemma channel block multiline",
			"<|channel>thought\nline1\nline2\n<channel|>\n\nfix: something\n\nwith body",
			"fix: something\n\nwith body",
		},
		{
			"gemma unclosed channel drops the thought",
			"fix: preamble\n<|channel>thought\nendless reasoning without end",
			"fix: preamble",
		},
		{
			"gemma unclosed channel at start drops everything",
			"<|channel>thought\nfix: something",
			"",
		},
		{
			"gemma sentinel tokens",
			"<|turn>user\nprompt<turn|>\n<|turn>model\nfix: something<turn|><eos>",
			"prompt\n\nfix: something",
		},
		{
			"gemma bare thought label",
			"thought\nfix: something",
			"fix: something",
		},
		{
			"empty",
			"",
			"",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stripThinkBlock(tt.input); got != tt.want {
				t.Errorf("stripThinkBlock(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
