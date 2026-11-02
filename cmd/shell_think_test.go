package cmd

import (
	"fmt"
	"testing"

	"edgebot/llm"
)

// Think-tag literals are assembled at runtime from rune values because the
// raw tags must not appear verbatim in this file (they get mangled in
// transit by the same text pipeline we are guarding the transcript against).
var thinkOpen = fmt.Sprintf("%c%s%c", 60, "think", 62)   // '<' "think" '>'
var thinkClose = fmt.Sprintf("%c/%s%c", 60, "think", 62) // '<' '/' "think" '>'

// appendLLMResult must strip think leakage (think blocks, Gemma channel
// tokens, leftover sentinels) from assistant content before it reaches the
// transcript, without mutating the API history it receives.
func TestAppendLLMResultStripsThinkTokens(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "closed think block",
			content: thinkOpen + "\nreasoning here\n" + thinkClose + "\n\nThe answer is 42.",
			want:    "The answer is 42.",
		},
		{
			name:    "gemma channel leak",
			content: "<|channel>thought<channel|>Final answer.",
			want:    "Final answer.",
		},
		{
			name:    "unclosed think tag drops rest",
			content: "Before\n" + thinkOpen + "capped mid-thought",
			want:    "Before",
		},
		{
			name:    "stray closing tag removed",
			content: "answer " + thinkClose + " noise",
			want:    "answer  noise",
		},
		{
			name:    "clean content untouched",
			content: "No thinking here, just <|retro> vibes of plain text.",
			want:    "No thinking here, just <|retro> vibes of plain text.",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			m := &ShellModel{}
			apiMsgs := []llm.Message{{Role: "assistant", Content: tt.content}}

			m.appendLLMResult(apiMsgs)

			if got := m.messages[len(m.messages)-1].content; got != tt.want {
				t.Errorf("displayed content = %q, want %q", got, tt.want)
			}
			// The API history must be untouched so follow-up hops replay it.
			if orig, ok := apiMsgs[0].Content.(string); !ok || orig != tt.content {
				t.Errorf("API history mutated: got %q, want original %q", orig, tt.content)
			}
		})
	}
}
