package cmd

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func FuzzSplitTelegramMessage(f *testing.F) {
	f.Add("hello world", 20)
	f.Add(strings.Repeat("a\nb", 50), 16)
	f.Add("è", 1)

	f.Fuzz(func(t *testing.T, text string, limit int) {
		// Telegram message text is JSON-decoded UTF-8; converting invalid
		// input to runes would replace bytes and make the invariant moot.
		if !utf8.ValidString(text) {
			return
		}
		if limit < 1 {
			limit = 1
		}
		if limit > 200 {
			limit = 200
		}

		chunks := splitTelegramMessage(text, limit)
		if strings.Join(chunks, "") != text {
			t.Fatalf("chunks lost content: %q from %q", chunks, text)
		}
		for _, c := range chunks {
			if n := utf8.RuneCountInString(c); n > limit {
				t.Fatalf("chunk of %d runes exceeds limit %d", n, limit)
			}
		}
	})
}

func FuzzWrapShellText(f *testing.F) {
	f.Add("hello world this is a long line that should wrap", 20)
	f.Add("short", 80)
	f.Add("multi\nline\ntext", 5)
	f.Add("unbrokenwordthatisverylongindeed", 3)

	f.Fuzz(func(t *testing.T, text string, width int) {
		if width < 1 {
			width = 1
		}
		if width > 200 {
			width = 200
		}

		for _, line := range strings.Split(wrapShellText(text, width), "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			if n := utf8.RuneCountInString(line); n > width {
				t.Fatalf("line of %d runes exceeds width %d: %q", n, width, line)
			}
		}
	})
}

func FuzzStripThinkBlock(f *testing.F) {
	f.Add("plain")
	f.Add("a <tag> b")
	f.Add("")
	f.Add("thought\ncontent")

	f.Fuzz(func(t *testing.T, s string) {
		_ = stripThinkBlock(s)
	})
}
