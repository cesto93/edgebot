package cmd

import (
	"strings"
	"testing"

	"edgebot/config"
)

func TestTranscriptCacheInvalidates(t *testing.T) {
	m := &ShellModel{cfg: &config.Config{}}
	m.messages = []Message{{role: "assistant", content: "hello"}}

	first := m.transcriptBody(m.messages, 80)
	if !strings.Contains(first, "hello") {
		t.Fatalf("transcript = %q, want it to contain the message", first)
	}
	if again := m.transcriptBody(m.messages, 80); again != first {
		t.Error("unchanged state produced a different transcript")
	}

	t.Run("reset and rewrite", func(t *testing.T) {
		m := &ShellModel{cfg: &config.Config{}}
		m.messages = []Message{{role: "assistant", content: "old"}}
		_ = m.transcriptBody(m.messages, 80)

		m.messages = nil
		_ = m.transcriptBody(m.messages, 80)
		m.messages = []Message{{role: "assistant", content: "new content"}}

		if got := m.transcriptBody(m.messages, 80); !strings.Contains(got, "new content") {
			t.Errorf("transcript after reset = %q, want the new message", got)
		}
	})

	t.Run("width", func(t *testing.T) {
		m := &ShellModel{cfg: &config.Config{}}
		long := strings.Repeat("word ", 40)
		m.messages = []Message{{role: "assistant", content: long}}

		narrow := m.transcriptBody(m.messages, 20)
		wide := m.transcriptBody(m.messages, 200)
		if strings.Count(narrow, "\n") <= strings.Count(wide, "\n") {
			t.Error("width change did not re-render the transcript")
		}
	})

	t.Run("suggestions", func(t *testing.T) {
		m := &ShellModel{cfg: &config.Config{}, showSuggestions: true}
		m.suggestions = []string{"@old.txt"}
		if got := m.transcriptBody(nil, 80); !strings.Contains(got, "@old.txt") {
			t.Fatalf("transcript = %q, want the suggestion", got)
		}

		m.suggestions = []string{"@new.txt"}
		if got := m.transcriptBody(nil, 80); !strings.Contains(got, "@new.txt") {
			t.Errorf("transcript = %q, want the updated suggestion", got)
		}
	})

	t.Run("menu selection", func(t *testing.T) {
		m := &ShellModel{cfg: &config.Config{}}
		m.menu = menuState{kind: menuConfig, options: []string{"one", "two"}}
		m.menu.selectedIdx = 0
		first := m.transcriptBody(nil, 80)

		m.menu.selectedIdx = 1
		if got := m.transcriptBody(nil, 80); got == first {
			t.Error("menu selection change did not re-render the transcript")
		}
	})
}
