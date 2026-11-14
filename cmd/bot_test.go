package cmd

import (
	"testing"

	"edgebot/llm"
)

func TestBotSessionLockForChatIsStable(t *testing.T) {
	s := newBotSession()
	want := s.lockForChat(42)
	if got := s.lockForChat(42); got != want {
		t.Fatal("same chat must map to the same mutex")
	}
	if got := s.lockForChat(42 + botChatLockStripes); got != want {
		t.Fatal("striped chats should reuse a lock")
	}
}

func TestLastAssistantOrLastContent(t *testing.T) {
	tests := []struct {
		name string
		msgs []llm.Message
		want string
	}{
		{
			name: "last assistant wins",
			msgs: []llm.Message{
				{Role: "assistant", Content: "first"},
				{Role: "tool", Content: "tool result"},
				{Role: "assistant", Content: "final"},
			},
			want: "final",
		},
		{
			name: "last assistant beats a trailing user message",
			msgs: []llm.Message{
				{Role: "assistant", Content: "answer"},
				{Role: "user", Content: "follow-up"},
			},
			want: "answer",
		},
		{
			name: "non-string content is stringified",
			msgs: []llm.Message{
				{Role: "assistant", Content: []any{"a"}},
			},
			want: "[a]",
		},
		{
			name: "empty",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := lastAssistantOrLastContent(tt.msgs); got != tt.want {
				t.Errorf("lastAssistantOrLastContent() = %q, want %q", got, tt.want)
			}
		})
	}
}
