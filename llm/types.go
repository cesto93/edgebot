package llm

import "context"

type Message struct {
	Role         string           `json:"role"`
	Content      any              `json:"content"`
	ToolCalls    []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID   string           `json:"tool_call_id,omitempty"`
	ExtraContent *ExtraContent    `json:"extra_content,omitempty"`
}

type ContentPart struct {
	Type         string        `json:"type"`
	Text         string        `json:"text,omitempty"`
	ImageURL     *ContentImage `json:"image_url,omitempty"`
	InputAudio   *InputAudio   `json:"input_audio,omitempty"`
	ExtraContent *ExtraContent `json:"extra_content,omitempty"`
}

type ContentImage struct {
	URL string `json:"url"`
}

type InputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type OpenAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	ExtraContent *ExtraContent `json:"extra_content,omitempty"`
}

// ExtraContent carries provider-specific extensions on OpenAI-compatible
// messages. Gemini thinking models (2.5/3 series) return
// extra_content.google.thought_signature on tool calls (and on the last part
// of text responses); it must be sent back verbatim on the next turn or
// Gemini 3 rejects the request with
// "Function call is missing a thought_signature in functionCall parts".
type ExtraContent struct {
	Google *GoogleExtraContent `json:"google,omitempty"`
}

type GoogleExtraContent struct {
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

type ToolCall struct {
	Name      string
	Arguments map[string]any
}

type ToolExecutor interface {
	ExecuteTool(ctx context.Context, call ToolCall) (string, error)
	IsAllowedCommand(cmd string) bool
	AskConfirmation(cmd string) bool
}

type Caller interface {
	Call(ctx context.Context, systemPrompt string, messages []Message, tools []any) ([]Message, error)
}

// RawCaller extends Caller with structured output support.
type RawCaller interface {
	Caller
	CallStructured(ctx context.Context, systemPrompt string, messages []Message, tools []any, responseFormat any) ([]Message, error)
}
