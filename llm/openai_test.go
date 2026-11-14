package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNewOpenAICaller(t *testing.T) {
	caller := NewOpenAICaller("http://example.com/v1", "test-api-key", "test-model", nil)

	if caller == nil {
		t.Fatal("NewOpenAICaller() returned nil")
	}
	if caller.BaseURL != "http://example.com/v1" {
		t.Errorf("BaseURL = %q, want %q", caller.BaseURL, "http://example.com/v1")
	}
	if caller.APIKey != "test-api-key" {
		t.Errorf("APIKey = %q, want %q", caller.APIKey, "test-api-key")
	}
	if caller.Model != "test-model" {
		t.Errorf("Model = %q, want %q", caller.Model, "test-model")
	}
	if caller.Client == nil {
		t.Error("Client is nil")
	}
	if caller.Executor != nil {
		t.Error("Executor should be nil when nil is passed")
	}
}

func TestNewOpenAICallerWithExecutor(t *testing.T) {
	mock := &mockExecutor{}
	caller := NewOpenAICaller("http://example.com/v1", "key", "model", mock)

	if caller.Executor != mock {
		t.Error("Executor not set correctly")
	}
}

func TestOpenAIRequestSerialization(t *testing.T) {
	req := OpenAIRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
		Tools: []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": "RunCommand",
				},
			},
		},
		Temperature: 0.7,
	}

	if req.Model != "test-model" {
		t.Errorf("Model = %q, want %q", req.Model, "test-model")
	}
	if len(req.Messages) != 2 {
		t.Errorf("Messages length = %d, want 2", len(req.Messages))
	}
	if len(req.Tools) != 1 {
		t.Errorf("Tools length = %d, want 1", len(req.Tools))
	}
	if req.Temperature != 0.7 {
		t.Errorf("Temperature = %f, want 0.7", req.Temperature)
	}
}

func TestOpenAIThinkEffortSent(t *testing.T) {
	var got OpenAIRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OpenAIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		got = req
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"}}]}`))
	}))
	defer srv.Close()

	caller := NewOpenAICallerWithThink(srv.URL, "", "m", &mockExecutor{}, ThinkEffortLow)
	if _, err := caller.Call(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.ReasoningEffort == nil || *got.ReasoningEffort != "low" {
		t.Errorf("reasoning_effort = %+v, want low", got.ReasoningEffort)
	}
	if got.Reasoning != nil {
		t.Errorf("reasoning = %+v, want omitted (non-OpenRouter sends reasoning_effort only)", got.Reasoning)
	}
}

func TestOpenAIThinkEffortSentOpenRouter(t *testing.T) {
	caller := NewOpenAICallerWithThink(
		"https://openrouter.ai/api/v1",
		"", "m", &mockExecutor{}, ThinkEffortLow,
	)
	reasoningEffort, reasoning := caller.reasoningFields()
	if reasoningEffort != nil {
		t.Errorf("reasoning_effort = %+v, want omitted on OpenRouter", reasoningEffort)
	}
	if reasoning == nil || reasoning.Effort != "low" {
		t.Errorf("reasoning = %+v, want {low} on OpenRouter", reasoning)
	}
}

func TestOpenAIThinkEffortSentGemini(t *testing.T) {
	caller := NewOpenAICallerWithThink(
		"https://generativelanguage.googleapis.com/v1beta/openai",
		"", "m", &mockExecutor{}, ThinkEffortLow,
	)
	reasoningEffort, reasoning := caller.reasoningFields()
	if reasoningEffort == nil || *reasoningEffort != "low" {
		t.Errorf("reasoning_effort = %+v, want low on Gemini", reasoningEffort)
	}
	if reasoning != nil {
		t.Errorf("reasoning = %+v, want omitted on Gemini (rejected with 400)", reasoning)
	}
}

func TestOpenAIThinkEffortOmittedByDefault(t *testing.T) {
	var got OpenAIRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OpenAIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		got = req
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi"}}]}`))
	}))
	defer srv.Close()

	caller := NewOpenAICaller(srv.URL, "", "m", &mockExecutor{})
	if _, err := caller.Call(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got.ReasoningEffort != nil {
		t.Errorf("reasoning_effort = %q, want omitted", *got.ReasoningEffort)
	}
	if got.Reasoning != nil {
		t.Errorf("reasoning = %+v, want omitted", got.Reasoning)
	}
}

func TestProviderCallerWithThinkPropagates(t *testing.T) {
	c := NewProviderCallerWithThink("ollama", "m", nil, ThinkEffortHigh)
	oc, ok := c.(*OpenAICaller)
	if !ok {
		t.Fatalf("caller type = %T, want *OpenAICaller", c)
	}
	if oc.ThinkEffort != ThinkEffortHigh {
		t.Errorf("ThinkEffort = %q, want high", oc.ThinkEffort)
	}
	lc := NewProviderCallerWithThink("llamacpp", "m", nil, ThinkEffortNone)
	llc, ok := lc.(*LlamacppCaller)
	if !ok {
		t.Fatalf("caller type = %T, want *LlamacppCaller", lc)
	}
	if !llc.NoThink {
		t.Error("llamacpp think=none should set NoThink")
	}
}

type mockExecutor struct{}

func (m *mockExecutor) ExecuteTool(context.Context, ToolCall) (string, error) {
	return "mock output", nil
}

func (m *mockExecutor) IsAllowedCommand(cmd string) bool {
	return true
}

func (m *mockExecutor) AskConfirmation(cmd string) bool {
	return true
}

func TestThoughtSignatureUnmarshalPreserved(t *testing.T) {
	raw := `{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"ReadFile","arguments":"{}"},"extra_content":{"google":{"thought_signature":"SIG_A"}}}]}` //nolint:lll
	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(msg.ToolCalls))
	}
	tc := msg.ToolCalls[0]
	if tc.ExtraContent == nil || tc.ExtraContent.Google == nil ||
		tc.ExtraContent.Google.ThoughtSignature != "SIG_A" {
		t.Fatalf("thought signature not preserved: %+v", tc.ExtraContent)
	}
	out, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var roundTrip Message
	if err := json.Unmarshal(out, &roundTrip); err != nil {
		t.Fatalf("re-unmarshal: %v", err)
	}
	sig := roundTrip.ToolCalls[0].ExtraContent.Google.ThoughtSignature
	if sig != "SIG_A" {
		t.Errorf("round-trip signature = %q, want SIG_A", sig)
	}
}

func TestEnsureGeminiThoughtSignatures(t *testing.T) {
	mkCall := func(sig string) OpenAIToolCall {
		var tc OpenAIToolCall
		tc.ID = "call_1"
		tc.Type = "function"
		tc.Function.Name = "ReadFile"
		tc.Function.Arguments = "{}"
		if sig != "" {
			tc.ExtraContent = &ExtraContent{
				Google: &GoogleExtraContent{ThoughtSignature: sig},
			}
		}
		return tc
	}

	// Existing signatures are preserved verbatim.
	msgs := []Message{{Role: "assistant", ToolCalls: []OpenAIToolCall{mkCall("SIG_A")}}}
	ensureGeminiThoughtSignatures(msgs)
	if got := msgs[0].ToolCalls[0].ExtraContent.Google.ThoughtSignature; got != "SIG_A" {
		t.Errorf("existing signature = %q, want SIG_A", got)
	}

	// Missing signatures get the skip sentinel so old histories don't 400.
	msgs = []Message{{Role: "assistant", ToolCalls: []OpenAIToolCall{mkCall("")}}}
	ensureGeminiThoughtSignatures(msgs)
	if got := msgs[0].ToolCalls[0].ExtraContent.Google.ThoughtSignature; got != geminiSkipThoughtSignature {
		t.Errorf("backfilled signature = %q, want %q", got, geminiSkipThoughtSignature)
	}

	// Parallel calls: only the first needs a signature; the rest stay empty.
	msgs = []Message{{Role: "assistant", ToolCalls: []OpenAIToolCall{mkCall("SIG_A"), mkCall("")}}}
	ensureGeminiThoughtSignatures(msgs)
	if got := msgs[0].ToolCalls[0].ExtraContent.Google.ThoughtSignature; got != "SIG_A" {
		t.Errorf("first signature = %q, want SIG_A", got)
	}
	if msgs[0].ToolCalls[1].ExtraContent != nil {
		t.Errorf("second parallel call should stay signature-free, got %+v",
			msgs[0].ToolCalls[1].ExtraContent)
	}

	// Messages without tool calls are untouched.
	msgs = []Message{{Role: "user", Content: "hi"}}
	ensureGeminiThoughtSignatures(msgs)
	if msgs[0].ExtraContent != nil {
		t.Errorf("non-tool message should stay untouched, got %+v", msgs[0].ExtraContent)
	}
}

func TestGeminiSecondHopReplaysSignature(t *testing.T) {
	var secondHop OpenAIRequest
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OpenAIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		calls++
		w.Header().Set("Content-Type", "application/json")
		if calls == 1 {
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"ReadFile","arguments":"{}"},"extra_content":{"google":{"thought_signature":"SIG_A"}}}]}}]}`)) //nolint:lll
			return
		}
		secondHop = req
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"done"}}]}`))
	}))
	defer srv.Close()

	// Signature preservation is provider-agnostic: the assistant message
	// returned by hop 0 must be replayed verbatim on hop 1.
	caller := NewOpenAICaller(srv.URL, "", "m", &mockExecutor{})
	if _, err := caller.Call(context.Background(), "sys",
		[]Message{{Role: "user", Content: "hi"}}, nil); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
	found := false
	for _, m := range secondHop.Messages {
		for _, tc := range m.ToolCalls {
			if tc.ExtraContent != nil && tc.ExtraContent.Google != nil &&
				tc.ExtraContent.Google.ThoughtSignature == "SIG_A" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("second hop replayed messages lack thought_signature SIG_A: %+v",
			secondHop.Messages)
	}
}

func TestOpenAIRetriesTransientFailures(t *testing.T) {
	orig := openAIRetryBaseDelay
	openAIRetryBaseDelay = time.Millisecond
	t.Cleanup(func() { openAIRetryBaseDelay = orig })

	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("slow down"))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	caller := NewOpenAICaller(srv.URL, "", "m", NoopExecutor{})
	msgs, err := caller.Call(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Call() = %v", err)
	}
	if attempts != 3 {
		t.Errorf("attempts = %d, want 3 (two retries)", attempts)
	}
	if len(msgs) == 0 || msgs[len(msgs)-1].Content != "ok" {
		t.Errorf("messages = %+v, want final content %q", msgs, "ok")
	}
}

func TestOpenAINoRetryOnClientError(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(strings.Repeat("x", 5000)))
	}))
	defer srv.Close()

	caller := NewOpenAICaller(srv.URL, "", "m", NoopExecutor{})
	_, err := caller.Call(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Call() = nil error, want API error")
	}
	if attempts != 1 {
		t.Errorf("attempts = %d, want 1 (4xx must not be retried)", attempts)
	}
	if len(err.Error()) > openAIMaxErrorBodyLen+200 {
		t.Errorf("error message is not bounded: %d bytes", len(err.Error()))
	}
}

func TestOpenAICancelStopsRetries(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	caller := NewOpenAICaller("http://127.0.0.1:1", "", "m", NoopExecutor{})
	_, err := caller.Call(ctx, "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Call() error = %v, want context.Canceled", err)
	}
}

func TestParseRetryAfter(t *testing.T) {
	if got := parseRetryAfter("2"); got != 2*time.Second {
		t.Errorf("parseRetryAfter(2) = %v, want 2s", got)
	}
	if got := parseRetryAfter(""); got != 0 {
		t.Errorf("parseRetryAfter(empty) = %v, want 0", got)
	}
	if got := parseRetryAfter("not-a-date"); got != 0 {
		t.Errorf("parseRetryAfter(invalid) = %v, want 0", got)
	}
}

func TestOpenAIRetryDelayJitterRange(t *testing.T) {
	orig := openAIRetryBaseDelay
	openAIRetryBaseDelay = 100 * time.Millisecond
	t.Cleanup(func() { openAIRetryBaseDelay = orig })

	for attempt := 0; attempt < 3; attempt++ {
		base := openAIRetryBaseDelay * time.Duration(1<<attempt)
		for range 20 {
			got := openAIRetryDelay(attempt)
			if got < base*3/4 || got > base*5/4 {
				t.Fatalf("openAIRetryDelay(%d) = %v, want between %v and %v", attempt, got, base*3/4, base*5/4)
			}
		}
	}
}
