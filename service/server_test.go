package service

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"edgebot/llm"
)

// startTestServer runs a Server on a fresh temp socket and swaps the package
// socket path for the duration of the test. Returns the socket path and a
// channel that is closed (with any Serve error) when the server exits.
func startTestServer(t *testing.T, srv *Server) (string, <-chan error) {
	t.Helper()

	path := filepath.Join(t.TempDir(), "service.sock")
	old := socketPathFunc
	socketPathFunc = func() (string, error) { return path, nil }
	t.Cleanup(func() { socketPathFunc = old })

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	done := make(chan error, 1)
	go func() { done <- Serve(ctx, srv) }()

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return path, done
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("service socket did not appear")
	return "", nil
}

func newTestClient(t *testing.T) *Client {
	t.Helper()
	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func TestServerPingAndStop(t *testing.T) {
	srv := NewServer()
	path, done := startTestServer(t, srv)

	if !IsActive() {
		t.Fatalf("IsActive() = false, want true (socket %s)", path)
	}

	c := newTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := c.Ping(ctx)
	if err != nil {
		t.Fatalf("Ping: %v", err)
	}
	if resp.Version != ServiceVersion {
		t.Errorf("version = %q, want %q", resp.Version, ServiceVersion)
	}

	if err := c.Stop(ctx); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Serve returned: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("service did not stop after Stop RPC")
	}

	if IsActive() {
		t.Error("IsActive() = true after stop, want false")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("socket file not removed after stop (err = %v)", err)
	}
}

func TestServerChatBuildsAgent(t *testing.T) {
	var gotAgent *llm.Agent
	srv := NewServer()
	srv.callLLM = func(_ context.Context, agent *llm.Agent, _ llm.ToolExecutor, messages []llm.Message) ([]llm.Message, error) {
		gotAgent = agent
		return messages, nil
	}
	startTestServer(t, srv)

	c := newTestClient(t)
	ctx := context.Background()
	result, err := c.Chat(ctx, ChatRequest{
		Messages:   []llm.Message{{Role: "user", Content: "hi"}},
		Agent:      "plan",
		Model:      "model-x",
		Provider:   "ollama",
		Tools:      map[string]bool{"ReadFile": true},
		AgentFiles: false,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(result) != 1 || result[0].Content != "hi" {
		t.Errorf("result = %+v, want the single echoed message", result)
	}
	if gotAgent == nil {
		t.Fatal("callLLM was not invoked")
	}
	if gotAgent.Prompt == "" {
		t.Error("agent prompt is empty, want the rendered agent prompt")
	}
	if gotAgent.Model != "model-x" || gotAgent.Provider != "ollama" {
		t.Errorf("agent = %+v, want model-x/ollama", gotAgent)
	}
}

func TestServerChatSystemPromptOverride(t *testing.T) {
	var gotAgent *llm.Agent
	srv := NewServer()
	srv.callLLM = func(_ context.Context, agent *llm.Agent, _ llm.ToolExecutor, messages []llm.Message) ([]llm.Message, error) {
		gotAgent = agent
		return []llm.Message{{Role: "assistant", Content: "done"}}, nil
	}
	startTestServer(t, srv)

	c := newTestClient(t)
	result, err := c.Chat(context.Background(), ChatRequest{
		Messages:     []llm.Message{{Role: "user", Content: "x"}},
		SystemPrompt: "custom system prompt",
		Model:        "m",
		Provider:     "ollama",
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(result) != 1 || result[0].Content != "done" {
		t.Errorf("result = %+v, want the fake response", result)
	}
	if gotAgent == nil {
		t.Fatal("callLLM was not invoked")
	}
	if gotAgent.Prompt != "custom system prompt" {
		t.Errorf("prompt = %q, want the system prompt override", gotAgent.Prompt)
	}
	if gotAgent.AgentFiles != "" {
		t.Errorf("agent files should be empty for system prompt override, got %q", gotAgent.AgentFiles)
	}
	if gotAgent.Skills != "" {
		t.Errorf("skills should be empty for system prompt override, got %q", gotAgent.Skills)
	}
	if len(gotAgent.Tools) != 0 {
		t.Errorf("tools should be empty for system prompt override, got %d", len(gotAgent.Tools))
	}
}

func TestServerChatThinkEffortPassthrough(t *testing.T) {
	var gotAgent *llm.Agent
	srv := NewServer()
	srv.callLLM = func(_ context.Context, agent *llm.Agent, _ llm.ToolExecutor, messages []llm.Message) ([]llm.Message, error) {
		gotAgent = agent
		return messages, nil
	}
	startTestServer(t, srv)

	c := newTestClient(t)
	_, err := c.Chat(context.Background(), ChatRequest{
		Messages:    []llm.Message{{Role: "user", Content: "x"}},
		Agent:       "build",
		Model:       "m",
		Provider:    "ollama",
		ThinkEffort: "low",
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if gotAgent == nil {
		t.Fatal("callLLM was not invoked")
	}
	if gotAgent.ThinkEffort != llm.ThinkEffortLow {
		t.Errorf("ThinkEffort = %q, want low", gotAgent.ThinkEffort)
	}
}

func TestServerChatInvalidThinkEffort(t *testing.T) {
	srv := NewServer()
	startTestServer(t, srv)

	c := newTestClient(t)
	_, err := c.Chat(context.Background(), ChatRequest{
		Messages:    []llm.Message{{Role: "user", Content: "x"}},
		Agent:       "build",
		Model:       "m",
		Provider:    "ollama",
		ThinkEffort: "ultra",
	})
	if err == nil {
		t.Fatal("Chat: want error for invalid think effort, got nil")
	}
}

func TestServerChatLLMErrorIsReturned(t *testing.T) {
	srv := NewServer()
	srv.callLLM = func(_ context.Context, _ *llm.Agent, _ llm.ToolExecutor, _ []llm.Message) ([]llm.Message, error) {
		return nil, os.ErrNotExist
	}
	startTestServer(t, srv)

	c := newTestClient(t)
	_, err := c.Chat(context.Background(), ChatRequest{
		Messages: []llm.Message{{Role: "user", Content: "x"}},
		Agent:    "build",
		Model:    "m",
		Provider: "ollama",
	})
	if err == nil {
		t.Fatal("Chat: want error from LLM call, got nil")
	}
	if err == ErrUnavailable {
		t.Fatalf("Chat: got ErrUnavailable, want the underlying LLM error (%v)", err)
	}
}

func TestChatUnavailable(t *testing.T) {
	// Point the socket at a path where nothing listens.
	path := filepath.Join(t.TempDir(), "missing.sock")
	old := socketPathFunc
	socketPathFunc = func() (string, error) { return path, nil }
	t.Cleanup(func() { socketPathFunc = old })

	if IsActive() {
		t.Error("IsActive() = true for a missing socket, want false")
	}

	_, err := Chat(context.Background(), ChatRequest{
		Messages: []llm.Message{{Role: "user", Content: "x"}},
		Agent:    "build",
		Model:    "m",
		Provider: "ollama",
	})
	if !errors.Is(err, ErrUnavailable) {
		t.Errorf("Chat err = %v, want ErrUnavailable", err)
	}
}
