package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/service"
)

func (e *ShellExecutorForLLM) ExecuteTool(ctx context.Context, call llm.ToolCall) (string, error) {
	policy := &llm.ToolExecutorPolicy{
		CommandTimeout: e.m.cfg.Shell.CommandTimeout,
		ReadDenylist:   e.m.cfg.Shell.ReadDenylist,
		ConfirmCommand: func(cmd string) bool {
			if !e.m.cfg.Shell.Confirm {
				return true
			}
			if config.IsAllowedCommand(cmd, e.m.cfg.Shell.AllowedCommands) {
				return true
			}
			return e.AskConfirmation(cmd)
		},
		ConfirmWriteFile: func(path string) bool {
			if !e.m.cfg.Shell.Confirm {
				return true
			}
			return e.AskConfirmation(fmt.Sprintf("Write to file %s?", path))
		},
		OnExecute: func(call llm.ToolCall) {
			e.m.mu.Lock()
			e.m.messages = append(e.m.messages, Message{role: "assistant", content: systemStyle.Render(e.toolActionMessage(call))})
			e.m.mu.Unlock()
		},
	}
	return policy.ExecuteTool(ctx, call)
}

// toolActionMessage renders the status line shown in the transcript before a
// tool executes.
func (e *ShellExecutorForLLM) toolActionMessage(call llm.ToolCall) string {
	switch call.Name {
	case "RunCommand":
		cmd, ok := call.Arguments["command"].(string)
		if !ok {
			return "[Executing: <invalid args>]"
		}
		return fmt.Sprintf("[Executing: %s]", cmd)
	case "WriteFile":
		path, ok := call.Arguments["path"].(string)
		if !ok {
			return "[Writing to file: <invalid args>]"
		}
		return fmt.Sprintf("[Writing to file: %s]", strings.TrimPrefix(path, "@"))
	case "ReadFile":
		path, ok := call.Arguments["path"].(string)
		if !ok {
			return "[Reading file: <invalid args>]"
		}
		return fmt.Sprintf("[Reading file: %s]", strings.TrimPrefix(path, "@"))
	case "KVSet":
		key, ok := call.Arguments["key"].(string)
		if !ok {
			return "[KV Store: Saving <invalid args>]"
		}
		return fmt.Sprintf("[KV Store: Saving %s]", key)
	case "KVGet":
		key, ok := call.Arguments["key"].(string)
		if !ok {
			return "[KV Store: Retrieving <invalid args>]"
		}
		return fmt.Sprintf("[KV Store: Retrieving %s]", key)
	case "KVList":
		return "[KV Store: Listing keys]"
	}
	return ""
}

func (e *ShellExecutorForLLM) IsAllowedCommand(cmd string) bool {
	return config.IsAllowedCommand(cmd, e.m.cfg.Shell.AllowedCommands)
}

func (e *ShellExecutorForLLM) AskConfirmation(cmd string) bool {
	m := e.m
	m.mu.Lock()
	m.pendingCommand = cmd
	m.waitingConfirm = true
	cancelChan := m.cancelChan
	m.mu.Unlock()

	if m.teaProgram != nil {
		m.teaProgram.Send(confirmationMsg{cmd: cmd})
	}
	if cancelChan == nil {
		m.mu.Lock()
		m.waitingConfirm = false
		m.mu.Unlock()
		return false
	}

	select {
	case result := <-m.confirmationChan:
		return result
	case <-cancelChan:
		m.mu.Lock()
		m.waitingConfirm = false
		m.mu.Unlock()
		return false
	}
}

func (m *ShellModel) ElaborateMessage(reqID uint64) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	m.mu.Lock()
	cancelChan := m.cancelChan
	m.mu.Unlock()
	if cancelChan != nil {
		go func() {
			select {
			case <-cancelChan:
				cancel()
			case <-ctx.Done():
			}
		}()
	}

	var commonMessages []llm.Message
	var structuredSchema string
	// Copy messages under lock to avoid race with Update goroutine.
	m.mu.Lock()
	msgsCopy := append([]Message(nil), m.messages...)
	m.mu.Unlock()
	for _, msg := range msgsCopy {
		if msg.role == "user" || msg.role == "assistant" || msg.role == "tool" {
			if msg.schema != "" && msg.role == "user" {
				structuredSchema = msg.schema
			}
			var content any = msg.content
			if len(msg.images) > 0 {
				parts := []llm.ContentPart{
					{Type: "text", Text: msg.content},
				}
				for _, img := range msg.images {
					parts = append(parts, llm.ContentPart{
						Type:     "image_url",
						ImageURL: &llm.ContentImage{URL: img},
					})
				}
				content = parts
			}
			commonMessages = append(commonMessages, llm.Message{Role: msg.role, Content: content, ToolCallID: msg.toolCallID, ToolCalls: msg.toolCalls})
		}
	}

	if structuredSchema != "" {
		m.runStructuredMessage(ctx, reqID, commonMessages, structuredSchema)
		return
	}

	if m.serviceClient != nil || service.IsActive() {
		resultMessages, err := m.serviceChat(ctx, commonMessages)
		if err != nil {
			switch {
			case isRequestCancellation(err):
				// The user cancelled; handleEscapeKey already reported it.
				m.finishCancelled(reqID)
				return
			case errors.Is(err, service.ErrUnavailable):
				slog.Debug("service unavailable, falling back to local execution", "err", err)
			default:
				m.failWithError(reqID, fmt.Sprintf("Error: %v", err))
				return
			}
		} else {
			m.appendLLMResult(resultMessages)
			m.finishSuccess(reqID)
			return
		}
	}

	agent := llm.NewAgentForSession(m.cfg.Agent, m.cfg.LLM.Model, m.cfg.LLM.Provider, m.cfg.Tools, m.cfg.LitertLM.Backend, m.cfg.AgentFiles, m.cfg.Skills, llm.ThinkEffort(m.cfg.LLM.ThinkEffort))

	executor := &ShellExecutorForLLM{m: m}

	resultMessages, err := agent.CallLLM(ctx, executor, commonMessages)

	if err != nil {
		if isRequestCancellation(err) {
			m.finishCancelled(reqID)
			return
		}
		m.failWithError(reqID, fmt.Sprintf("Error: %v", err))
		return
	}

	m.appendLLMResult(resultMessages)
	m.finishSuccess(reqID)
}

// beginRequest allocates a new request id and marks the shell as loading.
// Every request mutation is tagged with its id so a superseded goroutine can
// no longer append results or clear the loading state of a newer request.
func (m *ShellModel) beginRequest() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.loading = true
	m.cancelChan = make(chan struct{})
	m.cancelOnce = sync.Once{}
	m.reqSeq++
	return m.reqSeq
}

// isCurrentRequest reports whether reqID is still the active request.
func (m *ShellModel) isCurrentRequest(reqID uint64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.reqSeq == reqID
}

// isRequestCancellation reports whether err is the result of the user
// cancelling the current request (Esc / Ctrl+C). Cancellation is not a
// failure: the transcript already shows "Request cancelled.".
func isRequestCancellation(err error) bool {
	return errors.Is(err, context.Canceled)
}

// finishCancelled clears the loading state after a cancellation without
// appending an error bubble.
func (m *ShellModel) finishCancelled(reqID uint64) {
	m.mu.Lock()
	if m.reqSeq != reqID {
		m.mu.Unlock()
		return
	}
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

func (m *ShellModel) failWithError(reqID uint64, content string) {
	m.mu.Lock()
	if m.reqSeq != reqID {
		m.mu.Unlock()
		return
	}
	m.messages = append(m.messages, Message{role: "error", content: content})
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

func (m *ShellModel) finishSuccess(reqID uint64) {
	m.mu.Lock()
	if m.reqSeq != reqID {
		m.mu.Unlock()
		return
	}
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

// serviceChat routes a request through the running service, maintaining a
// cached client on the shell. Connectivity failures (ErrUnavailable) clear
// the cached client so the caller can fall back to local execution.
func (m *ShellModel) serviceChat(ctx context.Context, messages []llm.Message) ([]llm.Message, error) {
	if m.serviceClient == nil {
		c, err := service.NewClient()
		if err != nil {
			return nil, service.ErrUnavailable
		}
		m.serviceClient = c
	}

	req := chatRequestFromConfig(m.cfg, messages)

	result, err := m.serviceClient.Chat(ctx, req)
	if err != nil {
		if errors.Is(err, service.ErrUnavailable) {
			if closeErr := m.serviceClient.Close(); closeErr != nil {
				slog.Warn("service client close failed", "err", closeErr)
			}
			m.serviceClient = nil
		}
		return nil, err
	}
	return result, nil
}

// runStructuredMessage executes a structured command (a command carrying a
// JSON schema in its frontmatter) through CallStructured, bypassing the
// service like extract did, and renders the JSON result.
func (m *ShellModel) runStructuredMessage(ctx context.Context, reqID uint64, messages []llm.Message, schemaPath string) {
	safeAppendError := func(text string) {
		m.mu.Lock()
		if m.reqSeq != reqID {
			m.mu.Unlock()
			return
		}
		m.messages = append(m.messages, Message{role: "error", content: text})
		m.loading = false
		m.mu.Unlock()
		if m.teaProgram != nil {
			m.teaProgram.Send(responseReadyMsg{})
		}
	}
	cleanPath := filepath.Clean(schemaPath)
	if !filepath.IsAbs(cleanPath) {
		if cwd, err := os.Getwd(); err == nil {
			cleanPath = filepath.Join(cwd, cleanPath)
		}
	}
	schemaData, err := os.ReadFile(cleanPath)
	if err != nil {
		safeAppendError(fmt.Sprintf("Error: failed to read schema: %v", err))
		return
	}

	var schemaRaw any
	if err := json.Unmarshal(schemaData, &schemaRaw); err != nil {
		safeAppendError(fmt.Sprintf("Error: invalid JSON schema: %v", err))
		return
	}

	responseFormat := structuredResponseFormat(schemaRaw)

	systemPrompt := "You extract structured data from documents and images. Return only valid JSON matching the provided schema."

	caller := llm.NewProviderCallerRawWithThink(m.cfg.LLM.Provider, m.cfg.LLM.Model, llm.NoopExecutor{}, llm.ThinkEffort(m.cfg.LLM.ThinkEffort))
	if lc, ok := caller.(*llm.LitertLMCaller); ok {
		lc.Backend = m.cfg.LitertLM.Backend
	}
	resultMessages, err := caller.CallStructured(ctx, systemPrompt, messages, nil, responseFormat)
	if err != nil {
		if isRequestCancellation(err) {
			m.finishCancelled(reqID)
			return
		}
		safeAppendError(fmt.Sprintf("Error: LLM call failed: %v", err))
		return
	}

	if len(resultMessages) == 0 {
		safeAppendError("Error: no response from LLM")
		return
	}

	lastMsg := resultMessages[len(resultMessages)-1]
	contentStr, ok := lastMsg.Content.(string)
	if !ok || strings.TrimSpace(contentStr) == "" {
		safeAppendError("Error: empty response from LLM")
		return
	}

	result := prettyJSONOrRaw(stripThinkBlock(strings.TrimSpace(contentStr)))

	m.mu.Lock()
	if m.reqSeq != reqID {
		m.mu.Unlock()
		return
	}
	m.messages = append(m.messages, Message{role: "assistant", content: result})
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

// appendLLMResult renders the messages returned by the LLM (locally or via
// the service) into the shell transcript.
func (m *ShellModel) appendLLMResult(resultMessages []llm.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, msg := range resultMessages {
		contentStr := ""
		switch v := msg.Content.(type) {
		case string:
			contentStr = v
		case []llm.ContentPart:
			var sb strings.Builder
			for _, p := range v {
				if p.Type == "text" {
					sb.WriteString(p.Text)
				}
			}
			contentStr = sb.String()
		case nil:
			contentStr = ""
		default:
			contentStr = fmt.Sprintf("%v", msg.Content)
		}

		switch msg.Role {
		case "system":
			m.messages = append(m.messages, Message{role: "system", content: contentStr})
		case "user":
			m.messages = append(m.messages, Message{role: "user", content: contentStr})
		case "assistant":
			// Reasoning models leak their thought process into the message
			// (Wake up, Neo... blocks, Gemma channel tokens, unclosed think
			// tags mid-sentence), which must never reach the transcript.
			// Strip it for display only; the API history keeps the original.
			m.messages = append(m.messages, Message{role: "assistant", content: stripThinkBlock(contentStr), toolCalls: msg.ToolCalls})
		case "tool":
			m.messages = append(m.messages, Message{role: "tool", content: contentStr, toolCallID: msg.ToolCallID})
		}
	}
}
