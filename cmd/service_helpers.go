package cmd

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/service"
)

// chatRequestFromConfig builds a service.ChatRequest carrying the session
// configuration (agent, model, provider, tools, backend, think effort,
// AGENTS.md and skills support, and confirmation policy) from a loaded config.
func chatRequestFromConfig(cfg *config.Config, messages []llm.Message) service.ChatRequest {
	return service.ChatRequest{
		Messages:        messages,
		Agent:           cfg.Agent,
		Model:           cfg.LLM.Model,
		Provider:        cfg.LLM.Provider,
		Tools:           cfg.Tools,
		Backend:         cfg.LitertLM.Backend,
		AgentFiles:      cfg.AgentFiles,
		Skills:          cfg.Skills,
		Confirm:         cfg.Shell.Confirm,
		AllowedCommands: append([]string(nil), cfg.Shell.AllowedCommands...),
		ThinkEffort:     cfg.LLM.ThinkEffort,
	}
}

// chatWithServiceFallback routes a chat request through the running service
// when active, falling back to local execution when the service is
// unreachable. wrap, if non-nil, is applied to service-originated errors that
// are not ErrUnavailable. ctx bounds the request, so caller cancellation
// propagates to the service instead of being masked by a background context.
func chatWithServiceFallback(ctx context.Context, req service.ChatRequest, wrap func(error) error, local func() ([]llm.Message, error)) ([]llm.Message, error) {
	if !service.IsActive() {
		return local()
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	result, err := service.Chat(ctx, req)
	if err == nil {
		return result, nil
	}
	if !errors.Is(err, service.ErrUnavailable) {
		if wrap != nil {
			return nil, wrap(err)
		}
		return nil, err
	}
	slog.Debug("service unavailable, falling back to local execution", "err", err)
	return local()
}
