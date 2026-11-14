package llm

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"time"
)

// openAIDefaultRequestTimeout matches the historical hardcoded client timeout.
const openAIDefaultRequestTimeout = 60 * time.Second

type ProviderConfig struct {
	BaseURL string
	APIKey  string
	// Timeout bounds a single chat-completions request (LLM_TIMEOUT).
	Timeout time.Duration
}

func getProviderConfig(provider string) ProviderConfig {
	timeout := openAIDefaultRequestTimeout
	if v := strings.TrimSpace(os.Getenv("LLM_TIMEOUT")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			timeout = d
		} else {
			slog.Warn("invalid LLM_TIMEOUT, using default", "value", v, "default", timeout)
		}
	}

	switch provider {
	case "gemini":
		return ProviderConfig{
			BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai",
			APIKey:  os.Getenv("GEMINI_API_KEY"),
			Timeout: timeout,
		}
	case "openrouter":
		key := os.Getenv("OPENROUTER_API_KEY")
		if key == "" {
			key = os.Getenv("OPEN_ROUTE_KEY")
		}
		return ProviderConfig{
			BaseURL: "https://openrouter.ai/api/v1",
			APIKey:  key,
			Timeout: timeout,
		}
	case "litertlm", "llamacpp":
		return ProviderConfig{}
	default: // ollama
		baseURL := os.Getenv("OLLAMA_HOST")
		if baseURL == "" {
			baseURL = "http://localhost:11434"
		}
		if !strings.HasSuffix(baseURL, "/v1") {
			baseURL = strings.TrimSuffix(baseURL, "/") + "/v1"
		}
		return ProviderConfig{BaseURL: baseURL, Timeout: timeout}
	}
}

func newProviderCaller(provider, model string, executor ToolExecutor) (Caller, bool) {
	switch provider {
	case "llamacpp":
		return NewLlamacppCaller(model, executor), true
	case "litertlm":
		return NewLitertLMCaller(model, executor), true
	default:
		return nil, false
	}
}

func NewProviderCaller(provider, model string, executor ToolExecutor) Caller {
	return NewProviderCallerWithThink(provider, model, executor, "")
}

func NewProviderCallerRaw(provider, model string, executor ToolExecutor) RawCaller {
	return NewProviderCallerRawWithThink(provider, model, executor, "")
}

// NewProviderCallerWithThink is like NewProviderCaller with a unified think
// effort applied to the returned caller ("": provider default).
func NewProviderCallerWithThink(provider, model string, executor ToolExecutor, think ThinkEffort) Caller {
	if c, ok := newProviderCaller(provider, model, executor); ok {
		applyThinkEffort(c, think)
		return c
	}
	return newOpenAICallerFromProvider(provider, model, executor, think)
}

// NewProviderCallerRawWithThink is the RawCaller variant of
// NewProviderCallerWithThink.
func NewProviderCallerRawWithThink(provider, model string, executor ToolExecutor, think ThinkEffort) RawCaller {
	if c, ok := newProviderCaller(provider, model, executor); ok {
		applyThinkEffort(c, think)
		if rc, ok := c.(RawCaller); ok {
			return rc
		}
	}
	return newOpenAICallerFromProvider(provider, model, executor, think)
}

// newOpenAICallerFromProvider builds an HTTP caller using the provider's base
// URL, key and configured request timeout.
func newOpenAICallerFromProvider(provider, model string, executor ToolExecutor, think ThinkEffort) *OpenAICaller {
	cfg := getProviderConfig(provider)
	caller := NewOpenAICallerWithThink(cfg.BaseURL, cfg.APIKey, model, executor, think)
	caller.Provider = strings.ToLower(strings.TrimSpace(provider))
	if caller.Provider == "" {
		caller.Provider = "ollama"
	}
	if cfg.Timeout > 0 {
		caller.Client.Timeout = cfg.Timeout
	}
	return caller
}

// applyThinkEffort sets the think effort on in-process callers that support
// it. OpenAI-compatible callers are constructed with it directly.
func applyThinkEffort(c Caller, think ThinkEffort) {
	switch v := c.(type) {
	case *LlamacppCaller:
		v.ThinkEffort = think
		if think == ThinkEffortNone {
			v.NoThink = true
		}
	case *LitertLMCaller:
		v.ThinkEffort = think
	}
}

func (a *Agent) CallLLM(ctx context.Context, executor ToolExecutor, messages []Message) ([]Message, error) {
	caller := NewProviderCallerWithThink(a.Provider, a.Model, executor, a.ThinkEffort)
	if lc, ok := caller.(*LitertLMCaller); ok {
		lc.Backend = a.Backend
	}
	prompt := a.Prompt
	if a.AgentFiles != "" {
		prompt = prompt + "\n\n" + a.AgentFiles
	}
	if a.Skills != "" {
		prompt = prompt + "\n\n" + a.Skills
	}
	return caller.Call(ctx, prompt, messages, a.Tools)
}
