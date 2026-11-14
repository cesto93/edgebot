package llm

import (
	"os"
	"testing"
	"time"
)

func TestGetProviderConfigGemini(t *testing.T) {
	os.Setenv("GEMINI_API_KEY", "gemini-test-key")
	defer os.Unsetenv("GEMINI_API_KEY")

	cfg := getProviderConfig("gemini")

	if cfg.BaseURL != "https://generativelanguage.googleapis.com/v1beta/openai" {
		t.Errorf("gemini BaseURL = %q, want %q", cfg.BaseURL, "https://generativelanguage.googleapis.com/v1beta/openai")
	}
	if cfg.APIKey != "gemini-test-key" {
		t.Errorf("gemini APIKey = %q, want %q", cfg.APIKey, "gemini-test-key")
	}
}

func TestGetProviderConfigOpenRouter(t *testing.T) {
	os.Setenv("OPEN_ROUTE_KEY", "or-test-key")
	defer os.Unsetenv("OPEN_ROUTE_KEY")

	cfg := getProviderConfig("openrouter")

	if cfg.BaseURL != "https://openrouter.ai/api/v1" {
		t.Errorf("openrouter BaseURL = %q, want %q", cfg.BaseURL, "https://openrouter.ai/api/v1")
	}
	if cfg.APIKey != "or-test-key" {
		t.Errorf("openrouter APIKey = %q, want %q", cfg.APIKey, "or-test-key")
	}
}

func TestGetProviderConfigLiteRTLM(t *testing.T) {
	cfg := getProviderConfig("litertlm")

	if cfg.BaseURL != "" {
		t.Errorf("litertlm BaseURL = %q, want empty (in-process provider)", cfg.BaseURL)
	}
}

func TestGetProviderConfigLiteRTLMCustomURL(t *testing.T) {
	os.Setenv("LITERTLM_BASE_URL", "http://custom:8080")
	defer os.Unsetenv("LITERTLM_BASE_URL")

	cfg := getProviderConfig("litertlm")

	if cfg.BaseURL != "" {
		t.Errorf("litertlm BaseURL = %q, want empty (in-process provider)", cfg.BaseURL)
	}
}

func TestGetProviderConfigOllamaDefault(t *testing.T) {
	os.Unsetenv("OLLAMA_HOST")

	cfg := getProviderConfig("ollama")

	if cfg.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("ollama BaseURL = %q, want %q", cfg.BaseURL, "http://localhost:11434/v1")
	}
}

func TestGetProviderConfigOllamaCustomHost(t *testing.T) {
	os.Setenv("OLLAMA_HOST", "http://custom-host:5000")
	defer os.Unsetenv("OLLAMA_HOST")

	cfg := getProviderConfig("ollama")

	if cfg.BaseURL != "http://custom-host:5000/v1" {
		t.Errorf("ollama BaseURL = %q, want %q", cfg.BaseURL, "http://custom-host:5000/v1")
	}
}

func TestGetProviderConfigUnknownDefaultsToOllama(t *testing.T) {
	os.Unsetenv("OLLAMA_HOST")

	cfg := getProviderConfig("unknown-provider")

	if cfg.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("unknown provider BaseURL = %q, want %q", cfg.BaseURL, "http://localhost:11434/v1")
	}
}

func TestNewProviderCallerGemini(t *testing.T) {
	os.Setenv("GEMINI_API_KEY", "test-key")
	defer os.Unsetenv("GEMINI_API_KEY")

	caller := NewProviderCaller("gemini", "gemini-3.7-flash", nil)

	oac, ok := caller.(*OpenAICaller)
	if !ok {
		t.Fatalf("Expected *OpenAICaller, got %T", caller)
	}
	if oac.APIKey != "test-key" {
		t.Errorf("APIKey = %q, want %q", oac.APIKey, "test-key")
	}
	if oac.Model != "gemini-3.7-flash" {
		t.Errorf("Model = %q, want %q", oac.Model, "gemini-3.7-flash")
	}
}

func TestNewProviderCallerLiteRTLM(t *testing.T) {
	caller := NewProviderCaller("litertlm", "test-model", nil)

	lc, ok := caller.(*LitertLMCaller)
	if !ok {
		t.Fatalf("Expected *LitertLMCaller, got %T", caller)
	}
	if lc.Model != "test-model" {
		t.Errorf("Model = %q, want %q", lc.Model, "test-model")
	}
}

func TestNewProviderCallerRawLiteRTLM(t *testing.T) {
	caller := NewProviderCallerRaw("litertlm", "test-model", nil)

	lc, ok := caller.(*LitertLMCaller)
	if !ok {
		t.Fatalf("Expected *LitertLMCaller, got %T", caller)
	}
	if lc.Model != "test-model" {
		t.Errorf("Model = %q, want %q", lc.Model, "test-model")
	}
}

func TestNewProviderCallerOllama(t *testing.T) {
	os.Unsetenv("OLLAMA_HOST")

	caller := NewProviderCaller("ollama", "granite4:3b-h", nil)

	oac, ok := caller.(*OpenAICaller)
	if !ok {
		t.Fatalf("Expected *OpenAICaller, got %T", caller)
	}
	if oac.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("BaseURL = %q, want %q", oac.BaseURL, "http://localhost:11434/v1")
	}
}

func TestGetProviderConfigTimeout(t *testing.T) {
	t.Setenv("LLM_TIMEOUT", "30s")
	if got := getProviderConfig("ollama").Timeout; got != 30*time.Second {
		t.Errorf("Timeout = %v, want 30s", got)
	}

	t.Setenv("LLM_TIMEOUT", "not-a-duration")
	if got := getProviderConfig("ollama").Timeout; got != openAIDefaultRequestTimeout {
		t.Errorf("invalid LLM_TIMEOUT: Timeout = %v, want %v", got, openAIDefaultRequestTimeout)
	}
}

func TestNewProviderCallerSetsProviderIdentity(t *testing.T) {
	for _, provider := range []string{"ollama", "gemini", "openrouter"} {
		t.Run(provider, func(t *testing.T) {
			caller, ok := NewProviderCaller(provider, "m", NoopExecutor{}).(*OpenAICaller)
			if !ok {
				t.Fatalf("caller type = %T, want *OpenAICaller", caller)
			}
			if caller.Provider != provider {
				t.Errorf("Provider = %q, want %q", caller.Provider, provider)
			}
		})
	}

	raw, ok := NewProviderCallerRaw("openrouter", "m", NoopExecutor{}).(*OpenAICaller)
	if !ok {
		t.Fatalf("raw caller type = %T, want *OpenAICaller", raw)
	}
	if raw.Provider != "openrouter" {
		t.Errorf("raw Provider = %q, want openrouter", raw.Provider)
	}
}

func TestProviderIdentityOverridesURLSniffing(t *testing.T) {
	caller := &OpenAICaller{BaseURL: "https://gateway.example/v1", Provider: "openrouter"}
	if !caller.isOpenRouter() {
		t.Error("openrouter provider behind a custom URL was not recognized")
	}
	if caller.providerName() != "openrouter" {
		t.Errorf("providerName() = %q, want openrouter", caller.providerName())
	}

	caller.Provider = "gemini"
	if !caller.isGemini() {
		t.Error("gemini provider behind a custom URL was not recognized")
	}

	caller.Provider = "custom-gateway"
	if caller.providerName() != "custom-gateway" {
		t.Errorf("providerName() = %q, want custom-gateway", caller.providerName())
	}
}
