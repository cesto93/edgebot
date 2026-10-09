package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mockDecisionDirs isolates home/config/routing from the real machine and
// seeds the llamacpp models dir with one decision GGUF (lev-1) and one plain
// GGUF (plain-3b). No .env files are read and no network lookups run.
func mockDecisionDirs(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	cfgDir := t.TempDir()

	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	t.Cleanup(func() { userHomeDirFunc = origHome })

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) { return cfgDir, nil }
	t.Cleanup(func() { userConfigDirFunc = origUserConfigDirFunc })

	origConfigPaths := configPaths
	t.Cleanup(func() { configPaths = origConfigPaths })
	configPaths = []string{t.TempDir()}

	origLoadEnv := loadEnvFunc
	loadEnvFunc = func() error { return nil }
	t.Cleanup(func() { loadEnvFunc = origLoadEnv })

	origOpenRouter := getOpenRouterModelsFunc
	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	t.Cleanup(func() { getOpenRouterModelsFunc = origOpenRouter })

	origAvailable := getAvailableModelsFunc
	getAvailableModelsFunc = func() ([]ModelInfo, error) { return nil, nil }
	t.Cleanup(func() { getAvailableModelsFunc = origAvailable })

	t.Setenv("LLM_PROVIDER", "")
	t.Setenv("LLM_MODEL", "")
	t.Setenv("LLM_DECISION_MODEL", "")
	t.Setenv("LITERTLM_BACKEND", "")

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestGGUF(t, filepath.Join(modelsDir, "lev-1.gguf"), []ggufKV{
		{"lev.decision.type", ggufTypeString, ggufStr("lev")},
	})
	writeTestGGUF(t, filepath.Join(modelsDir, "plain-3b.gguf"), []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("llama")},
	})
}

func TestIsDecisionModel(t *testing.T) {
	mockDecisionDirs(t)
	if !IsDecisionModel("lev-1") {
		t.Error("IsDecisionModel(lev-1) = false, want true")
	}
	if IsDecisionModel("plain-3b") {
		t.Error("IsDecisionModel(plain-3b) = true, want false")
	}
	if IsDecisionModel("absent") {
		t.Error("IsDecisionModel(absent) = true, want false")
	}
}

func TestSaveDecisionModel(t *testing.T) {
	mockDecisionDirs(t)

	if err := SaveDecisionModel("lev-1"); err != nil {
		t.Fatalf("SaveDecisionModel(lev-1) = %v", err)
	}
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() = %v", err)
	}
	if cfg.LLM.DecisionModel != "lev-1" {
		t.Errorf("DecisionModel = %q, want %q", cfg.LLM.DecisionModel, "lev-1")
	}
	if cfg.LLM.Model == "lev-1" {
		t.Errorf("chat model was clobbered with the decision model")
	}

	for _, name := range []string{"plain-3b", "absent"} {
		if err := SaveDecisionModel(name); err == nil {
			t.Errorf("SaveDecisionModel(%q) succeeded, want error", name)
		}
	}
}

func TestSetDefaultModel(t *testing.T) {
	mockDecisionDirs(t)

	kind, err := SetDefaultModel("lev-1", "llamacpp")
	if err != nil {
		t.Fatalf("SetDefaultModel(lev-1) = %v", err)
	}
	if kind != ModelTypeDecision {
		t.Errorf("kind = %q, want %q", kind, ModelTypeDecision)
	}
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.DecisionModel != "lev-1" {
		t.Errorf("DecisionModel = %q, want lev-1", cfg.LLM.DecisionModel)
	}
	if cfg.LLM.Model == "lev-1" {
		t.Errorf("chat model was clobbered with the decision model")
	}

	kind, err = SetDefaultModel("plain-3b", "llamacpp")
	if err != nil {
		t.Fatalf("SetDefaultModel(plain-3b) = %v", err)
	}
	if kind != ModelTypeLLM {
		t.Errorf("kind = %q, want %q", kind, ModelTypeLLM)
	}
	cfg, err = LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.Model != "plain-3b" || cfg.LLM.Provider != "llamacpp" {
		t.Errorf("chat default = %s/%s, want plain-3b/llamacpp", cfg.LLM.Model, cfg.LLM.Provider)
	}
	if cfg.LLM.DecisionModel != "lev-1" {
		t.Errorf("DecisionModel = %q, want lev-1 (preserved)", cfg.LLM.DecisionModel)
	}
}

func TestDecisionModelEnvOverride(t *testing.T) {
	mockDecisionDirs(t)
	t.Setenv("LLM_DECISION_MODEL", "lev-1")
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig() = %v", err)
	}
	if cfg.LLM.DecisionModel != "lev-1" {
		t.Errorf("DecisionModel = %q, want lev-1", cfg.LLM.DecisionModel)
	}
}

func TestSaveConfigWritesDecisionModel(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatal(err)
	}
	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) { return configPath, nil }
	defer func() { getConfigPathFunc = origGetConfigPath }()

	cfg := &Config{
		ConfigFile: filepath.Join(configPath, "config.yaml"),
		LogLevel:   "info",
		LLM: struct {
			Provider      string   `mapstructure:"provider"`
			Model         string   `mapstructure:"model"`
			DecisionModel string   `mapstructure:"decision_model"`
			InputTypes    []string `mapstructure:"input_types"`
			ThinkEffort   string   `mapstructure:"think_effort"`
		}{
			Provider:      "ollama",
			Model:         "test-model",
			DecisionModel: "lev-1",
		},
		Shell: struct {
			Confirm         bool     `mapstructure:"confirm"`
			AllowedCommands []string `mapstructure:"allowed_commands"`
		}{
			Confirm:         true,
			AllowedCommands: []string{"ls"},
		},
	}
	if err := SaveConfig(cfg); err != nil {
		t.Fatalf("SaveConfig() = %v", err)
	}
	data, err := os.ReadFile(cfg.ConfigFile)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "decision_model: lev-1") {
		t.Errorf("config file missing decision_model, got:\n%s", data)
	}

	cfg.LLM.DecisionModel = ""
	if err := SaveConfig(cfg); err != nil {
		t.Fatalf("SaveConfig() = %v", err)
	}
	data, err = os.ReadFile(cfg.ConfigFile)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "decision_model") {
		t.Errorf("empty decision model should be omitted, got:\n%s", data)
	}
}
