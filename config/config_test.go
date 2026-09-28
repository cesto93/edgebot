package config

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatalf("Failed to create temp config dir: %v", err)
	}

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{tmpDir}

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) {
		return tmpDir, nil
	}
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	tests := []struct {
		name          string
		configContent string
		configExists  bool
		wantModel     string
		wantConfirm   bool
		wantAgent     string
	}{
		{
			name: "valid config file",
			configContent: `
llm:
  model: "custom-model:latest"
shell:
  confirm: false
`,
			configExists: true,
			wantModel:    "custom-model:latest",
			wantConfirm:  false,
			wantAgent:    "build",
		},
		{
			name:          "default config when file not found",
			configContent: "",
			configExists:  false,
			wantModel:     "granite4:3b-h",
			wantConfirm:   true,
			wantAgent:     "build",
		},
		{
			name: "partial config uses defaults",
			configContent: `
llm:
  model: "partial-model"
`,
			configExists: true,
			wantModel:    "partial-model",
			wantConfirm:  true,
			wantAgent:    "build",
		},
		{
			name: "custom agent",
			configContent: `
agent: "plan"
llm:
  model: "custom-model:latest"
`,
			configExists: true,
			wantModel:    "custom-model:latest",
			wantConfirm:  true,
			wantAgent:    "plan",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configFile := filepath.Join(configPath, "config.yaml")
			if tt.configExists {
				if err := os.WriteFile(configFile, []byte(tt.configContent), 0644); err != nil {
					t.Fatalf("Failed to write config file: %v", err)
				}
			} else {
				os.Remove(configFile)
			}

			cfg, err := LoadConfig()
			if err != nil {
				t.Fatalf("LoadConfig() error = %v", err)
			}

			if cfg.LLM.Model != tt.wantModel {
				t.Errorf("LoadConfig().LLM.Model = %q, want %q", cfg.LLM.Model, tt.wantModel)
			}

			if cfg.Shell.Confirm != tt.wantConfirm {
				t.Errorf("LoadConfig().Shell.Confirm = %v, want %v", cfg.Shell.Confirm, tt.wantConfirm)
			}

			if cfg.Agent != tt.wantAgent {
				t.Errorf("LoadConfig().Agent = %q, want %q", cfg.Agent, tt.wantAgent)
			}
		})
	}
}

func TestLoadConfigEnvOverrides(t *testing.T) {
	tmpDir := t.TempDir()
	home := t.TempDir()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{tmpDir}

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) { return tmpDir, nil }
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	// Hermetic: no real .env files, no network model lookups.
	origLoadEnv := loadEnvFunc
	loadEnvFunc = func() error { return nil }
	defer func() { loadEnvFunc = origLoadEnv }()

	origOpenRouter := getOpenRouterModelsFunc
	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	defer func() { getOpenRouterModelsFunc = origOpenRouter }()

	origAvailable := getAvailableModelsFunc
	getAvailableModelsFunc = func() ([]ModelInfo, error) { return nil, nil }
	defer func() { getAvailableModelsFunc = origAvailable }()

	writeFileConfig := func(provider, model, backend string) {
		content := "llm:\n  provider: \"" + provider + "\"\n  model: \"" +
			model + "\"\nlitertlm:\n  backend: \"" + backend + "\"\n"
		if err := os.WriteFile(filepath.Join(tmpDir, "config.yaml"),
			[]byte(content), 0644); err != nil {
			t.Fatalf("Failed to write config file: %v", err)
		}
	}

	t.Run("provider and model override file", func(t *testing.T) {
		writeFileConfig("ollama", "granite4:3b-h", "cpu")
		t.Setenv("LLM_PROVIDER", "litertlm")
		t.Setenv("LLM_MODEL", "gemma-4-E2B-it")
		cfg, err := LoadConfig()
		if err != nil {
			t.Fatalf("LoadConfig() error = %v", err)
		}
		if cfg.LLM.Provider != "litertlm" {
			t.Errorf("Provider = %q, want %q", cfg.LLM.Provider, "litertlm")
		}
		if cfg.LLM.Model != "gemma-4-E2B-it" {
			t.Errorf("Model = %q, want %q", cfg.LLM.Model, "gemma-4-E2B-it")
		}
	})

	t.Run("env applies without config file", func(t *testing.T) {
		os.Remove(filepath.Join(tmpDir, "config.yaml"))
		t.Setenv("LLM_PROVIDER", "litertlm")
		t.Setenv("LLM_MODEL", "gemma-4-E2B-it")
		cfg, err := LoadConfig()
		if err != nil {
			t.Fatalf("LoadConfig() error = %v", err)
		}
		if cfg.LLM.Provider != "litertlm" || cfg.LLM.Model != "gemma-4-E2B-it" {
			t.Errorf("got %q/%q, want litertlm/gemma-4-E2B-it",
				cfg.LLM.Provider, cfg.LLM.Model)
		}
	})

	t.Run("no env keeps file values", func(t *testing.T) {
		writeFileConfig("ollama", "granite4:3b-h", "cpu")
		cfg, err := LoadConfig()
		if err != nil {
			t.Fatalf("LoadConfig() error = %v", err)
		}
		if cfg.LLM.Provider != "ollama" || cfg.LLM.Model != "granite4:3b-h" {
			t.Errorf("got %q/%q, want ollama/granite4:3b-h",
				cfg.LLM.Provider, cfg.LLM.Model)
		}
	})

	t.Run("model only auto-detects litertlm from disk", func(t *testing.T) {
		writeFileConfig("ollama", "granite4:3b-h", "cpu")
		modelDir := filepath.Join(home, ".edgebot", "models", "litertlm")
		if err := os.MkdirAll(modelDir, 0755); err != nil {
			t.Fatalf("Failed to create models dir: %v", err)
		}
		if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-E2B-it.litertlm"),
			[]byte("model"), 0644); err != nil {
			t.Fatalf("Failed to write model file: %v", err)
		}
		t.Setenv("LLM_MODEL", "gemma-4-E2B-it")
		cfg, err := LoadConfig()
		if err != nil {
			t.Fatalf("LoadConfig() error = %v", err)
		}
		if cfg.LLM.Provider != "litertlm" {
			t.Errorf("Provider = %q, want %q", cfg.LLM.Provider, "litertlm")
		}
	})

	t.Run("backend override", func(t *testing.T) {
		writeFileConfig("ollama", "granite4:3b-h", "cpu")
		t.Setenv("LITERTLM_BACKEND", "gpu")
		cfg, err := LoadConfig()
		if err != nil {
			t.Fatalf("LoadConfig() error = %v", err)
		}
		if cfg.LitertLM.Backend != "gpu" {
			t.Errorf("Backend = %q, want %q", cfg.LitertLM.Backend, "gpu")
		}
	})
}

func TestThinkEffortNormalization(t *testing.T) {
	cases := []struct{ in, want string }{
		{"", ""},
		{"low", "low"},
		{" Low ", "low"},
		{"HIGH", "high"},
		{"xhigh", "xhigh"},
		{"max", "max"},
		{"none", "none"},
		{"minimal", "minimal"},
		{"ultra", ""},
		{"off", ""},
	}
	for _, tt := range cases {
		if got := normalizeThinkEffort(tt.in); got != tt.want {
			t.Errorf("normalizeThinkEffort(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestSaveConfigPreservesThinkEffort(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.yaml")
	cfg := &Config{ConfigFile: configFile, LogLevel: "info"}
	cfg.LLM.Provider = "ollama"
	cfg.LLM.Model = "m"
	cfg.LLM.ThinkEffort = "low"
	if err := SaveConfig(cfg); err != nil {
		t.Fatalf("SaveConfig: %v", err)
	}
	data, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("read saved config: %v", err)
	}
	if !strings.Contains(string(data), "think_effort") || !strings.Contains(string(data), "low") {
		t.Errorf("saved config missing think_effort: low:\n%s", data)
	}
}

func TestSaveModel(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatalf("Failed to create temp config dir: %v", err)
	}

	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) {
		return configPath, nil
	}
	defer func() { getConfigPathFunc = origGetConfigPath }()

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) {
		return tmpDir, nil
	}
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{configPath}

	configFile := filepath.Join(configPath, "config.yaml")
	initialConfig := `llm:
  model: "initial-model"
shell:
  confirm: true
  allowed_commands: "ls,pwd"
`

	if err := os.WriteFile(configFile, []byte(initialConfig), 0644); err != nil {
		t.Fatalf("Failed to write initial config: %v", err)
	}

	newModel := "new-model:latest"
	if err := SaveModelWithProvider(newModel, ""); err != nil {
		t.Fatalf("SaveModel() error = %v", err)
	}

	data, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("Failed to read config file: %v", err)
	}

	if !bytes.Contains(data, []byte(`model: new-model:latest`)) {
		t.Errorf("config file does not contain new model, got: %s", string(data))
	}
}

func TestSaveModelCreatesNewFile(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatalf("Failed to create temp config dir: %v", err)
	}
	configFile := filepath.Join(configPath, "config.yaml")

	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) {
		return configPath, nil
	}
	defer func() { getConfigPathFunc = origGetConfigPath }()

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) {
		return tmpDir, nil
	}
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{configPath}

	if _, err := os.Stat(configFile); err == nil {
		os.Remove(configFile)
	}

	newModel := "brand-new-model:latest"
	if err := SaveModelWithProvider(newModel, ""); err != nil {
		t.Fatalf("SaveModel() error = %v", err)
	}

	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		t.Fatal("SaveModel did not create config file")
	}

	data, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("Failed to read config file: %v", err)
	}

	if !bytes.Contains(data, []byte(`model: brand-new-model:latest`)) {
		t.Errorf("config file does not contain new model, got: %s", string(data))
	}
}

func TestSelectModelNoModels(t *testing.T) {
	origModels := getAvailableModelsFunc
	getAvailableModelsFunc = func() ([]ModelInfo, error) {
		return []ModelInfo{}, nil
	}
	defer func() { getAvailableModelsFunc = origModels }()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Failed to create pipe: %v", err)
	}
	origStdin := os.Stdin
	defer func() { os.Stdin = origStdin }()
	os.Stdin = r
	defer w.Close()

	go func() {
		w.WriteString("\n")
		w.Close()
	}()

	err = SelectModel()
	if err != nil {
		t.Fatalf("SelectModel() error = %v", err)
	}
}

func TestSelectModelEmptySelection(t *testing.T) {
	origModels := getAvailableModelsFunc
	getAvailableModelsFunc = func() ([]ModelInfo, error) {
		return []ModelInfo{
			{Name: "model1"},
			{Name: "model2"},
		}, nil
	}
	defer func() { getAvailableModelsFunc = origModels }()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Failed to create pipe: %v", err)
	}
	origStdin := os.Stdin
	defer func() { os.Stdin = origStdin }()
	os.Stdin = r
	defer w.Close()

	go func() {
		w.WriteString("\n")
		w.Close()
	}()

	err = SelectModel()
	if err != nil {
		t.Fatalf("SelectModel() error = %v", err)
	}
}

func TestSelectModelInvalidInput(t *testing.T) {
	origModels := getAvailableModelsFunc
	getAvailableModelsFunc = func() ([]ModelInfo, error) {
		return []ModelInfo{
			{Name: "model1"},
			{Name: "model2"},
		}, nil
	}
	defer func() { getAvailableModelsFunc = origModels }()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Failed to create pipe: %v", err)
	}
	origStdin := os.Stdin
	defer func() { os.Stdin = origStdin }()
	os.Stdin = r
	defer w.Close()

	go func() {
		w.WriteString("999\n")
		w.Close()
	}()

	err = SelectModel()
	if err != nil {
		t.Fatalf("SelectModel() error = %v", err)
	}
}

func TestSelectModelSavesNewModel(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatalf("Failed to create temp config dir: %v", err)
	}
	configFile := filepath.Join(configPath, "config.yaml")

	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) {
		return configPath, nil
	}
	defer func() { getConfigPathFunc = origGetConfigPath }()

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) {
		return tmpDir, nil
	}
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{configPath}

	initialConfig := `llm:
  model: "initial-model"
shell:
  confirm: true
`
	if err := os.WriteFile(configFile, []byte(initialConfig), 0644); err != nil {
		t.Fatalf("Failed to write initial config: %v", err)
	}

	origModels := getAvailableModelsFunc
	getAvailableModelsFunc = func() ([]ModelInfo, error) {
		return []ModelInfo{
			{Name: "model1"},
			{Name: "model2"},
			{Name: "model3"},
		}, nil
	}
	defer func() { getAvailableModelsFunc = origModels }()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Failed to create pipe: %v", err)
	}
	origStdin := os.Stdin
	defer func() { os.Stdin = origStdin }()
	os.Stdin = r
	defer w.Close()

	go func() {
		w.WriteString("2\n")
		w.Close()
	}()

	err = SelectModel()
	if err != nil {
		t.Fatalf("SelectModel() error = %v", err)
	}

	data, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("Failed to read config file: %v", err)
	}

	if !bytes.Contains(data, []byte(`model: model2`)) {
		t.Errorf("config file does not contain selected model 'model2', got: %s", string(data))
	}
}

func TestIsLitertLMModel(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	modelDir := filepath.Join(home, ".edgebot", "models", "litertlm")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-E2B-it.litertlm"), []byte("model"), 0644); err != nil {
		t.Fatalf("Failed to write model file: %v", err)
	}

	tests := []struct {
		name  string
		model string
		want  bool
	}{
		{
			name:  "litertlm model",
			model: "gemma-4-E2B-it",
			want:  true,
		},
		{
			name:  "not litertlm model",
			model: "granite4:3b-h",
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsLitertLMModel(tt.model)
			if got != tt.want {
				t.Errorf("IsLitertLMModel(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestGetLitertLMModels(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	modelDir := filepath.Join(home, ".edgebot", "models", "litertlm")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	for _, name := range []string{"gemma-4-E2B-it.litertlm", "gemma-4-E4B-it.litertlm", "notes.txt"} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("model"), 0644); err != nil {
			t.Fatalf("Failed to write file %s: %v", name, err)
		}
	}

	models := GetLitertLMModels()
	if len(models) != 2 {
		t.Fatalf("GetLitertLMModels() returned %d models, want 2: %+v", len(models), models)
	}
	found := map[string]bool{}
	for _, m := range models {
		if m.Provider != "litertlm" {
			t.Errorf("model %q provider = %q, want litertlm", m.Name, m.Provider)
		}
		found[m.Name] = true
	}
	if !found["gemma-4-E2B-it"] || !found["gemma-4-E4B-it"] {
		t.Errorf("GetLitertLMModels() did not list both models, got: %+v", found)
	}
}

func TestIsAllowedCommand(t *testing.T) {
	tests := []struct {
		name        string
		cmd         string
		allowedList []string
		want        bool
	}{
		{
			name:        "command is allowed",
			cmd:         "ls",
			allowedList: []string{"ls", "pwd", "cat"},
			want:        true,
		},
		{
			name:        "command is not allowed",
			cmd:         "rm",
			allowedList: []string{"ls", "pwd", "cat"},
			want:        false,
		},
		{
			name:        "empty allowed list",
			cmd:         "ls",
			allowedList: []string{},
			want:        false,
		},
		{
			name:        "command with spaces in allowed list",
			cmd:         "ls",
			allowedList: []string{"ls ", "pwd"},
			want:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsAllowedCommand(tt.cmd, tt.allowedList)
			if got != tt.want {
				t.Errorf("IsAllowedCommand(%q, %v) = %v, want %v", tt.cmd, tt.allowedList, got, tt.want)
			}
		})
	}
}

func TestModelInListGemini(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-3.7-flash", true},
		{"gemini-3.5-flash-lite", true},
		{"gemma-4-31b-it", true},
		{"gemma-4-26b-a4b-it", true},
		{"other-model", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := modelInList(tt.model, GeminiModels); got != tt.want {
				t.Errorf("modelInList(%q, GeminiModels) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestGetEnvPaths(t *testing.T) {
	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) {
		return "/tmp/test-config", nil
	}
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	paths := GetEnvPaths()

	if len(paths) < 2 {
		t.Fatalf("GetEnvPaths() returned %d paths, want at least 2", len(paths))
	}

	if paths[0] != "/tmp/test-config/edgebot/.env" {
		t.Errorf("First path = %q, want %q", paths[0], "/tmp/test-config/edgebot/.env")
	}
	if paths[1] != ".env" {
		t.Errorf("Second path = %q, want %q", paths[1], ".env")
	}
}

func TestGetEnvPathsConfigDirError(t *testing.T) {
	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) {
		return "", fmt.Errorf("no config dir")
	}
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	paths := GetEnvPaths()

	if len(paths) != 1 {
		t.Fatalf("GetEnvPaths() returned %d paths, want 1", len(paths))
	}
	if paths[0] != ".env" {
		t.Errorf("Path = %q, want %q", paths[0], ".env")
	}
}

func TestLoadCommandsFromDir(t *testing.T) {
	tmpDir := t.TempDir()

	commands, err := LoadCommandsFromDir(tmpDir)
	if err != nil {
		t.Fatalf("LoadCommandsFromDir() error = %v", err)
	}
	if len(commands) != 0 {
		t.Errorf("LoadCommandsFromDir() returned %d commands, want 0", len(commands))
	}
}

func TestLoadCommandsFromDirWithFiles(t *testing.T) {
	tmpDir := t.TempDir()

	mdContent := `---
description: A test command
---
This is the test prompt for the command.`

	if err := os.WriteFile(filepath.Join(tmpDir, "test-cmd.md"), []byte(mdContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	if err := os.WriteFile(filepath.Join(tmpDir, "not-markdown.txt"), []byte("ignored"), 0644); err != nil {
		t.Fatalf("Failed to write ignored file: %v", err)
	}

	commands, err := LoadCommandsFromDir(tmpDir)
	if err != nil {
		t.Fatalf("LoadCommandsFromDir() error = %v", err)
	}

	if len(commands) != 1 {
		t.Fatalf("LoadCommandsFromDir() returned %d commands, want 1", len(commands))
	}

	cmd := commands[0]
	if cmd.Name != "test-cmd" {
		t.Errorf("Command Name = %q, want %q", cmd.Name, "test-cmd")
	}
	if cmd.Description != "A test command" {
		t.Errorf("Command Description = %q, want %q", cmd.Description, "A test command")
	}
	if !strings.Contains(cmd.Prompt, "test prompt") {
		t.Errorf("Command Prompt missing expected content: %q", cmd.Prompt)
	}
}

func TestParseCommandFileWithFrontmatter(t *testing.T) {
	tmpDir := t.TempDir()
	content := `---
description: My custom command
---
Do something useful`

	path := filepath.Join(tmpDir, "cmd.md")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	cmd, err := parseCommandFile(path)
	if err != nil {
		t.Fatalf("parseCommandFile() error = %v", err)
	}
	if cmd.Description != "My custom command" {
		t.Errorf("Description = %q, want %q", cmd.Description, "My custom command")
	}
	if cmd.Prompt != "Do something useful" {
		t.Errorf("Prompt = %q, want %q", cmd.Prompt, "Do something useful")
	}
}

func TestParseCommandFileWithSchema(t *testing.T) {
	tmpDir := t.TempDir()
	content := `---
description: Extract invoice data
schema: invoice_schema.json
---
Extract the invoice`
	path := filepath.Join(tmpDir, "invoice.md")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	cmd, err := parseCommandFile(path)
	if err != nil {
		t.Fatalf("parseCommandFile() error = %v", err)
	}
	wantSchema := filepath.Join(tmpDir, "invoice_schema.json")
	if cmd.Schema != wantSchema {
		t.Errorf("Schema = %q, want %q (resolved relative to command dir)", cmd.Schema, wantSchema)
	}
	if cmd.Description != "Extract invoice data" {
		t.Errorf("Description = %q, want %q", cmd.Description, "Extract invoice data")
	}
}

func TestParseCommandFileWithoutFrontmatter(t *testing.T) {
	tmpDir := t.TempDir()
	content := `Just a simple prompt without frontmatter`

	path := filepath.Join(tmpDir, "simple.md")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	cmd, err := parseCommandFile(path)
	if err != nil {
		t.Fatalf("parseCommandFile() error = %v", err)
	}
	if cmd.Prompt != "Just a simple prompt without frontmatter" {
		t.Errorf("Prompt = %q, want %q", cmd.Prompt, "Just a simple prompt without frontmatter")
	}
	if cmd.Description != "" {
		t.Errorf("Description = %q, want empty", cmd.Description)
	}
}

func TestParseCommandFileNotFound(t *testing.T) {
	_, err := parseCommandFile("/nonexistent/file.md")
	if err == nil {
		t.Error("parseCommandFile() expected error for nonexistent file, got nil")
	}
}

func TestEnsureCommandsDir(t *testing.T) {
	tmpDir := t.TempDir()
	origDir, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(origDir)

	err := EnsureCommandsDir()
	if err != nil {
		t.Fatalf("EnsureCommandsDir() error = %v", err)
	}

	if _, err := os.Stat(filepath.Join(tmpDir, ".edgebot", "commands")); os.IsNotExist(err) {
		t.Error("EnsureCommandsDir() did not create .edgebot/commands directory")
	}
}

func TestLookupModelInfo(t *testing.T) {
	tests := []struct {
		name     string
		wantNil  bool
		wantName string
	}{
		{
			name:     "gemini model found",
			wantNil:  false,
			wantName: "gemini-3.7-flash",
		},
		{
			name:    "unknown model not found",
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := LookupModelInfo(tt.wantName)
			if tt.wantNil {
				if info != nil {
					t.Errorf("LookupModelInfo() = %v, want nil", info)
				}
			} else {
				if info == nil {
					t.Fatal("LookupModelInfo() returned nil")
				}
				if info.Name != tt.wantName {
					t.Errorf("LookupModelInfo().Name = %q, want %q", info.Name, tt.wantName)
				}
			}
		})
	}
}

func TestSaveConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatalf("Failed to create temp config dir: %v", err)
	}
	configFile := filepath.Join(configPath, "config.yaml")

	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) {
		return configPath, nil
	}
	defer func() { getConfigPathFunc = origGetConfigPath }()

	cfg := &Config{
		ConfigFile: configFile,
		LogLevel:   "info",
		LLM: struct {
			Provider    string   `mapstructure:"provider"`
			Model       string   `mapstructure:"model"`
			InputTypes  []string `mapstructure:"input_types"`
			ThinkEffort string   `mapstructure:"think_effort"`
		}{
			Provider: "ollama",
			Model:    "test-model",
		},
		Shell: struct {
			Confirm         bool     `mapstructure:"confirm"`
			AllowedCommands []string `mapstructure:"allowed_commands"`
		}{
			Confirm:         true,
			AllowedCommands: []string{"ls", "pwd"},
		},
	}

	err := SaveConfig(cfg)
	if err != nil {
		t.Fatalf("SaveConfig() error = %v", err)
	}

	data, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("Failed to read config file: %v", err)
	}

	if !bytes.Contains(data, []byte("model: test-model")) {
		t.Errorf("Config file missing model, got: %s", string(data))
	}
	if !bytes.Contains(data, []byte("provider: ollama")) {
		t.Errorf("Config file missing provider, got: %s", string(data))
	}
}

func TestSaveConfigEmptyPath(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatalf("Failed to create temp config dir: %v", err)
	}

	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) {
		return configPath, nil
	}
	defer func() { getConfigPathFunc = origGetConfigPath }()

	cfg := &Config{
		LogLevel: "info",
		LLM: struct {
			Provider    string   `mapstructure:"provider"`
			Model       string   `mapstructure:"model"`
			InputTypes  []string `mapstructure:"input_types"`
			ThinkEffort string   `mapstructure:"think_effort"`
		}{
			Provider: "gemini",
			Model:    "gemini-3.7-flash",
		},
	}

	err := SaveConfig(cfg)
	if err != nil {
		t.Fatalf("SaveConfig() error = %v", err)
	}

	configFile := filepath.Join(configPath, "config.yaml")
	data, err := os.ReadFile(configFile)
	if err != nil {
		t.Fatalf("Failed to read config file: %v", err)
	}

	if !bytes.Contains(data, []byte("model: gemini-3.7-flash")) {
		t.Errorf("Config file missing model, got: %s", string(data))
	}
}

func TestGetLlamacppModelsExcludesMMProj(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	// Isolate from the real ~/.config/edgebot config and avoid network
	// lookups inside LoadConfig.
	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) { return t.TempDir(), nil }
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{t.TempDir()}

	origOpenRouter := getOpenRouterModelsFunc
	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	defer func() { getOpenRouterModelsFunc = origOpenRouter }()

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	for _, name := range []string{
		"Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
		"mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
		"granite4-3b-h.Q4_K_M.gguf",
		"readme.txt",
	} {
		if err := os.WriteFile(filepath.Join(modelsDir, name), []byte("m"), 0644); err != nil {
			t.Fatalf("Failed to write file %s: %v", name, err)
		}
	}

	models := GetLlamacppModels()
	if len(models) != 2 {
		t.Fatalf("GetLlamacppModels() returned %d models, want 2: %+v", len(models), models)
	}
	for _, m := range models {
		if strings.Contains(strings.ToLower(m.Name), "mmproj") {
			t.Errorf("GetLlamacppModels() listed mmproj model %q", m.Name)
		}
		want := []string(nil)
		if m.Name == "Qwen2.5-VL-3B-Instruct-Q8_0" {
			want = []string{"text", "image"}
		}
		if !slices.Equal(m.InputTypes, want) {
			t.Errorf("GetLlamacppModels() %q InputTypes = %v, want %v", m.Name, m.InputTypes, want)
		}
	}
}

func TestGetLlamacppModelsMatchesMMProjWithDifferentQuant(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) { return t.TempDir(), nil }
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{t.TempDir()}

	origOpenRouter := getOpenRouterModelsFunc
	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	defer func() { getOpenRouterModelsFunc = origOpenRouter }()

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	for _, name := range []string{
		"Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
		"mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
		"granite4-3b-h.Q4_K_M.gguf",
	} {
		if err := os.WriteFile(filepath.Join(modelsDir, name), []byte("m"), 0644); err != nil {
			t.Fatalf("Failed to write file %s: %v", name, err)
		}
	}

	models := GetLlamacppModels()
	if len(models) != 2 {
		t.Fatalf("GetLlamacppModels() returned %d models, want 2: %+v", len(models), models)
	}
	for _, m := range models {
		want := []string(nil)
		if m.Name == "Qwen2.5-VL-3B-Instruct-Q8_0" {
			want = []string{"text", "image"}
		}
		if !slices.Equal(m.InputTypes, want) {
			t.Errorf("GetLlamacppModels() %q InputTypes = %v, want %v", m.Name, m.InputTypes, want)
		}
	}
}

func TestGetLlamacppModelsNoMMProjNoImage(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) { return t.TempDir(), nil }
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{t.TempDir()}

	origOpenRouter := getOpenRouterModelsFunc
	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	defer func() { getOpenRouterModelsFunc = origOpenRouter }()

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelsDir, "granite4-3b-h.Q4_K_M.gguf"), []byte("m"), 0644); err != nil {
		t.Fatalf("Failed to write model file: %v", err)
	}

	models := GetLlamacppModels()
	if len(models) != 1 {
		t.Fatalf("GetLlamacppModels() returned %d models, want 1: %+v", len(models), models)
	}
	if len(models[0].InputTypes) != 0 {
		t.Errorf("GetLlamacppModels() InputTypes = %v, want none (no mmproj present)", models[0].InputTypes)
	}
}

func TestFindLlamacppMMProj(t *testing.T) {
	home := t.TempDir()
	origHome := userHomeDirFunc
	userHomeDirFunc = func() (string, error) { return home, nil }
	defer func() { userHomeDirFunc = origHome }()

	// Isolate from the real ~/.config/edgebot config and avoid network
	// lookups inside LoadConfig.
	origUserConfigDirFunc := userConfigDirFunc
	userConfigDirFunc = func() (string, error) { return t.TempDir(), nil }
	defer func() { userConfigDirFunc = origUserConfigDirFunc }()

	origConfigPaths := configPaths
	defer func() { configPaths = origConfigPaths }()
	configPaths = []string{t.TempDir()}

	origOpenRouter := getOpenRouterModelsFunc
	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	defer func() { getOpenRouterModelsFunc = origOpenRouter }()

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatalf("Failed to create models dir: %v", err)
	}
	projFile := filepath.Join(modelsDir, "mmproj-qwen2.5-vl.gguf")
	if err := os.WriteFile(projFile, []byte("m"), 0644); err != nil {
		t.Fatalf("Failed to write mmproj: %v", err)
	}

	t.Run("auto-detect from models dir", func(t *testing.T) {
		got, err := FindLlamacppMMProj()
		if err != nil {
			t.Fatalf("FindLlamacppMMProj() error = %v", err)
		}
		if got != projFile {
			t.Errorf("FindLlamacppMMProj() = %q, want %q", got, projFile)
		}
	})

	t.Run("missing projector", func(t *testing.T) {
		if err := os.Remove(projFile); err != nil {
			t.Fatalf("Failed to remove mmproj: %v", err)
		}
		got, err := FindLlamacppMMProj()
		if err != nil {
			t.Fatalf("FindLlamacppMMProj() error = %v", err)
		}
		if got != "" {
			t.Errorf("FindLlamacppMMProj() = %q, want empty", got)
		}
	})
}
