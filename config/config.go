package config

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/api"
	"github.com/spf13/viper"
	"github.com/subosito/gotenv"
)

type CommandInfo struct {
	Name        string
	Description string
	Prompt      string
	Schema      string
}

type Config struct {
	ConfigFile string
	LogLevel   string `mapstructure:"log_level"`
	Agent      string `mapstructure:"agent"`
	AgentFiles bool   `mapstructure:"agent_files"`
	Skills     bool   `mapstructure:"skills"`
	LLM        struct {
		Provider    string   `mapstructure:"provider"`
		Model       string   `mapstructure:"model"`
		InputTypes  []string `mapstructure:"input_types"`
		ThinkEffort string   `mapstructure:"think_effort"`
	} `mapstructure:"llm"`
	Shell struct {
		Confirm         bool     `mapstructure:"confirm"`
		AllowedCommands []string `mapstructure:"allowed_commands"`
	} `mapstructure:"shell"`
	LitertLM struct {
		Backend string `mapstructure:"backend"`
	} `mapstructure:"litertlm"`
	Tools    map[string]bool   `mapstructure:"tools"`
	Commands map[string]string `mapstructure:"commands"`
}

var configPaths = []string{"."}

var userConfigDirFunc = os.UserConfigDir

var userHomeDirFunc = os.UserHomeDir

var loadEnvFunc = loadEnv

// defaultTools is the default set of enabled tools, used when a config has no
// tools map.
var defaultTools = map[string]bool{
	"RunCommand": true,
	"WriteFile":  true,
	"ReadFile":   true,
	"KVSet":      true,
	"KVGet":      true,
	"KVList":     true,
}

func loadEnv() error {
	// Load from user config directory first (global defaults)
	userConfigDir, err := userConfigDirFunc()
	if err == nil {
		globalEnvPath := filepath.Join(userConfigDir, "edgebot", ".env")
		if _, err := os.Stat(globalEnvPath); err == nil {
			if err := gotenv.Load(globalEnvPath); err != nil {
				return fmt.Errorf("error loading global .env file at %s: %w", globalEnvPath, err)
			}
		}
	}

	// Load from current directory (local overrides)
	envPath := ".env"
	if _, err := os.Stat(envPath); err == nil {
		if err := gotenv.Load(envPath); err != nil {
			return fmt.Errorf("error loading .env file: %w", err)
		}
	}

	return nil
}

func InitLogger(level string) {
	var slogLevel slog.Level
	switch level {
	case "debug":
		slogLevel = slog.LevelDebug
	case "info":
		slogLevel = slog.LevelInfo
	case "warn":
		slogLevel = slog.LevelWarn
	case "error":
		slogLevel = slog.LevelError
	default:
		slogLevel = slog.LevelInfo
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slogLevel})
	slog.SetDefault(slog.New(handler))
}

func LoadConfig() (*Config, error) {
	if err := loadEnvFunc(); err != nil {
		return nil, fmt.Errorf("error loading .env file: %w", err)
	}

	v := viper.New()
	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.SetDefault("llm.provider", "ollama")
	v.SetDefault("llm.model", "granite4:3b-h")
	v.SetDefault("shell.confirm", true)
	v.SetDefault("shell.allowed_commands", []string{"ls", "pwd", "git"})
	v.SetDefault("log_level", "info")
	v.SetDefault("agent", "build")
	v.SetDefault("agent_files", true)
	v.SetDefault("skills", true)
	v.SetDefault("litertlm.backend", "cpu")
	toolsCopy := make(map[string]bool, len(defaultTools))
	for k, v := range defaultTools {
		toolsCopy[k] = v
	}
	v.SetDefault("tools", toolsCopy)

	for _, path := range configPaths {
		v.AddConfigPath(path)
	}

	userConfigDir, err := userConfigDirFunc()
	var configPath string
	if err == nil {
		configPath = filepath.Join(userConfigDir, "edgebot")
		v.AddConfigPath(configPath)
	}

	if err := v.ReadInConfig(); err != nil {
		var notFound viper.ConfigFileNotFoundError
		if errors.As(err, &notFound) {
			toolsClone := make(map[string]bool, len(defaultTools))
			for k, v := range defaultTools {
				toolsClone[k] = v
			}
			defaultConfig := &Config{
				ConfigFile: "",
				LogLevel:   "info",
				Agent:      "build",
				AgentFiles: true,
				Skills:     true,
				LLM: struct {
					Provider    string   `mapstructure:"provider"`
					Model       string   `mapstructure:"model"`
					InputTypes  []string `mapstructure:"input_types"`
					ThinkEffort string   `mapstructure:"think_effort"`
				}{
					Provider:   "ollama",
					Model:      "granite4:3b-h",
					InputTypes: []string{"text"},
				},
				Shell: struct {
					Confirm         bool     `mapstructure:"confirm"`
					AllowedCommands []string `mapstructure:"allowed_commands"`
				}{
					Confirm:         true,
					AllowedCommands: []string{"ls", "pwd", "git"},
				},
				LitertLM: struct {
					Backend string `mapstructure:"backend"`
				}{
					Backend: "cpu",
				},
				Tools: toolsClone,
			}

			if configPath != "" {
				if mkErr := os.MkdirAll(configPath, 0o755); mkErr == nil {
					defaultConfigFile := filepath.Join(configPath, "config.yaml")
					if _, statErr := os.Stat(defaultConfigFile); os.IsNotExist(statErr) {
						content := "log_level: \"info\"\nagent: \"build\"\nagent_files: true\nskills: true\nllm:\n  provider: \"ollama\"\n  model: \"granite4:3b-h\"\n  input_types:\n    - \"text\"\nshell:\n  confirm: true\n  allowed_commands:\n    - \"ls\"\n    - \"pwd\"\n    - \"git\"\nlitertlm:\n  backend: \"cpu\"\ntools:\n  RunCommand: true\n  WriteFile: true\n  ReadFile: true\n  KVSet: true\n  KVGet: true\n  KVList: true\n"
						if writeErr := os.WriteFile(defaultConfigFile, []byte(content), 0o644); writeErr != nil {
							slog.Warn("failed to write default config", "err", writeErr)
						} else {
							defaultConfig.ConfigFile = defaultConfigFile
						}
					}
				} else {
					slog.Warn("failed to create config dir", "err", mkErr)
				}
			}

			applyEnvOverrides(defaultConfig)

			return defaultConfig, nil
		}
		return nil, fmt.Errorf("error reading config file: %w", err)
	}

	var config Config
	if err := v.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("error unmarshaling config: %w", err)
	}
	config.ConfigFile = v.ConfigFileUsed()

	applyEnvOverrides(&config)

	if config.Tools == nil {
		clone := make(map[string]bool, len(defaultTools))
		for k, v := range defaultTools {
			clone[k] = v
		}
		config.Tools = clone
	}

	if len(config.LLM.InputTypes) == 0 {
		if info := lookupModelInfo(config.LLM.Model); info != nil && len(info.InputTypes) > 0 {
			config.LLM.InputTypes = info.InputTypes
		}
	}

	config.LLM.ThinkEffort = normalizeThinkEffort(config.LLM.ThinkEffort)

	return &config, nil
}

// normalizeThinkEffort lowercases/trims llm.think_effort; unknown values are
// reset to "" (provider default) with a warning so old configs keep loading.
// applyEnvOverrides lets explicit environment variables override the loaded
// configuration (built-in defaults or config file), so containers can select
// the provider/model without editing config.yaml:
//
//	LLM_PROVIDER     overrides llm.provider
//	LLM_MODEL        overrides llm.model (when LLM_PROVIDER is unset, the
//	                 provider is auto-detected via LookupModelInfo, like
//	                 `edgebot config --model` does)
//	LITERTLM_BACKEND overrides litertlm.backend
func applyEnvOverrides(cfg *Config) {
	if v := strings.TrimSpace(os.Getenv("LITERTLM_BACKEND")); v != "" {
		cfg.LitertLM.Backend = v
	}
	if v := strings.TrimSpace(os.Getenv("LLM_MODEL")); v != "" {
		cfg.LLM.Model = v
		if strings.TrimSpace(os.Getenv("LLM_PROVIDER")) == "" {
			if info := LookupModelInfo(v); info != nil {
				cfg.LLM.Provider = info.Provider
				cfg.LLM.InputTypes = info.InputTypes
			}
		}
	}
	if v := strings.TrimSpace(os.Getenv("LLM_PROVIDER")); v != "" {
		cfg.LLM.Provider = strings.ToLower(v)
	}
}

func normalizeThinkEffort(s string) string {
	v := strings.ToLower(strings.TrimSpace(s))
	if v == "" {
		return ""
	}
	switch v {
	case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		return v
	default:
		slog.Warn("invalid llm.think_effort, using provider default",
			"value", s)
		return ""
	}
}

var getConfigPathFunc = getConfigPath

func getConfigPath() (string, error) {
	userConfigDir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	configPath := filepath.Join(userConfigDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		return "", err
	}
	return configPath, nil
}

// EdgebotDir returns ~/.edgebot, creating it if necessary.
func EdgebotDir() (string, error) {
	home, err := userHomeDirFunc()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".edgebot")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	return dir, nil
}

// LibDir returns ~/.edgebot/lib, the directory holding the in-process
// inference shared libraries (llama.cpp / LiteRT-LM).
func LibDir() (string, error) {
	dir, err := EdgebotDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "lib"), nil
}

// ModelsDir returns ~/.edgebot/models/<provider> for the given provider.
func ModelsDir(provider string) (string, error) {
	dir, err := EdgebotDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "models", provider), nil
}

func modelInList(modelName string, models []ModelInfo) bool {
	for _, m := range models {
		if m.Name == modelName {
			return true
		}
	}
	return false
}

func IsLitertLMModel(modelName string) bool {
	return modelInList(modelName, GetLitertLMModels())
}

func SaveConfig(cfg *Config) error {
	configFile := cfg.ConfigFile
	if configFile == "" {
		configPath, err := getConfigPathFunc()
		if err != nil {
			return fmt.Errorf("failed to get config path: %w", err)
		}
		configFile = filepath.Join(configPath, "config.yaml")
	}

	out := struct {
		LogLevel   string `yaml:"log_level"`
		Agent      string `yaml:"agent,omitempty"`
		AgentFiles bool   `yaml:"agent_files"`
		Skills     bool   `yaml:"skills"`
		LLM        struct {
			Provider    string   `yaml:"provider"`
			Model       string   `yaml:"model"`
			InputTypes  []string `yaml:"input_types,omitempty"`
			ThinkEffort string   `yaml:"think_effort,omitempty"`
		} `yaml:"llm"`
		Shell struct {
			Confirm         bool     `yaml:"confirm"`
			AllowedCommands []string `yaml:"allowed_commands,omitempty"`
		} `yaml:"shell"`
		LitertLM struct {
			Backend string `yaml:"backend"`
		} `yaml:"litertlm"`
		Tools    map[string]bool   `yaml:"tools,omitempty"`
		Commands map[string]string `yaml:"commands,omitempty"`
	}{
		LogLevel:   cfg.LogLevel,
		Agent:      cfg.Agent,
		AgentFiles: cfg.AgentFiles,
		Skills:     cfg.Skills,
		Tools:      cfg.Tools,
		Commands:   cfg.Commands,
	}
	out.LLM.Provider = cfg.LLM.Provider
	out.LLM.Model = cfg.LLM.Model
	out.LLM.InputTypes = cfg.LLM.InputTypes
	out.LLM.ThinkEffort = cfg.LLM.ThinkEffort
	out.Shell.Confirm = cfg.Shell.Confirm
	out.Shell.AllowedCommands = cfg.Shell.AllowedCommands
	out.LitertLM.Backend = cfg.LitertLM.Backend

	data, err := yaml.Marshal(out)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(configFile, data, 0644); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	return nil
}

func lookupModelInfo(modelName string) *ModelInfo {
	allModels := append([]ModelInfo{}, GeminiModels...)
	allModels = append(allModels, getOpenRouterModelsFunc()...)
	for _, m := range allModels {
		if m.Name == modelName {
			return &m
		}
	}
	return nil
}

func LookupModelInfo(modelName string) *ModelInfo {
	for _, m := range GetAllAvailableModels() {
		if m.Name == modelName {
			return &m
		}
	}
	return nil
}

func SaveModelWithProvider(modelName, provider string) error {
	cfg, err := LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	cfg.LLM.Model = modelName
	if provider != "" {
		cfg.LLM.Provider = provider
	} else if modelInList(modelName, GeminiModels) {
		cfg.LLM.Provider = "gemini"
	} else if IsLitertLMModel(modelName) {
		cfg.LLM.Provider = "litertlm"
	} else if modelInList(modelName, getOpenRouterModelsFunc()) {
		cfg.LLM.Provider = "openrouter"
	} else if IsLlamacppModel(modelName) {
		cfg.LLM.Provider = "llamacpp"
	} else {
		cfg.LLM.Provider = "ollama"
	}

	if info := LookupModelInfo(modelName); info != nil && len(info.InputTypes) > 0 {
		cfg.LLM.InputTypes = info.InputTypes
	} else {
		cfg.LLM.InputTypes = []string{"text"}
	}

	return SaveConfig(cfg)
}

type ModelInfo struct {
	Name       string
	Provider   string
	Size       string
	ModifiedAt string
	InputTypes []string
}

var GeminiModels = []ModelInfo{
	{Name: "gemini-3.7-flash", Provider: "gemini", InputTypes: []string{"text", "image", "audio"}},
	{Name: "gemini-3.5-flash-lite", Provider: "gemini", InputTypes: []string{"text", "image", "audio"}},
	{Name: "gemma-4-31b-it", Provider: "gemini", InputTypes: []string{"text", "image"}},
	{Name: "gemma-4-26b-a4b-it", Provider: "gemini", InputTypes: []string{"text", "image"}},
}

// GetLitertLMModels lists the native LiteRT-LM models available on disk by
// scanning ~/.edgebot/models/litertlm/ for .litertlm files.
func GetLitertLMModels() []ModelInfo {
	dir, err := ModelsDir("litertlm")
	if err != nil {
		return nil
	}
	return scanModels(dir, "litertlm", ".litertlm")
}

// GetLlamacppModels lists the GGUF models on disk (excluding vision projector
// files). Models whose base name matches an available mmproj file get the
// image input type.
func GetLlamacppModels() []ModelInfo {
	dir, err := ModelsDir("llamacpp")
	if err != nil {
		return nil
	}
	all := scanModels(dir, "llamacpp", ".gguf")
	visionKeys := map[string]bool{}
	var models []ModelInfo
	for _, m := range all {
		if strings.Contains(strings.ToLower(m.Name), "mmproj") {
			visionKeys[llamacppVisionKey(m.Name)] = true
		}
	}
	for _, m := range all {
		if strings.Contains(strings.ToLower(m.Name), "mmproj") {
			continue
		}
		if visionKeys[llamacppVisionKey(m.Name)] {
			m.InputTypes = []string{"text", "image"}
		}
		models = append(models, m)
	}
	return models
}

// llamacppQuantRe matches the quantization suffix of a GGUF base name (e.g.
// -Q8_0, -Q4_K_M, -f16) so a model can be matched to its vision projector
// even when their quantizations differ.
var llamacppQuantRe = regexp.MustCompile(`-[Qq][0-9][A-Za-z0-9._]*$|-(?:f16|F16|bf16|BF16)$`)

// llamacppVisionKey normalizes a GGUF base name (mmproj or model) to an
// identity used to pair a model with its vision projector: it strips the
// mmproj marker prefix/suffix and any quantization suffix.
func llamacppVisionKey(name string) string {
	s := strings.TrimPrefix(name, "mmproj-")
	s = strings.TrimSuffix(s, "-mmproj")
	return llamacppQuantRe.ReplaceAllString(s, "")
}

// FindLlamacppMMProj resolves the vision projector (mmproj) GGUF file used for
// image input by scanning the llamacpp models dir for a file whose name
// contains "mmproj". Returns "" when no projector is present.
func FindLlamacppMMProj() (string, error) {
	dir, err := ModelsDir("llamacpp")
	if err != nil {
		return "", err
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("scan mmproj: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		lower := strings.ToLower(entry.Name())
		if strings.HasSuffix(lower, ".gguf") && strings.Contains(lower, "mmproj") {
			return filepath.Join(dir, entry.Name()), nil
		}
	}
	return "", nil
}

// scanModels lists the files with the given extension in dir as ModelInfos.
func scanModels(dir, provider, ext string) []ModelInfo {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("failed to scan models", "dir", dir, "err", err)
		}
		return nil
	}
	var models []ModelInfo
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(strings.ToLower(name), strings.ToLower(ext)) {
			continue
		}
		// Trim suffix case-insensitively.
		modelName := name[:len(name)-len(ext)]
		info, err := entry.Info()
		size := ""
		if err == nil {
			size = FormatFileSize(info.Size())
		}
		models = append(models, ModelInfo{
			Name:     modelName,
			Provider: provider,
			Size:     size,
		})
	}
	return models
}

func IsLlamacppModel(modelName string) bool {
	return modelInList(modelName, GetLlamacppModels())
}

// DeleteLocalModel removes a locally downloaded model file (llamacpp GGUF or
// litertlm .litertlm) and returns the paths of the removed files. Deleting a
// llamacpp model also deletes the vision projector (mmproj) files paired with
// it (same vision key, e.g. mmproj-<model>-f16.gguf). Remote models
// (ollama/gemini/openrouter) cannot be deleted.
func DeleteLocalModel(modelName string) ([]string, error) {
	if IsLitertLMModel(modelName) {
		dir, err := ModelsDir("litertlm")
		if err != nil {
			return nil, err
		}
		path, err := removeModelFile(filepath.Join(dir, modelName+".litertlm"))
		if err != nil {
			return nil, err
		}
		return []string{path}, nil
	}
	if IsLlamacppModel(modelName) {
		dir, err := ModelsDir("llamacpp")
		if err != nil {
			return nil, err
		}
		path, err := removeModelFile(filepath.Join(dir, modelName+".gguf"))
		if err != nil {
			return nil, err
		}
		removed := []string{path}
		for _, proj := range llamacppMMProjsForModel(dir, modelName) {
			projPath := filepath.Join(dir, proj+".gguf")
			if _, err := removeModelFile(projPath); err != nil {
				if !os.IsNotExist(err) {
					slog.Warn("failed to delete vision projector", "path", projPath, "error", err)
				}
				continue
			}
			removed = append(removed, projPath)
		}
		return removed, nil
	}
	return nil, fmt.Errorf("model %q is not a locally downloaded model", modelName)
}

// llamacppMMProjsForModel lists the vision projector base names in dir whose
// vision key matches the given model's, so deleting the model also removes its
// paired mmproj files.
func llamacppMMProjsForModel(dir, modelName string) []string {
	key := llamacppVisionKey(modelName)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("failed to scan mmproj list", "dir", dir, "err", err)
		}
		return nil
	}
	var projs []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		lower := strings.ToLower(name)
		if !strings.HasSuffix(lower, ".gguf") || !strings.Contains(lower, "mmproj") {
			continue
		}
		// Case-insensitive suffix trim.
		base := name[:len(name)-len(".gguf")]
		if llamacppVisionKey(base) == key {
			projs = append(projs, base)
		}
	}
	return projs
}

func removeModelFile(path string) (string, error) {
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("model file not found at %s", path)
		}
		return "", fmt.Errorf("failed to delete model file: %w", err)
	}
	return path, nil
}

// openRouterModelsURL is the OpenRouter API endpoint listing all models.
var openRouterModelsURL = "https://openrouter.ai/api/v1/models"

// openRouterModelsCache caches the fetched list of free OpenRouter models for
// a short TTL to avoid hitting the API on every models listing.
var (
	openRouterModelsCache    []ModelInfo
	openRouterModelsCached   time.Time
	openRouterModelsCacheTTL = 10 * time.Minute
	openRouterModelsMu       sync.Mutex
)

// getOpenRouterModelsFunc is swappable for tests.
var getOpenRouterModelsFunc = GetOpenRouterModels

// GetOpenRouterModels returns the list of free OpenRouter models, fetched
// from the OpenRouter API and cached briefly.
func GetOpenRouterModels() []ModelInfo {
	openRouterModelsMu.Lock()
	if openRouterModelsCache != nil && time.Since(openRouterModelsCached) < openRouterModelsCacheTTL {
		cached := openRouterModelsCache
		openRouterModelsMu.Unlock()
		return cached
	}
	openRouterModelsMu.Unlock()

	fetched := fetchOpenRouterFreeModels()

	openRouterModelsMu.Lock()
	openRouterModelsCache = fetched
	openRouterModelsCached = time.Now()
	cached := openRouterModelsCache
	openRouterModelsMu.Unlock()
	return cached
}

type openRouterModelsResponse struct {
	Data []struct {
		ID      string `json:"id"`
		Pricing struct {
			Prompt     string `json:"prompt"`
			Completion string `json:"completion"`
		} `json:"pricing"`
		Architecture struct {
			InputModalities  []string `json:"input_modalities"`
			OutputModalities []string `json:"output_modalities"`
		} `json:"architecture"`
	} `json:"data"`
}

// fetchOpenRouterFreeModels GETs openRouterModelsURL and keeps only the models
// whose prompt and completion pricing is zero and whose output modalities do
// not include audio (excludes music/sound generation models like Lyria).
func fetchOpenRouterFreeModels() []ModelInfo {
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(openRouterModelsURL)
	if err != nil {
		slog.Debug("failed to fetch OpenRouter models", "error", err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		slog.Debug("openrouter models request failed", "status", resp.StatusCode)
		return nil
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		slog.Debug("failed to read OpenRouter models response", "error", err)
		return nil
	}
	var payload openRouterModelsResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		slog.Debug("failed to parse OpenRouter models response", "error", err)
		return nil
	}
	var models []ModelInfo
	for _, m := range payload.Data {
		if !isZeroPrice(m.Pricing.Prompt) || !isZeroPrice(m.Pricing.Completion) {
			continue
		}
		if slices.Contains(m.Architecture.OutputModalities, "audio") {
			continue
		}
		inputTypes := m.Architecture.InputModalities
		if len(inputTypes) == 0 {
			inputTypes = []string{"text"}
		}
		models = append(models, ModelInfo{Name: m.ID, Provider: "openrouter", InputTypes: inputTypes})
	}
	return models
}

func isZeroPrice(s string) bool {
	f, err := strconv.ParseFloat(s, 64)
	return err == nil && f == 0
}

var getAvailableModelsFunc = GetAvailableModels

func GetAllAvailableModels() []ModelInfo {
	var models []ModelInfo
	if ollamaModels, err := getAvailableModelsFunc(); err == nil {
		models = append(models, ollamaModels...)
	}
	models = append(models, GeminiModels...)
	models = append(models, getOpenRouterModelsFunc()...)
	models = append(models, GetLitertLMModels()...)
	models = append(models, GetLlamacppModels()...)
	return models
}

func GetAvailableModels() ([]ModelInfo, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	models, err := client.List(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	var modelList []ModelInfo
	for _, model := range models.Models {
		modelList = append(modelList, ModelInfo{
			Name:     model.Name,
			Provider: "ollama",
		})
	}

	return modelList, nil
}

func SelectModel() error {
	cfg, err := LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	models := GetAllAvailableModels()

	if len(models) == 0 {
		fmt.Printf("No models found. Please install models using 'ollama pull <model>'\n")
		return nil
	}

	fmt.Printf("Available Models:\n\n")

	for i, model := range models {
		marker := "  "
		if model.Name == cfg.LLM.Model {
			marker = "* "
		}
		fmt.Printf("[%d] %s%s (%s)\n", i+1, marker, model.Name, model.Provider)
	}

	fmt.Printf("\nEnter number to select model (or press Enter to cancel): ")

	var input string
	fmt.Scanln(&input)
	input = strings.TrimSpace(input)

	if input == "" {
		fmt.Printf("Selection cancelled.\n")
		return nil
	}

	choice, err := strconv.Atoi(input)
	if err != nil || choice < 1 || choice > len(models) {
		fmt.Printf("Invalid selection.\n")
		return nil
	}

	selectedModel := models[choice-1].Name
	selectedProvider := models[choice-1].Provider
	return SaveModelWithProvider(selectedModel, selectedProvider)
}

func IsAllowedCommand(cmd string, allowedList []string) bool {
	if len(allowedList) == 0 {
		return false
	}
	for _, a := range allowedList {
		a = strings.TrimSpace(a)
		if a == cmd {
			return true
		}
	}
	return false
}

// GetCommandName returns the first word of a shell command string.
func GetCommandName(cmd string) string {
	parts := strings.Fields(cmd)
	if len(parts) > 0 {
		return parts[0]
	}
	return cmd
}

func GetEnvPaths() []string {
	var paths []string

	userConfigDir, err := userConfigDirFunc()
	if err == nil {
		globalEnvPath := filepath.Join(userConfigDir, "edgebot", ".env")
		paths = append(paths, globalEnvPath)
	}

	paths = append(paths, ".env")

	return paths
}

var getUserConfigDirFunc = os.UserConfigDir

func LoadCommands(cfg *Config) []CommandInfo {
	var fileCmds []CommandInfo
	dirs := loadCommandDirs()
	for _, dir := range dirs {
		cmds, err := LoadCommandsFromDir(dir)
		if err != nil {
			slog.Debug("failed to load commands", "dir", dir, "err", err)
			continue
		}
		fileCmds = append(fileCmds, cmds...)
	}

	configCmds := make(map[string]CommandInfo)
	for name, prompt := range cfg.Commands {
		configCmds[name] = CommandInfo{
			Name:        name,
			Description: prompt,
			Prompt:      prompt,
		}
	}

	for _, cmd := range fileCmds {
		configCmds[cmd.Name] = cmd
	}

	var result []CommandInfo
	for _, cmd := range configCmds {
		result = append(result, cmd)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})
	return result
}

func LoadCommandsFromDir(dir string) ([]CommandInfo, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create commands directory: %w", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read commands directory: %w", err)
	}

	var commands []CommandInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		cmd, err := parseCommandFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			slog.Warn("skip command file", "path", filepath.Join(dir, entry.Name()), "err", err)
			continue
		}
		cmd.Name = strings.TrimSuffix(entry.Name(), ".md")
		commands = append(commands, cmd)
	}

	return commands, nil
}

func loadCommandDirs() []string {
	var dirs []string

	homeDir, err := os.UserHomeDir()
	if err == nil {
		dirs = append(dirs, filepath.Join(homeDir, ".edgebot", "commands"))
	}

	cwd, err := os.Getwd()
	if err == nil {
		dirs = append(dirs, filepath.Join(cwd, ".edgebot", "commands"))
	}

	return dirs
}

func parseCommandFile(path string) (CommandInfo, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return CommandInfo{}, err
	}

	content := string(data)
	var cmd CommandInfo

	if strings.HasPrefix(content, "---") {
		parts := strings.SplitN(content, "---", 3)
		if len(parts) >= 3 {
			frontmatter := strings.TrimSpace(parts[1])
			var meta struct {
				Description string `yaml:"description"`
				Schema      string `yaml:"schema"`
			}
			if err := yaml.Unmarshal([]byte(frontmatter), &meta); err != nil {
				slog.Warn("invalid command frontmatter", "path", path, "err", err)
			} else {
				cmd.Description = meta.Description
				if meta.Schema != "" {
					cmd.Schema = filepath.Join(filepath.Dir(path), meta.Schema)
				}
			}
			cmd.Prompt = strings.TrimSpace(parts[2])
			return cmd, nil
		}
	}

	cmd.Prompt = strings.TrimSpace(content)
	return cmd, nil
}

func EnsureCommandsDir() error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	localDir := filepath.Join(cwd, ".edgebot", "commands")
	return os.MkdirAll(localDir, 0755)
}

// FormatFileSize renders a byte count as a human-readable size (e.g. "1.5 MB").
func FormatFileSize(b int64) string {
	const unit = 1024
	if b < 0 {
		b = 0
	}
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
		if exp >= len("KMGTPE")-1 {
			break
		}
	}
	if exp >= len("KMGTPE") {
		exp = len("KMGTPE") - 1
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}
