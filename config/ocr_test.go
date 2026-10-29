package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mockOCRDirs isolates home/config/routing from the real machine and seeds the
// llamacpp models dir with a MiniCPM-style OCR GGUF, a plain multimodal GGUF,
// and an OCR vision projector whose filename does not match the model.
func mockOCRDirs(t *testing.T) string {
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
	t.Setenv("LLM_OCR_MODEL", "")
	t.Setenv("LLM_OCR_MMPROJ", "")
	t.Setenv("LITERTLM_BACKEND", "")

	modelsDir := filepath.Join(home, ".edgebot", "models", "llamacpp")
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestGGUF(t, filepath.Join(modelsDir, "MiniCPM-OCR-1_0-Q4_K_M.gguf"), []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("qwen35")},
		{"general.name", ggufTypeString, ggufStr("MiniCPM OCR 1.0")},
		{"general.basename", ggufTypeString, ggufStr("MiniCPM-OCR")},
	})
	writeTestGGUF(t, filepath.Join(modelsDir, "plain-vl.gguf"), []ggufKV{
		{"general.architecture", ggufTypeString, ggufStr("gemma")},
		{"general.name", ggufTypeString, ggufStr("Gemma 4 Vision")},
	})
	writeTestGGUF(t, filepath.Join(modelsDir, "mmproj-model-f16-4x.gguf"), []ggufKV{
		{"general.name", ggufTypeString, ggufStr("vision projector")},
	})
	return modelsDir
}

func TestIsOCRModelFile(t *testing.T) {
	dir := t.TempDir()

	nameMarker := filepath.Join(dir, "name.gguf")
	writeTestGGUF(t, nameMarker, []ggufKV{
		{"general.name", ggufTypeString, ggufStr("MiniCPM OCR 1.0")},
	})
	baseMarker := filepath.Join(dir, "basename.gguf")
	writeTestGGUF(t, baseMarker, []ggufKV{
		{"general.name", ggufTypeString, ggufStr("Some Model")},
		{"general.basename", ggufTypeString, ggufStr("MiniCPM-OCR")},
	})
	tmplMarker := filepath.Join(dir, "template.gguf")
	writeTestGGUF(t, tmplMarker, []ggufKV{
		{"general.name", ggufTypeString, ggufStr("Some Model")},
		{"tokenizer.chat_template", ggufTypeString, ggufStr("... Text Recognition: ...")},
	})
	plain := filepath.Join(dir, "plain.gguf")
	writeTestGGUF(t, plain, []ggufKV{
		{"general.name", ggufTypeString, ggufStr("Gemma 4 Vision")},
		{"tokenizer.chat_template", ggufTypeString, ggufStr("<image> describe")},
	})
	substring := filepath.Join(dir, "substring.gguf")
	writeTestGGUF(t, substring, []ggufKV{
		{"general.name", ggufTypeString, ggufStr("Docrata 7B")},
	})

	tests := []struct {
		name string
		path string
		want bool
	}{
		{"name marker", nameMarker, true},
		{"basename marker", baseMarker, true},
		{"chat template marker", tmplMarker, true},
		{"plain multimodal", plain, false},
		{"ocr substring is not a word", substring, false},
		{"missing file", filepath.Join(dir, "absent.gguf"), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsOCRModelFile(tc.path); got != tc.want {
				t.Errorf("IsOCRModelFile() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGetLlamacppModelsOCRType(t *testing.T) {
	mockOCRDirs(t)

	byName := map[string]ModelInfo{}
	for _, m := range GetLlamacppModels() {
		byName[m.Name] = m
	}
	ocr, ok := byName["MiniCPM-OCR-1_0-Q4_K_M"]
	if !ok {
		t.Fatalf("OCR model missing from GetLlamacppModels(): %+v", byName)
	}
	if got := ocr.ModelType(); got != ModelTypeOCR {
		t.Errorf("OCR ModelType() = %q, want %q", got, ModelTypeOCR)
	}
	if !contains(ocr.InputTypes, "image") {
		t.Errorf("OCR InputTypes = %v, want image via the unique projector", ocr.InputTypes)
	}
	plain := byName["plain-vl"]
	if got := plain.ModelType(); got != ModelTypeLLM {
		t.Errorf("plain ModelType() = %q, want %q", got, ModelTypeLLM)
	}
	if contains(plain.InputTypes, "image") {
		t.Errorf("plain InputTypes = %v, want text only (projector is not paired)", plain.InputTypes)
	}
}

func TestFindLlamacppMMProjForModel(t *testing.T) {
	t.Run("unique fallback", func(t *testing.T) {
		mockOCRDirs(t)
		got, err := FindLlamacppMMProjForModel("MiniCPM-OCR-1_0-Q4_K_M")
		if err != nil {
			t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
		}
		if filepath.Base(got) != "mmproj-model-f16-4x.gguf" {
			t.Errorf("projector = %q, want mmproj-model-f16-4x.gguf", got)
		}
	})

	t.Run("exact name wins", func(t *testing.T) {
		dir := mockOCRDirs(t)
		writeTestGGUF(t, filepath.Join(dir, "mmproj-other.gguf"), []ggufKV{
			{"general.name", ggufTypeString, ggufStr("other projector")},
		})
		writeTestGGUF(t, filepath.Join(dir, "mmproj-MiniCPM-OCR-1_0-Q4_K_M.gguf"), []ggufKV{
			{"general.name", ggufTypeString, ggufStr("matching projector")},
		})
		got, err := FindLlamacppMMProjForModel("MiniCPM-OCR-1_0-Q4_K_M")
		if err != nil {
			t.Fatalf("FindLlamacppMMProjForModel() = %v", err)
		}
		if filepath.Base(got) != "mmproj-MiniCPM-OCR-1_0-Q4_K_M.gguf" {
			t.Errorf("projector = %q, want the exact name match", got)
		}
	})

	t.Run("ambiguous", func(t *testing.T) {
		dir := mockOCRDirs(t)
		writeTestGGUF(t, filepath.Join(dir, "mmproj-other.gguf"), []ggufKV{
			{"general.name", ggufTypeString, ggufStr("other projector")},
		})
		if _, err := FindLlamacppMMProjForModel("plain-vl"); err == nil {
			t.Fatal("FindLlamacppMMProjForModel() = nil error, want ambiguity error")
		}
	})
}

func TestIsOCRModel(t *testing.T) {
	mockOCRDirs(t)
	if !IsOCRModel("MiniCPM-OCR-1_0-Q4_K_M") {
		t.Error("IsOCRModel(MiniCPM-OCR-1_0-Q4_K_M) = false, want true")
	}
	if IsOCRModel("plain-vl") {
		t.Error("IsOCRModel(plain-vl) = true, want false")
	}
	if IsOCRModel("absent") {
		t.Error("IsOCRModel(absent) = true, want false")
	}
}

func TestSaveOCRModel(t *testing.T) {
	mockOCRDirs(t)

	if err := SaveOCRModel("MiniCPM-OCR-1_0-Q4_K_M"); err != nil {
		t.Fatalf("SaveOCRModel() = %v", err)
	}
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.OCRModel != "MiniCPM-OCR-1_0-Q4_K_M" {
		t.Errorf("OCRModel = %q, want MiniCPM-OCR-1_0-Q4_K_M", cfg.LLM.OCRModel)
	}
	if cfg.LLM.Model == cfg.LLM.OCRModel {
		t.Error("chat model was clobbered with the OCR model")
	}

	if err := SaveOCRModel("plain-vl"); err == nil {
		t.Error("SaveOCRModel(plain-vl) succeeded, want error")
	}
}

func TestSetDefaultModelOCR(t *testing.T) {
	mockOCRDirs(t)

	kind, err := SetDefaultModel("MiniCPM-OCR-1_0-Q4_K_M", "llamacpp")
	if err != nil {
		t.Fatalf("SetDefaultModel() = %v", err)
	}
	if kind != ModelTypeOCR {
		t.Errorf("kind = %q, want %q", kind, ModelTypeOCR)
	}
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.OCRModel != "MiniCPM-OCR-1_0-Q4_K_M" {
		t.Errorf("OCRModel = %q, want MiniCPM-OCR-1_0-Q4_K_M", cfg.LLM.OCRModel)
	}
}

func TestOCRModelEnvOverride(t *testing.T) {
	mockOCRDirs(t)
	t.Setenv("LLM_OCR_MODEL", "MiniCPM-OCR-1_0-Q4_K_M")
	t.Setenv("LLM_OCR_MMPROJ", "mmproj-model-f16-4x")
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.OCRModel != "MiniCPM-OCR-1_0-Q4_K_M" {
		t.Errorf("OCRModel = %q, want MiniCPM-OCR-1_0-Q4_K_M", cfg.LLM.OCRModel)
	}
	if cfg.LLM.OCRMMProj != "mmproj-model-f16-4x" {
		t.Errorf("OCRMMProj = %q, want mmproj-model-f16-4x", cfg.LLM.OCRMMProj)
	}
}

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func TestIsOCRName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"MiniCPM OCR 1.0", true},
		{"MiniCPM-OCR", true},
		{"mistralai/mistral-ocr-latest", true},
		{"Docrata 7B", false},
		{"Gemma 4 Vision", false},
	}
	for _, tc := range tests {
		if got := IsOCRName(tc.name); got != tc.want {
			t.Errorf("IsOCRName(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestSaveConfigWritesOCRModel(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "edgebot")
	if err := os.MkdirAll(configPath, 0755); err != nil {
		t.Fatal(err)
	}
	origGetConfigPath := getConfigPathFunc
	getConfigPathFunc = func() (string, error) { return configPath, nil }
	defer func() { getConfigPathFunc = origGetConfigPath }()

	cfg := &Config{ConfigFile: filepath.Join(configPath, "config.yaml")}
	cfg.LLM.OCRModel = "MiniCPM-OCR-1_0-Q4_K_M"
	cfg.LLM.OCRMMProj = "mmproj-model-f16-4x"
	if err := SaveConfig(cfg); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(cfg.ConfigFile)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"ocr_model: MiniCPM-OCR-1_0-Q4_K_M", "ocr_mmproj: mmproj-model-f16-4x"} {
		if !strings.Contains(string(data), want) {
			t.Errorf("config missing %q:\n%s", want, data)
		}
	}
}

func TestModelTypeOCR(t *testing.T) {
	if got := (ModelInfo{Type: ModelTypeOCR}).ModelType(); got != ModelTypeOCR {
		t.Errorf("OCR ModelInfo ModelType() = %q, want %q", got, ModelTypeOCR)
	}
}
