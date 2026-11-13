package config

import (
	"os"
	"testing"
)

// TestMain stubs the live network fetchers for the whole package and points
// ~/.edgebot at an empty temp dir.
//
// Several tests exercise model lookup and config saving, which reach
// GetOpenRouterModels / GetAvailableModels and scan ~/.edgebot/models. Without
// the stubs they depend on the real openrouter.ai API and a local Ollama (the
// suite then fails on offline machines and in CI), and without the temp home
// they scan the developer's multi-GB GGUF files, which is slow and makes
// results depend on what is installed locally.
//
// Individual tests may still swap these funcs; they restore to the values
// installed here. The real fetchers and GGUF scanners stay covered by the
// httptest/table tests in openrouter_test.go and gguf_test.go.
func TestMain(m *testing.M) {
	home, err := os.MkdirTemp("", "edgebot-config-test-*")
	if err != nil {
		panic("config: cannot create test home: " + err.Error())
	}

	getOpenRouterModelsFunc = func() []ModelInfo { return nil }
	getAvailableModelsFunc = func() ([]ModelInfo, error) { return nil, nil }
	userHomeDirFunc = func() (string, error) { return home, nil }

	code := m.Run()

	_ = os.RemoveAll(home)
	os.Exit(code)
}
