package llm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLitertLMClientMissingLib(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LITERTLM_LIB", "")
	t.Setenv("LITERTLM_BACKEND", "")

	// Model file present so resolution gets past the model check and
	// reaches the shared-library pre-flight check with an empty lib dir.
	modelDir := filepath.Join(home, ".edgebot", "models", "litertlm")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "test.litertlm"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	l := &LitertLMCaller{Model: "test", Executor: NoopExecutor{}}
	_, err := l.client(context.Background())
	if err == nil {
		t.Fatal("client() = nil error, want missing-library error")
	}
	if !strings.Contains(err.Error(), "liblitert-lm.so") {
		t.Errorf("client() error = %q, want it to name the missing unified library", err)
	}
	if !strings.Contains(err.Error(), "edgebot litertlm install") {
		t.Errorf("client() error = %q, want remediation hint", err)
	}
}

func TestLitertLMClientMissingModel(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("LITERTLM_LIB", "")
	t.Setenv("LITERTLM_BACKEND", "")

	l := &LitertLMCaller{Model: "does-not-exist", Executor: NoopExecutor{}}
	_, err := l.client(context.Background())
	if err == nil {
		t.Fatal("client() = nil error, want missing-model error")
	}
	if !strings.Contains(err.Error(), "model file not found") {
		t.Errorf("client() error = %q, want missing-model error", err)
	}
}
