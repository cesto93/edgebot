package cmd

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunLitertLMInstallReusesUnifiedLibAndCleansLegacy(t *testing.T) {
	dir := t.TempDir()
	unified := filepath.Join(dir, "liblitert-lm.so")
	if err := os.WriteFile(unified, bytes.Repeat([]byte{0x7f}, 1024), 0644); err != nil {
		t.Fatal(err)
	}
	legacy := []string{"liblitertlm_c_cpu.so", "libGemmaModelConstraintProvider.so"}
	for _, name := range legacy {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	var out bytes.Buffer
	got, err := runLitertLMInstall(context.Background(), &out, litertlmInstallOptions{
		Version:  "v0.19.1",
		Platform: "linux_x86_64",
		Backend:  "cpu",
		Dir:      dir,
	})
	if err != nil {
		t.Fatalf("runLitertLMInstall() error = %v", err)
	}
	if got != dir {
		t.Errorf("runLitertLMInstall() = %q, want %q", got, dir)
	}
	if _, err := os.Stat(unified); err != nil {
		t.Errorf("unified library was removed: %v", err)
	}
	for _, name := range legacy {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Errorf("legacy library %s was not removed (err = %v)", name, err)
		}
	}
	if !strings.Contains(out.String(), "LiteRT-LM runtime ready") {
		t.Errorf("output = %q, want runtime-ready message", out.String())
	}
}

func TestLitertLMUnifiedLibName(t *testing.T) {
	tests := map[string]string{
		"linux_x86_64":   "liblitert-lm.so",
		"linux_arm64":    "liblitert-lm.so",
		"macos_arm64":    "liblitert-lm.dylib",
		"windows_x86_64": "litert-lm.dll",
	}
	for platform, want := range tests {
		if got := litertlmUnifiedLibName(platform); got != want {
			t.Errorf("litertlmUnifiedLibName(%q) = %q, want %q", platform, got, want)
		}
	}
}
