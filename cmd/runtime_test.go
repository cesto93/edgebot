package cmd

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/libfetch"
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

	record, err := readLitertlmInstallRecord(dir)
	if err != nil {
		t.Fatalf("readLitertlmInstallRecord() error = %v", err)
	}
	if record.Version != "v0.19.1" || record.Platform != "linux_x86_64" || record.Backend != "cpu" {
		t.Errorf("install record = %+v, want version/platform/backend recorded", record)
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

func TestRunRuntimeShowReportsInstalledAndMissing(t *testing.T) {
	dir := t.TempDir()
	platform, _ := libfetch.Platform()
	unified := litertlmUnifiedLibName(platform)
	if err := os.WriteFile(filepath.Join(dir, unified), bytes.Repeat([]byte{0x7f}, 1024), 0644); err != nil {
		t.Fatal(err)
	}
	record := litertlmInstallRecord{Version: "v0.19.1", Platform: platform, Backend: "cpu"}
	if err := writeLitertlmInstallRecord(dir, record); err != nil {
		t.Fatalf("writeLitertlmInstallRecord() error = %v", err)
	}

	var out bytes.Buffer
	if err := runRuntimeShow(&out, dir); err != nil {
		t.Fatalf("runRuntimeShow() error = %v", err)
	}
	text := out.String()
	if !strings.Contains(text, "litertlm") || !strings.Contains(text, "v0.19.1") {
		t.Errorf("show output missing installed litertlm row: %q", text)
	}
	if !strings.Contains(text, "llamacpp") || !strings.Contains(text, "missing") {
		t.Errorf("show output missing llamacpp status: %q", text)
	}
	if !strings.Contains(text, "edgebot runtime install llamacpp") {
		t.Errorf("show output missing install hint: %q", text)
	}
}

func TestResolveRuntimeDirHonoursOverrides(t *testing.T) {
	t.Setenv("LITERTLM_LIB", "/custom/litertlm")
	if got, err := resolveRuntimeDir(runtimeLitertLM, ""); err != nil || got != "/custom/litertlm" {
		t.Errorf("resolveRuntimeDir(litertlm) = %q, %v; want /custom/litertlm", got, err)
	}
	if got, err := resolveRuntimeDir(runtimeLitertLM, "/explicit"); err != nil || got != "/explicit" {
		t.Errorf("resolveRuntimeDir(litertlm, override) = %q, %v; want /explicit", got, err)
	}
	if got, err := resolveRuntimeDir(runtimeLlamacpp, "/explicit"); err != nil || got != "/explicit" {
		t.Errorf("resolveRuntimeDir(llamacpp, override) = %q, %v; want /explicit", got, err)
	}
}

func TestLlamacppLibraryName(t *testing.T) {
	tests := map[string]string{
		"linux":   "libllama.so",
		"darwin":  "libllama.dylib",
		"windows": "llama.dll",
	}
	for goos, want := range tests {
		if got := llamacppLibraryName(goos); got != want {
			t.Errorf("llamacppLibraryName(%q) = %q, want %q", goos, got, want)
		}
	}
}

func TestReadYzmaInstallRecord(t *testing.T) {
	dir := t.TempDir()
	body := `{"tag":"b10785","arch":"amd64","os":"linux","processor":"cuda"}`
	if err := os.WriteFile(filepath.Join(dir, yzmaInstallRecord), []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	record, err := readYzmaInstallRecord(dir)
	if err != nil {
		t.Fatalf("readYzmaInstallRecord() error = %v", err)
	}
	if record.Tag != "b10785" || record.Processor != "cuda" {
		t.Errorf("readYzmaInstallRecord() = %+v, want tag/processor parsed", record)
	}
}

func TestRunLlamacppInstallReusesExistingLibrary(t *testing.T) {
	dir := t.TempDir()
	libPath := filepath.Join(dir, llamacppLibraryName(runtime.GOOS))
	if err := os.WriteFile(libPath, bytes.Repeat([]byte{0x7f}, 1024), 0644); err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	got, err := runLlamacppInstall(context.Background(), &out, llamacppInstallOptions{Dir: dir})
	if err != nil {
		t.Fatalf("runLlamacppInstall() error = %v", err)
	}
	if got != dir {
		t.Errorf("runLlamacppInstall() = %q, want %q", got, dir)
	}
	if !strings.Contains(out.String(), "already installed") {
		t.Errorf("output = %q, want already-installed message", out.String())
	}
}

func TestYzmaBinaryInstallsMatchingCLIWhenMissing(t *testing.T) {
	pathDir := t.TempDir()
	fakeGo := filepath.Join(pathDir, "go")
	script := "#!/bin/sh\n: > \"$GOBIN/yzma\"\n"
	if err := os.WriteFile(fakeGo, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pathDir)
	t.Setenv("GOBIN", "")

	origExec := execCommandContext
	defer func() { execCommandContext = origExec }()
	execCommandContext = func(ctx context.Context, name string, args ...string) *exec.Cmd {
		return exec.CommandContext(ctx, name, args...)
	}

	bin, cleanup, err := yzmaBinary(context.Background(), io.Discard)
	if err != nil {
		t.Fatalf("yzmaBinary() error = %v", err)
	}
	dir := filepath.Dir(bin)
	if _, err := os.Stat(bin); err != nil {
		t.Errorf("yzmaBinary() = %q, stat error = %v", bin, err)
	}
	cleanup()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("cleanup left %s behind (err = %v)", dir, err)
	}
}

func TestYzmaBinaryWithoutCLIOrGoFails(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	_, _, err := yzmaBinary(context.Background(), io.Discard)
	if err == nil {
		t.Fatal("yzmaBinary() = nil error, want a missing-CLI error")
	}
	if !strings.Contains(err.Error(), "yzma CLI not found") {
		t.Errorf("yzmaBinary() error = %q, want a missing-CLI hint", err)
	}
}
