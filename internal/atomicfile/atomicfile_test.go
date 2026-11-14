package atomicfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteCreatesAndReplaces(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "config.yaml")

	if err := Write(path, []byte("first"), 0o600); err != nil {
		t.Fatalf("Write() = %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("mode = %o, want 600", info.Mode().Perm())
	}

	if err := Write(path, []byte("second"), 0o640); err != nil {
		t.Fatalf("overwrite Write() = %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(data) != "second" {
		t.Errorf("content = %q, want %q", data, "second")
	}

	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("directory has %d entries, want only the target (no temp leftovers)", len(entries))
	}
}

func TestWriteFailsForDirectoryTarget(t *testing.T) {
	dir := t.TempDir()
	if err := Write(dir, []byte("x"), 0o644); err == nil {
		t.Fatal("Write() = nil error, want a failure when the target is a directory")
	}
}
