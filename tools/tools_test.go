package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRunCommand(t *testing.T) {
	tests := []struct {
		name    string
		command string
		wantErr bool
		checkFn func(string) bool
	}{
		{
			name:    "echo hello",
			command: "echo hello",
			wantErr: false,
			checkFn: func(output string) bool {
				return strings.TrimSpace(output) == "hello"
			},
		},
		{
			name:    "echo with newline",
			command: "printf 'test output'",
			wantErr: false,
			checkFn: func(output string) bool {
				return strings.TrimSpace(output) == "test output"
			},
		},
		{
			name:    "invalid command",
			command: "nonexistentcommand12345",
			wantErr: true,
			checkFn: func(output string) bool {
				return strings.Contains(output, "not found") ||
					strings.Contains(output, "command not found")
			},
		},
		{
			name:    "exit code 1",
			command: "exit 1",
			wantErr: true,
			checkFn: func(output string) bool {
				return true
			},
		},
		{
			name:    "stderr output",
			command: "echo error >&2",
			wantErr: false,
			checkFn: func(output string) bool {
				return strings.Contains(output, "error")
			},
		},
		{
			name:    "combined stdout stderr",
			command: "echo stdout; echo stderr >&2",
			wantErr: false,
			checkFn: func(output string) bool {
				return strings.Contains(output, "stdout") &&
					strings.Contains(output, "stderr")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := RunCommand(tt.command)

			if (err != nil) != tt.wantErr {
				t.Errorf("RunCommand() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.checkFn != nil && !tt.checkFn(output) {
				t.Errorf("RunCommand() output = %q, did not match expected condition", output)
			}
		})
	}
}

func TestFileOperations(t *testing.T) {
	tempFile := "test_file.txt"
	content := "test content"

	t.Run("WriteFile", func(t *testing.T) {
		msg, err := WriteFile(tempFile, content)
		if err != nil {
			t.Errorf("WriteFile() error = %v", err)
		}
		if !strings.Contains(msg, "successfully") {
			t.Errorf("WriteFile() output = %q, expected success message", msg)
		}
	})

	t.Run("ReadFile", func(t *testing.T) {
		got, err := ReadFile(tempFile)
		if err != nil {
			t.Errorf("ReadFile() error = %v", err)
		}
		if got != content {
			t.Errorf("ReadFile() = %q, want %q", got, content)
		}
	})

	t.Run("ReadFile Non-existent", func(t *testing.T) {
		_, err := ReadFile("non_existent_file.txt")
		if err == nil {
			t.Error("ReadFile() expected error for non-existent file, got nil")
		}
	})

	// Cleanup
	os.Remove(tempFile)
}

func TestRunCommandContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	start := time.Now()
	_, err := RunCommandContext(ctx, "sleep 5")
	if err == nil {
		t.Fatal("RunCommandContext() = nil error, want cancellation error")
	}
	if !strings.Contains(err.Error(), "cancelled") {
		t.Errorf("error = %v, want a cancellation message", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("cancelled command took %v, want it to stop promptly", elapsed)
	}
}

func TestRunCommandContextTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := RunCommandContext(ctx, "sleep 5")
	if err == nil {
		t.Fatal("RunCommandContext() = nil error, want timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("error = %v, want a timeout message", err)
	}
}

func TestRunCommandOutputCap(t *testing.T) {
	// Emit more than the cap; the writer must keep the first
	// MaxCommandOutputBytes bytes and mark the truncation.
	out, err := RunCommandContext(context.Background(), "yes x | head -c 300000")
	if err != nil {
		t.Fatalf("RunCommandContext() error = %v", err)
	}
	if !strings.Contains(out, "output truncated") {
		t.Error("large output was not marked as truncated")
	}
	if len(out) > MaxCommandOutputBytes+200 {
		t.Errorf("output length = %d, want at most cap (%d) plus marker", len(out), MaxCommandOutputBytes)
	}
}

func TestWriteFilePreservesMode(t *testing.T) {
	path := filepath.Join(t.TempDir(), "script.sh")
	if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
		t.Fatalf("seed file: %v", err)
	}

	if _, err := WriteFile(path, "new"); err != nil {
		t.Fatalf("WriteFile() = %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("mode = %o, want 600 preserved", info.Mode().Perm())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(data) != "new" {
		t.Errorf("content = %q, want new", data)
	}
}
