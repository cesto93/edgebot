package llm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"edgebot/config"
)

// TestConfirmPolicyMatchesConfigAllowlist guards against the two allowlist
// implementations drifting apart: the shell, the bot and the service must all
// agree with the policy used by the tool executor.
func TestConfirmPolicyMatchesConfigAllowlist(t *testing.T) {
	allowed := []string{"ls", "pwd", "git"}

	commands := []string{
		"ls",
		"ls -la",
		"git commit -m \"fix: a; b\"",
		"ls; rm -rf /",
		"ls && curl http://evil",
		"ls | sh",
		"ls > /etc/passwd",
		"ls $(rm -rf /)",
		"rm -rf /",
		"",
	}

	for _, cmd := range commands {
		t.Run(cmd, func(t *testing.T) {
			want := config.IsAllowedCommand(cmd, allowed)
			policy := NewConfirmPolicy(true, allowed)
			if got := policy.ConfirmCommand(cmd); got != want {
				t.Errorf("policy.ConfirmCommand(%q) = %v, config.IsAllowedCommand = %v", cmd, got, want)
			}
			if got := policy.IsAllowedCommand(cmd); got != want {
				t.Errorf("policy.IsAllowedCommand(%q) = %v, config.IsAllowedCommand = %v", cmd, got, want)
			}
		})
	}
}

// TestConfirmPolicyDisabledAllowsEverything documents the confirm=false escape
// hatch shared by the bot and the service.
func TestConfirmPolicyDisabledAllowsEverything(t *testing.T) {
	policy := NewConfirmPolicy(false, nil)
	if !policy.ConfirmCommand("rm -rf /") {
		t.Error("ConfirmCommand with confirm=false = false, want true")
	}
	if !policy.ConfirmWriteFile("/etc/passwd") {
		t.Error("ConfirmWriteFile with confirm=false = false, want true")
	}
}

// TestConfirmPolicyDeniesWriteFile keeps the write guard covered: with
// confirm=true every WriteFile requires confirmation, whatever the allowlist.
func TestConfirmPolicyDeniesWriteFile(t *testing.T) {
	policy := NewConfirmPolicy(true, []string{"ls"})
	if policy.ConfirmWriteFile("/tmp/x") {
		t.Error("ConfirmWriteFile with confirm=true = true, want false")
	}
}

// TestPolicyCommandTimeout ensures a hung tool call is bounded and the error
// is reported back to the model as text instead of blocking the session.
func TestPolicyCommandTimeout(t *testing.T) {
	policy := &ToolExecutorPolicy{CommandTimeout: 100 * time.Millisecond}

	start := time.Now()
	out, err := policy.ExecuteTool(context.Background(), ToolCall{
		Name:      "RunCommand",
		Arguments: map[string]any{"command": "sleep 5"},
	})
	if err != nil {
		t.Fatalf("ExecuteTool() error = %v, want the failure in the output", err)
	}
	if !strings.Contains(out, "timed out") {
		t.Errorf("output = %q, want a timeout message", out)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("ExecuteTool took %v, want the timeout to stop it promptly", elapsed)
	}
}

func TestPolicyCommandCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	policy := NewConfirmPolicy(false, nil)
	out, err := policy.ExecuteTool(ctx, ToolCall{
		Name:      "RunCommand",
		Arguments: map[string]any{"command": "sleep 5"},
	})
	if err != nil {
		t.Fatalf("ExecuteTool() error = %v", err)
	}
	if !strings.Contains(out, "cancelled") {
		t.Errorf("output = %q, want a cancellation message", out)
	}
}

func TestPolicyReadDenylist(t *testing.T) {
	dir := t.TempDir()
	secret := filepath.Join(dir, ".env")
	if err := os.WriteFile(secret, []byte("TOKEN=1"), 0o600); err != nil {
		t.Fatalf("seed secret: %v", err)
	}

	policy := &ToolExecutorPolicy{ReadDenylist: []string{".env"}}
	out, err := policy.ExecuteTool(context.Background(), ToolCall{
		Name:      "ReadFile",
		Arguments: map[string]any{"path": secret},
	})
	if err != nil {
		t.Fatalf("ExecuteTool() error = %v", err)
	}
	if !strings.Contains(out, "read_denylist") {
		t.Errorf("output = %q, want a denylist error", out)
	}
	if strings.Contains(out, "TOKEN") {
		t.Error("denylisted content leaked into the output")
	}

	allowed := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(allowed, []byte("hello"), 0o600); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	out, err = policy.ExecuteTool(context.Background(), ToolCall{
		Name:      "ReadFile",
		Arguments: map[string]any{"path": allowed},
	})
	if err != nil {
		t.Fatalf("ExecuteTool() error = %v", err)
	}
	if out != "hello" {
		t.Errorf("output = %q, want %q", out, "hello")
	}
}
