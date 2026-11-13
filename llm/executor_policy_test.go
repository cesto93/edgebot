package llm

import (
	"testing"

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
