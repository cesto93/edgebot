package config

import (
	"strings"
	"testing"
)

// FuzzIsAllowedCommand checks the safety invariant: a command is never
// auto-approved when it contains a character that can expand into an
// arbitrary command (substitution, redirection, subshells, escapes), and an
// empty allowlist never approves anything.
func FuzzIsAllowedCommand(f *testing.F) {
	f.Add("ls -la", "ls")
	f.Add("ls; rm -rf /", "ls")
	f.Add("git commit -m \"fix: a; b\"", "git")
	f.Add("", "")
	f.Add("ls $HOME", "ls")

	f.Fuzz(func(t *testing.T, cmd, allowed string) {
		list := []string{}
		if allowed != "" {
			list = append(list, allowed)
		}

		if !IsAllowedCommand(cmd, list) {
			return
		}
		if strings.TrimSpace(allowed) == "" {
			t.Fatalf("empty allowlist approved %q", cmd)
		}
		if i := strings.IndexAny(cmd, shellUnsafeChars); i >= 0 {
			t.Fatalf("command with unsafe character %q approved: %q", cmd[i], cmd)
		}
	})
}
