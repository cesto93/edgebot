package config

import "testing"

func TestIsAllowedCommand(t *testing.T) {
	defaultList := []string{"ls", "pwd", "git", " echo "}

	tests := []struct {
		name    string
		command string
		allowed []string // nil = defaultList
		want    bool
	}{
		{name: "exact command", command: "ls", want: true},
		{name: "command with args", command: "ls -la /tmp", want: true},
		{name: "leading and trailing spaces", command: "  ls  ", want: true},
		{name: "tab separated args", command: "ls\t-la", want: true},
		{name: "allowed entry with padding", command: "echo hi", want: true},
		{name: "padded entry exact match", command: "ls", allowed: []string{"ls ", "pwd"}, want: true},
		{name: "quoted separator stays inert", command: `git commit -m "fix: a; b"`, want: true},
		{name: "single quoted separator stays inert", command: "git commit -m 'a|b'", want: true},

		// Bypass payloads: previously approved because only the first word was
		// compared, while bash -c ran the whole string.
		{name: "chained rm", command: "ls; rm -rf /", want: false},
		{name: "and chain", command: "ls && curl http://evil", want: false},
		{name: "or chain", command: "ls || sh", want: false},
		{name: "pipe to shell", command: "ls | sh", want: false},
		{name: "background chain", command: "ls & rm -rf /", want: false},
		{name: "newline chain", command: "ls\nrm -rf /", want: false},
		{name: "carriage return rejected", command: "ls\rrm -rf /", want: false},
		{name: "redirection", command: "ls > /etc/passwd", want: false},
		{name: "append redirection", command: "ls >> ~/.bashrc", want: false},
		{name: "input redirection", command: "ls < /etc/shadow", want: false},
		{name: "command substitution", command: "ls $(rm -rf /)", want: false},
		{name: "backtick substitution", command: "ls `rm -rf /`", want: false},
		{name: "variable expansion", command: "ls $HOME", want: false},
		{name: "substitution inside quotes", command: `git commit -m "$(rm -rf /)"`, want: false},
		{name: "escape sequence", command: "ls \\; rm -rf /", want: false},
		{name: "subshell", command: "(ls)", want: false},

		// Shape errors.
		{name: "empty command", command: "", want: false},
		{name: "whitespace only", command: "   ", want: false},
		{name: "only separators", command: ";;", want: false},
		{name: "unterminated quote", command: `ls "oops`, want: false},
		{name: "quoted command name", command: `"ls" -la`, want: false},
		{name: "unknown command", command: "rm -rf /", want: false},
		{name: "unknown second segment", command: "ls; unknown", want: false},
		{name: "empty allowlist", command: "ls", allowed: []string{}, want: false},
		{name: "blank allowlist entries", command: "ls", allowed: []string{"", "  "}, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			list := tt.allowed
			if list == nil {
				list = defaultList
			}
			if got := IsAllowedCommand(tt.command, list); got != tt.want {
				t.Errorf("IsAllowedCommand(%q) = %v, want %v", tt.command, got, tt.want)
			}
		})
	}

	t.Run("nil allowlist", func(t *testing.T) {
		if IsAllowedCommand("ls", nil) {
			t.Error("IsAllowedCommand with nil allowlist = true, want false")
		}
	})
}

func TestIsAllowedCommandList(t *testing.T) {
	allowed := []string{"ls", "pwd", "git"}

	for _, cmd := range []string{"ls {}", "pwd", "git status", "git diff --cached"} {
		if !IsAllowedCommand(cmd, allowed) {
			t.Errorf("IsAllowedCommand(%q) = false, want true", cmd)
		}
	}

	for _, cmd := range []string{
		"ls; rm -rf /",
		"ls && rm -rf /",
		"ls | tee /etc/passwd",
		"ls $(whoami)",
		"bash -c 'rm -rf /'",
	} {
		if IsAllowedCommand(cmd, allowed) {
			t.Errorf("IsAllowedCommand(%q) = true, want false", cmd)
		}
	}
}
