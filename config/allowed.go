package config

import "strings"

// shellSegmentSeparators split a `bash -c` string into independent simple
// commands. Each half of `&&` / `||` is a `&` / `|`, so both are covered.
const shellSegmentSeparators = ";&|\n"

// shellUnsafeChars can expand into an arbitrary command (or redirect output)
// without starting a new segment, so any command containing one of them can
// never be auto-approved from an allowlist. Backslash and subshell parens are
// included for the same reason, `$`/backtick are rejected even inside quotes
// because double quotes still expand them, and a carriage return is rejected
// because it is invisible in logs and never legitimate in a command.
const shellUnsafeChars = "`$\\<>()\r"

// IsAllowedCommand reports whether cmd may run without asking the user for
// confirmation.
//
// A command is auto-approved only when every simple command in it starts with
// one of the allowed names and it contains no construct we cannot reason
// about. Separators are split, so `ls; rm -rf /` is rejected unless both `ls`
// and `rm` are allowed; substitution, expansion, redirection, escaping and
// subshells reject the whole command. Quoted text stays inert, so
// `git commit -m "fix: a; b"` is still auto-approved.
//
// Entries are command names (first word), matching the documented
// `allowed_commands=ls,pwd` format.
func IsAllowedCommand(cmd string, allowedList []string) bool {
	if len(allowedList) == 0 {
		return false
	}
	segments, ok := splitCommandSegments(cmd)
	if !ok || len(segments) == 0 {
		return false
	}
	for _, seg := range segments {
		if !isAllowedName(GetCommandName(seg), allowedList) {
			return false
		}
	}
	return true
}

// isAllowedName compares a command name against the allowlist, ignoring
// surrounding whitespace and empty entries.
func isAllowedName(name string, allowedList []string) bool {
	for _, a := range allowedList {
		if a = strings.TrimSpace(a); a != "" && a == name {
			return true
		}
	}
	return false
}

// splitCommandSegments splits cmd on unquoted separators. ok is false when the
// command contains something that makes an allowlist decision unsafe: a
// substitution/redirection construct, an escape, or unbalanced quotes.
func splitCommandSegments(cmd string) ([]string, bool) {
	var (
		segments []string
		current  strings.Builder
		quote    rune
	)
	for _, r := range cmd {
		if strings.ContainsRune(shellUnsafeChars, r) {
			return nil, false
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			}
			current.WriteRune(r)
			continue
		}
		switch {
		case r == '\'' || r == '"':
			quote = r
			current.WriteRune(r)
		case strings.ContainsRune(shellSegmentSeparators, r):
			segments = append(segments, current.String())
			current.Reset()
		default:
			current.WriteRune(r)
		}
	}
	if quote != 0 {
		return nil, false
	}
	segments = append(segments, current.String())

	trimmed := make([]string, 0, len(segments))
	for _, seg := range segments {
		if seg = strings.TrimSpace(seg); seg != "" {
			trimmed = append(trimmed, seg)
		}
	}
	return trimmed, true
}
