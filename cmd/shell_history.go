package cmd

import (
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"strings"
)

func getHistoryFile() string {
	usr, err := user.Current()
	if err != nil {
		return ""
	}
	return filepath.Join(usr.HomeDir, ".edgebot-history")
}

func loadHistory(path string) []string {
	if path == "" {
		return []string{}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return []string{}
	}
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return []string{}
	}
	lines := strings.Split(trimmed, "\n")
	result := make([]string, 0, len(lines))
	for _, line := range lines {
		if line = strings.TrimSpace(line); line != "" {
			result = append(result, line)
		}
	}
	return result
}

func saveHistory(path string, history []string) error {
	if path == "" {
		return nil
	}
	if len(history) == 0 {
		// Truncate stale file.
		if err := os.WriteFile(path, []byte{}, 0o644); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}

	content := strings.Join(history, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Errorf("save history: %w", err)
	}
	return nil
}
