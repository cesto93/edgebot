//go:build integration

package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func mockGitExecCommand(diffPath string) func(string, ...string) *exec.Cmd {
	return func(command string, args ...string) *exec.Cmd {
		if command != "git" {
			return exec.Command(command, args...)
		}
		switch {
		case len(args) >= 2 && args[0] == "log" && args[1] == "--oneline":
			return exec.Command("sh", "-c",
				`echo "abc1234 fix: previous change"`)
		case len(args) >= 2 && args[0] == "diff" && args[1] == "--cached":
			return exec.Command("sh", "-c",
				"cat '"+diffPath+"'")
		case args[0] == "add":
			return exec.Command("true")
		case args[0] == "reset":
			return exec.Command("true")
		case args[0] == "commit":
			return exec.Command("true")
		default:
			return exec.Command("true")
		}
	}
}

func TestRunCommitWithRealLLM(t *testing.T) {
	// The commit command runs locally against the llm model in
	// ~/.edgebot/models/llamacpp (see integration_llamacpp_test.go).
	tests := []struct {
		name  string
		model string
	}{
		{
			name:  "llamacpp-" + integrationLLMModel,
			model: integrationLLMModel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requireLlamacppModel(t, tt.model)

			tmpDir := t.TempDir()

			origDir, err := os.Getwd()
			if err != nil {
				t.Fatal(err)
			}

			diffPath, err := filepath.Abs(filepath.Join("testdata", "commit_diff.txt"))
			if err != nil {
				t.Fatal(err)
			}

			os.Chdir(tmpDir)
			defer os.Chdir(origDir)

			origExecCommand := execCommand
			execCommand = mockGitExecCommand(diffPath)
			defer func() { execCommand = origExecCommand }()

			configYAML := fmt.Sprintf("llm:\n  provider: llamacpp\n  model: %s\n", tt.model)
			if err := os.WriteFile(filepath.Join(tmpDir, "config.yaml"), []byte(configYAML), 0644); err != nil {
				t.Fatal(err)
			}

			dryRun = false
			commitAll = false
			defer func() {
				dryRun = false
				commitAll = false
			}()

			if err := runCommit(); err != nil {
				t.Fatalf("runCommit() error = %v", err)
			}
		})
	}
}
