package service

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"edgebot/llm"
)

func TestServiceExecutorConfirmTrue(t *testing.T) {
	e := &ServiceExecutor{confirm: true, allowed: []string{"ls"}}

	out, err := e.ExecuteTool(context.Background(), llm.ToolCall{Name: "RunCommand", Arguments: map[string]any{"command": "ls"}})
	if err != nil {
		t.Errorf("allowed command failed: %v", err)
	}
	if strings.Contains(out, "denied") {
		t.Errorf("allowed command was denied: %s", out)
	}

	out, _ = e.ExecuteTool(context.Background(), llm.ToolCall{Name: "RunCommand", Arguments: map[string]any{"command": "echo hi"}})
	if !strings.Contains(out, "denied") {
		t.Errorf("non-allowed command not denied: %s", out)
	}

	out, _ = e.ExecuteTool(context.Background(), llm.ToolCall{
		Name: "WriteFile",
		Arguments: map[string]any{
			"path":    filepath.Join(t.TempDir(), "x.txt"),
			"content": "data",
		},
	})
	if !strings.Contains(out, "denied") {
		t.Errorf("WriteFile not denied with confirm=true: %s", out)
	}
}

func TestServiceExecutorConfirmFalse(t *testing.T) {
	e := &ServiceExecutor{confirm: false}

	out, err := e.ExecuteTool(context.Background(), llm.ToolCall{Name: "RunCommand", Arguments: map[string]any{"command": "echo hi"}})
	if err != nil {
		t.Errorf("RunCommand failed: %v", err)
	}
	if strings.Contains(out, "denied") {
		t.Errorf("RunCommand denied with confirm=false: %s", out)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "written.txt")
	out, _ = e.ExecuteTool(context.Background(), llm.ToolCall{
		Name: "WriteFile",
		Arguments: map[string]any{
			"path":    path,
			"content": "data",
		},
	})
	if strings.Contains(out, "denied") {
		t.Errorf("WriteFile denied with confirm=false: %s", out)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("WriteFile did not write the file: %v", err)
	}
}

func TestServiceExecutorReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "readme.txt")
	if err := os.WriteFile(path, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}

	e := &ServiceExecutor{confirm: true}
	out, err := e.ExecuteTool(context.Background(), llm.ToolCall{Name: "ReadFile", Arguments: map[string]any{"path": path}})
	if err != nil {
		t.Errorf("ReadFile failed: %v", err)
	}
	if !strings.Contains(out, "hello") {
		t.Errorf("ReadFile output = %q, want to contain %q", out, "hello")
	}
}

func TestServiceExecutorInvalidArgs(t *testing.T) {
	e := &ServiceExecutor{confirm: false}
	out, _ := e.ExecuteTool(context.Background(), llm.ToolCall{Name: "RunCommand", Arguments: map[string]any{}})
	if !strings.Contains(out, "Invalid") {
		t.Errorf("invalid RunCommand args output = %q, want error message", out)
	}

	out, _ = e.ExecuteTool(context.Background(), llm.ToolCall{Name: "Nope", Arguments: map[string]any{}})
	if !strings.Contains(out, "Unknown tool") {
		t.Errorf("unknown tool output = %q, want error message", out)
	}
}

func TestServiceExecutorConfirmHelpers(t *testing.T) {
	e := &ServiceExecutor{confirm: true, allowed: []string{"ls", "pwd"}}
	if !e.IsAllowedCommand("ls") {
		t.Error("IsAllowedCommand(ls) = false, want true")
	}
	if e.IsAllowedCommand("rm") {
		t.Error("IsAllowedCommand(rm) = true, want false")
	}
	if !e.AskConfirmation("ls -la") {
		t.Error("AskConfirmation(ls -la) = false, want true (allowed command name)")
	}
	if e.AskConfirmation("rm -rf /") {
		t.Error("AskConfirmation(rm -rf /) = true, want false")
	}

	auto := &ServiceExecutor{confirm: false}
	if !auto.AskConfirmation("rm -rf /") {
		t.Error("AskConfirmation with confirm=false should always allow")
	}
}
