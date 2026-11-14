package llm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"edgebot/config"
	"edgebot/tools"
)

// ToolExecutorPolicy implements ToolExecutor with pluggable confirmation hooks
// and an optional OnExecute hook invoked right before a tool actually runs.
// Nil hooks allow the operation unconditionally (and never run a hook).
type ToolExecutorPolicy struct {
	// ConfirmCommand decides whether a shell command may run.
	ConfirmCommand func(cmd string) bool
	// ConfirmWriteFile decides whether a file write may proceed.
	ConfirmWriteFile func(path string) bool
	// OnExecute is called immediately before a tool executes (used by the
	// interactive shell to render its status messages).
	OnExecute func(call ToolCall)
	// CommandTimeout bounds a single RunCommand call. Zero uses
	// tools.DefaultCommandTimeout.
	CommandTimeout time.Duration
	// ReadDenylist blocks ReadFile on matching paths (shell.read_denylist).
	ReadDenylist []string
}

func (p *ToolExecutorPolicy) ExecuteTool(ctx context.Context, call ToolCall) (string, error) {
	switch call.Name {
	case "RunCommand":
		cmd, ok := call.Arguments["command"].(string)
		if !ok {
			return "Error: Invalid tool arguments", nil
		}
		if p.ConfirmCommand != nil && !p.ConfirmCommand(cmd) {
			return "Error: Command execution denied by user", nil
		}
		p.exec(call)
		timeout := p.CommandTimeout
		if timeout <= 0 {
			timeout = tools.DefaultCommandTimeout
		}
		cmdCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		output, err := tools.RunCommandContext(cmdCtx, cmd)
		if err != nil {
			return fmt.Sprintf("Error: %v\nOutput: %s", err, output), nil
		}
		return output, nil

	case "WriteFile":
		path, ok1 := call.Arguments["path"].(string)
		content, ok2 := call.Arguments["content"].(string)
		if !ok1 || !ok2 {
			return "Error: Invalid tool arguments", nil
		}
		path = strings.TrimLeft(path, "@")
		if p.ConfirmWriteFile != nil && !p.ConfirmWriteFile(path) {
			return "Error: File write denied by user", nil
		}
		p.exec(call)
		return tools.WriteFile(path, content)

	case "ReadFile":
		path, ok := call.Arguments["path"].(string)
		if !ok {
			return "Error: Invalid tool arguments", nil
		}
		path = strings.TrimLeft(path, "@")
		if tools.IsReadDenied(path, p.ReadDenylist) {
			return fmt.Sprintf("Error: reading %s is blocked by shell.read_denylist", path), nil
		}
		p.exec(call)
		return tools.ReadFile(path)

	case "KVSet":
		key, ok1 := call.Arguments["key"].(string)
		value, ok2 := call.Arguments["value"].(string)
		if !ok1 || !ok2 {
			return "Error: Invalid tool arguments", nil
		}
		p.exec(call)
		return tools.KVSet(key, value)

	case "KVGet":
		key, ok := call.Arguments["key"].(string)
		if !ok {
			return "Error: Invalid tool arguments", nil
		}
		p.exec(call)
		return tools.KVGet(key)

	case "KVList":
		p.exec(call)
		return tools.KVList()

	default:
		return fmt.Sprintf("Error: Unknown tool %s", call.Name), nil
	}
}

func (p *ToolExecutorPolicy) exec(call ToolCall) {
	if p.OnExecute != nil {
		p.OnExecute(call)
	}
}

func (p *ToolExecutorPolicy) IsAllowedCommand(cmd string) bool {
	if p.ConfirmCommand == nil {
		return true
	}
	return p.ConfirmCommand(cmd)
}

func (p *ToolExecutorPolicy) AskConfirmation(cmd string) bool {
	return p.IsAllowedCommand(cmd)
}

// NewConfirmPolicy returns a ToolExecutorPolicy that mirrors the service/bot
// confirm logic: when confirm is false everything passes; otherwise RunCommand
// is limited to allowedCommands and WriteFile is denied.
func NewConfirmPolicy(confirm bool, allowedCommands []string) *ToolExecutorPolicy {
	return &ToolExecutorPolicy{
		ConfirmCommand: func(cmd string) bool {
			if !confirm {
				return true
			}
			return config.IsAllowedCommand(cmd, allowedCommands)
		},
		ConfirmWriteFile: func(path string) bool {
			return !confirm
		},
	}
}

// NoopExecutor is a ToolExecutor that runs nothing and allows everything.
type NoopExecutor struct{}

func (NoopExecutor) ExecuteTool(context.Context, ToolCall) (string, error) {
	return "", nil
}

func (NoopExecutor) IsAllowedCommand(cmd string) bool {
	return true
}

func (NoopExecutor) AskConfirmation(cmd string) bool {
	return true
}
