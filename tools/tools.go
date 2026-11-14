package tools

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// DefaultCommandTimeout bounds a single RunCommand tool invocation when the
// policy does not set an explicit timeout.
const DefaultCommandTimeout = 2 * time.Minute

// MaxCommandOutputBytes caps the output returned to the model. Commands like
// `cat huge.log` or `yes` otherwise flood the next request (or the local
// model's context window) and make the session unusable.
const MaxCommandOutputBytes = 100 << 10

// cappedBuffer keeps the first limit bytes written to it and records whether
// anything was dropped. Writes never fail so the child process is not killed
// by a broken pipe before its natural exit.
type cappedBuffer struct {
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	remaining := b.limit - b.buf.Len()
	if remaining <= 0 {
		b.truncated = true
		return n, nil
	}
	if len(p) > remaining {
		p = p[:remaining]
		b.truncated = true
	}
	_, _ = b.buf.Write(p)
	return n, nil
}

func (b *cappedBuffer) String() string {
	s := b.buf.String()
	if b.truncated {
		s += fmt.Sprintf("\n\n[... output truncated to %d bytes ...]", b.limit)
	}
	return s
}

// RunCommand executes a shell command with the default timeout.
func RunCommand(command string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), DefaultCommandTimeout)
	defer cancel()
	return RunCommandContext(ctx, command)
}

// RunCommandContext executes command via bash -c, bounded by ctx. On timeout
// or cancellation the whole process group is killed so pipelines and
// background children do not outlive the request.
func RunCommandContext(ctx context.Context, command string) (string, error) {
	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	// Run in its own process group so cancellation reaches children spawned
	// by bash (pipelines, subshells) instead of only the direct child.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	// Do not wait forever on descendants that inherited the output pipe.
	cmd.WaitDelay = 5 * time.Second

	out := &cappedBuffer{limit: MaxCommandOutputBytes}
	cmd.Stdout = out
	cmd.Stderr = out

	err := cmd.Run()
	text := out.String()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return text, fmt.Errorf("command %s: %w", ctxErrLabel(ctxErr), ctxErr)
		}
		return text, fmt.Errorf("command failed: %w", err)
	}

	return text, nil
}

func ctxErrLabel(err error) string {
	if err == context.DeadlineExceeded {
		return "timed out"
	}
	return "cancelled"
}

func WriteFile(path string, content string) (string, error) {
	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}
	return fmt.Sprintf("File written successfully to %s", path), nil
}

func ReadFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}
	return string(content), nil
}
