package llm

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
	"text/template"

	"edgebot/config"
	"edgebot/tools"
)

const BuildPrompt = `You are an expert shell assistant operating inside a shell on the user machine.
You help users by reading files, executing commands, editing code, and writing new files.

The user machine OS is {{.Distro}} and uses the {{.Shell}} shell.
Current working directory: {{.Cwd}}.

Available tools:
{{.Tools}}`

const PlanPrompt = `You are a planning agent operating inside a shell on the user machine.
Your job is to analyze problems, read files, and produce a clear, actionable plan.
You cannot execute shell commands or write files — focus on analysis and planning, not on doing the work yourself.

The user machine OS is {{.Distro}} and uses the {{.Shell}} shell.
Current working directory: {{.Cwd}}.

Available tools:
{{.Tools}}`

const BotPrompt = `You are a Telegram bot assistant operating on the user machine.
You help users via Telegram chat by reading files and using the persistent KV store.
You cannot execute shell commands or write files — you can only read files (ReadFile) and interact with the KV store (KVGet, KVList, KVSet) to recall and persist information.
Keep replies concise and suitable for Telegram (plain text, avoid heavy markdown, messages are split at 4096 characters).

The user machine OS is {{.Distro}} and uses the {{.Shell}} shell.
Current working directory: {{.Cwd}}.

Available tools:
{{.Tools}}`

const ChatPrompt = `You are a helpful conversational assistant.`

func init() {
	dir, err := config.EdgebotDir()
	if err != nil {
		return
	}

	writePromptFile(dir, "BUILDPROMPT.md", BuildPrompt)
	writePromptFile(dir, "PLANPROMPT.md", PlanPrompt)
	writePromptFile(dir, "BOTPROMPT.md", BotPrompt)
	writePromptFile(dir, "CHATPROMPT.md", ChatPrompt)
}

// writePromptFile writes the prompt to ~/.edgebot/<name> if it does not exist.
func writePromptFile(dir, name, prompt string) {
	dest := filepath.Join(dir, name)
	if _, err := os.Stat(dest); err == nil {
		return
	}
	if err := os.WriteFile(dest, []byte(prompt), 0o644); err != nil {
		slog.Warn("failed to write prompt file", "path", dest, "err", err)
	}
}

// readBotPromptFile reads BOTPROMPT.md from ~/.edgebot/BOTPROMPT.md.
// Falls back to the embedded bot prompt if the file cannot be read.
func readBotPromptFile() []byte {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("Cannot determine home directory, using embedded prompt", "err", err)
		return []byte(BotPrompt)
	}

	raw, err := os.ReadFile(filepath.Join(home, ".edgebot", "BOTPROMPT.md"))
	if err != nil {
		slog.Warn("Cannot read ~/.edgebot/BOTPROMPT.md, using embedded prompt", "err", err)
		return []byte(BotPrompt)
	}

	return raw
}

// readChatPromptFile reads CHATPROMPT.md from ~/.edgebot/CHATPROMPT.md.
// Falls back to the embedded chat prompt if the file cannot be read.
func readChatPromptFile() []byte {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("Cannot determine home directory, using embedded prompt", "err", err)
		return []byte(ChatPrompt)
	}

	raw, err := os.ReadFile(filepath.Join(home, ".edgebot", "CHATPROMPT.md"))
	if err != nil {
		slog.Warn("Cannot read ~/.edgebot/CHATPROMPT.md, using embedded prompt", "err", err)
		return []byte(ChatPrompt)
	}

	return raw
}

func GetDefaultPromptBytes() []byte {
	return []byte(BuildPrompt)
}

// GetAgentSystemPrompt returns the system prompt for the named agent, rendered
// with the given tool list. Falls back to the default prompt for unknown names.
func GetAgentSystemPrompt(agentName string, toolList []any) string {
	raw := []byte(BuildPrompt)
	switch agentName {
	case "plan":
		raw = readPlanPromptFile()
	case "bot":
		raw = readBotPromptFile()
	case "chat":
		raw = readChatPromptFile()
	case "", "build":
		raw = readPromptFile()
	}
	return renderPromptTemplate(raw, toolList)
}

// renderPromptTemplate renders a prompt template with the current distro,
// shell, cwd, and tool list.
func renderPromptTemplate(raw []byte, toolList []any) string {
	cwd, _ := os.Getwd()
	data := PromptData{
		Distro: tools.GetDistro(),
		Shell:  tools.GetShell(),
		Cwd:    cwd,
		Tools:  buildToolDescriptions(toolList),
	}

	tmpl, err := template.New("prompt").Parse(string(raw))
	if err != nil {
		return "You are a helpful shell assistant."
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "You are a helpful shell assistant."
	}

	return buf.String()
}
