package cmd

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"edgebot/config"
	"edgebot/llm"

	tea "github.com/charmbracelet/bubbletea"
)

func (m *ShellModel) handleSubmit() (tea.Model, tea.Cmd) {
	value := strings.TrimSpace(m.input.Value())
	if value == "" {
		return m, nil
	}

	var images []string
	if strings.Contains(value, "@") {
		parts := strings.Fields(value)
		for i, part := range parts {
			if strings.HasPrefix(part, "@") {
				raw := strings.TrimPrefix(part, "@")
				raw = strings.Trim(raw, "\"'")
				if raw == "" {
					m.mu.Lock()
					m.messages = append(m.messages, Message{role: "error", content: "Error: empty file path after @"})
					m.mu.Unlock()
					return m, nil
				}
				path := raw
				if isImage(path) {
					encoded, err := encodeImage(path)
					if err != nil {
						m.mu.Lock()
						m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error encoding image %s: %v", path, err)})
						m.mu.Unlock()
						return m, nil
					}
					images = append(images, encoded)
				}
				parts[i] = path
			}
		}
		value = strings.Join(parts, " ")
	}

	if m.loading {
		return m, nil
	}

	if m.allowedCmdMode.active {
		parts := strings.Split(value, ",")
		var cleaned []string
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				cleaned = append(cleaned, p)
			}
		}
		m.cfg.Shell.AllowedCommands = cleaned
		if err := config.SaveConfig(m.cfg); err != nil {
			m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error saving config: %v", err)})
		} else {
			m.messages = append(m.messages, Message{role: "system", content: fmt.Sprintf("Allowed commands updated to: %s", strings.Join(m.cfg.Shell.AllowedCommands, ","))})
		}
		m.allowedCmdMode.active = false
		m.input.Prompt = "edgebot > "
		m.input.SetValue("")
		return m, nil
	}

	if len(m.history) == 0 || m.history[len(m.history)-1] != value {
		m.history = append(m.history, value)
		if err := saveHistory(m.commandHistoryPath, m.history); err != nil {
			slog.Warn("save history failed", "err", err)
		}
	}
	m.historyIndex = -1

	m.input.SetValue("")

	if strings.HasPrefix(value, "/") {
		return m.handleCommand(strings.TrimPrefix(value, "/"))
	}

	if handled, model, cmd := m.handleBuiltinCommand(value); handled {
		return model, cmd
	}

	m.mu.Lock()
	m.messages = append(m.messages, Message{role: "user", content: value, images: images})
	m.mu.Unlock()

	reqID := m.beginRequest()
	go m.ElaborateMessage(reqID)

	return m, nil
}

func (m *ShellModel) handleBuiltinCommand(cmd string) (handled bool, model tea.Model, cmd2 tea.Cmd) {
	switch cmd {
	case "exit", "quit":
		m.quitting = true
		return true, m, tea.Quit
	case "get-config":
		m.showConfig()
	case "config":
		m.openConfigMenu()
	case "help":
		m.showHelp()
	case "models":
		m.openModelMenu()
	case "agent":
		m.openAgentMenu()
	case "reset":
		m.messages = nil
	default:
		return false, m, nil
	}
	return true, m, nil
}

func (m *ShellModel) handleCommand(input string) (tea.Model, tea.Cmd) {
	parts := strings.Fields(input)
	if len(parts) == 0 {
		return m, nil
	}
	cmd := parts[0]

	if handled, model, cmd2 := m.handleBuiltinCommand(cmd); handled {
		return model, cmd2
	}

	for _, c := range m.commands {
		if c.Name == cmd {
			text, images, err := buildCommandTextAndImages(c.Prompt, parts[1:])
			if err != nil {
				m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error: %v", err)})
				return m, nil
			}
			m.mu.Lock()
			m.messages = append(m.messages, Message{role: "user", content: text, images: images, schema: c.Schema})
			m.mu.Unlock()
			reqID := m.beginRequest()
			go m.ElaborateMessage(reqID)
			return m, nil
		}
	}
	m.mu.Lock()
	m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Unknown command: /%s", cmd)})
	m.mu.Unlock()
	return m, nil
}

func (m *ShellModel) handleAutocomplete() (tea.Model, tea.Cmd) {
	value := m.input.Value()

	if strings.HasPrefix(value, "/") && !strings.Contains(value, " ") {
		partial := strings.TrimPrefix(value, "/")
		for _, cmd := range availableCommands {
			if strings.HasPrefix(cmd, partial) {
				m.input.SetValue("/" + cmd)
				break
			}
		}
		if m.input.Value() == value {
			for _, c := range m.commands {
				if strings.HasPrefix(c.Name, partial) {
					m.input.SetValue("/" + c.Name)
					break
				}
			}
		}
	} else {
		lastAt := strings.LastIndex(value, "@")
		if lastAt != -1 {
			partial := value[lastAt+1:]
			dir, base := filepath.Split(partial)

			matches := m.completeFiles(dir, base)
			if len(matches) > 0 {
				completed := matches[0]
				newValue := value[:lastAt+1] + completed
				m.input.SetValue(newValue)
			}
		}
	}

	m.updateSuggestions()
	return m, nil
}

func (m *ShellModel) completeFiles(dir, prefix string) []string {
	var results []string

	absDir := dir
	if !filepath.IsAbs(dir) {
		cwd, err := os.Getwd()
		if err != nil {
			cwd = "."
			slog.Warn("getwd failed", "err", err)
		}
		absDir = filepath.Join(cwd, dir)
	}
	absDir = filepath.Clean(absDir)

	entries, err := os.ReadDir(absDir)
	if err != nil {
		return results
	}

	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, prefix) {
			fullPath := filepath.Join(dir, name)
			if entry.IsDir() {
				results = append(results, fullPath+"/")
			} else {
				results = append(results, fullPath)
			}
		}
	}

	return results
}

func (m *ShellModel) updateSuggestions() {
	value := m.input.Value()
	var matches []string

	if strings.HasPrefix(value, "/") && !strings.Contains(value, " ") {
		filter := strings.TrimPrefix(value, "/")
		for _, cmd := range availableCommands {
			if strings.HasPrefix(cmd, filter) {
				matches = append(matches, "/"+cmd)
			}
		}
		for _, c := range m.commands {
			if strings.HasPrefix(c.Name, filter) {
				exists := false
				for _, m := range matches {
					if m == "/"+c.Name {
						exists = true
						break
					}
				}
				if !exists {
					matches = append(matches, "/"+c.Name)
				}
			}
		}
	}

	if lastAt := strings.LastIndex(value, "@"); lastAt != -1 {
		partial := value[lastAt+1:]
		dir, base := filepath.Split(partial)

		fileMatches := m.completeFiles(dir, base)
		for _, fm := range fileMatches {
			matches = append(matches, value[:lastAt+1]+fm)
		}
	}

	if len(matches) > 0 {
		if len(matches) == 1 && matches[0] == value {
			m.showSuggestions = false
			return
		}
		m.suggestions = matches
		m.showSuggestions = true
		if m.selectedIndex >= len(m.suggestions) {
			m.selectedIndex = 0
		}
	} else {
		m.showSuggestions = false
	}
}

func (m *ShellModel) navigateSuggestions(dir int) (tea.Model, tea.Cmd) {
	newIndex := m.selectedIndex + dir
	if newIndex < 0 {
		newIndex = len(m.suggestions) - 1
	} else if newIndex >= len(m.suggestions) {
		newIndex = 0
	}
	m.selectedIndex = newIndex
	return m, nil
}

func (m *ShellModel) selectSuggestion() (tea.Model, tea.Cmd) {
	if m.selectedIndex >= 0 && m.selectedIndex < len(m.suggestions) {
		suggestion := m.suggestions[m.selectedIndex]
		m.input.SetValue(suggestion)
		m.showSuggestions = false

		if strings.HasPrefix(suggestion, "/") && !strings.Contains(suggestion, "@") {
			if m.loading {
				return m, nil
			}
			return m.handleSubmit()
		}
		return m, nil
	}
	return m, nil
}

func (m *ShellModel) navigateHistory(dir int) (tea.Model, tea.Cmd) {
	if len(m.history) == 0 {
		return m, nil
	}

	newIndex := m.historyIndex + dir

	if newIndex < -1 {
		newIndex = -1
	} else if newIndex >= len(m.history) {
		return m, nil
	}

	m.historyIndex = newIndex

	if m.historyIndex == -1 {
		m.input.SetValue("")
	} else {
		m.input.SetValue(m.history[len(m.history)-1-m.historyIndex])
	}

	m.updateSuggestions()
	return m, nil
}

func (m *ShellModel) sortedToolNames() []string {
	var names []string
	for name := range m.cfg.Tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (m *ShellModel) showHelp() {
	var sb strings.Builder
	sb.WriteString("\nCommands:\n")
	sb.WriteString("  /help         - Show this help message\n")
	sb.WriteString("  /config       - Show configuration menu\n")
	sb.WriteString("  /get-config   - Show current LLM settings\n")
	sb.WriteString("  /models       - Switch to a different model\n")
	sb.WriteString("  /agent        - Switch the active agent\n")
	sb.WriteString("  /reset        - Clear the screen and messages\n")
	sb.WriteString("  /exit, /quit  - Exit the shell\n")
	sb.WriteString("  /<command>    - Execute a shell command\n")
	sb.WriteString("  @<file>       - Autocomplete file paths\n")
	sb.WriteString("  <text>        - Send text to the AI for a response\n")
	sb.WriteString("  PgUp/PgDn     - Scroll the transcript (mouse wheel works too)\n")
	sb.WriteString("  Shift+Up/Down - Scroll a few lines\n")
	sb.WriteString("  Home/End      - Jump to top / follow latest\n")

	if len(m.commands) > 0 {
		sb.WriteString("\nUser Commands:\n")
		for _, c := range m.commands {
			if c.Description != "" {
				sb.WriteString(fmt.Sprintf("  /%-12s - %s\n", c.Name, c.Description))
			} else {
				sb.WriteString(fmt.Sprintf("  /%-12s - %s\n", c.Name, c.Prompt))
			}
		}
	}

	sb.WriteString("\nEnv Files:\n")
	envPaths := config.GetEnvPaths()
	for _, path := range envPaths {
		sb.WriteString(fmt.Sprintf("  %s\n", path))
	}

	m.messages = append(m.messages, Message{role: "system", content: sb.String()})
}

func (m *ShellModel) showConfig() {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Agent: %s\n", m.cfg.Agent))
	sb.WriteString(fmt.Sprintf("Agent Files: %v\n", m.cfg.AgentFiles))
	sb.WriteString(fmt.Sprintf("Skills: %v\n", m.cfg.Skills))
	sb.WriteString(fmt.Sprintf("Provider: %s\n", m.cfg.LLM.Provider))
	sb.WriteString(fmt.Sprintf("Model: %s\n", m.cfg.LLM.Model))
	sb.WriteString(fmt.Sprintf("Confirm Commands: %v\n", m.cfg.Shell.Confirm))
	sb.WriteString(fmt.Sprintf("Allowed Commands: %s\n", strings.Join(m.cfg.Shell.AllowedCommands, ",")))

	toolDescs := llm.GetToolDescriptions()
	agentDef := llm.GetAgentDef(m.cfg.Agent)
	sb.WriteString("\nTools:\n")
	for _, name := range m.sortedToolNames() {
		status := "disabled"
		if m.cfg.Tools[name] {
			status = "enabled"
		}
		if !agentDef.Tools[name] {
			status = "blocked by agent"
		}
		desc := toolDescs[name]
		sb.WriteString(fmt.Sprintf("  %s (%s)", name, status))
		if desc != "" {
			sb.WriteString(fmt.Sprintf(" - %s", desc))
		}
		sb.WriteString("\n")
	}

	if len(m.commands) > 0 {
		sb.WriteString("\nCommands:\n")
		for _, c := range m.commands {
			if c.Description != "" {
				sb.WriteString(fmt.Sprintf("  /%s - %s\n", c.Name, c.Description))
			} else {
				sb.WriteString(fmt.Sprintf("  /%s - %s\n", c.Name, c.Prompt))
			}
		}
	}

	m.messages = append(m.messages, Message{role: "system", content: sb.String()})
}

func (m *ShellModel) openModelMenu() {
	models := config.GetAllAvailableModels()

	if len(models) == 0 {
		m.messages = append(m.messages, Message{role: "system", content: "No models found. Please install models using 'ollama pull <model>'"})
		return
	}

	m.menu.models = models
	m.menu.open(menuModel)
	m.input.SetValue("")
}

func (m *ShellModel) openAgentMenu() {
	defs := llm.GetAgentDefs()
	names := make([]string, len(defs))
	for i, def := range defs {
		names[i] = def.Name
	}

	m.menu.options = names
	m.menu.open(menuAgent)
	m.input.SetValue("")
}

func (m *ShellModel) selectAgentOption() {
	if m.menu.selectedIdx < 0 || m.menu.selectedIdx >= len(m.menu.options) {
		m.menu.close()
		return
	}

	selected := m.menu.options[m.menu.selectedIdx]
	if selected == m.cfg.Agent {
		m.menu.close()
		return
	}

	m.cfg.Agent = selected
	if err := config.SaveConfig(m.cfg); err != nil {
		m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error saving config: %v", err)})
	} else {
		m.messages = append(m.messages, Message{role: "system", content: fmt.Sprintf("Switched to agent: %s", selected)})
	}

	m.menu.close()
}

func (m *ShellModel) openConfigMenu() {
	m.menu.options = []string{
		fmt.Sprintf("Confirm Commands: %v", m.cfg.Shell.Confirm),
		fmt.Sprintf("Allowed Commands: %s", strings.Join(m.cfg.Shell.AllowedCommands, ",")),
		"Manage Tools",
		"Manage Commands",
		"Change Model",
		fmt.Sprintf("Change Agent: %s", m.cfg.Agent),
		"Back",
	}
	m.menu.open(menuConfig)
	m.input.SetValue("")
}

func (m *ShellModel) selectConfigOption() {
	switch m.menu.selectedIdx {
	case 0:
		m.cfg.Shell.Confirm = !m.cfg.Shell.Confirm
		if err := config.SaveConfig(m.cfg); err != nil {
			m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error saving config: %v", err)})
		} else {
			m.messages = append(m.messages, Message{role: "system", content: fmt.Sprintf("Confirm Commands set to: %v", m.cfg.Shell.Confirm)})
		}
		m.menu.close()
	case 1:
		m.allowedCmdMode.active = true
		m.input.SetValue(strings.Join(m.cfg.Shell.AllowedCommands, ","))
		m.input.Prompt = "Allowed commands (comma separated): "
		m.menu.close()
	case 2:
		m.menu.close()
		m.openToolsMenu()
	case 3:
		m.menu.close()
		m.openCommandsMenu()
	case 4:
		m.menu.close()
		m.openModelMenu()
	case 5:
		m.menu.close()
		m.openAgentMenu()
	case 6:
		m.menu.close()
	}
}

func (m *ShellModel) openCommandsMenu() {
	m.menu.options = []string{}
	for _, c := range m.commands {
		if c.Description != "" {
			m.menu.options = append(m.menu.options, fmt.Sprintf("/%s - %s", c.Name, c.Description))
		} else {
			m.menu.options = append(m.menu.options, fmt.Sprintf("/%s - %s", c.Name, c.Prompt))
		}
	}
	m.menu.options = append(m.menu.options, "Back")

	m.menu.open(menuCommands)
	m.input.SetValue("")
}

func (m *ShellModel) selectCommandOption() {
	m.menu.close()
	m.openConfigMenu()
}

func (m *ShellModel) openToolsMenu() {
	toolDescs := llm.GetToolDescriptions()

	m.menu.options = []string{}
	for _, name := range m.sortedToolNames() {
		status := "Disabled"
		if m.cfg.Tools[name] {
			status = "Enabled"
		}
		desc := toolDescs[name]
		option := fmt.Sprintf("%s: %s", name, status)
		if desc != "" {
			option = fmt.Sprintf("%s: %s - %s", name, status, desc)
		}
		m.menu.options = append(m.menu.options, option)
	}
	m.menu.options = append(m.menu.options, "Back")

	m.menu.open(menuTools)
	m.input.SetValue("")
}

func (m *ShellModel) selectToolOption() {
	if m.menu.selectedIdx == len(m.menu.options)-1 {
		m.menu.close()
		m.openConfigMenu()
		return
	}

	selectedTool := m.sortedToolNames()[m.menu.selectedIdx]
	m.cfg.Tools[selectedTool] = !m.cfg.Tools[selectedTool]

	if err := config.SaveConfig(m.cfg); err != nil {
		m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error saving config: %v", err)})
	} else {
		status := "disabled"
		if m.cfg.Tools[selectedTool] {
			status = "enabled"
		}
		m.messages = append(m.messages, Message{role: "system", content: fmt.Sprintf("Tool %s %s", selectedTool, status)})
	}

	m.openToolsMenu()
}

func (m *ShellModel) selectModel() {
	if m.menu.selectedIdx < 0 || m.menu.selectedIdx >= len(m.menu.models) {
		return
	}

	modelInfo := m.menu.models[m.menu.selectedIdx]
	selectedModel := modelInfo.Name
	provider := modelInfo.Provider

	if kind, err := config.SetDefaultModel(selectedModel, provider); err != nil {
		m.messages = append(m.messages, Message{role: "error", content: fmt.Sprintf("Error saving model: %v", err)})
	} else if kind == config.ModelTypeDecision {
		m.messages = append(m.messages, Message{role: "system", content: fmt.Sprintf("Switched decision model to: %s (used by `edgebot decide`)", selectedModel)})
		if newCfg, err := config.LoadConfig(); err == nil {
			m.cfg = newCfg
		} else {
			slog.Warn("reload config failed", "err", err)
		}
	} else {
		m.messages = append(m.messages, Message{role: "system", content: fmt.Sprintf("Switched to model: %s", selectedModel)})
		if newCfg, err := config.LoadConfig(); err == nil {
			m.cfg = newCfg
		} else {
			slog.Warn("reload config failed", "err", err)
		}
	}

	m.menu.close()
}
