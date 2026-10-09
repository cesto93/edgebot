package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/user"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/service"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	systemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00BFFF"))

	userStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF"))

	aiStyle = lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00BFFF"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFD700"))

	cmdStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00FF00"))

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#888888"))

	dimStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#666666"))

	highlightStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(lipgloss.Color("#444444"))

	userHeaderStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FFFFFF"))

	aiHeaderStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#00BFFF"))

	userBlockStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#888888")).
			Padding(0, 1)

	aiBlockStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#00BFFF")).
			Padding(0, 1)
)

var availableCommands = []string{
	"help",
	"get-config",
	"config",
	"models",
	"agent",
	"reset",
	"exit",
	"quit",
}

type menuKind int

const (
	menuNone menuKind = iota
	menuModel
	menuAgent
	menuConfig
	menuTools
	menuCommands
)

type menuState struct {
	kind        menuKind
	selectedIdx int
	options     []string
	models      []config.ModelInfo
}

func (ms *menuState) open(kind menuKind) {
	ms.kind = kind
	ms.selectedIdx = 0
}

func (ms *menuState) close() {
	ms.kind = menuNone
	ms.options = nil
	ms.models = nil
}

func (ms *menuState) itemCount() int {
	switch ms.kind {
	case menuModel:
		return len(ms.models)
	case menuAgent, menuConfig, menuTools, menuCommands:
		return len(ms.options)
	}
	return 0
}

type ShellExecutorForLLM struct {
	m *ShellModel
}

func (e *ShellExecutorForLLM) ExecuteTool(call llm.ToolCall) (string, error) {
	policy := &llm.ToolExecutorPolicy{
		ConfirmCommand: func(cmd string) bool {
			if !e.m.cfg.Shell.Confirm {
				return true
			}
			if config.IsAllowedCommand(config.GetCommandName(cmd), e.m.cfg.Shell.AllowedCommands) {
				return true
			}
			return e.AskConfirmation(cmd)
		},
		ConfirmWriteFile: func(path string) bool {
			if !e.m.cfg.Shell.Confirm {
				return true
			}
			return e.AskConfirmation(fmt.Sprintf("Write to file %s?", path))
		},
		OnExecute: func(call llm.ToolCall) {
			e.m.mu.Lock()
			e.m.messages = append(e.m.messages, Message{role: "assistant", content: systemStyle.Render(e.toolActionMessage(call))})
			e.m.mu.Unlock()
		},
	}
	return policy.ExecuteTool(call)
}

// toolActionMessage renders the status line shown in the transcript before a
// tool executes.
func (e *ShellExecutorForLLM) toolActionMessage(call llm.ToolCall) string {
	switch call.Name {
	case "RunCommand":
		cmd, ok := call.Arguments["command"].(string)
		if !ok {
			return "[Executing: <invalid args>]"
		}
		return fmt.Sprintf("[Executing: %s]", cmd)
	case "WriteFile":
		path, ok := call.Arguments["path"].(string)
		if !ok {
			return "[Writing to file: <invalid args>]"
		}
		return fmt.Sprintf("[Writing to file: %s]", strings.TrimPrefix(path, "@"))
	case "ReadFile":
		path, ok := call.Arguments["path"].(string)
		if !ok {
			return "[Reading file: <invalid args>]"
		}
		return fmt.Sprintf("[Reading file: %s]", strings.TrimPrefix(path, "@"))
	case "KVSet":
		key, ok := call.Arguments["key"].(string)
		if !ok {
			return "[KV Store: Saving <invalid args>]"
		}
		return fmt.Sprintf("[KV Store: Saving %s]", key)
	case "KVGet":
		key, ok := call.Arguments["key"].(string)
		if !ok {
			return "[KV Store: Retrieving <invalid args>]"
		}
		return fmt.Sprintf("[KV Store: Retrieving %s]", key)
	case "KVList":
		return "[KV Store: Listing keys]"
	}
	return ""
}

func (e *ShellExecutorForLLM) IsAllowedCommand(cmd string) bool {
	return config.IsAllowedCommand(config.GetCommandName(cmd), e.m.cfg.Shell.AllowedCommands)
}

func (e *ShellExecutorForLLM) AskConfirmation(cmd string) bool {
	e.m.pendingCommand = cmd
	e.m.waitingConfirm = true
	if e.m.teaProgram != nil {
		e.m.teaProgram.Send(confirmationMsg{cmd: cmd})
	}

	select {
	case result := <-e.m.confirmationChan:
		return result
	case <-e.m.cancelChan:
		e.m.waitingConfirm = false
		return false
	}
}

func getHistoryFile() string {
	usr, err := user.Current()
	if err != nil {
		return ""
	}
	return filepath.Join(usr.HomeDir, ".edgebot-history")
}

type Message struct {
	role       string
	content    string
	images     []string // Base64 encoded images or paths
	schema     string   // JSON schema path for structured commands
	toolCallID string
	toolCalls  []llm.OpenAIToolCall
}

type responseReadyMsg struct{}
type confirmationMsg struct {
	cmd string
}

type ShellModel struct {
	mu                 sync.Mutex
	teaProgram         *tea.Program
	input              textinput.Model
	viewport           viewport.Model
	ready              bool
	followOutput       bool
	lastMsgCount       int
	messages           []Message
	history            []string
	historyIndex       int
	commandHistoryPath string
	width              int
	height             int
	quitting           bool
	cfg                *config.Config
	suggestions        []string
	selectedIndex      int
	showSuggestions    bool
	loading            bool
	cancelChan         chan struct{}
	cancelOnce         sync.Once
	confirmationChan   chan bool
	pendingCommand     string
	waitingConfirm     bool
	menu               menuState
	commands           []config.CommandInfo
	serviceClient      *service.Client
	allowedCmdMode     struct {
		active bool
	}
}

func NewShellModel() (*ShellModel, error) {
	cfg, err := config.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	ti := textinput.New()
	ti.Placeholder = "Ask the AI..."
	ti.Focus()
	ti.Prompt = "edgebot > "

	historyPath := getHistoryFile()

	m := &ShellModel{
		input:              ti,
		messages:           []Message{},
		history:            loadHistory(historyPath),
		historyIndex:       -1,
		commandHistoryPath: historyPath,
		cfg:                cfg,
		commands:           config.LoadCommands(cfg),
		confirmationChan:   make(chan bool, 1),
		followOutput:       true,
	}

	return m, nil
}

func (m *ShellModel) SetProgram(p *tea.Program) {
	m.teaProgram = p
}

func (m *ShellModel) Init() tea.Cmd {
	return textinput.Blink
}

func (m *ShellModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case responseReadyMsg:
		m.loading = false
		return m, nil

	case confirmationMsg:
		m.waitingConfirm = true
		m.pendingCommand = msg.cmd
		return m, nil

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		if !m.ready {
			h := msg.Height - 4
			if h < 1 {
				h = 1
			}
			m.viewport = viewport.New(msg.Width, h)
			m.ready = true
			m.followOutput = true
		} else {
			m.viewport.Width = msg.Width
		}
		return m, nil

	case tea.MouseMsg:
		if m.ready {
			var cmd tea.Cmd
			m.viewport, cmd = m.viewport.Update(msg)
			m.followOutput = m.viewport.AtBottom()
			return m, cmd
		}
		return m, nil

	case tea.KeyMsg:
		if m.waitingConfirm {
			return m.handleConfirmationKeys(msg)
		}

		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyCtrlD:
			m.quitting = true
			return m, tea.Quit

		case tea.KeyEnter:
			return m.handleEnterKey()

		case tea.KeyPgUp:
			return m.scrollTranscript(-1)

		case tea.KeyPgDown:
			return m.scrollTranscript(1)

		case tea.KeyShiftUp:
			return m.scrollLines(-3)

		case tea.KeyShiftDown:
			return m.scrollLines(3)

		case tea.KeyHome:
			return m.jumpTranscript(true)

		case tea.KeyEnd:
			return m.jumpTranscript(false)

		case tea.KeyUp:
			if m.menu.kind != menuNone {
				return m.handleMenuNav(-1)
			}
			if m.showSuggestions && len(m.suggestions) > 0 {
				return m.navigateSuggestions(-1)
			}
			return m.navigateHistory(1)

		case tea.KeyDown:
			if m.menu.kind != menuNone {
				return m.handleMenuNav(1)
			}
			if m.showSuggestions && len(m.suggestions) > 0 {
				return m.navigateSuggestions(1)
			}
			return m.navigateHistory(-1)

		case tea.KeyTab:
			return m.handleAutocomplete()

		case tea.KeyEscape:
			return m.handleEscapeKey()
		}

		if m.menu.kind != menuNone {
			return m.handleMenuKeys(msg.String())
		}
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)

	if !m.allowedCmdMode.active && m.menu.kind == menuNone {
		m.updateSuggestions()
	} else {
		m.showSuggestions = false
	}

	return m, cmd
}

func (m *ShellModel) handleConfirmationKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEnter, tea.KeyEscape:
		m.waitingConfirm = false
		select {
		case m.confirmationChan <- false:
		default:
		}
		return m, nil
	}
	switch msg.String() {
	case "y", "Y":
		m.waitingConfirm = false
		select {
		case m.confirmationChan <- true:
		default:
		}
	case "n", "N":
		m.waitingConfirm = false
		select {
		case m.confirmationChan <- false:
		default:
		}
	}
	return m, nil
}

func (m *ShellModel) handleEnterKey() (tea.Model, tea.Cmd) {
	switch m.menu.kind {
	case menuModel:
		m.selectModel()
		return m, nil
	case menuAgent:
		m.selectAgentOption()
		return m, nil
	case menuConfig:
		m.selectConfigOption()
		return m, nil
	case menuTools:
		m.selectToolOption()
		return m, nil
	case menuCommands:
		m.selectCommandOption()
		return m, nil
	}
	if m.showSuggestions && len(m.suggestions) > 0 {
		return m.selectSuggestion()
	}
	return m.handleSubmit()
}

func (m *ShellModel) handleEscapeKey() (tea.Model, tea.Cmd) {
	if m.loading {
		if m.cancelChan != nil {
			m.cancelOnce.Do(func() { close(m.cancelChan) })
		}
		m.loading = false
		m.mu.Lock()
		m.messages = append(m.messages, Message{role: "system", content: "Request cancelled."})
		m.mu.Unlock()
		return m, nil
	}
	if m.menu.kind != menuNone {
		m.menu.close()
		return m, nil
	}
	if m.allowedCmdMode.active {
		m.allowedCmdMode.active = false
		m.input.Prompt = "edgebot > "
		m.input.SetValue("")
		return m, nil
	}
	m.input.SetValue("")
	m.showSuggestions = false
	return m, nil
}

func (m *ShellModel) handleMenuNav(dir int) (tea.Model, tea.Cmd) {
	n := m.menu.itemCount()
	if dir < 0 && m.menu.selectedIdx > 0 {
		m.menu.selectedIdx--
	} else if dir > 0 && m.menu.selectedIdx < n-1 {
		m.menu.selectedIdx++
	}
	return m, nil
}

// scrollTranscript moves the transcript viewport by half a page. Scrolling up
// detaches the auto-follow tail; scrolling back to the bottom re-attaches it.
func (m *ShellModel) scrollTranscript(dir int) (tea.Model, tea.Cmd) {
	if !m.ready {
		return m, nil
	}
	if dir < 0 {
		m.viewport.HalfViewUp()
	} else {
		m.viewport.HalfViewDown()
	}
	m.followOutput = m.viewport.AtBottom()
	return m, nil
}

// scrollLines moves the transcript viewport by n lines (negative scrolls
// up). Used for fine-grained gestures like Shift+Up/Down.
func (m *ShellModel) scrollLines(n int) (tea.Model, tea.Cmd) {
	if !m.ready {
		return m, nil
	}
	if n < 0 {
		m.viewport.ScrollUp(-n)
	} else {
		m.viewport.ScrollDown(n)
	}
	m.followOutput = m.viewport.AtBottom()
	return m, nil
}

// jumpTranscript jumps to the top (top=true) or tails the latest output.
func (m *ShellModel) jumpTranscript(top bool) (tea.Model, tea.Cmd) {
	if !m.ready {
		return m, nil
	}
	if top {
		m.viewport.GotoTop()
		m.followOutput = false
	} else {
		m.viewport.GotoBottom()
		m.followOutput = true
	}
	return m, nil
}

func (m *ShellModel) handleMenuKeys(key string) (tea.Model, tea.Cmd) {
	switch key {
	case "j", "down":
		return m.handleMenuNav(1)
	case "k", "up":
		return m.handleMenuNav(-1)
	}
	return m, nil
}

func (m *ShellModel) viewWidth() int {
	if m.width > 0 {
		return m.width
	}
	return 80
}

// wrapShellText word-wraps text to the given display width, preserving
// existing newlines. Bubbletea renders the View in an alternate screen with
// explicit cursor positioning, so lines wider than the terminal are clipped
// instead of soft-wrapped by the terminal. Without this, long AI answers are
// cut off laterally.
func wrapShellText(text string, width int) string {
	if width <= 0 {
		return text
	}
	paras := strings.Split(text, "\n")
	out := make([]string, 0, len(paras))
	for _, para := range paras {
		out = append(out, wrapShellLine(para, width)...)
	}
	return strings.Join(out, "\n")
}

// wrapShellTextWithPrefix wraps text so that every line (including the
// prefix on the first line and a matching indent on continuations) fits in
// width. It returns the individual lines.
func wrapShellTextWithPrefix(prefix, text string, width int) []string {
	pw := utf8.RuneCountInString(prefix)
	bw := width - pw
	if bw < 1 {
		bw = 1
	}
	indent := strings.Repeat(" ", pw)
	paras := strings.Split(text, "\n")
	var out []string
	for pi, para := range paras {
		for li, l := range wrapShellLine(para, bw) {
			if pi == 0 && li == 0 {
				out = append(out, prefix+l)
			} else {
				out = append(out, indent+l)
			}
		}
	}
	return out
}

// wrapShellLine wraps a single line (no newlines) on spaces, hard-splitting
// words longer than the width. Leading indentation is preserved on
// continuation lines.
func wrapShellLine(line string, width int) []string {
	if width <= 0 {
		return []string{line}
	}
	if utf8.RuneCountInString(line) <= width {
		return []string{line}
	}
	trimmed := strings.TrimLeft(line, " \t")
	if trimmed == "" {
		return []string{line}
	}
	indentLen := utf8.RuneCountInString(line) - utf8.RuneCountInString(trimmed)
	var indent string
	effWidth := width
	if indentLen < width {
		indent = line[:len(line)-len(trimmed)]
		effWidth = width - indentLen
	}
	var lines []string
	var cur string
	flush := func() {
		if cur != "" {
			lines = append(lines, indent+cur)
			cur = ""
		}
	}
	for _, word := range strings.Fields(trimmed) {
		for utf8.RuneCountInString(word) > effWidth {
			flush()
			r := []rune(word)
			lines = append(lines, indent+string(r[:effWidth]))
			word = string(r[effWidth:])
		}
		switch {
		case cur == "":
			cur = word
		case utf8.RuneCountInString(cur)+1+utf8.RuneCountInString(word) <= effWidth:
			cur += " " + word
		default:
			flush()
			cur = word
		}
	}
	flush()
	if len(lines) == 0 {
		return []string{line}
	}
	return lines
}

// renderTurnBlock renders a user or AI turn as a rounded bordered block that
// fits the outer width (border + padding included). Note lipgloss Width
// covers content + padding but not the border, hence outer-2.
func renderTurnBlock(header string, headerStyle, blockStyle lipgloss.Style, content string, outer int) string {
	if outer < 20 {
		return headerStyle.Render(header+": ") + wrapShellText(content, outer)
	}
	inner := outer - 4 // border (2) + padding (2)
	wrapped := wrapShellText(content, inner)
	body := headerStyle.Render(header) + "\n" + wrapped
	return blockStyle.Width(outer - 2).Render(body)
}

func renderUserBlock(content string, outer int) string {
	return renderTurnBlock("You", userHeaderStyle, userBlockStyle, content, outer)
}

func renderAIBlock(content string, outer int) string {
	return renderTurnBlock("AI", aiHeaderStyle, aiBlockStyle, content, outer)
}

// renderTranscript builds the scrollable body: turn blocks, plain status
// lines, menus and suggestions. The input line and its status stay pinned in
// the footer (renderFooter).
func (m *ShellModel) renderTranscript(msgs []Message, w int) string {
	var sb strings.Builder

	for _, msg := range msgs {
		switch msg.role {
		case "system":
			sb.WriteString(systemStyle.Render(wrapShellText(msg.content, w)))
			sb.WriteString("\n")
		case "user":
			sb.WriteString(renderUserBlock(msg.content, w))
			sb.WriteString("\n")
		case "assistant":
			sb.WriteString(renderAIBlock(msg.content, w))
			sb.WriteString("\n")
		case "tool":
			sb.WriteString(cmdStyle.Render(wrapShellText(msg.content, w)))
			sb.WriteString("\n")
		case "error":
			sb.WriteString(errorStyle.Render(strings.Join(
				wrapShellTextWithPrefix("Error: ", msg.content, w), "\n")))
			sb.WriteString("\n")
		}
	}

	m.renderMenu(&sb)

	if m.showSuggestions && len(m.suggestions) > 0 {
		sb.WriteString(dimStyle.Render("Suggestions: "))
		for i, suggestion := range m.suggestions {
			display := suggestion
			if lastAt := strings.LastIndex(suggestion, "@"); lastAt != -1 {
				display = suggestion[lastAt:]
			}

			if i == m.selectedIndex {
				sb.WriteString(highlightStyle.Render(" " + display + " "))
			} else {
				sb.WriteString(helpStyle.Render(" " + display + " "))
			}
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// renderFooter builds the pinned bottom area: an optional scroll hint plus
// the confirmation prompt, loading status or input line.
func (m *ShellModel) renderFooter() string {
	var sb strings.Builder

	if hint := m.scrollHint(); hint != "" {
		sb.WriteString(dimStyle.Render(hint))
		sb.WriteString("\n")
	}

	if m.waitingConfirm {
		sb.WriteString(systemStyle.Render(wrapShellText(
			fmt.Sprintf("[LLM wants to execute: %s]", m.pendingCommand), m.viewWidth())))
		sb.WriteString("\n")
		sb.WriteString(dimStyle.Render("Confirm execution? [y/N]"))
		sb.WriteString("\n")
	} else if m.loading {
		sb.WriteString(systemStyle.Render("Thinking... (Press Esc to cancel)"))
		sb.WriteString("\n")
	} else {
		sb.WriteString(m.input.View())
		sb.WriteString("\n")
	}

	return sb.String()
}

// scrollHint reports the transcript scroll state for the footer. It returns
// "" when everything fits on screen. A persistent hint matters because the
// viewport has no visible scrollbar and plain Up/Down navigate input history,
// so users otherwise never discover PgUp/PgDn/wheel scrolling.
func (m *ShellModel) scrollHint() string {
	if !m.ready {
		return ""
	}
	if m.viewport.TotalLineCount() <= m.viewport.Height {
		return ""
	}
	if m.followOutput {
		pct := int(m.viewport.ScrollPercent()*100 + 0.5)
		return fmt.Sprintf("scroll %d%% • PgUp/PgDn or Shift+Up/Down or wheel", pct)
	}
	return "PgUp/PgDn scroll • End follows latest"
}

func (m *ShellModel) View() string {
	if m.quitting {
		return fmt.Sprintf("%sGoodbye!%s\n", systemStyle.Render(""), "")
	}

	m.mu.Lock()
	msgs := append([]Message(nil), m.messages...)
	m.mu.Unlock()

	w := m.viewWidth()
	body := m.renderTranscript(msgs, w)
	footer := m.renderFooter()

	if !m.ready || m.height <= 0 {
		return body + footer
	}

	// New turns re-attach the tail so answers stay visible.
	if len(msgs) != m.lastMsgCount {
		m.followOutput = true
		m.lastMsgCount = len(msgs)
	}

	m.viewport.Width = w
	m.viewport.SetContent(body)
	h := m.height - lipgloss.Height(footer) - 1
	if h < 1 {
		h = 1
	}
	m.viewport.Height = h
	if m.followOutput {
		m.viewport.GotoBottom()
	}

	return m.viewport.View() + "\n" + footer
}

func (m *ShellModel) renderMenu(sb *strings.Builder) {
	switch m.menu.kind {
	case menuModel:
		sb.WriteString(systemStyle.Render("Select Model (↑/↓ to navigate, Enter to select, Esc to cancel):"))
		sb.WriteString("\n")
		for i, model := range m.menu.models {
			marker := ""
			if model.Name == m.cfg.LLM.Model {
				marker += "*"
			}
			if model.Provider == "llamacpp" && m.cfg.LLM.DecisionModel != "" && model.Name == m.cfg.LLM.DecisionModel {
				marker += "D"
			}
			label := fmt.Sprintf(" %-3s%s (%s) ", marker, model.Name, model.Provider)
			if i == m.menu.selectedIdx {
				sb.WriteString(highlightStyle.Render(label))
			} else {
				sb.WriteString(userStyle.Render(label))
			}
			sb.WriteString("\n")
		}
	case menuAgent:
		sb.WriteString(systemStyle.Render("Select Agent (↑/↓ to navigate, Enter to select, Esc to cancel):"))
		sb.WriteString("\n")
		for i, opt := range m.menu.options {
			marker := " "
			if opt == m.cfg.Agent {
				marker = "*"
			}
			label := fmt.Sprintf(" %s %s ", marker, opt)
			if i == m.menu.selectedIdx {
				sb.WriteString(highlightStyle.Render(label))
			} else {
				sb.WriteString(userStyle.Render(label))
			}
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
		for _, def := range llm.GetAgentDefs() {
			sb.WriteString(dimStyle.Render(fmt.Sprintf("  %s: %s", def.Name, def.Description)))
			sb.WriteString("\n")
		}
	case menuTools:
		sb.WriteString(systemStyle.Render("Manage Tools (↑/↓ to navigate, Enter to toggle, Esc to back):"))
		sb.WriteString("\n")
		m.renderMenuOptions(sb)
	case menuCommands:
		sb.WriteString(systemStyle.Render("Manage Commands (↑/↓ to navigate, Enter to select, Esc to back):"))
		sb.WriteString("\n")
		m.renderMenuOptions(sb)
	case menuConfig:
		sb.WriteString(systemStyle.Render("Configuration Menu (↑/↓ to navigate, Enter to select, Esc to cancel):"))
		sb.WriteString("\n")
		m.renderMenuOptions(sb)
	}
}

func (m *ShellModel) renderMenuOptions(sb *strings.Builder) {
	for i, opt := range m.menu.options {
		if i == m.menu.selectedIdx {
			sb.WriteString(highlightStyle.Render(fmt.Sprintf(" > %s ", opt)))
		} else {
			sb.WriteString(userStyle.Render(fmt.Sprintf("   %s ", opt)))
		}
		sb.WriteString("\n")
	}
}

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

	m.loading = true
	m.cancelChan = make(chan struct{})
	m.cancelOnce = sync.Once{}

	go m.ElaborateMessage()

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
			m.loading = true
			m.cancelChan = make(chan struct{})
			m.cancelOnce = sync.Once{}
			go m.ElaborateMessage()
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

func (m *ShellModel) ElaborateMessage() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cancelChan := m.cancelChan
	if cancelChan != nil {
		go func() {
			select {
			case <-cancelChan:
				cancel()
			case <-ctx.Done():
			}
		}()
	}

	var commonMessages []llm.Message
	var structuredSchema string
	// Copy messages under lock to avoid race with Update goroutine.
	m.mu.Lock()
	msgsCopy := append([]Message(nil), m.messages...)
	m.mu.Unlock()
	for _, msg := range msgsCopy {
		if msg.role == "user" || msg.role == "assistant" || msg.role == "tool" {
			if msg.schema != "" && msg.role == "user" {
				structuredSchema = msg.schema
			}
			var content any = msg.content
			if len(msg.images) > 0 {
				parts := []llm.ContentPart{
					{Type: "text", Text: msg.content},
				}
				for _, img := range msg.images {
					parts = append(parts, llm.ContentPart{
						Type:     "image_url",
						ImageURL: &llm.ContentImage{URL: img},
					})
				}
				content = parts
			}
			commonMessages = append(commonMessages, llm.Message{Role: msg.role, Content: content, ToolCallID: msg.toolCallID, ToolCalls: msg.toolCalls})
		}
	}

	if structuredSchema != "" {
		m.runStructuredMessage(ctx, commonMessages, structuredSchema)
		return
	}

	if m.serviceClient != nil || service.IsActive() {
		resultMessages, err := m.serviceChat(ctx, commonMessages)
		if err != nil {
			if errors.Is(err, service.ErrUnavailable) {
				slog.Debug("service unavailable, falling back to local execution", "err", err)
			} else {
				m.failWithError(fmt.Sprintf("Error: %v", err))
				return
			}
		} else {
			m.appendLLMResult(resultMessages)
			m.finishSuccess()
			return
		}
	}

	agent := llm.NewAgentForSession(m.cfg.Agent, m.cfg.LLM.Model, m.cfg.LLM.Provider, m.cfg.Tools, m.cfg.LitertLM.Backend, m.cfg.AgentFiles, m.cfg.Skills, llm.ThinkEffort(m.cfg.LLM.ThinkEffort))

	executor := &ShellExecutorForLLM{m: m}

	resultMessages, err := agent.CallLLM(ctx, executor, commonMessages)

	if err != nil {
		m.failWithError(fmt.Sprintf("Error: %v", err))
		return
	}

	m.appendLLMResult(resultMessages)
	m.finishSuccess()
}

func (m *ShellModel) failWithError(content string) {
	m.mu.Lock()
	m.messages = append(m.messages, Message{role: "error", content: content})
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

func (m *ShellModel) finishSuccess() {
	m.mu.Lock()
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

// serviceChat routes a request through the running service, maintaining a
// cached client on the shell. Connectivity failures (ErrUnavailable) clear
// the cached client so the caller can fall back to local execution.
func (m *ShellModel) serviceChat(ctx context.Context, messages []llm.Message) ([]llm.Message, error) {
	if m.serviceClient == nil {
		c, err := service.NewClient()
		if err != nil {
			return nil, service.ErrUnavailable
		}
		m.serviceClient = c
	}

	req := chatRequestFromConfig(m.cfg, messages)

	result, err := m.serviceClient.Chat(ctx, req)
	if err != nil {
		if errors.Is(err, service.ErrUnavailable) {
			if closeErr := m.serviceClient.Close(); closeErr != nil {
				slog.Warn("service client close failed", "err", closeErr)
			}
			m.serviceClient = nil
		}
		return nil, err
	}
	return result, nil
}

// runStructuredMessage executes a structured command (a command carrying a
// JSON schema in its frontmatter) through CallStructured, bypassing the
// service like extract did, and renders the JSON result.
func (m *ShellModel) runStructuredMessage(ctx context.Context, messages []llm.Message, schemaPath string) {
	safeAppendError := func(text string) {
		m.mu.Lock()
		m.messages = append(m.messages, Message{role: "error", content: text})
		m.loading = false
		m.mu.Unlock()
		if m.teaProgram != nil {
			m.teaProgram.Send(responseReadyMsg{})
		}
	}
	cleanPath := filepath.Clean(schemaPath)
	if !filepath.IsAbs(cleanPath) {
		if cwd, err := os.Getwd(); err == nil {
			cleanPath = filepath.Join(cwd, cleanPath)
		}
	}
	schemaData, err := os.ReadFile(cleanPath)
	if err != nil {
		safeAppendError(fmt.Sprintf("Error: failed to read schema: %v", err))
		return
	}

	var schemaRaw any
	if err := json.Unmarshal(schemaData, &schemaRaw); err != nil {
		safeAppendError(fmt.Sprintf("Error: invalid JSON schema: %v", err))
		return
	}

	responseFormat := structuredResponseFormat(schemaRaw)

	systemPrompt := "You extract structured data from documents and images. Return only valid JSON matching the provided schema."

	caller := llm.NewProviderCallerRawWithThink(m.cfg.LLM.Provider, m.cfg.LLM.Model, llm.NoopExecutor{}, llm.ThinkEffort(m.cfg.LLM.ThinkEffort))
	if lc, ok := caller.(*llm.LitertLMCaller); ok {
		lc.Backend = m.cfg.LitertLM.Backend
	}
	resultMessages, err := caller.CallStructured(ctx, systemPrompt, messages, nil, responseFormat)
	if err != nil {
		safeAppendError(fmt.Sprintf("Error: LLM call failed: %v", err))
		return
	}

	if len(resultMessages) == 0 {
		safeAppendError("Error: no response from LLM")
		return
	}

	lastMsg := resultMessages[len(resultMessages)-1]
	contentStr, ok := lastMsg.Content.(string)
	if !ok || strings.TrimSpace(contentStr) == "" {
		safeAppendError("Error: empty response from LLM")
		return
	}

	result := prettyJSONOrRaw(strings.TrimSpace(contentStr))

	m.mu.Lock()
	m.messages = append(m.messages, Message{role: "assistant", content: result})
	m.loading = false
	m.mu.Unlock()
	if m.teaProgram != nil {
		m.teaProgram.Send(responseReadyMsg{})
	}
}

// appendLLMResult renders the messages returned by the LLM (locally or via
// the service) into the shell transcript.
func (m *ShellModel) appendLLMResult(resultMessages []llm.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, msg := range resultMessages {
		contentStr := ""
		switch v := msg.Content.(type) {
		case string:
			contentStr = v
		case []llm.ContentPart:
			var sb strings.Builder
			for _, p := range v {
				if p.Type == "text" {
					sb.WriteString(p.Text)
				}
			}
			contentStr = sb.String()
		case nil:
			contentStr = ""
		default:
			contentStr = fmt.Sprintf("%v", msg.Content)
		}

		switch msg.Role {
		case "system":
			m.messages = append(m.messages, Message{role: "system", content: contentStr})
		case "user":
			m.messages = append(m.messages, Message{role: "user", content: contentStr})
		case "assistant":
			m.messages = append(m.messages, Message{role: "assistant", content: contentStr, toolCalls: msg.ToolCalls})
		case "tool":
			m.messages = append(m.messages, Message{role: "tool", content: contentStr, toolCallID: msg.ToolCallID})
		}
	}
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

func RunShell() error {
	m, err := NewShellModel()
	if err != nil {
		return fmt.Errorf("failed to create shell model: %w", err)
	}

	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	m.SetProgram(p)

	_, err = p.Run()
	return err
}
