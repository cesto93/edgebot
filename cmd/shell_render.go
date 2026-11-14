package cmd

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"edgebot/llm"

	"github.com/charmbracelet/lipgloss"
)

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

// transcriptCache memoizes the rendered transcript. Bubbletea calls View on
// every message (keystrokes, ticks, async results), so re-wrapping and
// re-styling the whole conversation on each keystroke is pure waste in long
// sessions. The key covers everything renderTranscript reads.
type transcriptCache struct {
	valid         bool
	msgCount      int
	width         int
	menuKind      menuKind
	menuSelected  int
	menuItems     string
	showSuggests  bool
	suggestions   string
	suggestIndex  int
	model         string
	decisionModel string
	agent         string
	rendered      string
}

// transcriptBody returns the rendered transcript, reusing the previous render
// when nothing visible changed.
func (m *ShellModel) transcriptBody(msgs []Message, width int) string {
	key := transcriptCache{
		valid:        true,
		msgCount:     len(msgs),
		width:        width,
		menuKind:     m.menu.kind,
		menuSelected: m.menu.selectedIdx,
		menuItems:    menuSignature(m.menu),
		showSuggests: m.showSuggestions,
		suggestions:  strings.Join(m.suggestions, "\x00"),
		suggestIndex: m.selectedIndex,
	}
	if m.cfg != nil {
		key.model = m.cfg.LLM.Model
		key.decisionModel = m.cfg.LLM.DecisionModel
		key.agent = m.cfg.Agent
	}
	c := &m.transcript
	// The turn path is append-only, so any count change (including a
	// reset-to-empty) invalidates the cache.
	if c.valid && c.msgCount == key.msgCount &&
		c.width == key.width && c.menuKind == key.menuKind &&
		c.menuSelected == key.menuSelected && c.menuItems == key.menuItems &&
		c.showSuggests == key.showSuggests && c.suggestions == key.suggestions &&
		c.suggestIndex == key.suggestIndex && c.model == key.model &&
		c.decisionModel == key.decisionModel && c.agent == key.agent {
		return c.rendered
	}

	rendered := m.renderTranscript(msgs, width)
	*c = key
	c.rendered = rendered
	return rendered
}

// menuSignature captures the variable menu contents (options/models) so the
// transcript cache invalidates when a menu is rebuilt in place.
func menuSignature(menu menuState) string {
	var sb strings.Builder
	for _, opt := range menu.options {
		sb.WriteString(opt)
		sb.WriteByte('\x00')
	}
	for _, model := range menu.models {
		sb.WriteString(model.Name)
		sb.WriteByte('\x00')
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

	m.mu.Lock()
	waitingConfirm := m.waitingConfirm
	pendingCommand := m.pendingCommand
	loading := m.loading
	m.mu.Unlock()

	if waitingConfirm {
		sb.WriteString(systemStyle.Render(wrapShellText(
			fmt.Sprintf("[LLM wants to execute: %s]", pendingCommand), m.viewWidth())))
		sb.WriteString("\n")
		sb.WriteString(dimStyle.Render("Confirm execution? [y/N]"))
		sb.WriteString("\n")
	} else if loading {
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
	body := m.transcriptBody(msgs, w)
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
