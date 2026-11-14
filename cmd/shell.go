package cmd

import (
	"fmt"
	"sync"

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
	reqSeq             uint64
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
	answer := func(ok bool) (tea.Model, tea.Cmd) {
		m.mu.Lock()
		m.waitingConfirm = false
		m.mu.Unlock()
		select {
		case m.confirmationChan <- ok:
		default:
		}
		return m, nil
	}
	switch msg.Type {
	case tea.KeyEnter, tea.KeyEscape:
		return answer(false)
	}
	switch msg.String() {
	case "y", "Y":
		return answer(true)
	case "n", "N":
		return answer(false)
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
	m.mu.Lock()
	loading := m.loading
	if loading {
		if m.cancelChan != nil {
			m.cancelOnce.Do(func() { close(m.cancelChan) })
		}
		m.loading = false
		m.messages = append(m.messages, Message{role: "system", content: "Request cancelled."})
	}
	m.mu.Unlock()
	if loading {
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
		m.viewport.HalfPageUp()
	} else {
		m.viewport.HalfPageDown()
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
