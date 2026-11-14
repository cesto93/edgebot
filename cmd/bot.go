package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/service"

	"github.com/spf13/cobra"
)

var (
	botToken     string
	botAllowFrom string
)

var botCmd = &cobra.Command{
	Use:   "bot",
	Short: "Start a Telegram bot that forwards messages to the agent",
	Long: `Starts a Telegram bot that forwards incoming messages to the configured agent.

The bot token is read from --token or the TELEGRAM_BOT_TOKEN environment
variable (set it in ~/.config/edgebot/.env or ./.env). Create a bot via
@BotFather on Telegram to obtain a token.

Each Telegram chat gets its own conversation history. Use /reset in the chat
to clear it. Tool calls (RunCommand, ReadFile, etc.) run automatically;
when shell.confirm is true, RunCommand is restricted to allowed_commands
and WriteFile is denied (like the gRPC service).

Example:
  export TELEGRAM_BOT_TOKEN=123456:ABC...
  edgebot bot
  edgebot bot --token 123456:ABC... --allow-from 123456789,@myuser`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runBot()
	},
}

func init() {
	botCmd.Flags().StringVar(&botToken, "token", "", "Telegram bot token (defaults to $TELEGRAM_BOT_TOKEN)")
	botCmd.Flags().StringVar(&botAllowFrom, "allow-from", "", "Comma-separated allowlist of chat IDs or @usernames (empty allows everyone, also reads $TELEGRAM_ALLOWED_CHAT_IDS)")
	rootCmd.AddCommand(botCmd)
}

// telegramClient is a minimal Telegram Bot API client using only net/http.
type telegramClient struct {
	token      string
	httpClient *http.Client
	baseURL    string
}

func newTelegramClient(token string) *telegramClient {
	return &telegramClient{
		token:      token,
		httpClient: &http.Client{Timeout: 70 * time.Second},
		baseURL:    "https://api.telegram.org/bot" + token,
	}
}

// telegram types (only fields we need)
type tgUser struct {
	ID       int64  `json:"id"`
	Username string `json:"username"`
}

type tgChat struct {
	ID       int64  `json:"id"`
	Type     string `json:"type"`
	Username string `json:"username"`
	Title    string `json:"title"`
}

type tgMessage struct {
	MessageID int64   `json:"message_id"`
	From      *tgUser `json:"from"`
	Chat      tgChat  `json:"chat"`
	Text      string  `json:"text"`
	Date      int64   `json:"date"`
}

type tgUpdate struct {
	UpdateID int64      `json:"update_id"`
	Message  *tgMessage `json:"message"`
}

type tgGetUpdatesResponse struct {
	OK     bool       `json:"ok"`
	Result []tgUpdate `json:"result"`
}

func (c *telegramClient) getMe(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/getMe", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("getMe request failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("getMe read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("getMe failed: %s %s", resp.Status, string(body))
	}
	var r struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return fmt.Errorf("getMe parse failed: %w", err)
	}
	if !r.OK {
		return fmt.Errorf("getMe returned not ok: %s", string(body))
	}
	return nil
}

func (c *telegramClient) getUpdates(ctx context.Context, offset int64, timeoutSec int) ([]tgUpdate, error) {
	u, err := url.Parse(c.baseURL + "/getUpdates")
	if err != nil {
		return nil, fmt.Errorf("parse updates URL: %w", err)
	}
	q := u.Query()
	q.Set("offset", strconv.FormatInt(offset, 10))
	q.Set("timeout", strconv.Itoa(timeoutSec))
	q.Set("allowed_updates", `["message"]`)
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("getUpdates read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("getUpdates %s: %s", resp.Status, string(body))
	}
	var r tgGetUpdatesResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("getUpdates unmarshal: %w", err)
	}
	if !r.OK {
		return nil, fmt.Errorf("getUpdates not ok: %s", string(body))
	}
	return r.Result, nil
}

func (c *telegramClient) sendMessage(ctx context.Context, chatID int64, text string) error {
	for _, chunk := range splitTelegramMessage(text, 4096) {
		if err := c.sendMessageChunk(ctx, chatID, chunk); err != nil {
			return err
		}
	}
	return nil
}

func (c *telegramClient) sendMessageChunk(ctx context.Context, chatID int64, text string) error {
	payload := map[string]any{
		"chat_id": chatID,
		"text":    text,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal sendMessage: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/sendMessage", bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("sendMessage failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("sendMessage read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("sendMessage %s: %s", resp.Status, string(body))
	}
	var r struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return fmt.Errorf("sendMessage parse: %w", err)
	}
	if !r.OK {
		return fmt.Errorf("sendMessage not ok: %s", string(body))
	}
	return nil
}

func (c *telegramClient) sendChatAction(ctx context.Context, chatID int64, action string) {
	payload := map[string]any{
		"chat_id": chatID,
		"action":  action,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		slog.Debug("marshal sendChatAction failed", "err", err)
		return
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/sendChatAction", bytes.NewReader(data))
	if err != nil {
		slog.Debug("sendChatAction request failed", "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		slog.Debug("sendChatAction do failed", "err", err)
		return
	}
	defer resp.Body.Close()
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		slog.Debug("sendChatAction discard failed", "err", err)
	}
	if resp.StatusCode != http.StatusOK {
		slog.Debug("sendChatAction non-OK", "status", resp.Status)
	}
}

func splitTelegramMessage(text string, limit int) []string {
	runes := []rune(text)
	if len(runes) <= limit {
		return []string{text}
	}
	var chunks []string
	for len(runes) > 0 {
		if len(runes) <= limit {
			chunks = append(chunks, string(runes))
			break
		}
		cut := limit
		for i := limit - 1; i > limit/2; i-- {
			if runes[i] == '\n' {
				cut = i + 1
				break
			}
		}
		chunks = append(chunks, string(runes[:cut]))
		runes = runes[cut:]
	}
	return chunks
}

// botSession holds per-chat conversation history.
type botSession struct {
	mu       sync.Mutex
	messages map[int64][]llm.Message
	// chatLocks serializes message handling per chat. Sharding keeps the
	// lock table bounded; two chats occasionally sharing a stripe only
	// delays one of them.
	chatLocks [botChatLockStripes]sync.Mutex
}

const botChatLockStripes = 64

func newBotSession() *botSession {
	return &botSession{messages: make(map[int64][]llm.Message)}
}

// lockForChat returns the stripe mutex that serializes one chat's messages.
func (s *botSession) lockForChat(chatID int64) *sync.Mutex {
	return &s.chatLocks[uint64(chatID)%botChatLockStripes]
}

const botMaxHistory = 50

func (s *botSession) append(chatID int64, msgs ...llm.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.messages[chatID] = append(s.messages[chatID], msgs...)
	if len(s.messages[chatID]) > botMaxHistory {
		s.messages[chatID] = s.messages[chatID][len(s.messages[chatID])-botMaxHistory:]
	}
}

func (s *botSession) get(chatID int64) []llm.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	msgs := s.messages[chatID]
	out := make([]llm.Message, len(msgs))
	copy(out, msgs)
	return out
}

func (s *botSession) reset(chatID int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.messages, chatID)
}

// botExecutor runs tools for the telegram bot. It mirrors the service's
// confirm policy: when confirm is true, only allowed commands pass and
// WriteFile is denied (there is no interactive prompt in Telegram).
type botExecutor struct {
	cfg *config.Config
}

func (e *botExecutor) ExecuteTool(ctx context.Context, call llm.ToolCall) (string, error) {
	policy := llm.NewConfirmPolicy(e.cfg.Shell.Confirm, e.cfg.Shell.AllowedCommands)
	policy.CommandTimeout = e.cfg.Shell.CommandTimeout
	policy.ReadDenylist = e.cfg.Shell.ReadDenylist
	policy.OnExecute = func(call llm.ToolCall) {
		slog.Info("bot tool execution", "tool", call.Name, "args", call.Arguments)
	}
	return policy.ExecuteTool(ctx, call)
}

func (e *botExecutor) IsAllowedCommand(cmd string) bool {
	return config.IsAllowedCommand(cmd, e.cfg.Shell.AllowedCommands)
}

func (e *botExecutor) AskConfirmation(cmd string) bool {
	if !e.cfg.Shell.Confirm {
		return true
	}
	return e.IsAllowedCommand(cmd)
}

func resolveBotToken() string {
	if botToken != "" {
		return botToken
	}
	if v := os.Getenv("TELEGRAM_BOT_TOKEN"); v != "" {
		return v
	}
	return ""
}

func parseAllowList() (map[int64]bool, map[string]bool) {
	raw := botAllowFrom
	if raw == "" {
		raw = os.Getenv("TELEGRAM_ALLOWED_CHAT_IDS")
	}
	if raw == "" {
		raw = os.Getenv("TELEGRAM_ALLOWED_USERS")
	}
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	ids := make(map[int64]bool)
	users := make(map[string]bool)
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if id, err := strconv.ParseInt(strings.TrimPrefix(part, "@"), 10, 64); err == nil {
			ids[id] = true
			continue
		}
		// username
		part = strings.TrimPrefix(part, "@")
		users[strings.ToLower(part)] = true
	}
	if len(ids) == 0 && len(users) == 0 {
		return nil, nil
	}
	return ids, users
}

func isAllowed(chatID int64, username string, ids map[int64]bool, users map[string]bool) bool {
	if ids == nil && users == nil {
		return true
	}
	if ids != nil && ids[chatID] {
		return true
	}
	if users != nil && username != "" && users[strings.ToLower(username)] {
		return true
	}
	return false
}

func runBot() error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	token := resolveBotToken()
	if token == "" {
		return fmt.Errorf("telegram bot token not set: use --token or set TELEGRAM_BOT_TOKEN in .env / environment (create a bot via @BotFather)")
	}

	allowIDs, allowUsers := parseAllowList()

	tg := newTelegramClient(token)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// validate token
	validateCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	err = tg.getMe(validateCtx)
	cancel()
	if err != nil {
		return fmt.Errorf("invalid telegram token: %w", err)
	}

	slog.Info("telegram bot started", "provider", cfg.LLM.Provider, "model", cfg.LLM.Model, "agent", cfg.Agent)
	fmt.Printf("Telegram bot started (provider=%s model=%s agent=%s). Press Ctrl+C to stop.\n", cfg.LLM.Provider, cfg.LLM.Model, cfg.Agent)
	if allowIDs != nil || allowUsers != nil {
		fmt.Printf("Allowlist active: %s\n", botAllowFrom)
		if botAllowFrom == "" {
			envVal := os.Getenv("TELEGRAM_ALLOWED_CHAT_IDS")
			if envVal == "" {
				envVal = os.Getenv("TELEGRAM_ALLOWED_USERS")
			}
			fmt.Printf("Allowlist from env: %s\n", envVal)
		}
	}

	sessions := newBotSession()
	sem := make(chan struct{}, 10)
	var offset int64

	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nShutting down bot...")
			return nil
		default:
		}

		// long poll with timeout 30s; use a child context so Ctrl+C interrupts quickly
		pollCtx, pollCancel := context.WithTimeout(ctx, 35*time.Second)
		updates, err := tg.getUpdates(pollCtx, offset, 30)
		pollCancel()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			// transient network error: wait and retry
			slog.Warn("telegram getUpdates failed", "error", err)
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(3 * time.Second):
				continue
			}
		}
		for _, upd := range updates {
			offset = upd.UpdateID + 1
			if upd.Message == nil {
				continue
			}
			msg := upd.Message
			username := ""
			if msg.From != nil {
				username = msg.From.Username
			}
			if !isAllowed(msg.Chat.ID, username, allowIDs, allowUsers) {
				slog.Info("telegram message from non-allowed chat ignored", "chat_id", msg.Chat.ID, "username", username)
				continue
			}
			text := strings.TrimSpace(msg.Text)
			if text == "" {
				continue
			}
			// Handle per-message in a goroutine to not block polling. The
			// per-chat lock is taken before the global semaphore so a chat
			// with queued messages cannot starve the other chats.
			go dispatchTelegramMessage(ctx, tg, cfg, sessions, msg, sem)
		}
	}
}

// dispatchTelegramMessage serializes messages of the same chat, then runs
// them under the global concurrency limit.
func dispatchTelegramMessage(ctx context.Context, tg *telegramClient, cfg *config.Config, sessions *botSession, msg *tgMessage, sem chan struct{}) {
	mu := sessions.lockForChat(msg.Chat.ID)
	mu.Lock()
	defer mu.Unlock()

	sem <- struct{}{}
	defer func() { <-sem }()
	handleTelegramMessage(ctx, tg, cfg, sessions, msg)
}

func handleTelegramMessage(ctx context.Context, tg *telegramClient, cfg *config.Config, sessions *botSession, msg *tgMessage) {
	chatID := msg.Chat.ID
	text := strings.TrimSpace(msg.Text)

	// built-in bot commands
	switch text {
	case "/start", "/help":
		help := "Hi! Send me any message and I'll forward it to the agent.\n\nCommands:\n/reset - clear conversation history\n/help - show this help"
		if err := tg.sendMessage(ctx, chatID, help); err != nil {
			slog.Error("telegram sendMessage failed", "err", err, "chat_id", chatID)
		}
		return
	case "/reset", "/clear":
		sessions.reset(chatID)
		if err := tg.sendMessage(ctx, chatID, "Conversation reset."); err != nil {
			slog.Error("telegram sendMessage failed", "err", err, "chat_id", chatID)
		}
		return
	}
	if strings.HasPrefix(text, "/reset ") || strings.HasPrefix(text, "/clear ") {
		sessions.reset(chatID)
		if err := tg.sendMessage(ctx, chatID, "Conversation reset."); err != nil {
			slog.Error("telegram sendMessage failed", "err", err, "chat_id", chatID)
		}
		return
	}

	// Do not log message text at info level: it is private user content.
	slog.Debug("telegram message received", "chat_id", chatID, "chars", len(text))

	// indicate typing
	typingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	tg.sendChatAction(typingCtx, chatID, "typing")
	cancel()

	// keep typing indicator refreshed every 4s while LLM runs
	typingDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(4 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-typingDone:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				taCtx, taCancel := context.WithTimeout(ctx, 5*time.Second)
				tg.sendChatAction(taCtx, chatID, "typing")
				taCancel()
			}
		}
	}()

	// build conversation
	history := sessions.get(chatID)
	userMsg := llm.Message{Role: "user", Content: text}
	allMessages := append(history, userMsg)

	// call LLM (via service if active, else local)
	resultMessages, err := botCallLLM(ctx, cfg, allMessages)
	close(typingDone)
	if err != nil {
		slog.Error("telegram bot LLM call failed", "error", err, "chat_id", chatID)
		if sendErr := tg.sendMessage(ctx, chatID, fmt.Sprintf("Error: %v", err)); sendErr != nil {
			slog.Error("telegram sendMessage failed", "err", sendErr, "chat_id", chatID)
		}
		return
	}
	if len(resultMessages) == 0 {
		if sendErr := tg.sendMessage(ctx, chatID, "No response from LLM."); sendErr != nil {
			slog.Error("telegram sendMessage failed", "err", sendErr, "chat_id", chatID)
		}
		return
	}

	// resultMessages are the delta (from originalCount onward) when using the caller;
	// agent.CallLLM returns only the new messages. Persist them.
	sessions.append(chatID, userMsg)
	sessions.append(chatID, resultMessages...)

	reply := lastAssistantOrLastContent(resultMessages)
	if reply == "" {
		reply = "(empty response)"
	}

	if err := tg.sendMessage(ctx, chatID, reply); err != nil {
		slog.Error("telegram sendMessage failed", "error", err, "chat_id", chatID)
	}
}

func botCallLLM(ctx context.Context, cfg *config.Config, messages []llm.Message) ([]llm.Message, error) {
	// Prefer the gRPC service when active (honors the session's agent/tools/backend).
	if service.IsActive() {
		req := chatRequestFromConfig(cfg, messages)
		req.Agent = "bot"
		// use a 2-minute timeout like the shell helper
		cctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		result, err := service.Chat(cctx, req)
		if err == nil {
			return result, nil
		}
		if !errors.Is(err, service.ErrUnavailable) {
			return nil, err
		}
		slog.Debug("service unavailable, falling back to local bot execution", "err", err)
	}

	agent := llm.NewAgentForSession("bot", cfg.LLM.Model, cfg.LLM.Provider, cfg.Tools, cfg.LitertLM.Backend, cfg.AgentFiles, cfg.Skills, llm.ThinkEffort(cfg.LLM.ThinkEffort))
	executor := &botExecutor{cfg: cfg}
	return agent.CallLLM(ctx, executor, messages)
}
