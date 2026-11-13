package service

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sync"

	"edgebot/config"
	"edgebot/llm"
	"edgebot/service/proto"

	"google.golang.org/grpc"
)

// Server implements the AIService gRPC API, wrapping the edgebot logic
// (prompts, AGENTS.md files, tools, and LLM calls) on the service side.
type Server struct {
	proto.UnimplementedAIServiceServer

	// callLLM is swappable for tests.
	callLLM  func(ctx context.Context, agent *llm.Agent, executor llm.ToolExecutor, messages []llm.Message) ([]llm.Message, error)
	stop     chan struct{}
	stopOnce sync.Once
}

// NewServer creates a Server that delegates LLM calls to the agent.
func NewServer() *Server {
	return &Server{
		callLLM: func(ctx context.Context, agent *llm.Agent, executor llm.ToolExecutor, messages []llm.Message) ([]llm.Message, error) {
			return agent.CallLLM(ctx, executor, messages)
		},
		stop: make(chan struct{}),
	}
}

// Ping reports the service's protocol version.
func (s *Server) Ping(_ context.Context, _ *proto.PingRequest) (*proto.PingResponse, error) {
	return &proto.PingResponse{Version: ServiceVersion}, nil
}

// Chat builds the agent from the session config, wraps the prompt and
// AGENTS.md files, runs the LLM loop with the service-side executor, and
// returns the resulting messages. LLM errors are reported in the response
// (not as a gRPC error) so clients can distinguish them from connectivity
// failures.
func (s *Server) Chat(ctx context.Context, req *proto.ChatRequest) (*proto.ChatResponse, error) {
	slog.Info("service request received", "agent", req.Agent, "model", req.Model, "provider", req.Provider, "messages", len(req.Messages))

	messages := messagesFromProto(req.Messages)

	var agent *llm.Agent
	if think, err := llm.ParseThinkEffort(req.ThinkEffort); err != nil {
		return &proto.ChatResponse{Error: err.Error()}, nil
	} else if req.SystemPrompt != "" {
		agent = &llm.Agent{
			Prompt:      req.SystemPrompt,
			Model:       req.Model,
			Provider:    req.Provider,
			Backend:     req.Backend,
			ThinkEffort: think,
		}
	} else {
		agent = llm.NewAgentForSession(req.Agent, req.Model, req.Provider, req.Tools, req.Backend, req.AgentFiles, req.Skills, think)
	}

	executor := &ServiceExecutor{
		confirm: req.Confirm,
		allowed: req.AllowedCommands,
	}

	result, err := s.callLLM(ctx, agent, executor, messages)
	if err != nil {
		return &proto.ChatResponse{Error: err.Error()}, nil
	}
	return &proto.ChatResponse{Messages: messagesToProto(result)}, nil
}

// Stop triggers a graceful shutdown of the service.
func (s *Server) Stop(_ context.Context, _ *proto.StopRequest) (*proto.StopResponse, error) {
	s.stopOnce.Do(func() { close(s.stop) })
	return &proto.StopResponse{}, nil
}

// StopCh is closed when Stop is called.
func (s *Server) StopCh() <-chan struct{} {
	return s.stop
}

// Serve listens on the unix socket and serves the AIService until ctx is
// cancelled, Stop is called, or the listener fails. A stale socket file left
// by a crashed run is removed when no live service is listening on it.
func Serve(ctx context.Context, s *Server) error {
	path := SocketPath()
	if path == "" {
		return fmt.Errorf("cannot determine service socket path")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	lis, err := net.Listen("unix", path)
	if err != nil {
		if IsActive() {
			return fmt.Errorf("service already running at %s", path)
		}
		if rmErr := os.Remove(path); rmErr != nil && !os.IsNotExist(rmErr) {
			slog.Warn("failed to remove stale socket", "path", path, "err", rmErr)
		} else if rmErr == nil {
			lis, err = net.Listen("unix", path)
		}
		if err != nil {
			return fmt.Errorf("failed to listen on %s: %w", path, err)
		}
	}
	if err := os.Chmod(path, 0o600); err != nil {
		slog.Warn("chmod socket failed", "path", path, "err", err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(MaxMsgSize),
		grpc.MaxSendMsgSize(MaxMsgSize),
	)
	proto.RegisterAIServiceServer(grpcServer, s)

	slog.Info("edgebot service listening", "socket", path)

	errCh := make(chan error, 1)
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		grpcServer.Stop()
		if rmErr := os.Remove(path); rmErr != nil && !os.IsNotExist(rmErr) {
			slog.Warn("failed to remove socket", "path", path, "err", rmErr)
		}
		return err
	case <-ctx.Done():
	case <-s.StopCh():
	}

	grpcServer.GracefulStop()
	if rmErr := os.Remove(path); rmErr != nil && !os.IsNotExist(rmErr) {
		slog.Warn("failed to remove socket", "path", path, "err", rmErr)
	}
	return nil
}

// ServiceExecutor executes tools on the service side. It honors the session's
// confirm policy: when confirm is true, RunCommand is restricted to the
// allowed commands and WriteFile is denied (there is no interactive
// confirmation path); when confirm is false, everything runs automatically.
type ServiceExecutor struct {
	confirm bool
	allowed []string
}

func (e *ServiceExecutor) ExecuteTool(call llm.ToolCall) (string, error) {
	policy := llm.NewConfirmPolicy(e.confirm, e.allowed)
	return policy.ExecuteTool(call)
}

func (e *ServiceExecutor) IsAllowedCommand(cmd string) bool {
	return config.IsAllowedCommand(cmd, e.allowed)
}

func (e *ServiceExecutor) AskConfirmation(cmd string) bool {
	if !e.confirm {
		return true
	}
	return e.IsAllowedCommand(cmd)
}
