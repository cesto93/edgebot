package service

import (
	"context"
	"errors"
	"fmt"

	"edgebot/llm"
	"edgebot/service/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// ErrUnavailable is returned when the service cannot be reached. Sessions use
// it to fall back to local execution.
var ErrUnavailable = errors.New("edgebot service unavailable")

// ChatRequest carries the messages and session configuration sent to the
// service. When SystemPrompt is non-empty it overrides the agent prompt
// entirely (used by non-agent callers such as the commit command).
type ChatRequest struct {
	Messages        []llm.Message
	SystemPrompt    string
	Agent           string
	Model           string
	Provider        string
	Tools           map[string]bool
	Backend         string
	AgentFiles      bool
	Skills          bool
	Confirm         bool
	AllowedCommands []string
	// ThinkEffort is the unified reasoning-effort level
	// ("": provider default).
	ThinkEffort string
}

func dial() (*grpc.ClientConn, error) {
	path := SocketPath()
	if path == "" {
		return nil, fmt.Errorf("cannot determine service socket path")
	}
	return grpc.NewClient(
		"unix://"+path,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(MaxMsgSize),
			grpc.MaxCallSendMsgSize(MaxMsgSize),
		),
	)
}

func toProtoRequest(req ChatRequest) *proto.ChatRequest {
	return &proto.ChatRequest{
		Messages:        messagesToProto(req.Messages),
		SystemPrompt:    req.SystemPrompt,
		Agent:           req.Agent,
		Model:           req.Model,
		Provider:        req.Provider,
		Tools:           req.Tools,
		Backend:         req.Backend,
		AgentFiles:      req.AgentFiles,
		Skills:          req.Skills,
		Confirm:         req.Confirm,
		AllowedCommands: req.AllowedCommands,
		ThinkEffort:     req.ThinkEffort,
	}
}

// Client is a connection to a running edgebot service.
type Client struct {
	conn *grpc.ClientConn
	svc  proto.AIServiceClient
}

// NewClient dials the service socket. The connection is lazy: no traffic is
// sent until the first RPC.
func NewClient() (*Client, error) {
	conn, err := dial()
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, svc: proto.NewAIServiceClient(conn)}, nil
}

// Close closes the underlying connection.
func (c *Client) Close() error {
	return c.conn.Close()
}

// Ping reports the service's protocol version.
func (c *Client) Ping(ctx context.Context) (*proto.PingResponse, error) {
	return c.svc.Ping(ctx, &proto.PingRequest{})
}

// Chat runs the agent loop on the service and returns the resulting messages.
func (c *Client) Chat(ctx context.Context, req ChatRequest) ([]llm.Message, error) {
	resp, err := c.svc.Chat(ctx, toProtoRequest(req))
	if err != nil {
		return nil, classifyChatError(ctx, err)
	}
	if resp.Error != "" {
		return nil, errors.New(resp.Error)
	}
	return messagesFromProto(resp.Messages), nil
}

// classifyChatError maps a gRPC failure to an error callers can act on.
// Only connectivity failures are reported as ErrUnavailable (which triggers
// the local fallback): a cancelled or timed-out request must surface as
// context.Canceled/context.DeadlineExceeded so callers do not silently start
// a duplicate local run, and application-level failures must stay visible
// instead of being mistaken for a missing service.
func classifyChatError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	switch status.Code(err) {
	case codes.Unavailable:
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	default:
		return err
	}
}

// Stop asks the service to shut down gracefully.
func (c *Client) Stop(ctx context.Context) error {
	_, err := c.svc.Stop(ctx, &proto.StopRequest{})
	return err
}

// Chat is a convenience for one-shot callers: it dials, performs a single
// chat round-trip, and closes the connection.
func Chat(ctx context.Context, req ChatRequest) ([]llm.Message, error) {
	c, err := NewClient()
	if err != nil {
		return nil, ErrUnavailable
	}
	defer func() { _ = c.Close() }()
	return c.Chat(ctx, req)
}
