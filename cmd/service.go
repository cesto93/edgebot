package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"edgebot/config"
	"edgebot/service"

	"github.com/spf13/cobra"
)

var serviceStop bool
var serviceStatus bool

var serviceCmd = &cobra.Command{
	Use:   "service",
	Short: "Run an edgebot gRPC service over a unix socket",
	Long: `Runs a lightweight gRPC service listening on a unix socket
(~/.edgebot/service.sock). Other edgebot sessions route their requests to
the service when it is active, wrapping the prompts, AGENTS.md files, tools,
and LLM calls server-side.

Example:
  edgebot service           # start the service (foreground)
  edgebot service --stop    # stop a running service
  edgebot service --status  # show whether a service is running`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runService()
	},
}

func init() {
	serviceCmd.Flags().BoolVar(&serviceStop, "stop", false, "stop a running service")
	serviceCmd.Flags().BoolVar(&serviceStatus, "status", false, "print whether a service is running")
	rootCmd.AddCommand(serviceCmd)
}

func runService() error {
	cfg, err := config.LoadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	initLogger(cfg)

	if serviceStatus {
		if service.IsActive() {
			fmt.Printf("Service is running at %s\n", service.SocketPath())
		} else {
			fmt.Printf("Service is not running%s\n", socketSuffix())
		}
		return nil
	}

	if serviceStop {
		return stopService()
	}

	return startService(cfg)
}

func socketSuffix() string {
	path := service.SocketPath()
	if path == "" {
		return ""
	}
	return fmt.Sprintf(" (socket: %s)", path)
}

func stopService() error {
	if !service.IsActive() {
		return fmt.Errorf("no running service to stop")
	}
	c, err := service.NewClient()
	if err != nil {
		return fmt.Errorf("failed to connect to service: %w", err)
	}
	defer func() { _ = c.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop service: %w", err)
	}
	fmt.Println("Service stopped.")
	return nil
}

func startService(cfg *config.Config) error {
	path := service.SocketPath()
	if path == "" {
		return fmt.Errorf("cannot determine service socket path")
	}
	fmt.Printf("Starting edgebot service on %s\n", path)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := service.NewServer()
	srv.ReadDenylist = cfg.Shell.ReadDenylist
	return service.Serve(ctx, srv)
}
