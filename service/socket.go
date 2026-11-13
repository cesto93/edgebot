package service

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"edgebot/config"
)

// ServiceVersion is the protocol version reported by Ping.
const ServiceVersion = "1"

// MaxMsgSize bounds gRPC message sizes in bytes (base64 images can be large).
const MaxMsgSize = 64 << 20

// socketPathFunc is swappable for tests.
var socketPathFunc = defaultSocketPath

func defaultSocketPath() (string, error) {
	dir, err := config.EdgebotDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "service.sock"), nil
}

// SocketPath returns the unix socket path used by the service.
func SocketPath() string {
	path, err := socketPathFunc()
	if err != nil {
		return ""
	}
	return path
}

// IsActive reports whether a live service is listening on the unix socket.
func IsActive() bool {
	path := SocketPath()
	if path == "" {
		return false
	}
	if _, err := os.Stat(path); err != nil {
		return false
	}
	c, err := NewClient()
	if err != nil {
		return false
	}
	defer func() { _ = c.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err = c.Ping(ctx)
	return err == nil
}
