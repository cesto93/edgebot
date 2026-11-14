package service

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestClassifyChatError(t *testing.T) {
	tests := []struct {
		name      string
		ctx       context.Context
		err       error
		wantCtx   error // non-nil: errors.Is must match this
		wantUnav  bool
		wantInMsg string
	}{
		{
			name:    "cancelled request surfaces ctx error",
			ctx:     cancelledContext(),
			err:     status.Error(codes.Canceled, "canceled"),
			wantCtx: context.Canceled,
		},
		{
			name:    "deadline surfaces ctx error",
			ctx:     deadlineContext(),
			err:     status.Error(codes.DeadlineExceeded, "deadline"),
			wantCtx: context.DeadlineExceeded,
		},
		{
			name:      "unavailable maps to ErrUnavailable and keeps detail",
			ctx:       context.Background(),
			err:       status.Error(codes.Unavailable, "dial tcp: connection refused"),
			wantUnav:  true,
			wantInMsg: "connection refused",
		},
		{
			name:      "application error is not unavailable",
			ctx:       context.Background(),
			err:       status.Error(codes.Internal, "boom"),
			wantInMsg: "boom",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyChatError(tt.ctx, tt.err)
			if got == nil {
				t.Fatal("classifyChatError returned nil")
			}
			if tt.wantCtx != nil && !errors.Is(got, tt.wantCtx) {
				t.Fatalf("error = %v, want %v", got, tt.wantCtx)
			}
			if tt.wantUnav != errors.Is(got, ErrUnavailable) {
				t.Fatalf("error = %v, ErrUnavailable = %v, want %v", got, errors.Is(got, ErrUnavailable), tt.wantUnav)
			}
			if tt.wantInMsg != "" && !strings.Contains(got.Error(), tt.wantInMsg) {
				t.Fatalf("error = %q, want it to contain %q", got, tt.wantInMsg)
			}
		})
	}
}

func cancelledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

func deadlineContext() context.Context {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	return ctx
}
