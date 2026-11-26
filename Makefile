SHELL := /bin/bash
EDGEBOT_DIR := $(HOME)/.edgebot
EDGEBOT_LIB := $(EDGEBOT_DIR)/lib
# Yzma CLI version is taken from go.mod (keeps CLI and Go binding in sync).
YZMA_VERSION ?= $(shell go list -m -f '{{.Version}}' github.com/hybridgroup/yzma 2>/dev/null)
# LiteRT-LM C API 1.0.0 (unified liblitert-lm shared library). Keep in sync
# with the litertlm-go binding, which requires upstream v0.19.1+.
LITERTLM_VERSION ?= v0.19.1
# LITERTLM_PLAT (linux_x86_64 | linux_arm64) defaults to the host arch inside
# `edgebot litertlm install`; override e.g. `make install-litertlm LITERTLM_PLAT=linux_arm64`.
LITERTLM_PLAT ?=
LITERTLM_BACKEND ?= cpu

build:
	go build -o edgebot .
install:
	go install .
proto:
	protoc --go_out=. --go_opt=paths=source_relative \
	  --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	  service/proto/service.proto
	@echo "Regenerated service/proto/*.pb.go (commit the generated files; protoc is only needed for regeneration)"
install-yzma:
	@test -n "$(YZMA_VERSION)" || { echo "error: cannot resolve yzma version from go.mod" >&2; exit 1; }
	go install github.com/hybridgroup/yzma@$(YZMA_VERSION)
	mkdir -p $(EDGEBOT_LIB)
	yzma install --lib $(EDGEBOT_LIB) --upgrade
	@echo ""
	@echo "✓ yzma + llama.cpp libraries installed to $(EDGEBOT_LIB)"
	@echo "  Set YZMA_LIB=$(EDGEBOT_LIB) in your shell rc file if needed."
install-litertlm:
	@LITERTLM_VERSION="$(LITERTLM_VERSION)" LITERTLM_PLAT="$(LITERTLM_PLAT)" LITERTLM_BACKEND="$(LITERTLM_BACKEND)" \
		go run . litertlm install --dir "$(EDGEBOT_LIB)"
	@echo ""
	@echo "✓ LiteRT-LM runtime installed to $(EDGEBOT_LIB)"
	@echo "  Model files (.litertlm) go in $(EDGEBOT_DIR)/models/litertlm/"
coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
lint:
	go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run
dup:
	jscpd .
check:
	gofmt -l .
	go vet ./...
	go test -race ./...
integration:
	go test -tags=integration -v ./cmd/
docker-build:
	docker build -t edgebot:latest .
docker-up:
	docker compose up -d
docker-down:
	docker compose down
