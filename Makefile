SHELL := /bin/bash
EDGEBOT_DIR := $(HOME)/.edgebot
EDGEBOT_LIB := $(EDGEBOT_DIR)/lib
LITERTLM_TAG ?= main
LITERTLM_VERSION ?= v0.16.0
# LITERTLM_PLAT (linux_x86_64 | linux_arm64) defaults to the host arch inside
# scripts/litertlm-libs.sh; override e.g. `make install-litertlm LITERTLM_PLAT=linux_arm64`.
# (Wrong-arch .so files fail dlopen with a misleading "No such file or directory".)
LITERTLM_PLAT ?=

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
	go install github.com/hybridgroup/yzma@v1.27.0
	mkdir -p $(EDGEBOT_LIB)
	yzma install --lib $(EDGEBOT_LIB) --upgrade
	@echo ""
	@echo "✓ yzma + llama.cpp libraries installed to $(EDGEBOT_LIB)"
	@echo "  Set YZMA_LIB=$(EDGEBOT_LIB) in your shell rc file if needed."
install-litertlm:
	@LITERTLM_TAG="$(LITERTLM_TAG)" LITERTLM_VERSION="$(LITERTLM_VERSION)" LITERTLM_PLAT="$(LITERTLM_PLAT)" \
		bash -c 'set -euo pipefail; source scripts/litertlm-libs.sh; litertlm_ensure_libs "$(EDGEBOT_LIB)"'
	@echo ""
	@echo "✓ LiteRT-LM libraries installed to $(EDGEBOT_LIB)"
	@echo "  Model files (.litertlm) go in $(EDGEBOT_DIR)/models/litertlm/"
coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
integration:
	go test -tags=integration -v ./cmd/
docker-build:
	docker build -t edgebot:latest .
docker-up:
	docker compose up -d
docker-down:
	docker compose down
