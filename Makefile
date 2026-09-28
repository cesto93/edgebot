SHELL := /bin/bash
EDGEBOT_DIR := $(HOME)/.edgebot
EDGEBOT_LIB := $(EDGEBOT_DIR)/lib
LITERTLM_TAG ?= main
LITERTLM_VERSION ?= v0.16.0
# LiteRT-LM ships separate linux builds per CPU arch (see docker-entrypoint.sh:
# wrong-arch .so files fail dlopen with a misleading "No such file or directory").
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
LITERTLM_PLAT ?= linux_arm64
else ifeq ($(UNAME_M),arm64)
LITERTLM_PLAT ?= linux_arm64
else
LITERTLM_PLAT ?= linux_x86_64
endif
LITERTLM_PREBUILT := https://github.com/google-ai-edge/LiteRT-LM/raw/$(LITERTLM_TAG)/prebuilt/$(LITERTLM_PLAT)
LITERTLM_AUX_LIBS := libGemmaModelConstraintProvider.so libLiteRt.so libLiteRtWebGpuAccelerator.so libLiteRtTopKWebGpuSampler.so
LITERTLM_RELEASE_URL ?= https://github.com/google-ai-edge/LiteRT-LM/releases/download/$(LITERTLM_VERSION)/litert_lm_c_api-0.1.0.zip

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
	mkdir -p $(EDGEBOT_LIB)
	@if [ ! -s "$(EDGEBOT_LIB)/liblitertlm_c_cpu.so" ]; then \
		echo "Downloading liblitert-lm.so ($(LITERTLM_PLAT)) from $(LITERTLM_RELEASE_URL) ..."; \
		tmp=$$(mktemp -d); \
		curl -fsSL "$(LITERTLM_RELEASE_URL)" -o "$$tmp/litert_lm_c_api.zip" || { echo "✗ failed to download LiteRT-LM release $(LITERTLM_VERSION)"; exit 1; }; \
		unzip -jo "$$tmp/litert_lm_c_api.zip" "lib/$(LITERTLM_PLAT)/liblitert-lm.so" -d "$(EDGEBOT_LIB)" || { echo "✗ failed to extract lib/$(LITERTLM_PLAT)/liblitert-lm.so"; exit 1; }; \
		mv "$(EDGEBOT_LIB)/liblitert-lm.so" "$(EDGEBOT_LIB)/liblitertlm_c_cpu.so" || { echo "✗ failed to rename liblitert-lm.so"; exit 1; }; \
		rm -rf "$$tmp"; \
	fi
	@for lib in $(LITERTLM_AUX_LIBS); do \
		if [ ! -s "$(EDGEBOT_LIB)/$$lib" ]; then \
			echo "Downloading $$lib ..."; \
			curl -fsSL "$(LITERTLM_PREBUILT)/$$lib" -o "$(EDGEBOT_LIB)/$$lib" || { echo "✗ failed to download $$lib"; exit 1; }; \
		fi; \
	done
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
