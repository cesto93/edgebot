SHELL := /bin/bash

# Native in-process runtimes are installed with
# `edgebot runtime install llamacpp|litertlm` (see `edgebot runtime show`).

build:
	go build -o edgebot .
install:
	go install .
proto:
	protoc --go_out=. --go_opt=paths=source_relative \
	  --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	  service/proto/service.proto
	@echo "Regenerated service/proto/*.pb.go (commit the generated files; protoc is only needed for regeneration)"
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
