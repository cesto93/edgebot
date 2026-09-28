# syntax=docker/dockerfile:1
FROM golang:1.26-bookworm AS builder

WORKDIR /src

# Cache deps separately
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/edgebot .

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates bash git curl unzip libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/edgebot /usr/local/bin/edgebot
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY scripts/litertlm-libs.sh /usr/local/share/edgebot/litertlm-libs.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace

# Persist config/data inside image volumes by default; override with bind mounts in compose
ENV HOME=/root

ENTRYPOINT ["docker-entrypoint.sh"]
