#!/usr/bin/env bash
# Edgebot docker entrypoint: assicura backend litertlm pronto all'uso.
# - installa le shared lib LiteRT-LM in ~/.edgebot/lib se mancanti
#   (stessa logica del target `make install-litertlm`)
# - scarica il modello .litertlm da HuggingFace se mancante
#   (via `edgebot models pull`, che aggiorna anche la config)
# - forza la config su provider=litertlm, modello e backend desiderati
# - esegue `edgebot "$@"` (di default `bot` dal compose)
set -euo pipefail

EDGEBOT_DIR="${HOME:-/root}/.edgebot"
LIB_DIR="${LITERTLM_LIB:-$EDGEBOT_DIR/lib}"
MODEL_DIR="$EDGEBOT_DIR/models/litertlm"

MODEL_REPO="${LITERTLM_MODEL_REPO:-litert-community/gemma-4-E2B-it-litert-lm}"
MODEL_FILE="${LITERTLM_MODEL_FILE:-gemma-4-E2B-it.litertlm}"
MODEL_NAME="${MODEL_FILE%.litertlm}"
BACKEND="${LITERTLM_BACKEND:-cpu}"

LITERTLM_TAG="${LITERTLM_TAG:-main}"
LITERTLM_VERSION="${LITERTLM_VERSION:-v0.16.0}"
LITERTLM_PREBUILT="${LITERTLM_PREBUILT:-https://github.com/google-ai-edge/LiteRT-LM/raw/${LITERTLM_TAG}/prebuilt/linux_x86_64}"
LITERTLM_RELEASE_URL="${LITERTLM_RELEASE_URL:-https://github.com/google-ai-edge/LiteRT-LM/releases/download/${LITERTLM_VERSION}/litert_lm_c_api-0.1.0.zip}"
# shellcheck disable=SC2209
AUX_LIBS="libGemmaModelConstraintProvider.so libLiteRt.so libLiteRtWebGpuAccelerator.so libLiteRtTopKWebGpuSampler.so"

mkdir -p "$LIB_DIR" "$MODEL_DIR"

# 1. Shared libraries LiteRT-LM (serve liblitertlm_c_cpu.so + aux lib).
if [ ! -f "$LIB_DIR/liblitertlm_c_cpu.so" ]; then
  echo "Downloading LiteRT-LM runtime from $LITERTLM_RELEASE_URL ..."
  tmp="$(mktemp -d)"
  curl -fsSL "$LITERTLM_RELEASE_URL" -o "$tmp/litert_lm_c_api.zip"
  unzip -jo "$tmp/litert_lm_c_api.zip" "lib/linux_x86_64/liblitert-lm.so" -d "$LIB_DIR"
  mv "$LIB_DIR/liblitert-lm.so" "$LIB_DIR/liblitertlm_c_cpu.so"
  rm -rf "$tmp"
fi
# shellcheck disable=SC2086
for lib in $AUX_LIBS; do
  if [ ! -f "$LIB_DIR/$lib" ]; then
    echo "Downloading $lib ..."
    curl -fsSL "$LITERTLM_PREBUILT/$lib" -o "$LIB_DIR/$lib"
  fi
done

# 2. Modello .litertlm: download solo se assente (un fallimento non deve
# impedire di forzare la config di default al passo 3).
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Downloading model $MODEL_REPO/$MODEL_FILE ..."
  edgebot models pull "$MODEL_REPO" "$MODEL_FILE" || echo "warning: model download failed, continuing with default config" >&2
else
  echo "Model already present: $MODEL_DIR/$MODEL_FILE"
fi

# 3. Config: provider litertlm + modello + backend (idempotente).
edgebot config --provider litertlm --model "$MODEL_NAME" --backend "$BACKEND"

# 4. Avvia il comando richiesto (default: bot).
exec edgebot "$@"
