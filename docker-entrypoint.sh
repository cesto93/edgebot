#!/usr/bin/env bash
# Edgebot docker entrypoint: assicura backend litertlm pronto all'uso.
# - installa la runtime LiteRT-LM in ~/.edgebot/lib se mancante tramite
#   `edgebot litertlm install` (stesso downloader libfetch del binding
#   litertlm-go: platform linux_x86_64/linux_arm64 dedotta dal binario,
#   scrittura atomica, nessun curl/unzip richiesto)
# - scarica il modello .litertlm da HuggingFace se mancante
#   (via `edgebot models pull`, che aggiorna anche la config)
#   (un fallimento non deve impedire di forzare la config al passo 3)
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

main() {
  mkdir -p "$LIB_DIR" "$MODEL_DIR"

  # 1. Runtime LiteRT-LM (C API 1.0.0, unified liblitert-lm.so).
  edgebot litertlm install --dir "$LIB_DIR" || exit 1

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
}

# Sourced (not executed) by tests to exercise the helpers via the shared lib.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
