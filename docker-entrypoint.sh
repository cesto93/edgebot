#!/usr/bin/env bash
# Edgebot docker entrypoint: assicura backend litertlm pronto all'uso.
# - installa le shared lib LiteRT-LM in ~/.edgebot/lib se mancanti,
#   vuote, corrotte o per l'arch sbagliata (via scripts/litertlm-libs.sh,
#   la stessa logica del target `make install-litertlm`): la platform
#   (linux_x86_64/linux_arm64) segue `uname -m`, ogni file scaricato e'
#   verificato (non vuoto, dimensione minima, magic ELF, e_machine) prima
#   di sostituire quello esistente
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

# Shared LiteRT-LM download/verification helpers (platform selection, URLs,
# install orchestration). In the image the library lives at
# /usr/local/share/edgebot/litertlm-libs.sh (see Dockerfile); in a repo
# checkout fall back to scripts/ next to this file (also covers sourcing
# for tests).
LITERTLM_LIBS_SH="${LITERTLM_LIBS_SH:-/usr/local/share/edgebot/litertlm-libs.sh}"
if [ ! -f "$LITERTLM_LIBS_SH" ]; then
  LITERTLM_LIBS_SH="$(dirname "${BASH_SOURCE[0]}")/scripts/litertlm-libs.sh"
fi
# shellcheck disable=SC1090
source "$LITERTLM_LIBS_SH"

main() {
  mkdir -p "$LIB_DIR" "$MODEL_DIR"

  # 1. Shared libraries LiteRT-LM (serve liblitertlm_c_cpu.so + aux lib).
  litertlm_ensure_libs "$LIB_DIR" || exit 1

  # 1b. Fail fast when a required lib is still unusable (diagnostics inside).
  litertlm_check_required "$LIB_DIR" || exit 1

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
