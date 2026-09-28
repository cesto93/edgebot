#!/usr/bin/env bash
# Edgebot docker entrypoint: assicura backend litertlm pronto all'uso.
# - installa le shared lib LiteRT-LM in ~/.edgebot/lib se mancanti,
#   vuote, corrotte o per l'arch sbagliata (stessa logica del target
#   `make install-litertlm`): la platform (linux_x86_64/linux_arm64) segue
#   `uname -m`, ogni file scaricato e' verificato (non vuoto, dimensione
#   minima, magic ELF, e_machine) prima di sostituire quello esistente
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

LITERTLM_TAG="${LITERTLM_TAG:-main}"
LITERTLM_VERSION="${LITERTLM_VERSION:-v0.16.0}"
# LiteRT-LM ships separate linux builds per CPU arch. Pick the one matching
# the host: downloading the wrong arch yields a misleading dlopen
# "No such file or directory" at runtime even though the file exists and
# is non-empty (glibc reports EM mismatch as ENOENT). Overridable for
# testing or exotic setups (e.g. emulated containers).
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) LITERTLM_PLAT_DEFAULT="linux_x86_64" ;;
  aarch64|arm64) LITERTLM_PLAT_DEFAULT="linux_arm64" ;;
  *)
    echo "error: unsupported architecture $ARCH for litertlm (expected x86_64 or aarch64)" >&2
    exit 1
    ;;
esac
LITERTLM_PLAT="${LITERTLM_PLAT:-$LITERTLM_PLAT_DEFAULT}"
# Expected ELF e_machine for the selected platform (bytes at offset 18):
# x86-64 = 62 (0x3E), AArch64 = 183 (0xB7). lib_ok rejects .so files built
# for the other arch so stale volume contents are re-downloaded.
case "$LITERTLM_PLAT" in
  linux_x86_64) EXPECT_MACHINE="3e00" ;;
  linux_arm64) EXPECT_MACHINE="b700" ;;
  *)
    echo "error: unsupported LITERTLM_PLAT $LITERTLM_PLAT (expected linux_x86_64 or linux_arm64)" >&2
    exit 1
    ;;
esac
LITERTLM_PREBUILT="${LITERTLM_PREBUILT:-https://github.com/google-ai-edge/LiteRT-LM/raw/${LITERTLM_TAG}/prebuilt/${LITERTLM_PLAT}}"
LITERTLM_RELEASE_URL="${LITERTLM_RELEASE_URL:-https://github.com/google-ai-edge/LiteRT-LM/releases/download/${LITERTLM_VERSION}/litert_lm_c_api-0.1.0.zip}"
# shellcheck disable=SC2209
AUX_LIBS="libGemmaModelConstraintProvider.so libLiteRt.so libLiteRtWebGpuAccelerator.so libLiteRtTopKWebGpuSampler.so"
MAIN_LIB="liblitertlm_c_cpu.so"
# Libs without which litertlm-go refuses to start: the backend-selected
# main C-API lib plus the Gemma constraint provider (dlopen'd
# unconditionally; the remaining AUX_LIBS are optional GPU plugins whose
# absence only disables GPU backends).
REQUIRED_LIBS="$MAIN_LIB libGemmaModelConstraintProvider.so"
# Sanity floor for a valid .so (real LiteRT-LM libs are tens of MB).
# Overridable for tests that stub downloads with small ELF files.
MIN_LIB_BYTES="${MIN_LIB_BYTES:-1048576}"

# lib_ok <path>: true when the file is a plausible ELF shared library
# for the selected platform (exists, non-empty, >= MIN_LIB_BYTES, ELF
# magic, e_machine matching LITERTLM_PLAT). Existing empty, truncated or
# wrong-arch files from interrupted downloads or an arch-changed host must
# be re-downloaded, so callers use this instead of a plain `[ -f ]` test.
lib_ok() {
  [ -s "$1" ] || return 1
  [ "$(wc -c <"$1")" -ge "$MIN_LIB_BYTES" ] || return 1
  [ "$(head -c 4 "$1" | od -An -tx1 | tr -d ' \n')" = "7f454c46" ] || return 1
  [ "$(dd if="$1" bs=1 skip=18 count=2 2>/dev/null | od -An -tx1 | tr -d ' \n')" = "$EXPECT_MACHINE" ] || return 1
}

# download <url> <dest>: fetch url to dest atomically, verifying the
# payload before replacing any existing file. Returns non-zero on
# download or verification failure (dest is left untouched).
download() {
  echo "Downloading $(basename "$2") ..."
  tmpfile="$(mktemp "$LIB_DIR/.dl.XXXXXX")"
  if ! curl -fsSL "$1" -o "$tmpfile"; then
    rm -f "$tmpfile"
    echo "error: failed to download $1" >&2
    return 1
  fi
  if ! lib_ok "$tmpfile"; then
    rm -f "$tmpfile"
    echo "error: downloaded $(basename "$2") failed verification (not a valid shared library)" >&2
    return 1
  fi
  mv "$tmpfile" "$2"
}

install_main_lib() {
  echo "Downloading LiteRT-LM runtime ($LITERTLM_PLAT) from $LITERTLM_RELEASE_URL ..."
  tmp="$(mktemp -d)"
  if ! curl -fsSL "$LITERTLM_RELEASE_URL" -o "$tmp/litert_lm_c_api.zip"; then
    echo "error: failed to download $LITERTLM_RELEASE_URL" >&2
    rm -rf "$tmp"
    return 1
  fi
  if ! unzip -jo "$tmp/litert_lm_c_api.zip" "lib/${LITERTLM_PLAT}/liblitert-lm.so" -d "$LIB_DIR"; then
    echo "error: failed to extract lib/${LITERTLM_PLAT}/liblitert-lm.so from the release archive" >&2
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$tmp"
  if ! lib_ok "$LIB_DIR/liblitert-lm.so"; then
    echo "error: extracted liblitert-lm.so failed verification" >&2
    return 1
  fi
  mv "$LIB_DIR/liblitert-lm.so" "$LIB_DIR/$MAIN_LIB"
}

main() {
  mkdir -p "$LIB_DIR" "$MODEL_DIR"

  # 1. Shared libraries LiteRT-LM (serve liblitertlm_c_cpu.so + aux lib).
  if ! lib_ok "$LIB_DIR/$MAIN_LIB"; then
    install_main_lib || exit 1
  fi
  # shellcheck disable=SC2086
  for lib in $AUX_LIBS; do
    if ! lib_ok "$LIB_DIR/$lib"; then
      download "$LITERTLM_PREBUILT/$lib" "$LIB_DIR/$lib" || exit 1
    fi
  done

  # 1b. Fail fast with diagnostics when a required lib is still unusable.
  # Without this the bot would start and fail every LLM call at runtime
  # with a cryptic dlopen error.
  # shellcheck disable=SC2086
  for lib in $REQUIRED_LIBS; do
    if ! lib_ok "$LIB_DIR/$lib"; then
      echo "error: required LiteRT-LM library $LIB_DIR/$lib is missing or invalid" >&2
      echo "hint: remove the stale file and restart, or wipe the volume (docker compose down -v)" >&2
      ls -la "$LIB_DIR" >&2
      exit 1
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
}

# Sourced (not executed) by tests to exercise lib_ok/download.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
