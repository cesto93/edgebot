#!/usr/bin/env bash
# Shared LiteRT-LM native-library helpers (source-only library).
#
# Sourced by docker-entrypoint.sh and by the Makefile install-litertlm
# target (which runs `bash -c 'source scripts/litertlm-libs.sh; ...'`).
# Keeps platform selection, download URLs, verification and install
# orchestration in one place so the two callers cannot drift apart.
#
# Configuration via environment (all optional):
#   LITERTLM_TAG        git tag/branch for prebuilt aux libs (default main)
#   LITERTLM_VERSION    release tag for the main C-API lib (default v0.16.0)
#   LITERTLM_PLAT       linux_x86_64 | linux_arm64 (default from `uname -m`)
#   LITERTLM_PREBUILT   base URL for aux libs (default derived from tag+plat)
#   LITERTLM_RELEASE_URL  C-API release zip URL (default derived from version)
#   MIN_LIB_BYTES       sanity floor for a valid .so (default 1048576;
#                       override with a small value when testing with stubs)
#
# Provides: litertlm_lib_ok, litertlm_download, litertlm_install_main_lib,
# litertlm_ensure_libs, litertlm_check_required, litertlm_check_system_deps.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "error: scripts/litertlm-libs.sh is a library, source it instead of executing it" >&2
  exit 1
fi

LITERTLM_TAG="${LITERTLM_TAG:-main}"
LITERTLM_VERSION="${LITERTLM_VERSION:-v0.16.0}"
# LiteRT-LM ships separate linux builds per CPU arch. Pick the one matching
# the host: downloading the wrong arch yields a misleading dlopen
# "No such file or directory" at runtime even though the file exists and
# is non-empty (glibc reports EM mismatch as ENOENT). Overridable via
# LITERTLM_PLAT for testing or exotic setups (e.g. emulated containers).
_ARCH="$(uname -m)"
case "$_ARCH" in
  x86_64|amd64) LITERTLM_PLAT_DEFAULT="linux_x86_64" ;;
  aarch64|arm64) LITERTLM_PLAT_DEFAULT="linux_arm64" ;;
  *)
    echo "error: unsupported architecture $_ARCH for litertlm (expected x86_64 or aarch64)" >&2
    exit 1
    ;;
esac
unset _ARCH
LITERTLM_PLAT="${LITERTLM_PLAT:-$LITERTLM_PLAT_DEFAULT}"
# Expected ELF e_machine for the selected platform (bytes at offset 18):
# x86-64 = 62 (0x3E), AArch64 = 183 (0xB7). litertlm_lib_ok rejects .so
# files built for the other arch so stale contents are re-downloaded.
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

# litertlm_lib_ok <path>: true when the file is a plausible ELF shared
# library for the selected platform (exists, non-empty, >= MIN_LIB_BYTES,
# ELF magic, e_machine matching LITERTLM_PLAT). Existing empty, truncated
# or wrong-arch files from interrupted downloads or an arch-changed host
# must be re-downloaded, so callers use this instead of a plain `-s` test.
litertlm_lib_ok() {
  [ -s "$1" ] || return 1
  [ "$(wc -c <"$1")" -ge "$MIN_LIB_BYTES" ] || return 1
  [ "$(head -c 4 "$1" | od -An -tx1 | tr -d ' \n')" = "7f454c46" ] || return 1
  [ "$(dd if="$1" bs=1 skip=18 count=2 2>/dev/null | od -An -tx1 | tr -d ' \n')" = "$EXPECT_MACHINE" ] || return 1
}

# litertlm_download <url> <dest>: fetch url to dest atomically, verifying
# the payload before replacing any existing file. Returns non-zero on
# download or verification failure (dest is left untouched).
litertlm_download() {
  echo "Downloading $(basename "$2") ..."
  tmpfile="$(mktemp "$(dirname "$2")/.dl.XXXXXX")"
  if ! curl -fsSL "$1" -o "$tmpfile"; then
    rm -f "$tmpfile"
    echo "error: failed to download $1" >&2
    return 1
  fi
  if ! litertlm_lib_ok "$tmpfile"; then
    rm -f "$tmpfile"
    echo "error: downloaded $(basename "$2") failed verification (not a valid shared library)" >&2
    return 1
  fi
  mv "$tmpfile" "$2"
  chmod 644 "$2"
}

# litertlm_install_main_lib <libdir>: fetch the C-API release zip, extract
# the platform build of liblitert-lm.so and rename it to the filename the
# litertlm-go binding dlopens (liblitertlm_c_cpu.so).
litertlm_install_main_lib() {
  echo "Downloading LiteRT-LM runtime ($LITERTLM_PLAT) from $LITERTLM_RELEASE_URL ..."
  tmp="$(mktemp -d)"
  if ! curl -fsSL "$LITERTLM_RELEASE_URL" -o "$tmp/litert_lm_c_api.zip"; then
    echo "error: failed to download $LITERTLM_RELEASE_URL" >&2
    rm -rf "$tmp"
    return 1
  fi
  if ! unzip -jo "$tmp/litert_lm_c_api.zip" "lib/${LITERTLM_PLAT}/liblitert-lm.so" -d "$1"; then
    echo "error: failed to extract lib/${LITERTLM_PLAT}/liblitert-lm.so from the release archive" >&2
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$tmp"
  if ! litertlm_lib_ok "$1/liblitert-lm.so"; then
    echo "error: extracted liblitert-lm.so failed verification" >&2
    return 1
  fi
  mv "$1/liblitert-lm.so" "$1/$MAIN_LIB"
  chmod 644 "$1/$MAIN_LIB"
}

# litertlm_ensure_libs <libdir>: download the main lib (when missing or
# invalid) plus every aux lib. Returns non-zero on the first failure.
litertlm_ensure_libs() {
  mkdir -p "$1"
  if ! litertlm_lib_ok "$1/$MAIN_LIB"; then
    litertlm_install_main_lib "$1" || return 1
  fi
  # shellcheck disable=SC2086
  for lib in $AUX_LIBS; do
    if ! litertlm_lib_ok "$1/$lib"; then
      litertlm_download "$LITERTLM_PREBUILT/$lib" "$1/$lib" || return 1
    fi
  done
}

# litertlm_check_system_deps <libdir>: fail fast when a required lib has
# unresolvable system dependencies. The LiteRT-LM runtime links against
# the Vulkan loader (libvulkan.so.1, Debian package libvulkan1) even for
# the cpu backend, and debian:bookworm-slim does not ship it: without
# this check the container starts and every LLM call fails at dlopen
# with "libvulkan.so.1: cannot open shared object file". Only REQUIRED_LIBS
# are checked (optional GPU plugins may carry extra deps but their dlopen
# failure is non-fatal). Skipped silently when `ldd` is unavailable.
litertlm_check_system_deps() {
  command -v ldd >/dev/null 2>&1 || return 0
  # shellcheck disable=SC2086
  for lib in $REQUIRED_LIBS; do
    missing="$(ldd "$1/$lib" 2>/dev/null | grep 'not found' || true)"
    if [ -n "$missing" ]; then
      echo "error: $1/$lib has missing system dependencies:" >&2
      echo "$missing" >&2
      echo "hint: install the Vulkan loader (Debian/Ubuntu: apt-get install libvulkan1) and restart; docker images need a rebuild to pick up the Dockerfile fix" >&2
      return 1
    fi
  done
}

# litertlm_check_required <libdir>: fail fast with diagnostics when a
# required lib is still unusable. Without this the bot would start and fail
# every LLM call at runtime with a cryptic dlopen error.
litertlm_check_required() {
  # shellcheck disable=SC2086
  for lib in $REQUIRED_LIBS; do
    if ! litertlm_lib_ok "$1/$lib"; then
      echo "error: required LiteRT-LM library $1/$lib is missing or invalid" >&2
      echo "hint: remove the stale file and restart, or wipe the volume (docker compose down -v)" >&2
      ls -la "$1" >&2
      return 1
    fi
  done
}
