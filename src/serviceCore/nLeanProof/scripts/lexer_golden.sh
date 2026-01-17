#!/bin/bash

# Golden lexer tests (Mojo CLI).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GOLDEN_DIR="$PROJECT_ROOT/tests/lexer/golden"
LEXER_CLI="$PROJECT_ROOT/core/lexer/lexer_cli.mojo"

usage() {
    echo "Usage: $0 [--disable-crashpad] [--mojo-home PATH]"
}

disable_crashpad=0
mojo_home=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disable-crashpad)
            disable_crashpad=1
            shift
            ;;
        --mojo-home)
            mojo_home="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$mojo_home" && -n "${MOJO_HOME_OVERRIDE:-}" ]]; then
    mojo_home="$MOJO_HOME_OVERRIDE"
fi
if [[ -z "$mojo_home" && -n "${MOJO_HOME:-}" ]]; then
    mojo_home="$MOJO_HOME"
fi

mojo_env=()
if [[ "$disable_crashpad" -eq 1 || "${MOJO_DISABLE_CRASHPAD:-}" == "1" ]]; then
    mojo_env=(MOJO_DISABLE_CRASHPAD=1 MODULAR_DISABLE_CRASHPAD=1 CRASHPAD_DISABLE=1 CHROME_CRASHPAD_DISABLE=1 LLVM_DISABLE_CRASH_REPORT=1)
fi
if [[ -n "$mojo_home" ]]; then
    mkdir -p "$mojo_home" "$mojo_home/.cache" "$mojo_home/.config" "$mojo_home/.local/state"
    mojo_env+=(HOME="$mojo_home" XDG_CACHE_HOME="$mojo_home/.cache" XDG_CONFIG_HOME="$mojo_home/.config" XDG_STATE_HOME="$mojo_home/.local/state" MOJO_HOME="$mojo_home" MODULAR_HOME="$mojo_home" MAX_HOME="$mojo_home")
fi

run_mojo() {
    if [[ ${#mojo_env[@]} -gt 0 ]]; then
        env "${mojo_env[@]}" mojo "$@"
    else
        mojo "$@"
    fi
}

if ! command -v mojo &> /dev/null; then
    echo "mojo not found; skipping lexer golden tests"
    exit 0
fi

if [[ ! -d "$GOLDEN_DIR" ]]; then
    echo "Golden test directory missing: $GOLDEN_DIR" >&2
    exit 1
fi

if [[ ! -f "$LEXER_CLI" ]]; then
    echo "Lexer CLI missing: $LEXER_CLI" >&2
    exit 1
fi

shopt -s nullglob

fail=0
found=0

for input in "$GOLDEN_DIR"/*.lean; do
    found=1
    base="$(basename "$input" .lean)"
    expected="$GOLDEN_DIR/$base.tokens"
    if [[ ! -f "$expected" ]]; then
        echo "Missing expected tokens: $expected" >&2
        fail=1
        continue
    fi

    text="$(cat "$input")"
    output="$(run_mojo run "$LEXER_CLI" --text "$text")"
    if ! diff -u "$expected" - <<<"$output"; then
        fail=1
    fi
done

if [[ $found -eq 0 ]]; then
    echo "No golden lexer tests found in $GOLDEN_DIR"
    exit 0
fi

if [[ $fail -ne 0 ]]; then
    echo "Golden lexer tests failed" >&2
    exit 1
fi

echo "Golden lexer tests passed"
