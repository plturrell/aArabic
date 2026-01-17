#!/bin/bash

# Run leanShimmy tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

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
if [[ "$disable_crashpad" -eq 1 ]]; then
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

mojo_args=()
if [[ "$disable_crashpad" -eq 1 ]]; then
    mojo_args+=(--disable-crashpad)
fi
if [[ -n "$mojo_home" ]]; then
    mojo_args+=(--mojo-home "$mojo_home")
fi

echo "Running Zig unit tests..."
zig build test

if command -v mojo &> /dev/null; then
    if [ -d "tests/mojo" ]; then
        echo "Running Mojo tests..."
        run_mojo test tests/mojo
    fi
    if [ -f "scripts/lexer_golden.sh" ]; then
        echo "Running lexer golden tests..."
        ./scripts/lexer_golden.sh "${mojo_args[@]}"
    fi
    if [ -f "scripts/parser_golden.sh" ]; then
        echo "Running parser golden tests..."
        ./scripts/parser_golden.sh "${mojo_args[@]}"
    fi
fi
