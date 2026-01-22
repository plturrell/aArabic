#!/bin/bash

# Run lexer over a subset of Lean4 tests and store token output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_ROOT/../../.." && pwd)"
TESTS_ROOT="$PROJECT_ROOT/tests/lean4"
LEXER_CLI="$PROJECT_ROOT/core/lexer/lexer_cli.mojo"
DISCOVER_BIN="$PROJECT_ROOT/zig-out/bin/lean4-discover"

limit=10
suite="lean"
max_bytes=65535
disable_crashpad=0
mojo_home=""

usage() {
    echo "Usage: $0 [--limit N] [--suite NAME] [--max-bytes N] [--disable-crashpad] [--mojo-home PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            limit="$2"
            shift 2
            ;;
        --suite)
            suite="$2"
            shift 2
            ;;
        --max-bytes)
            max_bytes="$2"
            shift 2
            ;;
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

if ! command -v mojo &> /dev/null; then
    echo "mojo not found; skipping conformance lexer run"
    exit 0
fi

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

if [[ ! -d "$TESTS_ROOT" ]]; then
    echo "Lean4 tests not found: $TESTS_ROOT" >&2
    exit 1
fi

if [[ ! -f "$LEXER_CLI" ]]; then
    echo "Lexer CLI missing: $LEXER_CLI" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -x "$DISCOVER_BIN" ]]; then
    zig build conformance-discover
fi

scan_root="$TESTS_ROOT/$suite"
if [[ ! -d "$scan_root" ]]; then
    echo "Lean4 suite not found: $scan_root" >&2
    exit 1
fi

out_dir="$PROJECT_ROOT/tmp/conformance/lexer"
mkdir -p "$out_dir"

processed=0
skipped=0

while IFS= read -r rel; do
    if [[ -z "$rel" ]]; then
        continue
    fi
    src="$scan_root/$rel"
    if [[ ! -f "$src" ]]; then
        continue
    fi
    size=$(wc -c < "$src" | tr -d " ")
    if [[ "$size" -gt "$max_bytes" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    output="$(run_mojo run "$LEXER_CLI" --file "$src")"
    if [[ "$output" == Failed\ to\ read\ file* ]]; then
        text="$(cat "$src")"
        output="$(run_mojo run "$LEXER_CLI" --text "$text")"
    fi
    out_path="$out_dir/$rel.tokens"
    mkdir -p "$(dirname "$out_path")"
    printf "%s\n" "$output" > "$out_path"
    processed=$((processed + 1))
done < <("$DISCOVER_BIN" --root "$TESTS_ROOT" --suite "$suite" --limit "$limit")

echo "Lexer outputs: $processed"
echo "Skipped (size): $skipped"
echo "Output dir: $out_dir"
