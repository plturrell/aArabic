#!/bin/bash

# Run upstream Lean4 as an oracle for a small subset of tests.
# Output is stored under tmp/conformance/oracle/.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_ROOT/../../.." && pwd)"
TESTS_ROOT="$REPO_ROOT/vendor/layerIntelligence/lean4/tests"

SUITE="lean"
LIMIT=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Usage: conformance_oracle.sh [--suite NAME] [--limit N]"
            exit 1
            ;;
    esac
done

LEAN_BIN="${LEAN_BIN:-lean}"
if [[ -x "$LEAN_BIN" ]]; then
    :
elif command -v "$LEAN_BIN" >/dev/null 2>&1; then
    LEAN_BIN="$(command -v "$LEAN_BIN")"
else
    echo "Lean binary not found. Set LEAN_BIN or ensure lean is in PATH."
    exit 1
fi

OUT_DIR="$PROJECT_ROOT/tmp/conformance/oracle"
mkdir -p "$OUT_DIR"

DISCOVER_CMD=("zig" "build" "conformance-discover" "--" "--root" "$TESTS_ROOT" "--suite" "$SUITE" "--limit" "$LIMIT" "--absolute")
TESTS=()
while IFS= read -r line; do
    TESTS+=("$line")
done < <("${DISCOVER_CMD[@]}")

for file in "${TESTS[@]}"; do
    rel="${file#$TESTS_ROOT/}"
    out_path="$OUT_DIR/${rel}.out"
    out_dir="$(dirname "$out_path")"
    mkdir -p "$out_dir"

    set +e
    LEAN_BACKTRACE=0 "$LEAN_BIN" --root="$TESTS_ROOT" -DprintMessageEndPos=true -Dlinter.all=false -DElab.inServer=true "$file" 2>&1 \
        | perl -pe 's/(\?(\w|_\w+))\.[0-9]+/\1/g' \
        | perl -pe 's/https:\/\/lean-lang\.org\/doc\/reference\/(v?[0-9.]+(-rc[0-9]+)?|latest)/REFERENCE/g' \
        > "$out_path"

    ret=${PIPESTATUS[0]}
    echo "$ret" > "${out_path}.ret"
    set -e
done

echo "Oracle outputs written to $OUT_DIR"
