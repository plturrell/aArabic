#!/bin/bash

# Run elaboration conformance tests against upstream Lean4 test suite.
# Output is stored under tmp/conformance/elaboration/.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_ROOT/../../.." && pwd)"
TESTS_ROOT="$PROJECT_ROOT/tests/lean4"

SUITE="lean"
LIMIT=50
VERBOSE=""
OUTPUT=""

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
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Usage: conformance_elaboration.sh [--suite NAME] [--limit N] [--verbose] [--output PATH]"
            exit 1
            ;;
    esac
done

# Build first
echo "Building elaboration conformance harness..."
cd "$PROJECT_ROOT"
zig build conformance-elaboration

OUT_DIR="$PROJECT_ROOT/tmp/conformance/elaboration"
mkdir -p "$OUT_DIR"

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$OUT_DIR/report.json"
fi

echo "Running elaboration conformance tests..."
echo "  Test root: $TESTS_ROOT"
echo "  Suite: $SUITE"
echo "  Limit: $LIMIT"
echo ""

./zig-out/bin/lean4-elaboration \
    --root "$TESTS_ROOT" \
    --suite "$SUITE" \
    --limit "$LIMIT" \
    --output "$OUTPUT" \
    $VERBOSE

echo ""
echo "Report written to: $OUTPUT"
