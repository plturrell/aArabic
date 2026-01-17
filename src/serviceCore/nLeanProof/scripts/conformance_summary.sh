#!/bin/bash

# Generate Lean4 conformance summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_ROOT/../../.." && pwd)"
TESTS_ROOT="$REPO_ROOT/vendor/layerIntelligence/lean4/tests"

if [[ ! -d "$TESTS_ROOT" ]]; then
    echo "Lean4 tests not found: $TESTS_ROOT" >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/tmp/conformance"
cd "$PROJECT_ROOT"

zig build conformance-summary -- --root "$TESTS_ROOT" --suite lean --output "$PROJECT_ROOT/tmp/conformance/summary.json"
