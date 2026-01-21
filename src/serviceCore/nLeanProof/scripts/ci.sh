#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_ROOT/../../.." && pwd)"
TESTS_ROOT="$PROJECT_ROOT/tests/lean4"
CI_DIR="$PROJECT_ROOT/tmp/ci"

if [[ ! -d "$TESTS_ROOT" ]]; then
    echo "Lean4 tests not found: $TESTS_ROOT" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

zig build -Doptimize=ReleaseFast
zig build test
./scripts/smoke_openai.sh

mkdir -p "$CI_DIR"

"$PROJECT_ROOT/zig-out/bin/lean4-discover" --root "$TESTS_ROOT" --suite lean --limit 10 --json > "$CI_DIR/discover.json"
"$PROJECT_ROOT/zig-out/bin/lean4-baseline" --root "$TESTS_ROOT" --suite lean --sample 10 --output "$CI_DIR/baseline.json"
"$PROJECT_ROOT/zig-out/bin/lean4-manifest" --root "$TESTS_ROOT" --suite lean --output "$CI_DIR/manifest.json"
"$PROJECT_ROOT/zig-out/bin/lean4-summary" --root "$TESTS_ROOT" --suite lean --output "$CI_DIR/summary.json"
