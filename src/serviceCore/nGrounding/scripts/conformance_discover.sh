#!/bin/bash

# Discover Lean4 upstream tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

zig build conformance-discover -- --root tests/lean4 --suite lean --limit 10
