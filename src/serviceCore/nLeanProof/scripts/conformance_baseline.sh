#!/bin/bash

# Generate a baseline report for Lean4 upstream tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

zig build conformance-baseline -- --root tests/lean4 --suite lean --sample 10
