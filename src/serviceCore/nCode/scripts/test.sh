#!/bin/bash

# Run nCode tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running Zig unit tests..."
zig build test

echo ""
echo "All tests passed!"

