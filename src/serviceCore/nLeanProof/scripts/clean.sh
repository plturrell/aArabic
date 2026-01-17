#!/bin/bash

# Clean leanShimmy build artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Cleaning build artifacts..."

rm -rf zig-out .zig-cache lib/*.so lib/*.dylib lib/*.a 2>/dev/null || true

echo "Done."
