#!/bin/bash

# Start leanShimmy server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

BIN_PATH="zig-out/bin/leanshimmy"
if [ ! -f "$BIN_PATH" ]; then
    echo "Server binary not found: $BIN_PATH"
    echo "Build it first: ./scripts/build_all.sh"
    exit 1
fi

echo "Starting leanShimmy server..."
echo "Host: ${LEANSHIMMY_HOST:-0.0.0.0}"
echo "Port: ${LEANSHIMMY_PORT:-8002}"
echo ""

exec "$BIN_PATH"
