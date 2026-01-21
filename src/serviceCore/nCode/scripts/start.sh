#!/bin/bash

# Start nCode server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

BIN_PATH="zig-out/bin/ncode-server"
if [ ! -f "$BIN_PATH" ]; then
    echo "Server binary not found: $BIN_PATH"
    echo "Building first..."
    zig build
fi

echo "Starting nCode server..."
echo "Host: 0.0.0.0"
echo "Port: ${NCODE_PORT:-18003}"
echo ""

exec "$BIN_PATH"

