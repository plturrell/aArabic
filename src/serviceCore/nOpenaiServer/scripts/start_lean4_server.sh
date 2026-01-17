#!/bin/bash
# Start Lean4 HTTP Service (Zig)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "Starting Lean4 HTTP Service"
echo "================================================================================"
echo ""

BIN_PATH="./shimmy_lean4_server"
if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Lean4 server binary not found: $BIN_PATH"
    echo "Build it with: ./scripts/build_zig.sh"
    exit 1
fi

echo "Configuration:"
echo "  Host: 0.0.0.0 (override with SHIMMY_LEAN4_HOST)"
echo "  Port: 8002 (override with SHIMMY_LEAN4_PORT)"
echo "  Work dir: tmp/lean4 (override with SHIMMY_LEAN4_WORK_DIR)"
echo "  Lean bin: lean (override with LEAN4_BIN)"
echo "  Lean root: (override with LEAN4_ROOT)"
echo ""

echo "Starting server..."
echo ""

./shimmy_lean4_server
