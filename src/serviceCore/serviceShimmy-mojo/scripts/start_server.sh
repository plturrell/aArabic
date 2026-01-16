#!/bin/bash
# Start Shimmy-Mojo HTTP Server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "🚀 Starting Shimmy-Mojo HTTP Server"
echo "================================================================================"
echo ""

# Check for Mojo
if ! command -v mojo &> /dev/null; then
    echo "❌ Mojo is required but not installed"
    exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi

LIB_PATH="./libzig_http_shimmy.$LIB_EXT"
if [ ! -f "$LIB_PATH" ]; then
    echo "❌ Zig HTTP library not found: $LIB_PATH"
    echo "   Build it with: ./scripts/build_zig.sh"
    exit 1
fi

echo "🔧 Configuration:"
echo "   Host: 0.0.0.0"
echo "   Port: 11434"
echo "   Models dir: ./models (default)"
echo ""

echo "🌐 Starting server..."
echo ""

# Start the Zig+Mojo server
mojo run services/llm/handlers.mojo
