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

if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi

BIN_PATH="./shimmy_openai_server"
if [ ! -f "$BIN_PATH" ]; then
    echo "❌ OpenAI server binary not found: $BIN_PATH"
    echo "   Build it with: ./scripts/build_zig.sh"
    exit 1
fi

INFERENCE_LIB="./inference/engine/zig-out/lib/libinference.$LIB_EXT"
if [ ! -f "$INFERENCE_LIB" ]; then
    echo "⚠️  Inference library not found: $INFERENCE_LIB"
    echo "   The server will still start, but inference will fail until it exists."
fi

echo "🔧 Configuration:"
echo "   Host: 0.0.0.0"
echo "   Port: 11434"
echo "   Models dir: ./models (default)"
echo "   Model path: \$SHIMMY_MODEL_PATH (optional)"
echo "   Model id:   \$SHIMMY_MODEL_ID (optional)"
echo "   GGUF model: \$SHIMMY_MODEL (optional, loads this file)"
echo ""

echo "🌐 Starting server..."
echo ""

# Start the Zig OpenAI server with optional model path and inference lib
if [ -n "${SHIMMY_INFERENCE_LIB:-}" ]; then
    export SHIMMY_INFERENCE_LIB
fi

if [ -n "${SHIMMY_MODEL:-}" ]; then
    echo "📦 Loading model: $SHIMMY_MODEL"
    export SHIMMY_MODEL_PATH="$SHIMMY_MODEL"
fi

exec ./shimmy_openai_server
