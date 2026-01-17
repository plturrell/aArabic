#!/bin/bash
# Build Zig HTTP Server for Shimmy-Mojo

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "🔨 Building Zig HTTP Server for Shimmy-Mojo"
echo "================================================================================"
echo ""

# Check for Zig
if ! command -v zig &> /dev/null; then
    echo "❌ Zig is not installed"
    echo ""
    echo "Install Zig:"
    echo "  macOS:  brew install zig"
    echo "  Linux:  Download from https://ziglang.org/download/"
    echo ""
    exit 1
fi

# Show Zig version
ZIG_VERSION=$(zig version)
echo "✅ Zig found: $ZIG_VERSION"
echo ""

echo "🔨 Building inference engine (libinference)..."
echo "   Path: inference/engine"
echo ""

pushd "$PROJECT_DIR/inference/engine" >/dev/null
zig build -Doptimize=ReleaseFast
popd >/dev/null

echo ""
echo "🔨 Building OpenAI-compatible server..."
echo "   Input: openai_http_server.zig"
echo "   Output: shimmy_openai_server"
echo ""

zig build-exe openai_http_server.zig \
    -OReleaseFast \
    -femit-bin=shimmy_openai_server

echo "✅ Build successful!"
echo ""

if [ -f "shimmy_openai_server" ]; then
    FILE_SIZE=$(du -h "shimmy_openai_server" | cut -f1)
    echo "📦 Server binary created:"
    echo "   File: shimmy_openai_server"
    echo "   Size: $FILE_SIZE"
    echo ""
fi

echo "🔨 Building Lean4 service..."
echo "   Input: lean4_http_server.zig"
echo "   Output: shimmy_lean4_server"
echo ""

zig build-exe lean4_http_server.zig \
    -OReleaseFast \
    -femit-bin=shimmy_lean4_server

echo "✅ Build successful!"
echo ""

if [ -f "shimmy_lean4_server" ]; then
    FILE_SIZE=$(du -h "shimmy_lean4_server" | cut -f1)
    echo "📦 Server binary created:"
    echo "   File: shimmy_lean4_server"
    echo "   Size: $FILE_SIZE"
    echo ""
fi

echo "================================================================================"
echo "✅ Servers Built Successfully!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Run server:"
echo "     ./shimmy_openai_server"
echo "     ./shimmy_lean4_server"
echo ""
echo "  2. Test with curl:"
echo "     curl http://localhost:11434/health"
echo "     curl http://localhost:8002/health"
echo ""
echo "  3. Use with OpenAI SDK:"
echo "     from openai import OpenAI"
echo "     client = OpenAI(base_url=\"http://localhost:11434/v1\")"
echo ""
echo "Environment:"
echo "  SHIMMY_MODEL_PATH=...  # Optional model directory path"
echo "  SHIMMY_MODEL_ID=...    # Optional model id in responses"
echo "  SHIMMY_MODEL_DIR=...   # Optional model root dir (default: ./models)"
echo "  SHIMMY_INFERENCE_LIB=... # Optional libinference path"
echo ""
echo "Lean4 environment:"
echo "  SHIMMY_LEAN4_HOST=0.0.0.0"
echo "  SHIMMY_LEAN4_PORT=8002"
echo "  SHIMMY_LEAN4_WORK_DIR=tmp/lean4"
echo "  SHIMMY_LEAN4_MAX_OUTPUT_BYTES=1048576"
echo "  LEAN4_BIN=lean"
echo "  LEAN4_ROOT=/path/to/lean/root"
echo ""
echo "================================================================================"
