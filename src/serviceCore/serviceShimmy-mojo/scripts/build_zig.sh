#!/bin/bash
# Build Zig HTTP Server for Shimmy-Mojo

set -e

cd "$(dirname "$0")"

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

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    LIB_EXT="so"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "🖥️  Operating System: $OS"
echo "📦 Library extension: $LIB_EXT"
echo ""

# Build Zig HTTP server library
echo "🔨 Building Zig HTTP server..."
echo "   Input: zig_http_shimmy.zig"
echo "   Output: libzig_http_shimmy.$LIB_EXT"
echo ""

zig build-lib zig_http_shimmy.zig \
    -dynamic \
    -OReleaseFast \
    -femit-bin=libzig_http_shimmy.$LIB_EXT

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    
    # Show file info
    if [ -f "libzig_http_shimmy.$LIB_EXT" ]; then
        FILE_SIZE=$(du -h "libzig_http_shimmy.$LIB_EXT" | cut -f1)
        echo "📦 Library created:"
        echo "   File: libzig_http_shimmy.$LIB_EXT"
        echo "   Size: $FILE_SIZE"
        echo ""
    fi
    
    echo "================================================================================"
    echo "✅ Zig HTTP Server Built Successfully!"
    echo "================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run server:"
    echo "     mojo run server_shimmy.mojo"
    echo ""
    echo "  2. Test with curl:"
    echo "     curl http://localhost:11434/health"
    echo ""
    echo "  3. Use with OpenAI SDK:"
    echo "     from openai import OpenAI"
    echo "     client = OpenAI(base_url=\"http://localhost:11434/v1\")"
    echo ""
    echo "================================================================================"
else
    echo "❌ Build failed!"
    echo ""
    echo "Common issues:"
    echo "  • Make sure Zig is installed: zig version"
    echo "  • Try updating Zig: brew upgrade zig"
    echo "  • Check zig_http_shimmy.zig for syntax errors"
    echo ""
    exit 1
fi
