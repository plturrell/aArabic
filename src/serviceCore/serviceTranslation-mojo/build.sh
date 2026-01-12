#!/bin/bash
# Build script for Mojo Translation Service

set -e

echo "================================================================================"
echo "🔥 Building Mojo Translation Service"
echo "================================================================================"

# Check Mojo installation
if ! command -v mojo &> /dev/null; then
    echo "❌ Mojo not found. Please install Mojo first:"
    echo "   https://docs.modular.com/mojo/manual/get-started/"
    exit 1
fi

MOJO_VERSION=$(mojo --version | head -n 1)
echo "✅ Found: $MOJO_VERSION"

# Build main Mojo module
echo ""
echo "📦 Building main.mojo..."
mojo build main.mojo -o mojo-translation

if [ -f "mojo-translation" ]; then
    chmod +x mojo-translation
    echo "✅ Built: mojo-translation"
else
    echo "❌ Build failed"
    exit 1
fi

# Test the binary
echo ""
echo "🧪 Testing Mojo binary..."
./mojo-translation --help 2>/dev/null || echo "⚠️  Binary built but may need runtime dependencies"

echo ""
echo "================================================================================"
echo "✅ Build Complete!"
echo "================================================================================"
echo ""
echo "📝 Files created:"
echo "  • mojo-translation  - Mojo executable"
echo ""
echo "🚀 To run:"
echo "  ./mojo-translation              # Run Mojo translation CLI"
echo "  python3 server_mojo.py          # Run FastAPI server with Mojo backend"
echo ""
echo "================================================================================"
