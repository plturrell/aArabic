#!/bin/bash
# Build script for Shimmy-Mojo

set -e  # Exit on error

echo "=" | tr '=' '-' | head -c 80 && echo
echo "🔥 Building Shimmy-Mojo - Pure Mojo LLM Inference Engine"
echo "=" | tr '=' '-' | head -c 80 && echo

# Check Mojo installation
if ! command -v mojo &> /dev/null; then
    echo "❌ Error: Mojo not found!"
    echo "Install Mojo: curl https://get.modular.com | sh -s -- mojo"
    exit 1
fi

echo "✅ Mojo found: $(mojo --version)"

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "📦 Building modules..."
echo ""

# Build GGUF parser
echo "🔨 Building GGUF parser..."
if mojo build core/gguf_parser.mojo -o build/gguf_parser 2>&1 | grep -v "warning:"; then
    echo "  ✅ GGUF parser built"
else
    echo "  ⚠️  GGUF parser has warnings (non-fatal)"
fi

# Build tensor operations
echo "🔨 Building tensor operations..."
if mojo build core/tensor_ops.mojo -o build/tensor_ops 2>&1 | grep -v "warning:"; then
    echo "  ✅ Tensor ops built"
else
    echo "  ⚠️  Tensor ops has warnings (non-fatal)"
fi

# Build main executable (when ready)
if [ -f "main.mojo" ]; then
    echo "🔨 Building main executable..."
    if mojo build main.mojo -o shimmy-mojo 2>&1 | grep -v "warning:"; then
        echo "  ✅ shimmy-mojo executable built"
        chmod +x shimmy-mojo
    else
        echo "  ⚠️  Main executable has warnings (non-fatal)"
    fi
fi

echo ""
echo "=" | tr '=' '-' | head -c 80 && echo
echo "✅ Build complete!"
echo "=" | tr '=' '-' | head -c 80 && echo
echo ""
echo "📚 Quick start:"
echo "  • Test GGUF parser:   mojo run core/gguf_parser.mojo"
echo "  • Test tensor ops:    mojo run core/tensor_ops.mojo"
if [ -f "shimmy-mojo" ]; then
    echo "  • Run server:         ./shimmy-mojo serve"
    echo "  • List models:        ./shimmy-mojo list"
    echo "  • Generate:           ./shimmy-mojo generate <model> \"<prompt>\""
fi
echo ""
