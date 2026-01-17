#!/bin/bash

# Clean Build Artifacts for HyperShimmy

echo "======================================================================"
echo "🧹 Cleaning HyperShimmy Build Artifacts"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Remove Zig build artifacts
if [ -d "zig-cache" ]; then
    echo "  • Removing zig-cache/"
    rm -rf zig-cache/
fi

if [ -d "zig-out" ]; then
    echo "  • Removing zig-out/"
    rm -rf zig-out/
fi

# Remove compiled libraries
if [ -d "lib" ]; then
    echo "  • Cleaning lib/*.dylib, lib/*.so"
    rm -f lib/*.dylib lib/*.so lib/*.dll 2>/dev/null || true
fi

# Remove Mojo build artifacts
echo "  • Cleaning Mojo artifacts"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.mojopkg" -delete 2>/dev/null || true

# Remove log files
if [ -d "logs" ]; then
    echo "  • Cleaning logs/"
    rm -rf logs/
fi

# Remove temporary files
echo "  • Cleaning temporary files"
find . -type f -name "*.tmp" -delete 2>/dev/null || true
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name "*.swo" -delete 2>/dev/null || true

# Remove test artifacts
if [ -d "test-results" ]; then
    echo "  • Removing test-results/"
    rm -rf test-results/
fi

if [ -d "coverage" ]; then
    echo "  • Removing coverage/"
    rm -rf coverage/
fi

# Remove runtime data (optional - uncomment if needed)
# if [ -d "data" ]; then
#     echo "  • Removing data/"
#     rm -rf data/
# fi

# if [ -d "uploads" ]; then
#     echo "  • Removing uploads/"
#     rm -rf uploads/
# fi

echo ""
echo "======================================================================"
echo "✅ Clean Complete!"
echo "======================================================================"
echo ""
echo "All build artifacts have been removed."
echo "Run ./scripts/build_all.sh to rebuild."
echo ""
