#!/bin/bash

# Build all components for nLeanProof

set -e

echo "======================================================================"
echo "Building nLeanProof - complete build"
echo "======================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "Step 1/3: Building Mojo core modules"
echo "----------------------------------------------------------------------"

if command -v mojo &> /dev/null; then
    echo "Mojo found: $(mojo --version 2>&1 | head -n1)"
    mkdir -p lib

    # Determine extension
    LIB_EXT="dylib"
    if [[ "$(uname -s)" == "Linux" ]]; then
        LIB_EXT="so"
    fi

    if [ -f "bridge/lean4_ffi.mojo" ]; then
        echo "Compiling Mojo FFI library..."
        mojo build bridge/lean4_ffi.mojo -I . -o "lib/liblean4_ffi.${LIB_EXT}" || \
            echo "Mojo build failed"
    else
        echo "Bridge module not found"
    fi
else
    echo "WARNING: Mojo not found. Skipping Mojo build."
fi

echo ""
echo "Step 2/3: Building Zig server and libraries"
echo "----------------------------------------------------------------------"

if command -v zig &> /dev/null; then
    echo "Zig found: $(zig version)"
    zig build -Doptimize=ReleaseFast
    echo "Zig build complete"
else
    echo "ERROR: Zig not found. Install Zig 0.15.2+"
    exit 1
fi

echo ""
echo "Step 3/3: Preparing optional UI"
echo "----------------------------------------------------------------------"

if [ -d "webapp" ]; then
    ui_files=$(find webapp -type f 2>/dev/null | wc -l | xargs)
    echo "webapp directory found ($ui_files files)"
else
    echo "webapp directory not created yet"
fi

echo ""
echo "======================================================================"
echo "Build complete"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Run server: ./scripts/start.sh"
echo "  2. Run tests:  ./scripts/test.sh"
echo ""
echo "======================================================================"
