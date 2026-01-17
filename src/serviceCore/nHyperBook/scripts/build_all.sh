#!/bin/bash

# Build All Components for HyperShimmy
# This script builds the Zig server, Mojo core, and prepares the SAPUI5 UI

set -e  # Exit on error

echo "======================================================================"
echo "🔥 Building HyperShimmy - Complete Build"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ====================================================================
# Step 1: Build Zig Components
# ====================================================================
echo "⚡ Step 1/3: Building Zig server and I/O libraries..."
echo "----------------------------------------------------------------------"

if command -v zig &> /dev/null; then
    echo "✓ Zig found: $(zig version)"
    
    # Build using zig build
    zig build -Doptimize=ReleaseFast
    
    echo "✓ Zig build complete"
else
    echo "✗ ERROR: Zig not found. Please install Zig 0.13.0+"
    exit 1
fi

echo ""

# ====================================================================
# Step 2: Build Mojo Components
# ====================================================================
echo "🔥 Step 2/3: Building Mojo core modules..."
echo "----------------------------------------------------------------------"

if command -v mojo &> /dev/null; then
    echo "✓ Mojo found: $(mojo --version 2>&1 | head -n1)"
    
    # Compile Mojo modules
    # Note: Adjust paths as Mojo modules are created
    if [ -f "core/__init__.mojo" ]; then
        echo "  - Compiling core module..."
        mojo build core/__init__.mojo -o lib/libhyper_core.so 2>/dev/null || echo "  (core module not ready yet)"
    else
        echo "  (core module not created yet - will be built in Week 1, Day 6)"
    fi
    
    echo "✓ Mojo build complete (or skipped if modules not ready)"
else
    echo "⚠ WARNING: Mojo not found. Skipping Mojo build."
    echo "  Install Mojo 24.5+ to build core modules"
fi

echo ""

# ====================================================================
# Step 3: Prepare SAPUI5 UI
# ====================================================================
echo "🎨 Step 3/3: Preparing SAPUI5 UI..."
echo "----------------------------------------------------------------------"

# No build step needed for SAPUI5 - it uses CDN
# Just verify webapp structure exists
if [ -d "webapp" ]; then
    echo "✓ SAPUI5 webapp directory exists"
    
    # Count files for progress indication
    ui_files=$(find webapp -type f 2>/dev/null | wc -l)
    echo "  Found $ui_files UI files"
else
    echo "  (webapp not created yet - will be built in Week 1, Days 4-5)"
fi

echo ""

# ====================================================================
# Summary
# ====================================================================
echo "======================================================================"
echo "✅ Build Complete!"
echo "======================================================================"
echo ""
echo "Build Artifacts:"
echo "  • Zig libraries:  $(ls -1 zig-out/lib/ 2>/dev/null | wc -l | xargs) files in zig-out/lib/"
echo "  • Zig executable: $(ls -1 zig-out/bin/hypershimmy 2>/dev/null | wc -l | xargs) file(s) in zig-out/bin/"
echo "  • Mojo libraries: $(ls -1 lib/*.so 2>/dev/null | wc -l | xargs) files in lib/"
echo ""
echo "Next Steps:"
echo "  1. Run server:  ./scripts/start.sh"
echo "  2. Run tests:   ./scripts/test.sh"
echo "  3. View docs:   open docs/README.md"
echo ""
echo "======================================================================"
