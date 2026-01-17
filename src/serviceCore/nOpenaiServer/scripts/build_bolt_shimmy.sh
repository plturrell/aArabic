#!/bin/bash
# Build script for Zig Bolt Protocol Client

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "================================================================================"
echo "🔌 Building Zig Bolt Protocol Client"
echo "================================================================================"
echo ""

# Build the library
echo "📦 Compiling zig_bolt_shimmy.zig..."
zig build-lib zig_bolt_shimmy.zig -dynamic -OReleaseFast

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    # Show library details
    if [ -f "libzig_bolt_shimmy.dylib" ]; then
        echo "📊 Library created:"
        ls -lh libzig_bolt_shimmy.dylib
        echo ""
        
        echo "🔍 Exported symbols:"
        nm -gU libzig_bolt_shimmy.dylib | grep zig_bolt
        echo ""
    elif [ -f "libzig_bolt_shimmy.so" ]; then
        echo "📊 Library created:"
        ls -lh libzig_bolt_shimmy.so
        echo ""
        
        echo "🔍 Exported symbols:"
        nm -gD libzig_bolt_shimmy.so | grep zig_bolt
        echo ""
    fi
    
    echo "✅ Build complete!"
    echo ""
    echo "Usage from Mojo:"
    echo "  from sys.ffi import DLHandle"
    echo "  var lib = DLHandle(\"./libzig_bolt_shimmy.dylib\")"
    echo ""
else
    echo "❌ Compilation failed!"
    exit 1
fi
