#!/bin/bash
# Build script for Zig Data Types Library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "================================================================================"
echo "🔢 Building Zig Data Types Library"
echo "================================================================================"
echo ""

# Build the library
echo "📦 Compiling zig_data_types.zig..."
zig build-lib zig_data_types.zig -dynamic -OReleaseFast

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    # Show library details
    if [ -f "libzig_data_types.dylib" ]; then
        echo "📊 Library created:"
        ls -lh libzig_data_types.dylib
        echo ""
        
        echo "🔍 Exported symbols:"
        nm -gU libzig_data_types.dylib | grep zig_
        echo ""
    elif [ -f "libzig_data_types.so" ]; then
        echo "📊 Library created:"
        ls -lh libzig_data_types.so
        echo ""
        
        echo "🔍 Exported symbols:"
        nm -gD libzig_data_types.so | grep zig_
        echo ""
    fi
    
    echo "✅ Build complete!"
    echo ""
    echo "Features:"
    echo "  • Variant type system (Null, Bool, Int, Float, String, List, Map)"
    echo "  • Graph types (Node, Relationship, Path)"
    echo "  • JSON serialization"
    echo "  • OData v4 compatibility"
    echo ""
else
    echo "❌ Compilation failed!"
    exit 1
fi
