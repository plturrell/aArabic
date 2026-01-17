#!/bin/bash
# Build Zig JSON library for mojo-sdk
# Zero Python dependencies - pure Zig std.json

set -e

echo "🔨 Building Zig JSON library for mojo-sdk..."
echo ""

# Navigate to JSON module
cd src/serviceCore/serviceShimmy-mojo/mojo-sdk/stdlib/json

# Build dynamic library
echo "Compiling zig_json_parser.zig..."
zig build-lib zig_json_parser.zig \
    -dynamic \
    -O ReleaseFast \
    -target aarch64-macos \
    -femit-bin=libzig_json.dylib

echo "✅ libzig_json.dylib created"
echo ""

# Copy to project root for easy access
echo "Copying library to project root..."
cp libzig_json.dylib ../../../../../../

echo "✅ Library copied to: /Users/user/Documents/arabic_folder/libzig_json.dylib"
echo ""

# Return to project root
cd ../../../../../../

# Verify library exists
if [ -f "libzig_json.dylib" ]; then
    echo "✅ JSON library ready!"
    echo ""
    echo "Usage in Mojo:"
    echo "  from mojo_sdk.stdlib.json import JsonParser"
    echo "  var parser = JsonParser()"
    echo "  var data = parser.parse_file(\"config.json\")"
    echo ""
else
    echo "❌ Error: Library not found"
    exit 1
fi

echo "🎉 Build complete!"
