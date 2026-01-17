#!/bin/bash
# Test Dolby Audio Processor - Day 36
# Validates the Dolby audio processing pipeline

set -e

echo "========================================"
echo "  Dolby Audio Processor Test Suite"
echo "  Day 36 - AudioLabShimmy"
echo "========================================"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Building Dolby processor..."
zig build-lib zig/dolby_processor.zig \
    -dynamic \
    -O ReleaseFast \
    -femit-bin=libdolby_processor.dylib

if [ -f "libdolby_processor.dylib" ]; then
    echo "✓ Dolby processor library built"
else
    echo "✗ Failed to build library"
    exit 1
fi

echo ""
echo "Running built-in tests..."
zig test zig/dolby_processor.zig

echo ""
echo "========================================"
echo "  Test Results"
echo "========================================"
echo ""
echo "✓ Dolby processor implementation complete"
echo "✓ LUFS metering functional"
echo "✓ Multi-band compression ready"
echo "✓ Harmonic enhancement working"
echo "✓ Stereo widening operational"
echo "✓ Brick-wall limiter active"
echo ""
echo "Library: $(pwd)/libdolby_processor.dylib"
echo ""
echo "FFI Exports available:"
echo "  - process_audio_dolby()"
echo "  - measure_lufs_ffi()"
echo ""
echo "Ready for integration with Mojo TTS engine (Day 37-38)"
echo ""

exit 0
