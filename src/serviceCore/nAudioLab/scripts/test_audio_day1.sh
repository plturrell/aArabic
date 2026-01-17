#!/bin/bash

# Day 1 Audio Infrastructure Test Script
# Tests WAV file I/O, audio types, and basic processing

set -e

echo "=========================================="
echo "AudioLabShimmy Day 1 Test Suite"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Create test output directory
mkdir -p test_output

echo "Step 1: Building audio library..."
zig build audio

echo ""
echo "Step 2: Running unit tests..."
zig build test

echo ""
echo "Step 3: Creating test program..."

# Create a test program that exercises the audio I/O
cat > test_output/test_audio.zig << 'EOF'
const std = @import("std");
const audio_io = @import("../zig/audio_io.zig");
const audio_types = @import("../zig/audio_types.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Test 1: Generate 440Hz test tone ===\n", .{});
    var tone = try audio_io.generateTestTone(allocator, 440.0, 2.0, 48000, 2);
    defer tone.deinit();
    
    std.debug.print("Generated {d} frames at {d}Hz, {d} channels\n", .{
        tone.frameCount(), tone.sample_rate, tone.channels
    });
    std.debug.print("Duration: {d:.2}s\n", .{tone.duration()});

    std.debug.print("\n=== Test 2: Write WAV file (48kHz, 24-bit, stereo) ===\n", .{});
    try audio_io.writeWAV(tone, "test_output/tone_440hz_48k_24bit.wav");
    std.debug.print("✓ Written: test_output/tone_440hz_48k_24bit.wav\n", .{});

    std.debug.print("\n=== Test 3: Read WAV file back ===\n", .{});
    var tone_read = try audio_io.readWAV("test_output/tone_440hz_48k_24bit.wav", allocator);
    defer tone_read.deinit();
    
    std.debug.print("Read {d} frames at {d}Hz, {d} channels, {d}-bit\n", .{
        tone_read.frameCount(), tone_read.sample_rate, tone_read.channels, tone_read.bit_depth
    });

    std.debug.print("\n=== Test 4: Get audio file info ===\n", .{});
    const info = try audio_io.getAudioInfo("test_output/tone_440hz_48k_24bit.wav", allocator);
    std.debug.print("{}\n", .{info});

    std.debug.print("\n=== Test 5: Convert to mono ===\n", .{});
    var mono = try tone_read.toMono(allocator);
    defer mono.deinit();
    std.debug.print("Converted to mono: {d} channels, {d} frames\n", .{
        mono.channels, mono.frameCount()
    });

    std.debug.print("\n=== Test 6: Apply gain ===\n", .{});
    var tone_copy = try tone.clone(allocator);
    defer tone_copy.deinit();
    tone_copy.applyGain(-6.0);
    std.debug.print("Applied -6dB gain\n", .{});

    std.debug.print("\n=== Test 7: Generate multi-frequency test ===\n", .{});
    var tone1 = try audio_io.generateTestTone(allocator, 261.63, 1.0, 48000, 2); // C4
    defer tone1.deinit();
    try audio_io.writeWAV(tone1, "test_output/tone_C4.wav");
    
    var tone2 = try audio_io.generateTestTone(allocator, 329.63, 1.0, 48000, 2); // E4
    defer tone2.deinit();
    try audio_io.writeWAV(tone2, "test_output/tone_E4.wav");
    
    var tone3 = try audio_io.generateTestTone(allocator, 392.00, 1.0, 48000, 2); // G4
    defer tone3.deinit();
    try audio_io.writeWAV(tone3, "test_output/tone_G4.wav");
    
    std.debug.print("✓ Generated C-E-G chord (individual files)\n", .{});

    std.debug.print("\n=== Test 8: 16-bit WAV compatibility ===\n", .{});
    var tone_16bit = try audio_io.generateTestTone(allocator, 880.0, 0.5, 44100, 2);
    tone_16bit.bit_depth = 16; // Set to 16-bit
    defer tone_16bit.deinit();
    try audio_io.writeWAV(tone_16bit, "test_output/tone_880hz_44k_16bit.wav");
    std.debug.print("✓ Written 16-bit WAV: test_output/tone_880hz_44k_16bit.wav\n", .{});
    
    var tone_16bit_read = try audio_io.readWAV("test_output/tone_880hz_44k_16bit.wav", allocator);
    defer tone_16bit_read.deinit();
    std.debug.print("✓ Read back 16-bit WAV successfully\n", .{});

    std.debug.print("\n========================================\n", .{});
    std.debug.print("All tests completed successfully! ✓\n", .{});
    std.debug.print("========================================\n", .{});
}
EOF

echo ""
echo "Step 4: Compiling test program..."
zig build-exe test_output/test_audio.zig \
    --mod audio_io::zig/audio_io.zig \
    --mod audio_types::zig/audio_types.zig \
    --mod wav_format::zig/wav_format.zig \
    --deps audio_io,audio_types,wav_format

echo ""
echo "Step 5: Running test program..."
./test_audio

echo ""
echo "Step 6: Verifying generated files..."
if [ -f "test_output/tone_440hz_48k_24bit.wav" ]; then
    FILE_SIZE=$(stat -f%z "test_output/tone_440hz_48k_24bit.wav" 2>/dev/null || stat -c%s "test_output/tone_440hz_48k_24bit.wav" 2>/dev/null)
    echo "✓ tone_440hz_48k_24bit.wav exists (${FILE_SIZE} bytes)"
else
    echo "✗ tone_440hz_48k_24bit.wav not found"
    exit 1
fi

if [ -f "test_output/tone_C4.wav" ]; then
    echo "✓ tone_C4.wav exists"
else
    echo "✗ tone_C4.wav not found"
    exit 1
fi

if [ -f "test_output/tone_E4.wav" ]; then
    echo "✓ tone_E4.wav exists"
else
    echo "✗ tone_E4.wav not found"
    exit 1
fi

if [ -f "test_output/tone_G4.wav" ]; then
    echo "✓ tone_G4.wav exists"
else
    echo "✗ tone_G4.wav not found"
    exit 1
fi

if [ -f "test_output/tone_880hz_44k_16bit.wav" ]; then
    echo "✓ tone_880hz_44k_16bit.wav exists"
else
    echo "✗ tone_880hz_44k_16bit.wav not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Day 1 Implementation Complete!"
echo "=========================================="
echo ""
echo "Deliverables:"
echo "  • audio_types.zig - Audio data structures"
echo "  • wav_format.zig - WAV file format handling"
echo "  • audio_io.zig - Audio I/O operations"
echo "  • Professional 48kHz/24-bit audio support"
echo "  • WAV read/write functionality"
echo "  • Audio buffer operations"
echo "  • Test suite with generated audio files"
echo ""
echo "Generated test files in test_output/:"
ls -lh test_output/*.wav 2>/dev/null || true
echo ""
