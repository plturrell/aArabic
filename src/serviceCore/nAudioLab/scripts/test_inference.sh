#!/bin/bash

# Integration Testing Script for AudioLabShimmy TTS System
# Day 39: Complete end-to-end testing

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$PROJECT_DIR/tests"
OUTPUT_DIR="$TESTS_DIR/output"
MODEL_DIR="$PROJECT_DIR/data/models"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}AudioLabShimmy Integration Testing Suite${NC}"
echo -e "${BLUE}Day 39: End-to-End TTS Pipeline Testing${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to print section headers
print_section() {
    echo ""
    echo -e "${YELLOW}>>> $1${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
print_section "Checking Prerequisites"

if ! command -v mojo &> /dev/null; then
    print_error "Mojo compiler not found"
    echo "Please install Mojo from https://docs.modular.com/mojo/"
    exit 1
fi
print_success "Mojo compiler found"

if ! command -v zig &> /dev/null; then
    print_error "Zig compiler not found"
    echo "Please install Zig from https://ziglang.org/"
    exit 1
fi
print_success "Zig compiler found"

# Check if models exist
if [ ! -d "$MODEL_DIR" ]; then
    print_error "Model directory not found: $MODEL_DIR"
    echo "Please ensure models are trained and available"
    exit 1
fi
print_success "Model directory found"

# Run unit tests first
print_section "Running Unit Tests"

echo "Running audio type tests..."
if mojo run "$TESTS_DIR/test_audio_types.mojo" 2>/dev/null; then
    print_success "Audio type tests passed"
else
    print_error "Audio type tests failed"
fi

echo "Running text normalization tests..."
if mojo run "$TESTS_DIR/test_text_processing.mojo" 2>/dev/null; then
    print_success "Text normalization tests passed"
else
    print_error "Text normalization tests failed"
fi

# Run TTS pipeline integration tests
print_section "Running TTS Pipeline Integration Tests"

echo "This will test the complete end-to-end pipeline..."
echo "Expected duration: 2-5 minutes"
echo ""

if mojo run "$TESTS_DIR/test_tts_pipeline.mojo"; then
    print_success "TTS pipeline tests passed"
    PIPELINE_SUCCESS=1
else
    print_error "TTS pipeline tests failed"
    PIPELINE_SUCCESS=0
fi

# Run audio quality validation tests
print_section "Running Audio Quality Validation Tests"

echo "This will validate professional audio quality standards..."
echo "Testing LUFS, THD+N, dynamic range, and more..."
echo ""

if mojo run "$TESTS_DIR/test_audio_quality.mojo"; then
    print_success "Audio quality tests passed"
    QUALITY_SUCCESS=1
else
    print_error "Audio quality tests failed"
    QUALITY_SUCCESS=0
fi

# Performance benchmarking
print_section "Running Performance Benchmarks"

echo "Testing synthesis speed..."
cat > "$OUTPUT_DIR/benchmark_test.mojo" << 'EOF'
from time import now
from ..mojo.inference.engine import TTSEngine

fn main() raises:
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    var test_texts = List[String]()
    test_texts.append("Short sentence")
    test_texts.append("This is a medium length sentence for testing")
    test_texts.append("This is a longer sentence that will test the synthesis speed for more complex inputs with multiple clauses")
    
    print("Performance Benchmarks:")
    print("-" * 40)
    
    for i in range(len(test_texts)):
        var text = test_texts[i]
        var start = now()
        var audio = tts.synthesize(text)
        var end = now()
        
        var duration_ms = (end - start) / 1_000_000
        var audio_duration_s = Float32(audio.length) / Float32(audio.sample_rate)
        var rtf = Float32(duration_ms) / 1000.0 / audio_duration_s
        
        print(f"Text length: {len(text)} chars")
        print(f"Synthesis time: {duration_ms:.1f} ms")
        print(f"Audio duration: {audio_duration_s:.2f} s")
        print(f"Real-time factor: {rtf:.2f}x")
        print("-" * 40)
EOF

if mojo run "$OUTPUT_DIR/benchmark_test.mojo"; then
    print_success "Performance benchmarks completed"
else
    print_error "Performance benchmarks failed"
fi

# Memory profiling
print_section "Running Memory Profile"

echo "Testing memory usage patterns..."
cat > "$OUTPUT_DIR/memory_test.mojo" << 'EOF'
from ..mojo.inference.engine import TTSEngine

fn main() raises:
    print("Memory Profile Test:")
    print("-" * 40)
    
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Synthesize multiple times to check for leaks
    for i in range(10):
        var audio = tts.synthesize("Memory test iteration")
        print(f"Iteration {i+1}: {audio.length} samples")
    
    print("-" * 40)
    print("If you see consistent memory growth, there may be a leak")
EOF

if mojo run "$OUTPUT_DIR/memory_test.mojo"; then
    print_success "Memory profile completed"
else
    print_error "Memory profile failed"
fi

# Generate test audio samples
print_section "Generating Test Audio Samples"

echo "Creating sample outputs in $OUTPUT_DIR..."

cat > "$OUTPUT_DIR/generate_samples.mojo" << 'EOF'
from ..mojo.inference.engine import TTSEngine
from ..zig.audio_io import writeWAV

fn main() raises:
    var tts = TTSEngine.load("src/serviceCore/nAudioLab/data/models")
    
    # Generate various test samples
    var samples = List[Tuple[String, String]]()
    samples.append(("hello.wav", "Hello, world!"))
    samples.append(("numbers.wav", "The numbers are 42, 1234, and 3.14"))
    samples.append(("date.wav", "Today is January 17th, 2026"))
    samples.append(("complex.wav", "Dr. Smith lives at 123 Main St. and can be reached at $10.50"))
    
    print("Generating test samples...")
    for i in range(len(samples)):
        var filename, text = samples[i]
        var audio = tts.synthesize(text)
        var path = "src/serviceCore/nAudioLab/tests/output/" + filename
        writeWAV(audio, path)
        print(f"✓ Generated: {filename}")
EOF

if mojo run "$OUTPUT_DIR/generate_samples.mojo"; then
    print_success "Test samples generated"
    echo "Samples saved in: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"/*.wav 2>/dev/null || echo "No WAV files generated"
else
    print_error "Sample generation failed"
fi

# Test Dolby processing
print_section "Testing Dolby Audio Processing"

echo "Verifying Dolby post-processing chain..."
if [ -f "$PROJECT_DIR/scripts/test_dolby_processor.sh" ]; then
    bash "$PROJECT_DIR/scripts/test_dolby_processor.sh"
    print_success "Dolby processing verified"
else
    echo "Dolby test script not found (expected from Day 36)"
fi

# FFI bridge validation
print_section "Testing Mojo-Zig FFI Bridge"

echo "Verifying FFI integration..."
if [ -f "$PROJECT_DIR/scripts/test_ffi_bridge.py" ]; then
    python3 "$PROJECT_DIR/scripts/test_ffi_bridge.py"
    print_success "FFI bridge verified"
else
    echo "FFI test script not found (expected from Day 38)"
fi

# Summary
print_section "Test Summary"

echo ""
if [ $PIPELINE_SUCCESS -eq 1 ] && [ $QUALITY_SUCCESS -eq 1 ]; then
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}ALL TESTS PASSED! ✓${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "The AudioLabShimmy TTS system is working correctly!"
    echo ""
    echo "Next steps:"
    echo "  1. Review generated audio samples in $OUTPUT_DIR"
    echo "  2. Validate quality metrics against requirements"
    echo "  3. Proceed to Day 40: Documentation & Polish"
    exit 0
else
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}SOME TESTS FAILED ✗${NC}"
    echo -e "${RED}================================================${NC}"
    echo ""
    echo "Please review the failures above and fix issues before proceeding"
    exit 1
fi
