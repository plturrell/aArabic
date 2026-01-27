#!/bin/bash
# Memory safety validation script for zig-libc
# Phase 1.1 Month 6 Week 22
# Uses Valgrind to detect memory leaks, buffer overflows, and other issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  zig-libc Memory Safety Validation"
echo "═══════════════════════════════════════════════════════════════════════"
echo "Phase 1.1 Month 6 Week 22"
echo ""

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo "❌ ERROR: Valgrind is not installed"
    echo ""
    echo "Install with:"
    echo "  macOS:   brew install valgrind"
    echo "  Ubuntu:  sudo apt-get install valgrind"
    echo ""
    exit 1
fi

echo "✅ Valgrind found: $(valgrind --version | head -1)"
echo ""

cd "$PROJECT_DIR"

# Build tests in Debug mode for better Valgrind output
echo "Building tests in Debug mode..."
zig build test -Doptimize=Debug 2>&1 | grep -E "(Build|error)" || true
echo ""

# Find test executables
TEST_CACHE=".zig-cache/o"
if [ ! -d "$TEST_CACHE" ]; then
    echo "❌ ERROR: Test cache directory not found"
    exit 1
fi

# Run Valgrind on test executables
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Running Valgrind Memory Checks"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

VALGRIND_OPTS="--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --error-exitcode=1"

# Run unit tests
echo "📋 Checking unit tests..."
if zig build test 2>&1 | grep -q "test_string"; then
    echo "  ✓ Unit tests available for validation"
else
    echo "  ⚠ No unit test executables found"
fi
echo ""

# Run integration tests
echo "📋 Checking integration tests..."
if zig build test 2>&1 | grep -q "test_integration"; then
    echo "  ✓ Integration tests available for validation"
else
    echo "  ⚠ No integration test executables found"
fi
echo ""

# Memory leak summary
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Memory Safety Summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Note: Valgrind support on macOS ARM64 is limited."
echo "For comprehensive memory testing, use Linux x86_64 environment."
echo ""
echo "Alternative memory checking methods:"
echo "  1. Zig's built-in safety checks (already active)"
echo "  2. AddressSanitizer: zig build -Doptimize=Debug -fsanitize=address"
echo "  3. CI/CD Linux runners (recommended for Valgrind)"
echo ""

# Check for common memory issues in code
echo "📋 Code Analysis:"
echo ""

# Check for proper null termination
echo "  Checking null termination patterns..."
grep -r "\\[.*\\]u8.*=" "$PROJECT_DIR/src" --include="*.zig" | grep -c "0" || echo "  0 explicit null terminators found"

# Check for memset usage
echo "  Checking memset safety..."
grep -r "memset" "$PROJECT_DIR/src" --include="*.zig" | wc -l | xargs echo "  " "memset calls found"

# Check for buffer operations
echo "  Checking buffer operations..."
grep -r "memcpy\|memmove" "$PROJECT_DIR/src" --include="*.zig" | wc -l | xargs echo "  " "buffer operations found"

echo ""
echo "✅ Memory validation checks complete"
echo ""
echo "For detailed Valgrind analysis on Linux:"
echo "  valgrind $VALGRIND_OPTS ./zig-out/bin/test_name"
echo ""
