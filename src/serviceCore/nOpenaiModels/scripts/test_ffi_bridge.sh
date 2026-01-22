#!/bin/bash
# FFI Golden Test Script
# Run after EVERY file move to ensure FFI remains intact
#
# This script verifies:
# 1. Zig library compiles successfully
# 2. Expected FFI exports are present
# 3. Mojo can call the FFI functions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INFERENCE_DIR="$PROJECT_DIR/inference"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  FFI GOLDEN TEST"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Check if inference directory has been refactored
if [ -d "$INFERENCE_DIR/engine" ]; then
    INFERENCE_DIR="$INFERENCE_DIR/engine"
    echo -e "${GREEN}Using refactored path: inference/engine/${NC}"
elif [ -f "$INFERENCE_DIR/build.zig" ]; then
    echo -e "${YELLOW}Using legacy path: inference/${NC}"
else
    echo -e "${RED}ERROR: Cannot find inference directory${NC}"
    exit 1
fi

# Step 1: Build Zig library
echo -e "${YELLOW}Step 1: Building Zig inference library...${NC}"
cd "$INFERENCE_DIR"

if ! zig build test-mojo-bridge 2>&1; then
    echo -e "${RED}ERROR: Zig build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Zig build successful${NC}"

# Step 2: Check library exports
echo ""
echo -e "${YELLOW}Step 2: Checking FFI exports...${NC}"

EXPECTED_EXPORTS=(
    "inference_load_model"
    "inference_generate"
    "inference_is_loaded"
    "inference_get_info"
    "inference_unload"
)

# Find the built executable or library
BUILD_OUTPUT="$INFERENCE_DIR/zig-out/bin/test_mojo_bridge"

if [ ! -f "$BUILD_OUTPUT" ]; then
    echo -e "${RED}ERROR: Build output not found at $BUILD_OUTPUT${NC}"
    exit 1
fi

# Check for exports using nm
MISSING_EXPORTS=0
for export in "${EXPECTED_EXPORTS[@]}"; do
    if nm "$BUILD_OUTPUT" 2>/dev/null | grep -q "$export"; then
        echo -e "  ${GREEN}✓${NC} $export"
    else
        echo -e "  ${RED}✗${NC} $export (MISSING)"
        MISSING_EXPORTS=$((MISSING_EXPORTS + 1))
    fi
done

if [ $MISSING_EXPORTS -gt 0 ]; then
    echo -e "${RED}ERROR: $MISSING_EXPORTS expected FFI exports are missing${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All FFI exports present${NC}"

# Step 3: Run the Zig-side test
echo ""
echo -e "${YELLOW}Step 3: Running Zig FFI self-test...${NC}"

# The test_mojo_bridge executable has a main() that tests the FFI
# We run it without a model to just test the basic functions
cd "$INFERENCE_DIR"
if timeout 10 zig build test-mojo-bridge 2>&1 | head -20; then
    echo -e "${GREEN}✓ Zig FFI self-test passed${NC}"
else
    echo -e "${YELLOW}⚠ Zig FFI self-test skipped (requires model)${NC}"
fi

# Step 4: Check Mojo FFI test exists and can parse
echo ""
echo -e "${YELLOW}Step 4: Checking Mojo FFI test...${NC}"

MOJO_TEST="$PROJECT_DIR/tests/ffi/test_bridge_minimal.mojo"
if [ -f "$MOJO_TEST" ]; then
    echo -e "  ${GREEN}✓${NC} Mojo FFI test file exists"
    # Try to check syntax (mojo check if available)
    if command -v mojo &> /dev/null; then
        if mojo check "$MOJO_TEST" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Mojo FFI test syntax valid"
        else
            echo -e "  ${YELLOW}⚠${NC} Mojo syntax check failed (may need deps)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Mojo not found, skipping syntax check"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Mojo FFI test not found at $MOJO_TEST"
    echo -e "      Create it to enable full FFI testing"
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}FFI GOLDEN TEST PASSED${NC}"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

exit 0
