#!/bin/bash

# Day 30: Comprehensive Image Processing Tests
# Tests all components from Days 26-29

set -e

echo "=============================================="
echo " Day 30: Image Processing Integration Tests"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to nExtract directory
cd "$(dirname "$0")/.."

echo -e "${BLUE}Building test suite...${NC}"
zig test zig/tests/image_processing_test.zig 2>&1

TEST_RESULT=$?

echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} ✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Test Coverage Summary:"
    echo "  ✓ Integration tests (Full pipelines)"
    echo "  ✓ Quality assessment (PSNR, mean, std dev)"
    echo "  ✓ Performance benchmarks"
    echo "  ✓ Memory usage validation"
    echo "  ✓ Edge case handling"
    echo "  ✓ Component comparisons"
    echo ""
    echo "Components Tested:"
    echo "  ✓ Color space conversions (Day 26)"
    echo "  ✓ Image filters (Day 27)"
    echo "  ✓ Image transformations (Day 28)"
    echo "  ✓ Thresholding methods (Day 29)"
    echo ""
    echo -e "${GREEN}Day 30 complete! Ready for OCR engine (Days 31-35)${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED} ✗ TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
