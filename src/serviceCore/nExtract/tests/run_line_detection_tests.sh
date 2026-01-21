#!/bin/bash

# Day 31: Text Line Detection Tests
# Tests connected component analysis, line segmentation, and skew detection

set -e

echo "=============================================="
echo " Day 31: Text Line Detection Tests"
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
zig test zig/tests/line_detection_test.zig 2>&1

TEST_RESULT=$?

echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} ✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Test Coverage Summary:"
    echo "  ✓ Connected component analysis (4 and 8-connectivity)"
    echo "  ✓ Component extraction and bounding boxes"
    echo "  ✓ Component filtering (noise removal)"
    echo "  ✓ Horizontal projection profiles"
    echo "  ✓ Vertical projection profiles"
    echo "  ✓ Line segmentation"
    echo "  ✓ Skew detection (Projection method)"
    echo "  ✓ Skew detection (Hough transform)"
    echo "  ✓ Baseline detection"
    echo "  ✓ Full pipeline integration"
    echo "  ✓ Edge cases"
    echo "  ✓ Performance benchmarks"
    echo ""
    echo "Algorithms Implemented:"
    echo "  ✓ Two-pass connected component labeling"
    echo "  ✓ Projection profile line segmentation"
    echo "  ✓ Variance-based skew detection"
    echo "  ✓ Hough transform skew detection"
    echo "  ✓ Baseline detection via density analysis"
    echo ""
    echo -e "${GREEN}Day 31 complete! Ready for Day 32 (Character Segmentation)${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED} ✗ TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
