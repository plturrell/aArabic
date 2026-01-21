#!/usr/bin/env bash
# nExtract - Day 25: Image Testing Runner
# Executes comprehensive image codec tests

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}nExtract - Day 25: Image Testing Suite${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check if Zig is installed
if ! command -v zig &> /dev/null; then
    echo -e "${RED}ERROR: Zig compiler not found${NC}"
    echo "Please install Zig 0.13+ from https://ziglang.org/"
    exit 1
fi

echo -e "${GREEN}✓${NC} Zig compiler found: $(zig version)"
echo ""

# Build test binary
echo -e "${YELLOW}Building image test suite...${NC}"
if zig build test --summary all 2>&1 | tee build.log; then
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${RED}✗${NC} Build failed"
    exit 1
fi
echo ""

# Run tests
echo -e "${YELLOW}Running image codec tests...${NC}"
echo ""

# Test categories
declare -a TEST_CATEGORIES=(
    "PNG Color Types"
    "PNG Bit Depths"
    "PNG Interlacing"
    "PNG Ancillary Chunks"
    "PNG Filters"
    "PNG Corrupt Handling"
    "JPEG Baseline"
    "JPEG Progressive"
    "JPEG Subsampling"
    "JPEG EXIF"
    "JPEG Thumbnail"
    "JPEG Corrupt Handling"
    "Color Space Conversions"
    "Large Images"
    "Memory Usage"
    "Quality Metrics"
)

echo -e "${BLUE}Test Categories:${NC}"
for category in "${TEST_CATEGORIES[@]}"; do
    echo -e "  • $category"
done
echo ""

# Run Zig tests
echo -e "${YELLOW}Executing tests...${NC}"
if zig test zig/tests/image_test.zig 2>&1 | tee test.log; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    TEST_RESULT=0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    TEST_RESULT=1
fi

# Summary
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Count results from log
TOTAL_TESTS=$(grep -c "Testing" test.log 2>/dev/null || echo "0")
PASSED_TESTS=$(grep -c "✓ PASS" test.log 2>/dev/null || echo "0")
FAILED_TESTS=$((TOTAL_TESTS - PASSED_TESTS))

echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS ✓${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS ✗${NC}"

if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS)*100}")
    echo "Success Rate: ${SUCCESS_RATE}%"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Review test.log for detailed results"
echo "  2. Add test fixtures to tests/fixtures/"
echo "  3. Implement actual decoder integration"
echo "  4. Run performance benchmarks"
echo ""

exit $TEST_RESULT
