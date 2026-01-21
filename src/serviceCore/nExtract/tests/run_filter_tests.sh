#!/bin/bash
# Test script for Day 27: Image Filtering

set -e

echo "======================================"
echo "Day 27: Image Filtering Tests"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

echo "Running all filter tests..."
echo ""

cd /Users/user/Documents/arabic_folder

# Run all tests in the filters module
if zig test src/serviceCore/nExtract/zig/ocr/filters.zig 2>&1 | tee /tmp/filter_tests.log; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    TESTS_PASSED=4
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    TESTS_FAILED=1
fi

echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "Tests Passed: ${TESTS_PASSED}"
echo "Tests Failed: ${TESTS_FAILED}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
