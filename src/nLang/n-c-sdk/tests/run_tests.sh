#!/bin/bash

# Test runner script for n-c-sdk
# Runs unit, integration, and load tests with proper configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "  n-c-sdk Test Suite Runner"
echo "================================================"
echo ""

# Check if zig is available
if ! command -v zig &> /dev/null; then
    echo -e "${RED}Error: zig compiler not found${NC}"
    exit 1
fi

echo "Zig version: $(zig version)"
echo ""

# Parse arguments
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_LOAD=false
OPTIMIZE="ReleaseSafe"

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_INTEGRATION=false
            RUN_LOAD=false
            shift
            ;;
        --integration-only)
            RUN_UNIT=false
            RUN_LOAD=false
            shift
            ;;
        --load-only)
            RUN_UNIT=false
            RUN_INTEGRATION=false
            RUN_LOAD=true
            shift
            ;;
        --with-load)
            RUN_LOAD=true
            shift
            ;;
        --debug)
            OPTIMIZE="Debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--unit-only|--integration-only|--load-only|--with-load] [--debug]"
            exit 1
            ;;
    esac
done

FAILED=0

# Unit tests
if [ "$RUN_UNIT" = true ]; then
    echo -e "${YELLOW}Running Unit Tests...${NC}"
    echo "-------------------------------------------"
    if zig build test-unit -Doptimize=$OPTIMIZE; then
        echo -e "${GREEN}✓ Unit tests passed${NC}"
    else
        echo -e "${RED}✗ Unit tests failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# Integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${YELLOW}Running Integration Tests...${NC}"
    echo "-------------------------------------------"
    if zig build test-integration -Doptimize=$OPTIMIZE; then
        echo -e "${GREEN}✓ Integration tests passed${NC}"
    else
        echo -e "${RED}✗ Integration tests failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# Load tests
if [ "$RUN_LOAD" = true ]; then
    echo -e "${YELLOW}Running Load Tests...${NC}"
    echo "-------------------------------------------"
    echo "Note: Load tests may take several minutes"
    if zig build test-load -Doptimize=ReleaseFast; then
        echo -e "${GREEN}✓ Load tests passed${NC}"
    else
        echo -e "${RED}✗ Load tests failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# Summary
echo "================================================"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All test suites passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED test suite(s) failed${NC}"
    exit 1
fi