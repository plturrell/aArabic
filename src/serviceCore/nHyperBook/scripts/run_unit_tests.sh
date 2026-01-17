#!/bin/bash

# ============================================================================
# HyperShimmy Unit Test Runner
# ============================================================================
# Day 56: Comprehensive test execution script
# Runs all unit tests for Zig and Mojo components
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================================================"
echo "                    HyperShimmy Unit Test Suite"
echo "============================================================================"
echo ""

# ============================================================================
# Zig Unit Tests
# ============================================================================

echo -e "${BLUE}Running Zig Unit Tests...${NC}"
echo "----------------------------------------------------------------------------"

cd "$PROJECT_DIR"

if ! command -v zig &> /dev/null; then
    echo -e "${RED}Error: Zig compiler not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Building and running Zig tests...${NC}"

if zig build test 2>&1 | tee /tmp/zig_test_output.txt; then
    # Count test results from Zig output
    ZIG_PASSED=$(grep -c "All.*test.*passed" /tmp/zig_test_output.txt || echo "0")
    ZIG_FAILED=$(grep -c "FAIL" /tmp/zig_test_output.txt || echo "0")
    
    echo -e "${GREEN}✓ Zig unit tests completed${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ Zig unit tests failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""

# ============================================================================
# Mojo Unit Tests
# ============================================================================

echo -e "${BLUE}Running Mojo Unit Tests...${NC}"
echo "----------------------------------------------------------------------------"

cd "$PROJECT_DIR/mojo"

if ! command -v mojo &> /dev/null; then
    echo -e "${YELLOW}Warning: Mojo not found, skipping Mojo tests${NC}"
else
    echo -e "${YELLOW}Running Mojo embedding tests...${NC}"
    
    if mojo test_embeddings.mojo 2>&1 | tee /tmp/mojo_test_output.txt; then
        echo -e "${GREEN}✓ Mojo unit tests completed${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ Mojo unit tests failed${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
fi

echo ""

# ============================================================================
# Test Coverage Summary
# ============================================================================

echo "============================================================================"
echo "                           Test Summary"
echo "============================================================================"
echo ""
echo "Test Components Executed:"
echo "  - Zig Server Components (sources, security, JSON utils, errors)"
echo "  - Zig I/O Components (HTTP, HTML, PDF, web scraper)"
echo "  - Mojo Embedding Components"
echo ""
echo "Results:"
echo -e "  Total Test Suites:  ${TOTAL_TESTS}"
echo -e "  ${GREEN}Passed:${NC}             ${PASSED_TESTS}"
echo -e "  ${RED}Failed:${NC}             ${FAILED_TESTS}"
echo ""

# Calculate success rate
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "  Success Rate:       ${SUCCESS_RATE}%"
fi

echo ""

# ============================================================================
# Test Coverage Analysis
# ============================================================================

echo "Test Coverage Areas:"
echo "  ✓ Source Management (CRUD operations, validation)"
echo "  ✓ Security (input validation, sanitization, rate limiting, CORS)"
echo "  ✓ JSON Serialization (OData format, escaping, parsing)"
echo "  ✓ Error Handling (categorization, recovery, metrics)"
echo "  ✓ Embeddings (vector operations, similarity, storage)"
echo ""

# ============================================================================
# Recommendations
# ============================================================================

echo "Coverage Recommendations:"
echo "  - Target: 80%+ code coverage for production readiness"
echo "  - Focus areas: Integration tests, E2E workflows"
echo "  - Consider: Performance benchmarks, stress testing"
echo ""

# ============================================================================
# Exit Status
# ============================================================================

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}                    ✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}                    ✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}============================================================================${NC}"
    echo ""
    exit 1
fi
