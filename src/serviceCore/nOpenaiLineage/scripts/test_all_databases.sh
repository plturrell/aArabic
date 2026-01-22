#!/bin/bash

# Cross-Database Test Runner for nMetaData
# Tests API functionality across PostgreSQL, SAP HANA, and SQLite

set -e

echo "========================================"
echo "nMetaData Cross-Database Test Suite"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_URL="${POSTGRES_URL:-postgresql://test:test@localhost:5432/nmetadata_test}"
HANA_URL="${HANA_URL:-hana://test:test@localhost:39013}"
SQLITE_URL="${SQLITE_URL:-:memory:}"

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run tests for a specific database
run_database_tests() {
    local db_name=$1
    local db_url=$2
    
    echo -e "${BLUE}Testing with $db_name...${NC}"
    echo "Connection: $db_url"
    echo ""
    
    cd "$(dirname "$0")/../zig"
    
    # Run cross-database tests with specific database
    if DB_TYPE=$db_name DB_URL=$db_url zig test api/cross_db_test.zig 2>&1; then
        echo -e "${GREEN}✓ $db_name tests PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ $db_name tests FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Test PostgreSQL
if [ -n "$POSTGRES_URL" ]; then
    run_database_tests "PostgreSQL" "$POSTGRES_URL"
else
    echo -e "${YELLOW}Skipping PostgreSQL (not configured)${NC}"
fi

# Test SAP HANA
if [ -n "$HANA_URL" ]; then
    run_database_tests "HANA" "$HANA_URL"
else
    echo -e "${YELLOW}Skipping SAP HANA (not configured)${NC}"
fi

# Test SQLite (always available)
run_database_tests "SQLite" "$SQLITE_URL"

# Run performance comparison
echo -e "${BLUE}Running Performance Comparison...${NC}"
cd "$(dirname "$0")/../zig"
zig test api/cross_db_test.zig --test-filter "comparePerformance"
echo ""

# Print summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total Databases Tested: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✓ All cross-database tests PASSED!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some tests FAILED${NC}"
    exit 1
fi
