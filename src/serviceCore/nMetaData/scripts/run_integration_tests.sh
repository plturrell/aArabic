#!/bin/bash

# Integration Test Runner for nMetaData API
# Runs comprehensive integration tests against the API

set -e

echo "=================================="
echo "nMetaData Integration Test Runner"
echo "=================================="
echo ""

# Configuration
API_URL="${API_URL:-http://localhost:3000}"
TEST_TIMEOUT="${TEST_TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if API is running
echo "Checking API availability at $API_URL..."
if ! curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: API is not running at $API_URL${NC}"
    echo "Please start the API server first"
    exit 1
fi
echo -e "${GREEN}✓ API is running${NC}"
echo ""

# Run Zig tests
echo "Running Zig integration tests..."
cd "$(dirname "$0")/../zig"

zig test api/integration_test.zig \
    --test-filter "IntegrationTests" \
    --test-timeout "$TEST_TIMEOUT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All integration tests passed${NC}"
else
    echo -e "${RED}✗ Integration tests failed${NC}"
    exit 1
fi

echo ""
echo "=================================="
echo "Integration Tests Complete"
echo "=================================="
