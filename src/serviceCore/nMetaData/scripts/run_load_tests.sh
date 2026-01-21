#!/bin/bash

# Load Test Runner for nMetaData API
# Runs various load testing scenarios

set -e

echo "=================================="
echo "nMetaData Load Test Runner"
echo "=================================="
echo ""

# Configuration
API_URL="${API_URL:-http://localhost:3000}"
CONCURRENT_USERS="${CONCURRENT_USERS:-10}"
DURATION="${DURATION:-60}"
SCENARIO="${SCENARIO:-mixed}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Display configuration
echo -e "${BLUE}Load Test Configuration:${NC}"
echo "  API URL: $API_URL"
echo "  Concurrent Users: $CONCURRENT_USERS"
echo "  Duration: ${DURATION}s"
echo "  Scenario: $SCENARIO"
echo ""

# Function to run load test
run_load_test() {
    local scenario=$1
    local name=$2
    
    echo -e "${YELLOW}Running $name load test...${NC}"
    
    # Create temporary Zig test file
    cat > /tmp/load_test_runner.zig <<EOF
const std = @import("std");
const load_test = @import("api/load_test.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = load_test.LoadTestConfig{
        .concurrent_users = $CONCURRENT_USERS,
        .duration_seconds = $DURATION,
        .target_rps = 0,
        .base_url = "$API_URL",
        .auth_token = null,
    };

    const scenarios = try load_test.Scenarios.$scenario(allocator);
    defer allocator.free(scenarios);

    var runner = load_test.LoadTestRunner.init(allocator, config, scenarios);
    defer runner.deinit();

    try runner.run();
}
EOF

    cd "$(dirname "$0")/../zig"
    zig run /tmp/load_test_runner.zig
    
    rm /tmp/load_test_runner.zig
    echo ""
}

# Run specified scenario or all scenarios
case "$SCENARIO" in
    "auth"|"authentication")
        run_load_test "authentication" "Authentication"
        ;;
    "crud"|"dataset")
        run_load_test "datasetCRUD" "Dataset CRUD"
        ;;
    "lineage")
        run_load_test "lineage" "Lineage Tracking"
        ;;
    "graphql")
        run_load_test "graphql" "GraphQL"
        ;;
    "mixed")
        run_load_test "mixed" "Mixed Workload"
        ;;
    "all")
        echo -e "${BLUE}Running all load test scenarios...${NC}"
        echo ""
        run_load_test "authentication" "Authentication"
        run_load_test "datasetCRUD" "Dataset CRUD"
        run_load_test "lineage" "Lineage Tracking"
        run_load_test "graphql" "GraphQL"
        run_load_test "mixed" "Mixed Workload"
        ;;
    *)
        echo -e "${RED}Unknown scenario: $SCENARIO${NC}"
        echo "Available scenarios: auth, crud, lineage, graphql, mixed, all"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=================================="
echo "Load Tests Complete"
echo "==================================${NC}"
