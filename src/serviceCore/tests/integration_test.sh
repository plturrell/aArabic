#!/bin/bash
# Integration Tests for Rust Clients
set -e
echo "========================================"
echo "Rust Clients Integration Tests"
echo "========================================"
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
PASSED=0
FAILED=0
run_test() {
    local test_name=$1
    local command=$2
    echo -n "Testing: $test_name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
}
echo ""
echo "Phase 1: Binary Existence"
run_test "Qdrant CLI" "test -f src/serviceAutomation/qdrant-api-client/target/release/qdrant-cli"
run_test "Memgraph CLI" "test -f src/serviceAutomation/memgraph-api-client/target/release/aimo-memgraph-cli"
run_test "DragonflyDB CLI" "test -f src/serviceAutomation/dragonflydb-api-client/target/release/dragonfly-cli"
echo ""
echo "Phase 2: Dockerfiles"
run_test "Qdrant Dockerfile" "test -f src/serviceAutomation/qdrant-api-client/Dockerfile"
run_test "Memgraph Dockerfile" "test -f src/serviceAutomation/memgraph-api-client/Dockerfile"
run_test "DragonflyDB Dockerfile" "test -f src/serviceAutomation/dragonflydb-api-client/Dockerfile"
echo ""
echo "========================================"
echo -e "${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"
[ $FAILED -eq 0 ] && echo -e "${GREEN}All tests passed!${NC}" || echo -e "${RED}Some tests failed!${NC}"
