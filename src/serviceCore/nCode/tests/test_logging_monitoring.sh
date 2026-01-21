#!/bin/bash
# Test script for nCode logging and monitoring features (Day 8)
# Tests structured logging, metrics endpoints, and health checks

set -e

echo "========================================================================"
echo "nCode Logging & Monitoring Tests - Day 8"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function to print test results
print_test_result() {
    local test_name=$1
    local result=$2
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Check if server is running
SERVER_URL="http://localhost:18003"

echo "Prerequisites Check:"
echo "-------------------"

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo -e "${RED}✗ curl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ curl available${NC}"

# Check if jq is available (for JSON parsing)
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠ jq not found (JSON parsing will be limited)${NC}"
    HAS_JQ=false
else
    echo -e "${GREEN}✓ jq available${NC}"
    HAS_JQ=true
fi

echo ""
echo "Testing Endpoints:"
echo "-----------------"

# Test 1: Health endpoint
echo -n "Testing /health endpoint... "
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVER_URL/health" 2>/dev/null || echo "000")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n 1)
BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    if echo "$BODY" | grep -q "\"status\":\"ok\""; then
        print_test_result "Health endpoint returns 200 OK" "PASS"
        
        # Check for enhanced fields
        if echo "$BODY" | grep -q "\"version\""; then
            print_test_result "Health endpoint includes version" "PASS"
        else
            print_test_result "Health endpoint includes version" "FAIL"
        fi
        
        if echo "$BODY" | grep -q "\"uptime_seconds\""; then
            print_test_result "Health endpoint includes uptime" "PASS"
        else
            print_test_result "Health endpoint includes uptime" "FAIL"
        fi
    else
        print_test_result "Health endpoint returns valid JSON" "FAIL"
    fi
else
    print_test_result "Health endpoint returns 200 OK" "FAIL"
    echo -e "${RED}Server may not be running at $SERVER_URL${NC}"
    exit 1
fi

# Test 2: Metrics endpoint (Prometheus format)
echo -n "Testing /metrics endpoint... "
METRICS_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVER_URL/metrics" 2>/dev/null || echo "000")
HTTP_CODE=$(echo "$METRICS_RESPONSE" | tail -n 1)
BODY=$(echo "$METRICS_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    print_test_result "Metrics endpoint returns 200 OK" "PASS"
    
    # Check for key metrics
    if echo "$BODY" | grep -q "ncode_requests_total"; then
        print_test_result "Metrics include request counter" "PASS"
    else
        print_test_result "Metrics include request counter" "FAIL"
    fi
    
    if echo "$BODY" | grep -q "ncode_uptime_seconds"; then
        print_test_result "Metrics include uptime gauge" "PASS"
    else
        print_test_result "Metrics include uptime gauge" "FAIL"
    fi
    
    if echo "$BODY" | grep -q "# HELP"; then
        print_test_result "Metrics include Prometheus help text" "PASS"
    else
        print_test_result "Metrics include Prometheus help text" "FAIL"
    fi
    
    if echo "$BODY" | grep -q "# TYPE"; then
        print_test_result "Metrics include Prometheus type declarations" "PASS"
    else
        print_test_result "Metrics include Prometheus type declarations" "FAIL"
    fi
else
    print_test_result "Metrics endpoint returns 200 OK" "FAIL"
fi

# Test 3: Metrics JSON endpoint
echo -n "Testing /metrics.json endpoint... "
METRICS_JSON_RESPONSE=$(curl -s -w "\n%{http_code}" "$SERVER_URL/metrics.json" 2>/dev/null || echo "000")
HTTP_CODE=$(echo "$METRICS_JSON_RESPONSE" | tail -n 1)
BODY=$(echo "$METRICS_JSON_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    print_test_result "Metrics JSON endpoint returns 200 OK" "PASS"
    
    if [ "$HAS_JQ" = true ]; then
        # Validate JSON structure
        if echo "$BODY" | jq -e '.requests' >/dev/null 2>&1; then
            print_test_result "Metrics JSON includes requests section" "PASS"
        else
            print_test_result "Metrics JSON includes requests section" "FAIL"
        fi
        
        if echo "$BODY" | jq -e '.cache' >/dev/null 2>&1; then
            print_test_result "Metrics JSON includes cache section" "PASS"
        else
            print_test_result "Metrics JSON includes cache section" "FAIL"
        fi
        
        if echo "$BODY" | jq -e '.server.uptime_seconds' >/dev/null 2>&1; then
            print_test_result "Metrics JSON includes server uptime" "PASS"
        else
            print_test_result "Metrics JSON includes server uptime" "FAIL"
        fi
    else
        if echo "$BODY" | grep -q "\"requests\""; then
            print_test_result "Metrics JSON includes requests section" "PASS"
        else
            print_test_result "Metrics JSON includes requests section" "FAIL"
        fi
    fi
else
    print_test_result "Metrics JSON endpoint returns 200 OK" "FAIL"
fi

# Test 4: Make several requests and verify metrics update
echo ""
echo "Testing Metrics Updates:"
echo "-----------------------"

# Get initial request count
if [ "$HAS_JQ" = true ]; then
    INITIAL_REQUESTS=$(curl -s "$SERVER_URL/metrics.json" | jq -r '.requests.total' 2>/dev/null || echo "0")
else
    INITIAL_REQUESTS="0"
fi

echo "Initial request count: $INITIAL_REQUESTS"

# Make 5 test requests
for i in {1..5}; do
    curl -s "$SERVER_URL/health" > /dev/null 2>&1
done

sleep 1

# Get updated request count
if [ "$HAS_JQ" = true ]; then
    UPDATED_REQUESTS=$(curl -s "$SERVER_URL/metrics.json" | jq -r '.requests.total' 2>/dev/null || echo "0")
    echo "Updated request count: $UPDATED_REQUESTS"
    
    if [ "$UPDATED_REQUESTS" -gt "$INITIAL_REQUESTS" ]; then
        print_test_result "Metrics update correctly after requests" "PASS"
    else
        print_test_result "Metrics update correctly after requests" "FAIL"
    fi
else
    print_test_result "Metrics update correctly after requests" "SKIP (jq not available)"
fi

# Test 5: Content-Type headers
echo ""
echo "Testing HTTP Headers:"
echo "--------------------"

HEALTH_HEADERS=$(curl -s -I "$SERVER_URL/health" 2>/dev/null)
if echo "$HEALTH_HEADERS" | grep -qi "Content-Type: application/json"; then
    print_test_result "Health endpoint has correct Content-Type" "PASS"
else
    print_test_result "Health endpoint has correct Content-Type" "FAIL"
fi

METRICS_HEADERS=$(curl -s -I "$SERVER_URL/metrics" 2>/dev/null)
if echo "$METRICS_HEADERS" | grep -qi "Content-Type: text/plain"; then
    print_test_result "Metrics endpoint has correct Content-Type" "PASS"
else
    print_test_result "Metrics endpoint has correct Content-Type" "FAIL"
fi

# Test 6: Server version header
if echo "$HEALTH_HEADERS" | grep -qi "Server: nCode"; then
    print_test_result "Server includes version header" "PASS"
else
    print_test_result "Server includes version header" "FAIL"
fi

# Test 7: CORS headers
if echo "$HEALTH_HEADERS" | grep -qi "Access-Control-Allow-Origin"; then
    print_test_result "Server includes CORS headers" "PASS"
else
    print_test_result "Server includes CORS headers" "FAIL"
fi

# Performance test
echo ""
echo "Performance Test:"
echo "----------------"

START_TIME=$(date +%s%N)
for i in {1..10}; do
    curl -s "$SERVER_URL/health" > /dev/null 2>&1
done
END_TIME=$(date +%s%N)

DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))
AVG_MS=$(( DURATION_MS / 10 ))

echo "10 requests completed in ${DURATION_MS}ms (avg: ${AVG_MS}ms per request)"

if [ "$AVG_MS" -lt 50 ]; then
    print_test_result "Average response time < 50ms" "PASS"
else
    print_test_result "Average response time < 50ms" "FAIL"
fi

# Summary
echo ""
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo ""
echo "Total Tests: $TESTS_TOTAL"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
else
    echo -e "Failed: $TESTS_FAILED"
fi
echo ""

SUCCESS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))
echo "Success Rate: ${SUCCESS_RATE}%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Next Steps:"
    echo "1. Review metrics in Prometheus: curl http://localhost:18003/metrics"
    echo "2. Check JSON metrics: curl http://localhost:18003/metrics.json | jq"
    echo "3. Monitor logs with: NCODE_LOG_LEVEL=DEBUG ./zig-out/bin/ncode-server"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Ensure nCode server is running: ./zig-out/bin/ncode-server"
    echo "2. Check server logs for errors"
    echo "3. Verify server is accessible at $SERVER_URL"
    echo ""
    exit 1
fi
