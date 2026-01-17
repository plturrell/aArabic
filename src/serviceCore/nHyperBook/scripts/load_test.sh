#!/bin/bash
# HyperShimmy Load Testing Script
# Tests system behavior under heavy load

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8080}"
CONCURRENT_USERS="${CONCURRENT_USERS:-50}"
DURATION="${DURATION:-300}"  # 5 minutes
RAMP_UP="${RAMP_UP:-30}"     # 30 seconds ramp-up

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  HyperShimmy Load Testing${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}→ Checking server availability...${NC}"
if ! curl -s -f "${BASE_URL}/health" > /dev/null; then
    echo -e "${RED}✗ Server not available at ${BASE_URL}${NC}"
    echo -e "${YELLOW}  Please start the server first${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is available${NC}"
echo ""

# Check dependencies
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 is not installed${NC}"
        echo -e "${YELLOW}  Install with: brew install $2${NC}"
        exit 1
    fi
}

echo -e "${YELLOW}→ Checking dependencies...${NC}"
check_command "ab" "apache-bench"
check_command "vegeta" "vegeta"
check_command "jq" "jq"
echo -e "${GREEN}✓ All dependencies available${NC}"
echo ""

# Create results directory
RESULTS_DIR="load_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}✓ Results directory: ${RESULTS_DIR}${NC}"
echo ""

# Test 1: Health Endpoint Baseline
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 1: Health Endpoint Baseline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Running 10,000 requests with 100 concurrent...${NC}"
ab -n 10000 -c 100 -g "$RESULTS_DIR/health_baseline.tsv" \
   "${BASE_URL}/health" > "$RESULTS_DIR/health_baseline.txt" 2>&1
echo -e "${GREEN}✓ Health endpoint test complete${NC}"
echo ""

# Test 2: OData Metadata
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 2: OData Metadata${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Running 5,000 requests with 50 concurrent...${NC}"
ab -n 5000 -c 50 -g "$RESULTS_DIR/metadata.tsv" \
   "${BASE_URL}/odata/\$metadata" > "$RESULTS_DIR/metadata.txt" 2>&1
echo -e "${GREEN}✓ Metadata test complete${NC}"
echo ""

# Test 3: Sources List
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 3: Sources List${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Running 5,000 requests with 50 concurrent...${NC}"
ab -n 5000 -c 50 -g "$RESULTS_DIR/sources_list.tsv" \
   "${BASE_URL}/odata/Sources" > "$RESULTS_DIR/sources_list.txt" 2>&1
echo -e "${GREEN}✓ Sources list test complete${NC}"
echo ""

# Test 4: Mixed Workload with Vegeta
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 4: Mixed Workload${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Running mixed workload for ${DURATION}s...${NC}"

# Create Vegeta targets file
cat > "$RESULTS_DIR/targets.txt" << EOF
GET ${BASE_URL}/health
GET ${BASE_URL}/odata/\$metadata
GET ${BASE_URL}/odata/Sources
GET ${BASE_URL}/odata/Sources?$top=10
GET ${BASE_URL}/odata/Sources?$top=10&$skip=10
GET ${BASE_URL}/odata/Summaries
GET ${BASE_URL}/odata/AudioFiles
GET ${BASE_URL}/odata/Presentations
EOF

# Run Vegeta attack
vegeta attack -targets="$RESULTS_DIR/targets.txt" \
    -duration=${DURATION}s \
    -rate=50 \
    -workers=10 \
    > "$RESULTS_DIR/vegeta_results.bin"

# Generate Vegeta report
vegeta report "$RESULTS_DIR/vegeta_results.bin" > "$RESULTS_DIR/vegeta_report.txt"
vegeta plot "$RESULTS_DIR/vegeta_results.bin" > "$RESULTS_DIR/vegeta_plot.html"
echo -e "${GREEN}✓ Mixed workload test complete${NC}"
echo ""

# Test 5: Spike Test
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 5: Spike Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Simulating traffic spike (200 concurrent)...${NC}"
ab -n 5000 -c 200 -g "$RESULTS_DIR/spike_test.tsv" \
   "${BASE_URL}/health" > "$RESULTS_DIR/spike_test.txt" 2>&1
echo -e "${GREEN}✓ Spike test complete${NC}"
echo ""

# Test 6: Sustained Load Test
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test 6: Sustained Load Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}→ Running sustained load (60 seconds)...${NC}"
vegeta attack -targets="$RESULTS_DIR/targets.txt" \
    -duration=60s \
    -rate=100 \
    -workers=20 \
    > "$RESULTS_DIR/sustained_results.bin"
vegeta report "$RESULTS_DIR/sustained_results.bin" > "$RESULTS_DIR/sustained_report.txt"
echo -e "${GREEN}✓ Sustained load test complete${NC}"
echo ""

# Generate summary report
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Load Test Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > "$RESULTS_DIR/SUMMARY.md" << 'EOFSUM'
# HyperShimmy Load Test Results

## Test Configuration

- **Base URL**: BASE_URL_PLACEHOLDER
- **Date**: DATE_PLACEHOLDER
- **Duration**: DURATION_PLACEHOLDER seconds
- **Concurrent Users**: CONCURRENT_PLACEHOLDER

## Test Results

### Test 1: Health Endpoint Baseline
**Purpose**: Baseline performance for simple endpoint

HEALTH_RESULTS_PLACEHOLDER

### Test 2: OData Metadata
**Purpose**: Test metadata endpoint performance

METADATA_RESULTS_PLACEHOLDER

### Test 3: Sources List
**Purpose**: Test database query performance

SOURCES_RESULTS_PLACEHOLDER

### Test 4: Mixed Workload
**Purpose**: Realistic traffic pattern

VEGETA_RESULTS_PLACEHOLDER

### Test 5: Spike Test
**Purpose**: System behavior under sudden load

SPIKE_RESULTS_PLACEHOLDER

### Test 6: Sustained Load
**Purpose**: Long-term stability

SUSTAINED_RESULTS_PLACEHOLDER

## Performance Metrics

### Response Times (percentiles)
- P50 (median): Extracted from tests
- P95: Extracted from tests
- P99: Extracted from tests

### Throughput
- Requests per second: Extracted from tests
- Failed requests: Extracted from tests

### Resource Utilization
- CPU usage: Monitor during tests
- Memory usage: Monitor during tests
- Connection pool: Monitor during tests

## Recommendations

Based on load test results:

1. **Performance**: 
   - Target: < 100ms p95 for simple endpoints
   - Target: < 500ms p95 for complex queries

2. **Concurrency**:
   - Stable up to N concurrent users
   - Consider horizontal scaling beyond N users

3. **Bottlenecks**:
   - Identify slowest endpoints
   - Database query optimization needed?
   - Connection pool sizing adequate?

4. **Errors**:
   - Error rate should be < 0.1%
   - Investigate any 5xx errors
   - Check timeout settings

## Next Steps

- [ ] Review all failed requests
- [ ] Optimize slow endpoints
- [ ] Tune connection pools
- [ ] Configure auto-scaling thresholds
- [ ] Set up monitoring alerts
EOFSUM

# Extract statistics and populate summary
sed -i '' "s|BASE_URL_PLACEHOLDER|${BASE_URL}|g" "$RESULTS_DIR/SUMMARY.md"
sed -i '' "s|DATE_PLACEHOLDER|$(date)|g" "$RESULTS_DIR/SUMMARY.md"
sed -i '' "s|DURATION_PLACEHOLDER|${DURATION}|g" "$RESULTS_DIR/SUMMARY.md"
sed -i '' "s|CONCURRENT_PLACEHOLDER|${CONCURRENT_USERS}|g" "$RESULTS_DIR/SUMMARY.md"

# Extract key metrics from Apache Bench results
for test in health_baseline metadata sources_list spike_test; do
    if [ -f "$RESULTS_DIR/${test}.txt" ]; then
        echo "## ${test} Results" >> "$RESULTS_DIR/${test}_summary.txt"
        grep "Requests per second" "$RESULTS_DIR/${test}.txt" >> "$RESULTS_DIR/${test}_summary.txt" || true
        grep "Time per request" "$RESULTS_DIR/${test}.txt" >> "$RESULTS_DIR/${test}_summary.txt" || true
        grep "Failed requests" "$RESULTS_DIR/${test}.txt" >> "$RESULTS_DIR/${test}_summary.txt" || true
        grep -A 5 "Percentage of the requests" "$RESULTS_DIR/${test}.txt" >> "$RESULTS_DIR/${test}_summary.txt" || true
    fi
done

echo -e "${GREEN}✓ Load test complete!${NC}"
echo ""
echo -e "${BLUE}Results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${YELLOW}→ View summary: cat ${RESULTS_DIR}/SUMMARY.md${NC}"
echo -e "${YELLOW}→ View plots: open ${RESULTS_DIR}/vegeta_plot.html${NC}"
echo ""

# Display quick summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Quick Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -f "$RESULTS_DIR/vegeta_report.txt" ]; then
    echo -e "${YELLOW}Mixed Workload Results:${NC}"
    head -20 "$RESULTS_DIR/vegeta_report.txt"
    echo ""
fi

echo -e "${GREEN}✓ All tests completed successfully${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
