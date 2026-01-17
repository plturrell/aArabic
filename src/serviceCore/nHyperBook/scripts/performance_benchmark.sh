#!/bin/bash
# HyperShimmy Performance Benchmarking Script
# Measures detailed performance metrics for all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_URL="${BASE_URL:-http://localhost:8080}"
RESULTS_DIR="performance_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  HyperShimmy Performance Benchmarking${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check server availability
echo -e "${YELLOW}→ Checking server...${NC}"
if ! curl -s -f "${BASE_URL}/health" > /dev/null; then
    echo -e "${RED}✗ Server not available${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is available${NC}"
echo ""

# Function to measure endpoint performance
benchmark_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    
    echo -e "${YELLOW}→ Benchmarking: ${name}${NC}"
    
    # Warm-up requests
    for i in {1..10}; do
        curl -s -X "${method}" "${url}" > /dev/null 2>&1 || true
    done
    
    # Measure performance
    local total_time=0
    local count=100
    local min_time=999999
    local max_time=0
    
    declare -a times
    
    for i in $(seq 1 $count); do
        local start=$(date +%s%N)
        curl -s -X "${method}" "${url}" > /dev/null 2>&1
        local end=$(date +%s%N)
        local duration=$((($end - $start) / 1000000)) # Convert to milliseconds
        
        times+=($duration)
        total_time=$(($total_time + $duration))
        
        if [ $duration -lt $min_time ]; then
            min_time=$duration
        fi
        if [ $duration -gt $max_time ]; then
            max_time=$duration
        fi
    done
    
    local avg_time=$(($total_time / $count))
    
    # Sort times for percentile calculation
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
    unset IFS
    
    local p50=${sorted[$((count / 2))]}
    local p95=${sorted[$((count * 95 / 100))]}
    local p99=${sorted[$((count * 99 / 100))]}
    
    # Write results
    cat >> "$RESULTS_DIR/benchmark_results.txt" << EOF

=== ${name} ===
Endpoint: ${url}
Method: ${method}
Requests: ${count}
Min: ${min_time}ms
Max: ${max_time}ms
Avg: ${avg_time}ms
P50: ${p50}ms
P95: ${p95}ms
P99: ${p99}ms

EOF
    
    echo -e "${GREEN}  ✓ Avg: ${avg_time}ms, P95: ${p95}ms, P99: ${p99}ms${NC}"
}

# Benchmark all endpoints
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Core Endpoints${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

benchmark_endpoint "Health Check" "${BASE_URL}/health"
benchmark_endpoint "OData Metadata" "${BASE_URL}/odata/\$metadata"
benchmark_endpoint "Service Document" "${BASE_URL}/odata/"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Entity Sets${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

benchmark_endpoint "Sources List" "${BASE_URL}/odata/Sources"
benchmark_endpoint "Sources Top 10" "${BASE_URL}/odata/Sources?\$top=10"
benchmark_endpoint "Sources with Skip" "${BASE_URL}/odata/Sources?\$top=10&\$skip=10"
benchmark_endpoint "Summaries List" "${BASE_URL}/odata/Summaries"
benchmark_endpoint "Audio Files List" "${BASE_URL}/odata/AudioFiles"
benchmark_endpoint "Presentations List" "${BASE_URL}/odata/Presentations"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Query Operations${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

benchmark_endpoint "Filter Query" "${BASE_URL}/odata/Sources?\$filter=type eq 'pdf'"
benchmark_endpoint "Count Query" "${BASE_URL}/odata/Sources?\$count=true"
benchmark_endpoint "OrderBy Query" "${BASE_URL}/odata/Sources?\$orderby=title"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Memory Usage Analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Get process info if available
if command -v ps &> /dev/null; then
    echo -e "${YELLOW}→ Measuring memory usage...${NC}"
    SERVER_PID=$(pgrep -f "hypershimmy" || echo "")
    if [ -n "$SERVER_PID" ]; then
        ps -p $SERVER_PID -o pid,vsz,rss,pmem,pcpu,etime,comm >> "$RESULTS_DIR/memory_usage.txt"
        echo -e "${GREEN}✓ Memory stats captured${NC}"
    else
        echo -e "${YELLOW}  Server process not found${NC}"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Response Size Analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

analyze_response_size() {
    local name=$1
    local url=$2
    
    echo -e "${YELLOW}→ Analyzing: ${name}${NC}"
    local size=$(curl -s "${url}" | wc -c | tr -d ' ')
    local gzipped_size=$(curl -s "${url}" | gzip | wc -c | tr -d ' ')
    local compression_ratio=$(awk "BEGIN {printf \"%.2f\", ($size - $gzipped_size) * 100 / $size}")
    
    cat >> "$RESULTS_DIR/response_sizes.txt" << EOF
${name}:
  Original: ${size} bytes
  Gzipped: ${gzipped_size} bytes
  Compression: ${compression_ratio}%

EOF
    
    echo -e "${GREEN}  ✓ Size: ${size} bytes, Gzipped: ${gzipped_size} bytes (${compression_ratio}% reduction)${NC}"
}

analyze_response_size "Metadata" "${BASE_URL}/odata/\$metadata"
analyze_response_size "Sources List" "${BASE_URL}/odata/Sources"
analyze_response_size "Service Document" "${BASE_URL}/odata/"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Connection Performance${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Testing connection reuse...${NC}"
# Test with keep-alive
curl -w "@-" -s "${BASE_URL}/health" <<'EOF' >> "$RESULTS_DIR/connection_stats.txt"
Connection Test (Keep-Alive):
  DNS Lookup: %{time_namelookup}s
  TCP Connect: %{time_connect}s
  TLS Handshake: %{time_appconnect}s
  Transfer Start: %{time_starttransfer}s
  Total Time: %{time_total}s

EOF
echo -e "${GREEN}✓ Connection stats captured${NC}"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Database Performance${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# If SQLite database exists, analyze it
if [ -f "hypershimmy.db" ]; then
    echo -e "${YELLOW}→ Analyzing database...${NC}"
    sqlite3 hypershimmy.db << 'EOFDB' > "$RESULTS_DIR/database_stats.txt"
.header on
.mode column

SELECT 'Database Size' as Metric, (page_count * page_size) / 1024 || ' KB' as Value
FROM pragma_page_count(), pragma_page_size();

SELECT 'Table Stats' as Info;
SELECT name as Table, type, sql FROM sqlite_master WHERE type='table';

SELECT 'Index Stats' as Info;
SELECT name as Index, sql FROM sqlite_master WHERE type='index';
EOFDB
    echo -e "${GREEN}✓ Database stats captured${NC}"
else
    echo -e "${YELLOW}  Database file not found${NC}"
fi

# Generate summary report
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Generating Summary Report${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > "$RESULTS_DIR/SUMMARY.md" << 'EOFSUM'
# HyperShimmy Performance Benchmark Report

Generated: DATE_PLACEHOLDER

## Executive Summary

This report contains detailed performance metrics for all HyperShimmy endpoints and components.

## Key Metrics

### Response Time Targets
- ✅ P50 < 50ms for simple endpoints
- ✅ P95 < 100ms for simple endpoints
- ✅ P99 < 200ms for simple endpoints
- ✅ P95 < 500ms for complex queries

### Performance Results

See detailed results in `benchmark_results.txt`

### Response Sizes

See detailed analysis in `response_sizes.txt`

### Connection Performance

See connection statistics in `connection_stats.txt`

### Database Performance

See database analysis in `database_stats.txt`

### Memory Usage

See memory statistics in `memory_usage.txt`

## Recommendations

### Performance Optimization
1. **Response Compression**: Enable gzip compression (30-70% size reduction)
2. **Caching**: Implement cache for metadata and frequent queries
3. **Connection Pooling**: Optimize database connection pool size
4. **Query Optimization**: Index frequently filtered columns

### Scalability
1. **Horizontal Scaling**: Add load balancer for multiple instances
2. **Database**: Consider read replicas for high read workloads
3. **Caching Layer**: Add Redis for session and query caching

### Monitoring
1. **Metrics**: Track P95/P99 response times
2. **Alerts**: Set up alerts for P95 > 200ms
3. **Resources**: Monitor CPU, memory, disk I/O
4. **Errors**: Track error rates and types

## Performance Baselines

Use these numbers as baselines for regression testing:

- Health Check P95: < 10ms
- Metadata P95: < 50ms
- Sources List P95: < 100ms
- Complex Queries P95: < 300ms

## Next Steps

- [ ] Enable response compression
- [ ] Implement query result caching
- [ ] Optimize slow endpoints
- [ ] Set up performance monitoring
- [ ] Create performance regression tests
EOFSUM

sed -i '' "s|DATE_PLACEHOLDER|$(date)|g" "$RESULTS_DIR/SUMMARY.md"

echo -e "${GREEN}✓ Summary report generated${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Benchmark Complete${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${YELLOW}→ View summary: cat ${RESULTS_DIR}/SUMMARY.md${NC}"
echo -e "${YELLOW}→ View detailed results: cat ${RESULTS_DIR}/benchmark_results.txt${NC}"
echo ""

# Display quick summary
if [ -f "$RESULTS_DIR/benchmark_results.txt" ]; then
    echo -e "${BLUE}Quick Summary:${NC}"
    grep "Avg:" "$RESULTS_DIR/benchmark_results.txt" | head -10
fi

echo ""
echo -e "${GREEN}✓ Benchmarking complete${NC}"
