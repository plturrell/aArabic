#!/bin/bash
# Performance benchmark script for HANA Cloud operations
# Measures latency, throughput, and resource utilization

set -e

BASE_URL="http://localhost:11434"
API_BASE="/api/v1/prompts"
ITERATIONS=50

echo "======================================"
echo "HANA Cloud Performance Benchmark"
echo "======================================"
echo ""
echo "Configuration:"
echo "  - Base URL: $BASE_URL"
echo "  - Iterations: $ITERATIONS"
echo "  - Test mode: Sequential operations"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Arrays to store timing data
declare -a save_times
declare -a load_times
declare -a search_times
declare -a delete_times

echo -e "${CYAN}Phase 1: Save Performance Test${NC}"
echo "Saving $ITERATIONS prompts to HANA..."

for i in $(seq 1 $ITERATIONS); do
    START=$(date +%s%N)
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
      "${BASE_URL}${API_BASE}" \
      -H "Content-Type: application/json" \
      -d "{
        \"prompt_text\": \"Performance test prompt #$i - Calculate fibonacci(10)\",
        \"model_name\": \"lfm2.5-1.2b-q4_0\",
        \"user_id\": \"benchmark-user\",
        \"prompt_mode_id\": 1,
        \"tags\": \"benchmark,test\"
      }")
    
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))  # Convert to milliseconds
    save_times+=($ELAPSED)
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    
    if [ "$HTTP_CODE" -ne 201 ]; then
        echo "Error: Save failed with HTTP $HTTP_CODE"
        exit 1
    fi
    
    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i/$ITERATIONS saves..."
    fi
done

echo -e "${GREEN}✓ Phase 1 complete${NC}"
echo ""

echo -e "${CYAN}Phase 2: Load Performance Test${NC}"
echo "Loading history $ITERATIONS times..."

for i in $(seq 1 $ITERATIONS); do
    START=$(date +%s%N)
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X GET \
      "${BASE_URL}/v1/prompts/history?limit=20")
    
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))
    load_times+=($ELAPSED)
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    
    if [ "$HTTP_CODE" -ne 200 ]; then
        echo "Error: Load failed with HTTP $HTTP_CODE"
        exit 1
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i/$ITERATIONS loads..."
    fi
done

echo -e "${GREEN}✓ Phase 2 complete${NC}"
echo ""

echo -e "${CYAN}Phase 3: Search Performance Test${NC}"
echo "Searching $ITERATIONS times..."

SEARCH_TERMS=("fibonacci" "performance" "test" "calculate" "benchmark")

for i in $(seq 1 $ITERATIONS); do
    TERM=${SEARCH_TERMS[$((i % 5))]}
    
    START=$(date +%s%N)
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X GET \
      "${BASE_URL}${API_BASE}/search?q=${TERM}&limit=10")
    
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))
    search_times+=($ELAPSED)
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    
    if [ "$HTTP_CODE" -ne 200 ]; then
        echo "Error: Search failed with HTTP $HTTP_CODE"
        exit 1
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Completed $i/$ITERATIONS searches..."
    fi
done

echo -e "${GREEN}✓ Phase 3 complete${NC}"
echo ""

# Calculate statistics
calculate_stats() {
    local arr=("$@")
    local sum=0
    local min=999999
    local max=0
    local count=${#arr[@]}
    
    for val in "${arr[@]}"; do
        sum=$((sum + val))
        if [ $val -lt $min ]; then min=$val; fi
        if [ $val -gt $max ]; then max=$val; fi
    done
    
    local avg=$((sum / count))
    local p95_idx=$((count * 95 / 100))
    
    # Sort array for percentiles
    IFS=$'\n' sorted=($(sort -n <<<"${arr[*]}"))
    unset IFS
    local p95=${sorted[$p95_idx]}
    
    echo "$avg $min $max $p95"
}

echo "======================================"
echo -e "${YELLOW}Performance Results${NC}"
echo "======================================"
echo ""

# Save operations
stats=$(calculate_stats "${save_times[@]}")
read avg min max p95 <<< "$stats"
echo "Save Operations (POST /api/v1/prompts):"
echo "  Average:   ${avg}ms"
echo "  Min:       ${min}ms"
echo "  Max:       ${max}ms"
echo "  P95:       ${p95}ms"
echo "  Throughput: $((1000 * ITERATIONS / (avg * ITERATIONS / 1000))) ops/sec"
echo ""

# Load operations
stats=$(calculate_stats "${load_times[@]}")
read avg min max p95 <<< "$stats"
echo "Load Operations (GET /v1/prompts/history):"
echo "  Average:   ${avg}ms"
echo "  Min:       ${min}ms"
echo "  Max:       ${max}ms"
echo "  P95:       ${p95}ms"
echo "  Throughput: $((1000 * ITERATIONS / (avg * ITERATIONS / 1000))) ops/sec"
echo ""

# Search operations
stats=$(calculate_stats "${search_times[@]}")
read avg min max p95 <<< "$stats"
echo "Search Operations (GET /api/v1/prompts/search):"
echo "  Average:   ${avg}ms"
echo "  Min:       ${min}ms"
echo "  Max:       ${max}ms"
echo "  P95:       ${p95}ms"
echo "  Throughput: $((1000 * ITERATIONS / (avg * ITERATIONS / 1000))) ops/sec"
echo ""

echo "======================================"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo "======================================"
echo ""
echo "Summary:"
echo "  - Total operations: $((ITERATIONS * 3))"
echo "  - Test data created: $ITERATIONS prompts"
echo ""
echo "Note: Test data remains in HANA. Clean up with:"
echo "  DELETE FROM NUCLEUS.PROMPTS WHERE USER_ID = 'benchmark-user';"
