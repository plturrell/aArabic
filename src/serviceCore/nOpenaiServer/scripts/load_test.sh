#!/bin/bash
# Load Testing Script for Shimmy-Mojo HTTP Server
# Tests server performance with configurable parameters

set -e

# Default configuration
HOST="localhost"
PORT="11434"
DURATION="10"
CONCURRENCY="10"
TOOL=""
MODE="all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST       Target host (default: localhost)"
    echo "  -p, --port PORT       Target port (default: 11434)"
    echo "  -d, --duration SECS   Test duration in seconds (default: 10)"
    echo "  -c, --concurrency N   Number of concurrent connections (default: 10)"
    echo "  -t, --tool TOOL       Force tool: curl, hey, or wrk (default: auto-detect)"
    echo "  -m, --mode MODE       Test mode: health, concurrent, sustained, all (default: all)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Use defaults"
    echo "  $0 -c 50 -d 30               # 50 concurrent, 30 seconds"
    echo "  $0 -t hey                    # Use hey tool if available"
    echo "  $0 -m health -c 100          # Health check with 100 concurrent"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host) HOST="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        -d|--duration) DURATION="$2"; shift 2 ;;
        -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
        -t|--tool) TOOL="$2"; shift 2 ;;
        -m|--mode) MODE="$2"; shift 2 ;;
        --help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🔬 Load Testing Script for Shimmy-Mojo HTTP Server${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check for required tools
check_tools() {
    echo -e "${YELLOW}📦 Checking available tools...${NC}"
    
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}❌ curl is required but not installed${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} curl available"
    
    HAS_HEY=false
    HAS_WRK=false
    
    if command -v hey &> /dev/null; then
        HAS_HEY=true
        echo -e "  ${GREEN}✓${NC} hey available (recommended for load testing)"
    else
        echo -e "  ${YELLOW}○${NC} hey not available (install: go install github.com/rakyll/hey@latest)"
    fi
    
    if command -v wrk &> /dev/null; then
        HAS_WRK=true
        echo -e "  ${GREEN}✓${NC} wrk available"
    else
        echo -e "  ${YELLOW}○${NC} wrk not available (install: brew install wrk)"
    fi
    
    # Auto-select tool if not specified
    if [ -z "$TOOL" ]; then
        if $HAS_HEY; then
            TOOL="hey"
        elif $HAS_WRK; then
            TOOL="wrk"
        else
            TOOL="curl"
        fi
    fi
    
    echo -e "  ${BLUE}→${NC} Using tool: ${TOOL}"
    echo ""
}

# Check server is running
check_server() {
    echo -e "${YELLOW}🔍 Checking server availability...${NC}"
    if curl -s --connect-timeout 5 "${BASE_URL}/" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Server is responding at ${BASE_URL}"
        return 0
    else
        echo -e "  ${RED}❌${NC} Server not responding at ${BASE_URL}"
        echo -e "  ${YELLOW}→${NC} Start server with: ./scripts/start_server.sh"
        exit 1
    fi
}

print_config() {
    echo -e "${YELLOW}⚙️  Test Configuration:${NC}"
    echo -e "   Host:        ${HOST}"
    echo -e "   Port:        ${PORT}"
    echo -e "   Duration:    ${DURATION}s"
    echo -e "   Concurrency: ${CONCURRENCY}"
    echo -e "   Tool:        ${TOOL}"
    echo -e "   Mode:        ${MODE}"
    echo ""
}

# Run health check load test with curl
run_curl_health_test() {
    local endpoint="$1"
    local requests="$2"
    local concurrent="$3"
    
    echo -e "${BLUE}Running curl-based load test (${requests} requests, ${concurrent} concurrent)...${NC}"
    
    local success=0
    local failed=0
    local total_time=0
    local start_time=$(date +%s.%N)
    
    # Create temp files for results
    local results_file=$(mktemp)
    
    # Run concurrent requests using background jobs
    for ((i=1; i<=requests; i++)); do
        (
            local req_start=$(date +%s.%N)
            local http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")
            local req_end=$(date +%s.%N)
            local req_time=$(echo "$req_end - $req_start" | bc)
            echo "${http_code} ${req_time}" >> "$results_file"
        ) &
        
        # Limit concurrent jobs
        if (( i % concurrent == 0 )); then
            wait
        fi
    done
    wait
    
    local end_time=$(date +%s.%N)
    local total_duration=$(echo "$end_time - $start_time" | bc)
    
    # Parse results
    while read -r code time; do
        if [[ "$code" =~ ^2[0-9][0-9]$ ]]; then
            ((success++))
            total_time=$(echo "$total_time + $time" | bc)
        else
            ((failed++))
        fi
    done < "$results_file"
    
    rm -f "$results_file"
    
    local total=$((success + failed))
    local rps=$(echo "scale=2; $total / $total_duration" | bc)
    local avg_latency="N/A"
    if [ "$success" -gt 0 ]; then
        avg_latency=$(echo "scale=3; $total_time / $success * 1000" | bc)
        avg_latency="${avg_latency}ms"
    fi
    local error_rate=$(echo "scale=2; $failed * 100 / $total" | bc)
    
    echo ""
    echo -e "${GREEN}📊 Results:${NC}"
    echo -e "   Total Requests:     ${total}"
    echo -e "   Successful:         ${success}"
    echo -e "   Failed:             ${failed}"
    echo -e "   Duration:           ${total_duration}s"
    echo -e "   Requests/sec:       ${rps}"
    echo -e "   Avg Latency:        ${avg_latency}"
    echo -e "   Error Rate:         ${error_rate}%"
}

run_hey_test() {
    local endpoint="$1"
    local duration="$2"
    local concurrent="$3"
    local name="$4"

    echo -e "${BLUE}Running hey load test: ${name}${NC}"
    echo -e "   Endpoint: ${BASE_URL}${endpoint}"
    echo ""

    hey -z "${duration}s" -c "$concurrent" "${BASE_URL}${endpoint}" 2>&1 | tee /tmp/hey_results.txt

    # Extract key metrics
    local rps=$(grep "Requests/sec:" /tmp/hey_results.txt | awk '{print $2}')
    local avg_latency=$(grep "Average:" /tmp/hey_results.txt | head -1 | awk '{print $2}')
    local total=$(grep "Total:" /tmp/hey_results.txt | head -1 | awk '{print $2}')
    local status_200=$(grep "\[200\]" /tmp/hey_results.txt | awk '{print $2}')
    local errors=$(grep "Error distribution:" -A 100 /tmp/hey_results.txt | grep -v "Error distribution:" | head -5)

    echo ""
    echo -e "${GREEN}📊 Summary - ${name}:${NC}"
    echo -e "   Requests/sec:  ${rps:-N/A}"
    echo -e "   Avg Latency:   ${avg_latency:-N/A}"
    echo ""
}

run_wrk_test() {
    local endpoint="$1"
    local duration="$2"
    local concurrent="$3"
    local name="$4"

    echo -e "${BLUE}Running wrk load test: ${name}${NC}"
    echo -e "   Endpoint: ${BASE_URL}${endpoint}"
    echo ""

    wrk -t2 -c"$concurrent" -d"${duration}s" "${BASE_URL}${endpoint}" 2>&1 | tee /tmp/wrk_results.txt

    echo ""
}

# Health check test
run_health_test() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🏥 TEST 1: Health Check Load Test${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    case $TOOL in
        hey)
            run_hey_test "/" "$DURATION" "$CONCURRENCY" "Health Check"
            ;;
        wrk)
            run_wrk_test "/" "$DURATION" "$CONCURRENCY" "Health Check"
            ;;
        curl)
            local requests=$((DURATION * 10))
            run_curl_health_test "/" "$requests" "$CONCURRENCY"
            ;;
    esac
}

# Concurrent connections test
run_concurrent_test() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🔗 TEST 2: Concurrent Connections Test${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local levels=("10" "50" "100" "200")
    local short_duration=$((DURATION / 4))
    [ "$short_duration" -lt 2 ] && short_duration=2

    for level in "${levels[@]}"; do
        if [ "$level" -gt "$CONCURRENCY" ]; then
            continue
        fi
        echo -e "${BLUE}Testing with ${level} concurrent connections...${NC}"
        case $TOOL in
            hey)
                hey -z "${short_duration}s" -c "$level" "${BASE_URL}/v1/models" 2>&1 | grep -E "(Requests/sec|Average:)" | head -2
                ;;
            wrk)
                wrk -t2 -c"$level" -d"${short_duration}s" "${BASE_URL}/v1/models" 2>&1 | grep -E "(Req/Sec|Latency)"
                ;;
            curl)
                local requests=$((short_duration * 5))
                run_curl_health_test "/v1/models" "$requests" "$level"
                ;;
        esac
        echo ""
    done
}

# Sustained throughput test
run_sustained_test() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}📈 TEST 3: Sustained Throughput Test${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    case $TOOL in
        hey)
            run_hey_test "/v1/models" "$DURATION" "$CONCURRENCY" "Sustained Throughput"
            ;;
        wrk)
            run_wrk_test "/v1/models" "$DURATION" "$CONCURRENCY" "Sustained Throughput"
            ;;
        curl)
            local requests=$((DURATION * 20))
            run_curl_health_test "/v1/models" "$requests" "$CONCURRENCY"
            ;;
    esac
}

# Main execution
main() {
    check_tools
    check_server
    print_config

    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}🚀 Starting Load Tests${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""

    case $MODE in
        health)
            run_health_test
            ;;
        concurrent)
            run_concurrent_test
            ;;
        sustained)
            run_sustained_test
            ;;
        all)
            run_health_test
            echo ""
            run_concurrent_test
            echo ""
            run_sustained_test
            ;;
        *)
            echo -e "${RED}Unknown mode: ${MODE}${NC}"
            print_usage
            exit 1
            ;;
    esac

    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${GREEN}✅ Load testing complete${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

main

