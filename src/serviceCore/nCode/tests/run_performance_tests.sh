#!/bin/bash
# nCode Performance Test Runner - Day 9
# 
# Automated performance testing suite covering:
# - SCIP parsing performance
# - Database loading performance
# - API endpoint response times
# - Concurrent request handling
#
# Usage:
#   ./tests/run_performance_tests.sh              # Basic benchmarks
#   ./tests/run_performance_tests.sh --large      # Include large-scale tests
#   ./tests/run_performance_tests.sh --profile    # Enable memory profiling

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🚀 nCode Performance Test Suite - Day 9${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Parse arguments
LARGE_PROJECT=false
ENABLE_PROFILE=false
SCIP_FILE="$SCRIPT_DIR/sample.scip"

while [[ $# -gt 0 ]]; do
    case $1 in
        --large)
            LARGE_PROJECT=true
            shift
            ;;
        --profile)
            ENABLE_PROFILE=true
            shift
            ;;
        --scip-file)
            SCIP_FILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check required Python packages
REQUIRED_PACKAGES=("requests" "psutil" "protobuf")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing Python packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}Installing missing packages...${NC}"
    pip3 install "${MISSING_PACKAGES[@]}"
fi
echo -e "${GREEN}✓ Python packages available${NC}"

# Check SCIP file
if [ ! -f "$SCIP_FILE" ]; then
    echo -e "${YELLOW}⚠️  SCIP file not found: $SCIP_FILE${NC}"
    echo -e "${YELLOW}Creating sample SCIP index...${NC}"
    
    # Try to create a sample index from the project itself
    if command -v scip-python &> /dev/null; then
        cd "$PROJECT_ROOT"
        scip-python index --project-name ncode-test --output "$SCIP_FILE" . 2>/dev/null || true
    fi
    
    if [ ! -f "$SCIP_FILE" ]; then
        echo -e "${RED}❌ Could not create SCIP file${NC}"
        echo -e "${YELLOW}💡 Please create a SCIP index first or specify --scip-file${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ SCIP file found: $SCIP_FILE${NC}"

# Check if nCode server is running
SERVER_RUNNING=false
if curl -s http://localhost:18003/health > /dev/null 2>&1; then
    SERVER_RUNNING=true
    echo -e "${GREEN}✓ nCode server is running${NC}"
else
    echo -e "${YELLOW}⚠️  nCode server not running${NC}"
    echo -e "${YELLOW}💡 API tests will be skipped. Start server with: cd $PROJECT_ROOT && zig build run${NC}"
fi

# Check databases
echo ""
echo -e "${YELLOW}📊 Checking database services...${NC}"

# Qdrant
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant is running${NC}"
else
    echo -e "${YELLOW}⚠️  Qdrant not running (vector search tests will be skipped)${NC}"
fi

# Memgraph
if curl -s http://localhost:7687 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Memgraph is running${NC}"
else
    echo -e "${YELLOW}⚠️  Memgraph not running (graph tests will be skipped)${NC}"
fi

# Marquez
if curl -s http://localhost:5000/api/v1/namespaces > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Marquez is running${NC}"
else
    echo -e "${YELLOW}⚠️  Marquez not running (lineage tests will be skipped)${NC}"
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🏃 Running Performance Benchmarks${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Build command
CMD="python3 $SCRIPT_DIR/performance_benchmark.py --scip-file $SCIP_FILE"

if [ "$LARGE_PROJECT" = true ]; then
    CMD="$CMD --large-project"
    echo -e "${YELLOW}📈 Including large-scale tests${NC}"
fi

if [ "$ENABLE_PROFILE" = true ]; then
    CMD="$CMD --profile"
    echo -e "${YELLOW}📊 Memory profiling enabled${NC}"
fi

echo ""
echo -e "${YELLOW}Executing: $CMD${NC}"
echo ""

# Run benchmarks
if $CMD; then
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}✅ Performance Benchmarks Complete${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    
    # Find the most recent report
    LATEST_REPORT=$(ls -t "$SCRIPT_DIR"/performance_report_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        echo -e "${GREEN}📄 Report saved: $LATEST_REPORT${NC}"
        
        # Extract key metrics
        if command -v jq &> /dev/null; then
            echo ""
            echo -e "${BLUE}📊 Key Metrics Summary:${NC}"
            echo -e "${BLUE}----------------------${NC}"
            
            PASSED=$(jq -r '.passed' "$LATEST_REPORT")
            FAILED=$(jq -r '.failed' "$LATEST_REPORT")
            AVG_DURATION=$(jq -r '.summary.avg_duration_ms' "$LATEST_REPORT")
            AVG_THROUGHPUT=$(jq -r '.summary.avg_throughput' "$LATEST_REPORT")
            
            echo -e "  Tests Passed:      ${GREEN}$PASSED${NC}"
            echo -e "  Tests Failed:      ${RED}$FAILED${NC}"
            echo -e "  Avg Duration:      ${YELLOW}${AVG_DURATION} ms${NC}"
            echo -e "  Avg Throughput:    ${YELLOW}${AVG_THROUGHPUT} items/sec${NC}"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}💡 Next Steps:${NC}"
    echo -e "  • Review the detailed report JSON file"
    echo -e "  • Compare with previous benchmark results"
    echo -e "  • Identify optimization opportunities"
    if [ "$SERVER_RUNNING" = false ]; then
        echo -e "  • Start nCode server for complete API testing"
    fi
    echo ""
    
    exit 0
else
    echo ""
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}❌ Performance Benchmarks Failed${NC}"
    echo -e "${RED}======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo -e "  • Check that all required services are running"
    echo -e "  • Verify SCIP file is valid: $SCIP_FILE"
    echo -e "  • Check Python dependencies: pip3 install -r requirements.txt"
    echo -e "  • Review logs for error details"
    echo ""
    exit 1
fi
