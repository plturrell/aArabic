#!/bin/bash

echo "🧪 DEPLOYMENT VERIFICATION TEST"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

EMBEDDING_PORT=${EMBEDDING_PORT:-8007}
REDIS_PORT=${REDIS_PORT:-6379}
QDRANT_PORT=${QDRANT_PORT:-6333}

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "  Testing $test_name... "
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Service Health
echo -e "${BLUE}1. Health Checks${NC}"
run_test "Embedding service" "curl -sf http://localhost:$EMBEDDING_PORT/health | grep -q healthy"
run_test "Redis cache" "docker exec redis-cache redis-cli ping | grep -q PONG"
run_test "Qdrant vector DB" "curl -sf http://localhost:$QDRANT_PORT/readyz"

# Test 2: Embedding Generation
echo ""
echo -e "${BLUE}2. Embedding Generation${NC}"
run_test "Single embedding" "curl -sf -X POST http://localhost:$EMBEDDING_PORT/embed/single -H 'Content-Type: application/json' -d '{\"text\":\"test\"}' | grep -q embedding"
run_test "Batch embedding" "curl -sf -X POST http://localhost:$EMBEDDING_PORT/embed/batch -H 'Content-Type: application/json' -d '{\"texts\":[\"test1\",\"test2\"]}' | grep -q embeddings"

# Test 3: Model Support
echo ""
echo -e "${BLUE}3. Model Support${NC}"
run_test "General model (384d)" "curl -sf -X POST http://localhost:$EMBEDDING_PORT/embed/single -H 'Content-Type: application/json' -d '{\"text\":\"test\",\"model_type\":\"general\"}' | grep -q '\"dimensions\": 384'"
run_test "Financial model (768d)" "curl -sf -X POST http://localhost:$EMBEDDING_PORT/embed/single -H 'Content-Type: application/json' -d '{\"text\":\"test\",\"model_type\":\"financial\"}' | grep -q '\"dimensions\": 768'"

# Test 4: Cache Functionality
echo ""
echo -e "${BLUE}4. Cache Performance${NC}"
echo -n "  Testing cache speedup... "

# First call (uncached)
TIME1=$(curl -s -X POST http://localhost:$EMBEDDING_PORT/embed/single \
    -H "Content-Type: application/json" \
    -d '{"text":"cache_test_unique_12345"}' | \
    python3 -c "import json,sys; print(json.load(sys.stdin)['processing_time_ms'])" 2>/dev/null)

# Second call (should be cached)
TIME2=$(curl -s -X POST http://localhost:$EMBEDDING_PORT/embed/single \
    -H "Content-Type: application/json" \
    -d '{"text":"cache_test_unique_12345"}' | \
    python3 -c "import json,sys; print(json.load(sys.stdin)['processing_time_ms'])" 2>/dev/null)

if [ ! -z "$TIME1" ] && [ ! -z "$TIME2" ]; then
    SPEEDUP=$(python3 -c "print(int(float($TIME1) / float($TIME2)))" 2>/dev/null)
    if [ "$SPEEDUP" -gt 10 ]; then
        echo -e "${GREEN}✅ ${SPEEDUP}x speedup${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌ Only ${SPEEDUP}x speedup${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    echo -e "${RED}❌ Failed to measure${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 5: Metrics
echo ""
echo -e "${BLUE}5. Metrics & Monitoring${NC}"
run_test "Metrics endpoint" "curl -sf http://localhost:$EMBEDDING_PORT/metrics | grep -q requests_total"
run_test "Cache type" "curl -sf http://localhost:$EMBEDDING_PORT/metrics | grep -q 'Redis'"

# Summary
echo ""
echo "======================================"
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED ($TESTS_PASSED/$((TESTS_PASSED + TESTS_FAILED)))${NC}"
    echo "======================================"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"
    echo "======================================"
    exit 1
fi
