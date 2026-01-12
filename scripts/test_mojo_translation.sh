#!/bin/bash

echo "🧪 MOJO TRANSLATION SERVICE - TEST SUITE"
echo "=========================================="
echo ""

BASE_URL="http://localhost:8008"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

# Test function
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

# Test 1: Health Check
echo -e "${BLUE}1. Service Health${NC}"
run_test "Health endpoint" "curl -sf $BASE_URL/health | grep -q healthy"

# Test 2: Arabic to English Translation
echo ""
echo -e "${BLUE}2. Arabic → English Translation${NC}"

echo -n "  Testing basic translation... "
RESULT=$(curl -s -X POST $BASE_URL/translate \
    -H "Content-Type: application/json" \
    -d '{"text":"مرحبا","source_lang":"ar","target_lang":"en"}')

if echo "$RESULT" | grep -q "translated_text"; then
    TRANSLATION=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['translated_text'])" 2>/dev/null)
    echo -e "${GREEN}✅${NC}"
    echo "     Arabic: مرحبا"
    echo "     English: $TRANSLATION"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 3: English to Arabic Translation
echo ""
echo -e "${BLUE}3. English → Arabic Translation${NC}"

echo -n "  Testing reverse translation... "
RESULT=$(curl -s -X POST $BASE_URL/translate \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello","source_lang":"en","target_lang":"ar"}')

if echo "$RESULT" | grep -q "translated_text"; then
    TRANSLATION=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['translated_text'])" 2>/dev/null)
    echo -e "${GREEN}✅${NC}"
    echo "     English: Hello"
    echo "     Arabic: $TRANSLATION"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 4: Financial Text Translation
echo ""
echo -e "${BLUE}4. Financial Text Translation${NC}"

echo -n "  Testing financial domain... "
RESULT=$(curl -s -X POST $BASE_URL/translate \
    -H "Content-Type: application/json" \
    -d '{"text":"فاتورة مالية بمبلغ 5000 ريال","source_lang":"ar","target_lang":"en"}')

if echo "$RESULT" | grep -q "translated_text"; then
    TRANSLATION=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['translated_text'])" 2>/dev/null)
    SEMANTIC=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('semantic_score', 'N/A'))" 2>/dev/null)
    echo -e "${GREEN}✅${NC}"
    echo "     Arabic: فاتورة مالية بمبلغ 5000 ريال"
    echo "     English: $TRANSLATION"
    if [ "$SEMANTIC" != "N/A" ] && [ "$SEMANTIC" != "None" ]; then
        echo "     Semantic score: $SEMANTIC"
    fi
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 5: Batch Translation
echo ""
echo -e "${BLUE}5. Batch Translation${NC}"

echo -n "  Testing batch of 3 texts... "
RESULT=$(curl -s -X POST $BASE_URL/translate/batch \
    -H "Content-Type: application/json" \
    -d '{"texts":["فاتورة","دفع","تاريخ"],"source_lang":"ar","target_lang":"en"}')

if echo "$RESULT" | grep -q "translations"; then
    COUNT=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['count'])" 2>/dev/null)
    echo -e "${GREEN}✅${NC}"
    echo "     Translated $COUNT texts"
    
    # Show translations
    echo "$RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for t in data['translations']:
    print(f'     {t[\"index\"]+1}. {t[\"source\"]} → {t[\"translation\"]}')
" 2>/dev/null
    
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 6: Embedding Enhancement
echo ""
echo -e "${BLUE}6. Embedding Enhancement${NC}"

if curl -s http://localhost:8007/health > /dev/null 2>&1; then
    echo -n "  Testing semantic scoring... "
    RESULT=$(curl -s -X POST $BASE_URL/translate \
        -H "Content-Type: application/json" \
        -d '{"text":"مرحبا بك","source_lang":"ar","target_lang":"en","use_embeddings":true}')
    
    SEMANTIC=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('semantic_score', 'None'))" 2>/dev/null)
    
    if [ "$SEMANTIC" != "None" ] && [ "$SEMANTIC" != "null" ]; then
        echo -e "${GREEN}✅${NC}"
        echo "     Semantic score: $SEMANTIC"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌${NC}"
        echo "     No semantic score returned"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    echo "  ⚠️  Skipped (embedding service not running)"
fi

# Test 7: RAG Translation
echo ""
echo -e "${BLUE}7. RAG Translation Memory${NC}"

if curl -s http://localhost:6333/readyz > /dev/null 2>&1; then
    echo -n "  Testing RAG-enhanced translation... "
    RESULT=$(curl -s -X POST $BASE_URL/translate/with_rag \
        -H "Content-Type: application/json" \
        -d '{"text":"فاتورة جديدة","source_lang":"ar","target_lang":"en"}')
    
    if echo "$RESULT" | grep -q "method"; then
        echo -e "${GREEN}✅${NC}"
        METHOD=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('method', 'N/A'))" 2>/dev/null)
        echo "     Method: $METHOD"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    echo "  ⚠️  Skipped (Qdrant not running)"
fi

# Test 8: Models Endpoint
echo ""
echo -e "${BLUE}8. Models Information${NC}"

echo -n "  Testing models endpoint... "
RESULT=$(curl -s $BASE_URL/models)

if echo "$RESULT" | grep -q "Helsinki-NLP"; then
    echo -e "${GREEN}✅${NC}"
    echo "$RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('     Available models:')
for m in data['models']:
    print(f'       • {m[\"direction\"]}: {m[\"name\"]}')
" 2>/dev/null
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 9: Performance Check
echo ""
echo -e "${BLUE}9. Performance Metrics${NC}"

echo -n "  Testing translation speed... "
START=$(python3 -c "import time; print(time.time())")
RESULT=$(curl -s -X POST $BASE_URL/translate \
    -H "Content-Type: application/json" \
    -d '{"text":"اختبار السرعة","source_lang":"ar","target_lang":"en"}')
END=$(python3 -c "import time; print(time.time())")

ELAPSED=$(python3 -c "print(int(($END - $START) * 1000))")

if [ "$ELAPSED" -lt 5000 ]; then
    echo -e "${GREEN}✅${NC}"
    echo "     Total time: ${ELAPSED}ms"
    
    TRANSLATION_TIME=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('translation_time_ms', 'N/A'))" 2>/dev/null)
    if [ "$TRANSLATION_TIME" != "N/A" ]; then
        echo "     Translation only: ${TRANSLATION_TIME}ms"
    fi
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌${NC}"
    echo "     Too slow: ${ELAPSED}ms (expected < 5000ms)"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Summary
echo ""
echo "=========================================="
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED ($TESTS_PASSED/$((TESTS_PASSED + TESTS_FAILED)))${NC}"
    echo "=========================================="
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"
    echo "=========================================="
    exit 1
fi
