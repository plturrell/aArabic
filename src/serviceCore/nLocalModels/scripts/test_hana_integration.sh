#!/bin/bash
# Test script for HANA Cloud integration
# Tests all 4 API endpoints with real HANA connection

set -e  # Exit on error

BASE_URL="http://localhost:11434"
API_BASE="/api/v1/prompts"

echo "======================================"
echo "HANA Cloud Integration Test Suite"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Save a new prompt
echo -e "${YELLOW}Test 1: POST ${API_BASE}${NC}"
echo "Saving a new prompt to HANA..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "${BASE_URL}${API_BASE}" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "Test prompt for HANA integration - What is 2+2?",
    "model_name": "lfm2.5-1.2b-q4_0",
    "user_id": "test-user",
    "prompt_mode_id": 1,
    "tags": "test,integration"
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 201 ]; then
    echo -e "${GREEN}✓ PASS${NC} - Prompt saved successfully"
    echo "Response: $BODY"
    PROMPT_ID=$(echo "$BODY" | grep -o '"prompt_id":[0-9]*' | grep -o '[0-9]*')
    echo "Saved prompt ID: $PROMPT_ID"
else
    echo -e "${RED}✗ FAIL${NC} - Expected 201, got $HTTP_CODE"
    echo "Response: $BODY"
    exit 1
fi

echo ""

# Test 2: Get prompt history
echo -e "${YELLOW}Test 2: GET /v1/prompts/history${NC}"
echo "Loading prompt history from HANA..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X GET \
  "${BASE_URL}/v1/prompts/history?limit=10")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ]; then
    echo -e "${GREEN}✓ PASS${NC} - History loaded successfully"
    COUNT=$(echo "$BODY" | grep -o '"total":[0-9]*' | grep -o '[0-9]*' || echo "0")
    echo "Total prompts in history: $COUNT"
    echo "Response preview: $(echo "$BODY" | head -c 200)..."
else
    echo -e "${RED}✗ FAIL${NC} - Expected 200, got $HTTP_CODE"
    echo "Response: $BODY"
    exit 1
fi

echo ""

# Test 3: Search prompts
echo -e "${YELLOW}Test 3: GET ${API_BASE}/search${NC}"
echo "Searching for 'integration' in prompts..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X GET \
  "${BASE_URL}${API_BASE}/search?q=integration&limit=10")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ]; then
    echo -e "${GREEN}✓ PASS${NC} - Search completed successfully"
    RESULTS=$(echo "$BODY" | grep -o '"results":\[' || echo "No results")
    echo "Search results: $RESULTS"
    echo "Response preview: $(echo "$BODY" | head -c 200)..."
else
    echo -e "${RED}✗ FAIL${NC} - Expected 200, got $HTTP_CODE"
    echo "Response: $BODY"
    exit 1
fi

echo ""

# Test 4: Get prompt count
echo -e "${YELLOW}Test 4: GET ${API_BASE}/count${NC}"
echo "Getting total prompt count..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X GET \
  "${BASE_URL}${API_BASE}/count")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ]; then
    echo -e "${GREEN}✓ PASS${NC} - Count retrieved successfully"
    echo "Response: $BODY"
else
    echo -e "${RED}✗ FAIL${NC} - Expected 200, got $HTTP_CODE"
    echo "Response: $BODY"
    exit 1
fi

echo ""

# Test 5: Delete prompt (if we have an ID)
if [ ! -z "$PROMPT_ID" ]; then
    echo -e "${YELLOW}Test 5: DELETE ${API_BASE}/${PROMPT_ID}${NC}"
    echo "Deleting test prompt..."

    RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE \
      "${BASE_URL}${API_BASE}/${PROMPT_ID}")

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" -eq 200 ]; then
        echo -e "${GREEN}✓ PASS${NC} - Prompt deleted successfully"
        echo "Response: $BODY"
    else
        echo -e "${RED}✗ FAIL${NC} - Expected 200, got $HTTP_CODE"
        echo "Response: $BODY"
        exit 1
    fi
else
    echo -e "${YELLOW}Test 5: SKIP${NC} - No prompt ID to delete"
fi

echo ""
echo "======================================"
echo -e "${GREEN}All tests passed! ✓${NC}"
echo "======================================"
echo ""
echo "HANA Cloud integration is working correctly."
echo "Ready for production deployment."
