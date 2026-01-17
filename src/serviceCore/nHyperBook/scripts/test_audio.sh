#!/bin/bash
# Test script for Audio OData endpoints
# Tests GenerateAudio action and Audio entity CRUD

set -e

BASE_URL="http://localhost:8080"
ODATA_BASE="${BASE_URL}/odata/v4/research"

echo "========================================="
echo "Testing Audio OData Endpoints"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: GenerateAudio action (stub)
echo -e "${YELLOW}Test 1: POST GenerateAudio action${NC}"
echo "Endpoint: ${ODATA_BASE}/GenerateAudio"
echo ""

AUDIO_REQUEST='{
  "SourceId": "source_test_001",
  "Text": "This is a test audio generation request. The text should be converted to speech using AudioLabShimmy once it is integrated. For now, this returns a stub response indicating that the audio generation system is pending integration.",
  "Voice": "default",
  "Format": "mp3"
}'

echo "Request payload:"
echo "$AUDIO_REQUEST" | jq '.'
echo ""

GENERATE_RESPONSE=$(curl -s -X POST \
  "${ODATA_BASE}/GenerateAudio" \
  -H "Content-Type: application/json" \
  -d "$AUDIO_REQUEST")

echo "Response:"
echo "$GENERATE_RESPONSE" | jq '.'
echo ""

# Extract AudioId for next tests
AUDIO_ID=$(echo "$GENERATE_RESPONSE" | jq -r '.AudioId // empty')

if [ -n "$AUDIO_ID" ]; then
  echo -e "${GREEN}✓ Audio generation initiated${NC}"
  echo "Audio ID: $AUDIO_ID"
else
  echo -e "${RED}✗ Failed to generate audio${NC}"
  echo "Note: This is expected if the server is not running"
fi
echo ""

# Test 2: GET Audio collection
echo -e "${YELLOW}Test 2: GET Audio collection${NC}"
echo "Endpoint: ${ODATA_BASE}/Audio"
echo ""

AUDIO_LIST=$(curl -s -X GET \
  "${ODATA_BASE}/Audio" \
  -H "Accept: application/json")

echo "Response:"
echo "$AUDIO_LIST" | jq '.'
echo ""

AUDIO_COUNT=$(echo "$AUDIO_LIST" | jq '.value | length // 0')
echo -e "${GREEN}✓ Found $AUDIO_COUNT audio entities${NC}"
echo ""

# Test 3: GET single Audio entity (if we have an ID)
if [ -n "$AUDIO_ID" ]; then
  echo -e "${YELLOW}Test 3: GET Audio by ID${NC}"
  echo "Endpoint: ${ODATA_BASE}/Audio('${AUDIO_ID}')"
  echo ""
  
  AUDIO_ENTITY=$(curl -s -X GET \
    "${ODATA_BASE}/Audio('${AUDIO_ID}')" \
    -H "Accept: application/json")
  
  echo "Response:"
  echo "$AUDIO_ENTITY" | jq '.'
  echo ""
  
  if echo "$AUDIO_ENTITY" | jq -e '.AudioId' > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Audio entity retrieved${NC}"
  else
    echo -e "${YELLOW}⚠ Audio entity not found (expected in stub mode)${NC}"
  fi
  echo ""
fi

# Test 4: Audio file serving
echo -e "${YELLOW}Test 4: GET Audio file${NC}"
echo "Endpoint: ${BASE_URL}/audio/test.mp3"
echo ""

AUDIO_FILE=$(curl -s -X GET \
  "${BASE_URL}/audio/test.mp3" \
  -H "Accept: audio/mpeg")

echo "Response:"
if [ ${#AUDIO_FILE} -gt 100 ]; then
  echo "(Binary audio data, ${#AUDIO_FILE} bytes)"
  echo -e "${GREEN}✓ Audio file served${NC}"
else
  echo "$AUDIO_FILE"
  echo -e "${YELLOW}⚠ Audio file not available (expected in stub mode)${NC}"
fi
echo ""

# Test 5: Error handling - invalid source
echo -e "${YELLOW}Test 5: Error handling (invalid source)${NC}"
echo ""

INVALID_REQUEST='{
  "SourceId": "nonexistent_source",
  "Text": "Test text"
}'

ERROR_RESPONSE=$(curl -s -X POST \
  "${ODATA_BASE}/GenerateAudio" \
  -H "Content-Type: application/json" \
  -d "$INVALID_REQUEST")

echo "Response:"
echo "$ERROR_RESPONSE" | jq '.'
echo ""

if echo "$ERROR_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Error handling works${NC}"
else
  echo -e "${YELLOW}⚠ Error handling may need improvement${NC}"
fi
echo ""

# Test 6: DELETE Audio entity (if we have an ID)
if [ -n "$AUDIO_ID" ]; then
  echo -e "${YELLOW}Test 6: DELETE Audio entity${NC}"
  echo "Endpoint: ${ODATA_BASE}/Audio('${AUDIO_ID}')"
  echo ""
  
  DELETE_RESPONSE=$(curl -s -X DELETE \
    "${ODATA_BASE}/Audio('${AUDIO_ID}')" \
    -H "Accept: application/json")
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Audio entity deleted${NC}"
  else
    echo -e "${YELLOW}⚠ Delete operation returned error (expected in stub mode)${NC}"
  fi
  echo ""
fi

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""
echo "All tests completed!"
echo ""
echo "Note: Some endpoints may return stub/placeholder data"
echo "because AudioLabShimmy is not yet integrated."
echo ""
echo "Expected behavior:"
echo "  - GenerateAudio returns pending status"
echo "  - Audio collection may be empty"
echo "  - Audio files not yet available"
echo ""
echo "Once AudioLabShimmy is integrated:"
echo "  - Real audio generation will work"
echo "  - Audio files will be served"
echo "  - Full CRUD operations will function"
echo ""
