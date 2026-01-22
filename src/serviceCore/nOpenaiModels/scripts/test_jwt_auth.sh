#!/bin/bash
# Comprehensive JWT Authentication Test Suite (Day 14)
# Tests the complete authentication flow with the Zig backend

set -e

# Configuration
JWT_SECRET="${JWT_SECRET:-test-secret-key-12345}"
API_BASE="${API_BASE:-http://localhost:11434}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

echo "🧪 JWT Authentication Test Suite"
echo "================================="
echo ""
echo "Configuration:"
echo "  API Base: $API_BASE"
echo "  JWT Secret: ${JWT_SECRET:0:10}..."
echo ""

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}▶ Test: $test_name${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Generate test JWT token
echo -e "${YELLOW}Generating test JWT token...${NC}"
TEST_USER="test-user-$(date +%s)"

# Generate token using Python (inline)
TEST_TOKEN=$(python3 - "$TEST_USER" "$JWT_SECRET" <<'PYTHON'
import hmac, hashlib, base64, json, sys, time

def base64url_encode(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

user_id, secret = sys.argv[1], sys.argv[2]
now = int(time.time())
header = base64url_encode(json.dumps({"alg":"HS256","typ":"JWT"}, separators=(',',':')))
payload = base64url_encode(json.dumps({"user_id":user_id,"iat":now,"exp":now+86400}, separators=(',',':')))
signature = base64url_encode(hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
print(f"{header}.{payload}.{signature}")
PYTHON
)

echo -e "${GREEN}✓ Token generated for user: $TEST_USER${NC}"
echo "Token: ${TEST_TOKEN:0:50}..."
echo ""

# Test 1: Health check
run_test "Health Check" "
    curl -sf $API_BASE/health >/dev/null 2>&1
"

# Test 2: Authenticated prompt save
run_test "Authenticated Prompt Save" "
    RESPONSE=\$(curl -s -X POST $API_BASE/api/v1/prompts \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer $TEST_TOKEN' \
      -d '{
        \"prompt_text\": \"Test authenticated prompt\",
        \"model_name\": \"test-model\",
        \"prompt_mode_id\": 1,
        \"tags\": \"test\"
      }')
    
    echo \"\$RESPONSE\" | grep -q '\"user_id\":\"$TEST_USER\"'
"

# Test 3: Anonymous prompt save
run_test "Anonymous Prompt Save" "
    RESPONSE=\$(curl -s -X POST $API_BASE/api/v1/prompts \
      -H 'Content-Type: application/json' \
      -d '{
        \"prompt_text\": \"Test anonymous prompt\",
        \"model_name\": \"test-model\",
        \"prompt_mode_id\": 1
      }')
    
    echo \"\$RESPONSE\" | grep -q '\"user_id\":\"anonymous\"'
"

# Test 4: Invalid token (should fall back to anonymous)
run_test "Invalid Token Fallback" "
    RESPONSE=\$(curl -s -X POST $API_BASE/api/v1/prompts \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer invalid-token-12345' \
      -d '{
        \"prompt_text\": \"Test with invalid token\",
        \"model_name\": \"test-model\",
        \"prompt_mode_id\": 1
      }')
    
    echo \"\$RESPONSE\" | grep -q '\"user_id\":\"anonymous\"'
"

# Test 5: Expired token (should fall back to anonymous)
echo -e "${BLUE}▶ Test: Expired Token Fallback${NC}"

# Generate expired token
EXPIRED_TOKEN=$(python3 - "$TEST_USER" "$JWT_SECRET" <<'PYTHON'
import hmac, hashlib, base64, json, sys, time

def base64url_encode(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

user_id, secret = sys.argv[1], sys.argv[2]
now = int(time.time())
header = base64url_encode(json.dumps({"alg":"HS256","typ":"JWT"}, separators=(',',':')))
# Token expired 1 hour ago
payload = base64url_encode(json.dumps({"user_id":user_id,"iat":now-7200,"exp":now-3600}, separators=(',',':')))
signature = base64url_encode(hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
print(f"{header}.{payload}.{signature}")
PYTHON
)

RESPONSE=$(curl -s -X POST $API_BASE/api/v1/prompts \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $EXPIRED_TOKEN" \
  -d '{
    "prompt_text": "Test with expired token",
    "model_name": "test-model",
    "prompt_mode_id": 1
  }')

if echo "$RESPONSE" | grep -q '"user_id":"anonymous"'; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi
echo ""

# Test 6: Multiple users
echo -e "${BLUE}▶ Test: Multiple User Differentiation${NC}"

# Generate tokens for two different users
USER1="user1-$(date +%s)"
USER2="user2-$(date +%s)"

TOKEN1=$(python3 - "$USER1" "$JWT_SECRET" <<'PYTHON'
import hmac, hashlib, base64, json, sys, time
def base64url_encode(data):
    if isinstance(data, str): data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')
user_id, secret = sys.argv[1], sys.argv[2]
now = int(time.time())
header = base64url_encode(json.dumps({"alg":"HS256","typ":"JWT"}, separators=(',',':')))
payload = base64url_encode(json.dumps({"user_id":user_id,"iat":now,"exp":now+86400}, separators=(',',':')))
signature = base64url_encode(hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
print(f"{header}.{payload}.{signature}")
PYTHON
)

TOKEN2=$(python3 - "$USER2" "$JWT_SECRET" <<'PYTHON'
import hmac, hashlib, base64, json, sys, time
def base64url_encode(data):
    if isinstance(data, str): data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')
user_id, secret = sys.argv[1], sys.argv[2]
now = int(time.time())
header = base64url_encode(json.dumps({"alg":"HS256","typ":"JWT"}, separators=(',',':')))
payload = base64url_encode(json.dumps({"user_id":user_id,"iat":now,"exp":now+86400}, separators=(',',':')))
signature = base64url_encode(hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest())
print(f"{header}.{payload}.{signature}")
PYTHON
)

# Save prompts with both tokens
RESP1=$(curl -s -X POST $API_BASE/api/v1/prompts -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN1" -d '{"prompt_text":"User 1 prompt","model_name":"test","prompt_mode_id":1}')
RESP2=$(curl -s -X POST $API_BASE/api/v1/prompts -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN2" -d '{"prompt_text":"User 2 prompt","model_name":"test","prompt_mode_id":1}')

if echo "$RESP1" | grep -q "\"user_id\":\"$USER1\"" && echo "$RESP2" | grep -q "\"user_id\":\"$USER2\""; then
    echo -e "${GREEN}✓ PASS - Users correctly differentiated${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL - User differentiation failed${NC}"
    ((TESTS_FAILED++))
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total Tests:  $((TESTS_PASSED + TESTS_FAILED))"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed.${NC}"
    exit 1
fi
