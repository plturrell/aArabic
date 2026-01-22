#!/bin/bash
# Generate test JWT tokens for authentication testing (Day 14)
# Generates valid HS256-signed tokens that the Zig backend can verify

set -e

# Configuration
JWT_SECRET="${JWT_SECRET:-test-secret-key-12345}"
EXPIRATION_HOURS="${JWT_EXPIRATION_HOURS:-24}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔐 JWT Token Generator for nOpenai Server"
echo "=========================================="
echo ""

# Check for required tools
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Get user ID (parameter or default)
USER_ID="${1:-demo-user-$(date +%s)}"

# Create Python script for JWT generation
cat > /tmp/generate_jwt.py << 'PYTHON_SCRIPT'
import hmac
import hashlib
import base64
import json
import sys
import time

def base64url_encode(data):
    """Base64 URL-safe encode"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    encoded = base64.urlsafe_b64encode(data)
    return encoded.rstrip(b'=').decode('utf-8')

def generate_jwt(user_id, secret, expiration_hours=24):
    """Generate a valid HS256-signed JWT token"""
    
    # Header
    header = {
        "alg": "HS256",
        "typ": "JWT"
    }
    
    # Payload
    now = int(time.time())
    payload = {
        "user_id": user_id,
        "iat": now,
        "exp": now + (expiration_hours * 3600)
    }
    
    # Encode header and payload
    header_encoded = base64url_encode(json.dumps(header, separators=(',', ':')))
    payload_encoded = base64url_encode(json.dumps(payload, separators=(',', ':')))
    
    # Create signature
    message = f"{header_encoded}.{payload_encoded}".encode('utf-8')
    signature = hmac.new(
        secret.encode('utf-8'),
        message,
        hashlib.sha256
    ).digest()
    signature_encoded = base64url_encode(signature)
    
    # Combine parts
    token = f"{header_encoded}.{payload_encoded}.{signature_encoded}"
    
    return token, payload

if __name__ == "__main__":
    user_id = sys.argv[1]
    secret = sys.argv[2]
    expiration_hours = int(sys.argv[3])
    
    token, payload = generate_jwt(user_id, secret, expiration_hours)
    
    print("TOKEN:", token)
    print("USER_ID:", payload['user_id'])
    print("ISSUED_AT:", payload['iat'])
    print("EXPIRES_AT:", payload['exp'])
    print("EXPIRES_IN:", expiration_hours, "hours")

PYTHON_SCRIPT

# Generate token
echo -e "${YELLOW}Generating JWT for user: ${USER_ID}${NC}"
OUTPUT=$(python3 /tmp/generate_jwt.py "$USER_ID" "$JWT_SECRET" "$EXPIRATION_HOURS")

# Parse output
TOKEN=$(echo "$OUTPUT" | grep "TOKEN:" | cut -d' ' -f2)
ISSUED_AT=$(echo "$OUTPUT" | grep "ISSUED_AT:" | cut -d' ' -f2)
EXPIRES_AT=$(echo "$OUTPUT" | grep "EXPIRES_AT:" | cut -d' ' -f2)

echo ""
echo -e "${GREEN}✓ Token generated successfully!${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "User ID:       $USER_ID"
echo "Issued At:     $(date -r $ISSUED_AT 2>/dev/null || date -d @$ISSUED_AT 2>/dev/null || echo $ISSUED_AT)"
echo "Expires At:    $(date -r $EXPIRES_AT 2>/dev/null || date -d @$EXPIRES_AT 2>/dev/null || echo $EXPIRES_AT)"
echo "Lifetime:      $EXPIRATION_HOURS hours"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Token:"
echo "$TOKEN"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Save to file
TOKEN_FILE="./test_jwt_token.txt"
echo "$TOKEN" > "$TOKEN_FILE"
echo -e "${GREEN}✓ Token saved to: ${TOKEN_FILE}${NC}"
echo ""

# Test with API
echo -e "${YELLOW}Testing authentication with API...${NC}"
echo ""

# Test 1: Save prompt with authentication
echo "Test 1: Saving prompt with JWT token"
RESPONSE=$(curl -s -X POST http://localhost:11434/api/v1/prompts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt_text": "Test prompt with JWT authentication",
    "model_name": "lfm2.5-1.2b-q4_0",
    "prompt_mode_id": 1,
    "tags": "test,jwt"
  }')

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Check if user_id matches
if echo "$RESPONSE" | grep -q "\"user_id\":\"$USER_ID\""; then
    echo -e "${GREEN}✓ Authentication successful! User ID matches.${NC}"
else
    echo -e "${RED}✗ Authentication failed. User ID doesn't match.${NC}"
fi
echo ""

# Test 2: Anonymous request (no token)
echo "Test 2: Saving prompt without JWT token (anonymous)"
RESPONSE_ANON=$(curl -s -X POST http://localhost:11434/api/v1/prompts \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "Test prompt without authentication",
    "model_name": "lfm2.5-1.2b-q4_0",
    "prompt_mode_id": 1,
    "tags": "test,anonymous"
  }')

echo "$RESPONSE_ANON" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_ANON"
echo ""

if echo "$RESPONSE_ANON" | grep -q "\"user_id\":\"anonymous\""; then
    echo -e "${GREEN}✓ Anonymous mode working correctly.${NC}"
else
    echo -e "${RED}✗ Anonymous mode failed.${NC}"
fi
echo ""

# Usage instructions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Usage Instructions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Use token in curl:"
echo "   curl -H \"Authorization: Bearer \$(cat test_jwt_token.txt)\" \\"
echo "        http://localhost:11434/api/v1/prompts"
echo ""
echo "2. Use token in browser console:"
echo "   sap.ui.require([\"llm/server/dashboard/utils/TokenManager\"], function(TM) {"
echo "       TM.setToken('$TOKEN', true);"
echo "       console.log('User:', TM.getCurrentUser());"
echo "   });"
echo ""
echo "3. Generate new token with custom user:"
echo "   ./scripts/generate_test_jwt.sh user@example.com"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Cleanup
rm -f /tmp/generate_jwt.py

exit 0
