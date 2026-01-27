#!/bin/bash
# ==============================================================================
# SAP HANA Cloud & AI Core Connection Test Script
# ==============================================================================
# 
# Tests connectivity to:
# 1. SAP HANA Cloud (SQL API)
# 2. SAP AI Core (OAuth + Model deployment)
#
# Usage: ./test_connections.sh
# Requires: curl, jq (optional), source ../.env

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       SAP HANA Cloud & AI Core Connection Test                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Load environment variables from .env file
ENV_FILE="$(dirname "$0")/../../../../.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from: $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    # Try root .env
    ENV_FILE="$(dirname "$0")/../../../../../.env"
    if [ -f "$ENV_FILE" ]; then
        echo "Loading environment from: $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo -e "${RED}❌ No .env file found${NC}"
        echo "Please create .env with HANA and AI Core credentials"
        exit 1
    fi
fi

echo ""
echo "═══ Test 1: Environment Variables ═══"
echo ""

# Check HANA variables
if [ -n "$HANA_HOST" ]; then
    echo -e "${GREEN}✅ HANA_HOST:${NC} ${HANA_HOST:0:40}..."
else
    echo -e "${RED}❌ HANA_HOST not set${NC}"
fi

if [ -n "$HANA_USER" ]; then
    echo -e "${GREEN}✅ HANA_USER:${NC} ${HANA_USER:0:30}..."
else
    echo -e "${RED}❌ HANA_USER not set${NC}"
fi

if [ -n "$HANA_PASSWORD" ]; then
    echo -e "${GREEN}✅ HANA_PASSWORD:${NC} [set - hidden]"
else
    echo -e "${RED}❌ HANA_PASSWORD not set${NC}"
fi

if [ -n "$HANA_DATABASE" ]; then
    echo -e "${GREEN}✅ HANA_DATABASE:${NC} ${HANA_DATABASE}"
else
    echo -e "${YELLOW}⚠️  HANA_DATABASE not set (using default)${NC}"
fi

# Check AI Core variables
if [ -n "$AICORE_CLIENT_ID" ]; then
    echo -e "${GREEN}✅ AICORE_CLIENT_ID:${NC} ${AICORE_CLIENT_ID:0:30}..."
else
    echo -e "${RED}❌ AICORE_CLIENT_ID not set${NC}"
fi

if [ -n "$AICORE_CLIENT_SECRET" ]; then
    echo -e "${GREEN}✅ AICORE_CLIENT_SECRET:${NC} [set - hidden]"
else
    echo -e "${RED}❌ AICORE_CLIENT_SECRET not set${NC}"
fi

if [ -n "$AICORE_AUTH_URL" ]; then
    echo -e "${GREEN}✅ AICORE_AUTH_URL:${NC} ${AICORE_AUTH_URL}"
else
    echo -e "${RED}❌ AICORE_AUTH_URL not set${NC}"
fi

if [ -n "$AICORE_BASE_URL" ]; then
    echo -e "${GREEN}✅ AICORE_BASE_URL:${NC} ${AICORE_BASE_URL}"
else
    echo -e "${RED}❌ AICORE_BASE_URL not set${NC}"
fi

echo ""
echo "═══ Test 2: SAP HANA Cloud Connection ═══"
echo ""

if [ -n "$HANA_HOST" ] && [ -n "$HANA_USER" ] && [ -n "$HANA_PASSWORD" ]; then
    # Build HANA URL
    HANA_URL="https://${HANA_HOST}:${HANA_PORT:-443}"
    
    # Add database parameter if set
    if [ -n "$HANA_DATABASE" ]; then
        HANA_SQL_URL="${HANA_URL}/sql?databaseName=${HANA_DATABASE}"
    else
        HANA_SQL_URL="${HANA_URL}/sql"
    fi
    
    echo "Testing: $HANA_SQL_URL"
    echo "Query: SELECT 1 FROM DUMMY"
    echo ""
    
    # Create Basic Auth header
    AUTH_STRING=$(echo -n "${HANA_USER}:${HANA_PASSWORD}" | base64)
    
    # Execute test query
    RESPONSE=$(curl -s -X POST "$HANA_SQL_URL" \
        -H "Authorization: Basic $AUTH_STRING" \
        -H "Content-Type: application/json" \
        -d '{"statements": ["SELECT 1 AS TEST FROM DUMMY"]}' \
        --connect-timeout 10 \
        -w "\n%{http_code}" 2>&1) || true
    
    # Extract HTTP status code
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✅ HANA connection successful!${NC}"
        echo "Response: $BODY"
    elif [ "$HTTP_CODE" = "401" ]; then
        echo -e "${RED}❌ HANA authentication failed (401)${NC}"
        echo "Check your HANA_USER and HANA_PASSWORD"
    elif [ "$HTTP_CODE" = "404" ]; then
        echo -e "${YELLOW}⚠️  HANA SQL API endpoint not found (404)${NC}"
        echo "The SQL API may not be enabled. Trying alternative endpoint..."
        
        # Try alternative - basic HTTPS connection
        ALT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "https://${HANA_HOST}:${HANA_PORT:-443}" --connect-timeout 5) || true
        if [ "$ALT_RESPONSE" = "000" ]; then
            echo -e "${RED}❌ Cannot reach HANA host${NC}"
        else
            echo -e "${GREEN}✅ HANA host reachable (HTTP $ALT_RESPONSE)${NC}"
        fi
    elif [ "$HTTP_CODE" = "000" ]; then
        echo -e "${RED}❌ Cannot connect to HANA (network error)${NC}"
        echo "Check if the host is reachable and firewall allows connection"
    else
        echo -e "${RED}❌ HANA request failed with HTTP $HTTP_CODE${NC}"
        echo "Response: $BODY"
    fi
else
    echo -e "${RED}❌ HANA credentials not configured${NC}"
fi

echo ""
echo "═══ Test 3: SAP AI Core OAuth Token ═══"
echo ""

if [ -n "$AICORE_CLIENT_ID" ] && [ -n "$AICORE_CLIENT_SECRET" ] && [ -n "$AICORE_AUTH_URL" ]; then
    echo "Requesting OAuth token from: $AICORE_AUTH_URL"
    echo ""
    
    # Create Basic Auth header
    AICORE_AUTH_STRING=$(echo -n "${AICORE_CLIENT_ID}:${AICORE_CLIENT_SECRET}" | base64)
    
    # Request token
    TOKEN_RESPONSE=$(curl -s -X POST "$AICORE_AUTH_URL" \
        -H "Authorization: Basic $AICORE_AUTH_STRING" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "grant_type=client_credentials" \
        --connect-timeout 10 \
        -w "\n%{http_code}" 2>&1) || true
    
    TOKEN_HTTP_CODE=$(echo "$TOKEN_RESPONSE" | tail -n1)
    TOKEN_BODY=$(echo "$TOKEN_RESPONSE" | sed '$d')
    
    if [ "$TOKEN_HTTP_CODE" = "200" ]; then
        # Try to extract access_token
        if echo "$TOKEN_BODY" | grep -q "access_token"; then
            echo -e "${GREEN}✅ AI Core OAuth token received!${NC}"
            
            # Extract token for next test (using grep/sed since jq may not be available)
            ACCESS_TOKEN=$(echo "$TOKEN_BODY" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
            TOKEN_TYPE=$(echo "$TOKEN_BODY" | grep -o '"token_type":"[^"]*' | cut -d'"' -f4)
            EXPIRES_IN=$(echo "$TOKEN_BODY" | grep -o '"expires_in":[0-9]*' | cut -d':' -f2)
            
            echo "   Token Type: $TOKEN_TYPE"
            echo "   Expires In: ${EXPIRES_IN}s"
            echo "   Token: ${ACCESS_TOKEN:0:30}..."
        else
            echo -e "${YELLOW}⚠️  Response received but no access_token found${NC}"
            echo "Response: $TOKEN_BODY"
        fi
    elif [ "$TOKEN_HTTP_CODE" = "401" ]; then
        echo -e "${RED}❌ AI Core authentication failed (401)${NC}"
        echo "Check your AICORE_CLIENT_ID and AICORE_CLIENT_SECRET"
    elif [ "$TOKEN_HTTP_CODE" = "000" ]; then
        echo -e "${RED}❌ Cannot connect to AI Core auth server${NC}"
    else
        echo -e "${RED}❌ Token request failed with HTTP $TOKEN_HTTP_CODE${NC}"
        echo "Response: $TOKEN_BODY"
    fi
    
    # Test AI Core API endpoint if we got a token
    if [ -n "$ACCESS_TOKEN" ] && [ -n "$AICORE_BASE_URL" ]; then
        echo ""
        echo "═══ Test 4: AI Core API Endpoint ═══"
        echo ""
        
        DEPLOYMENTS_URL="${AICORE_BASE_URL}/v2/lm/deployments"
        echo "Testing: $DEPLOYMENTS_URL"
        
        API_RESPONSE=$(curl -s -X GET "$DEPLOYMENTS_URL" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -H "AI-Resource-Group: ${AICORE_RESOURCE_GROUP:-default}" \
            --connect-timeout 10 \
            -w "\n%{http_code}" 2>&1) || true
        
        API_HTTP_CODE=$(echo "$API_RESPONSE" | tail -n1)
        API_BODY=$(echo "$API_RESPONSE" | sed '$d')
        
        if [ "$API_HTTP_CODE" = "200" ]; then
            echo -e "${GREEN}✅ AI Core API accessible!${NC}"
            # Count deployments if possible
            if echo "$API_BODY" | grep -q "resources"; then
                DEPLOY_COUNT=$(echo "$API_BODY" | grep -o '"id"' | wc -l | tr -d ' ')
                echo "   Deployments found: $DEPLOY_COUNT"
            fi
        else
            echo -e "${YELLOW}⚠️  AI Core API returned HTTP $API_HTTP_CODE${NC}"
            echo "Response: ${API_BODY:0:200}..."
        fi
    fi
else
    echo -e "${RED}❌ AI Core credentials not configured${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        Test Summary                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Summary
if [ -n "$HANA_HOST" ]; then
    echo -e "HANA Cloud:  ${GREEN}✅ Configured${NC}"
else
    echo -e "HANA Cloud:  ${RED}❌ Not Configured${NC}"
fi

if [ -n "$ACCESS_TOKEN" ]; then
    echo -e "AI Core:     ${GREEN}✅ Token Acquired${NC}"
elif [ -n "$AICORE_CLIENT_ID" ]; then
    echo -e "AI Core:     ${YELLOW}⚠️  Configured but token failed${NC}"
else
    echo -e "AI Core:     ${RED}❌ Not Configured${NC}"
fi

echo ""