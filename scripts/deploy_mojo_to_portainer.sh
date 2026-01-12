#!/bin/bash
# Deploy Mojo Embedding Service to Portainer
# This script deploys the service as a Portainer stack

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/compose/docker-compose.mojo-embedding.yml"

echo "🚀 DEPLOYING MOJO EMBEDDING TO PORTAINER"
echo "========================================"
echo ""

# Configuration
PORTAINER_URL="http://localhost:9000"
ADMIN_USER="admin"
ADMIN_PASS="Standard2026"
STACK_NAME="mojo-embedding"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Authenticate
echo -e "${BLUE}Step 1: Authenticating with Portainer...${NC}"
AUTH_RESPONSE=$(curl -s -X POST "$PORTAINER_URL/api/auth" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"$ADMIN_USER\", \"password\": \"$ADMIN_PASS\"}")

JWT_TOKEN=$(echo "$AUTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['jwt'])" 2>/dev/null)

if [ -z "$JWT_TOKEN" ]; then
    echo -e "${RED}✗ Failed to authenticate${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Authenticated successfully${NC}"

# Step 2: Get endpoint ID
echo ""
echo -e "${BLUE}Step 2: Getting Docker endpoint...${NC}"
ENDPOINTS=$(curl -s -X GET "$PORTAINER_URL/api/endpoints" \
    -H "Authorization: Bearer $JWT_TOKEN")

ENDPOINT_ID=$(echo "$ENDPOINTS" | python3 -c "import sys, json; eps = json.load(sys.stdin); print(eps[0]['Id'] if eps else '')" 2>/dev/null)

if [ -z "$ENDPOINT_ID" ]; then
    echo -e "${RED}✗ No Docker endpoint found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Endpoint ID: $ENDPOINT_ID${NC}"

# Step 3: Check if stack exists
echo ""
echo -e "${BLUE}Step 3: Checking for existing stack...${NC}"
STACKS=$(curl -s -X GET "$PORTAINER_URL/api/stacks" \
    -H "Authorization: Bearer $JWT_TOKEN")

EXISTING_STACK_ID=$(echo "$STACKS" | python3 -c "
import sys, json
stacks = json.load(sys.stdin)
for s in stacks:
    if s['Name'] == '$STACK_NAME':
        print(s['Id'])
        break
" 2>/dev/null)

if [ -n "$EXISTING_STACK_ID" ]; then
    echo -e "${YELLOW}⚠ Stack '$STACK_NAME' already exists (ID: $EXISTING_STACK_ID)${NC}"
    echo "  Updating existing stack..."
    
    # Update existing stack
    STACK_CONTENT=$(cat "$COMPOSE_FILE")
    
    UPDATE_RESPONSE=$(curl -s -X PUT "$PORTAINER_URL/api/stacks/$EXISTING_STACK_ID?endpointId=$ENDPOINT_ID" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"stackFileContent\": $(echo "$STACK_CONTENT" | jq -Rs .),
            \"prune\": false,
            \"pullImage\": false
        }")
    
    if echo "$UPDATE_RESPONSE" | grep -q "Id"; then
        echo -e "${GREEN}✓ Stack updated successfully${NC}"
    else
        echo -e "${RED}✗ Failed to update stack${NC}"
        echo "$UPDATE_RESPONSE"
        exit 1
    fi
else
    echo -e "${BLUE}Creating new stack '$STACK_NAME'...${NC}"
    
    # Create new stack
    STACK_CONTENT=$(cat "$COMPOSE_FILE")
    
    CREATE_RESPONSE=$(curl -s -X POST "$PORTAINER_URL/api/stacks?endpointId=$ENDPOINT_ID&type=2&method=string" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$STACK_NAME\",
            \"stackFileContent\": $(echo "$STACK_CONTENT" | jq -Rs .),
            \"env\": []
        }")
    
    if echo "$CREATE_RESPONSE" | grep -q "Id"; then
        echo -e "${GREEN}✓ Stack created successfully${NC}"
        NEW_STACK_ID=$(echo "$CREATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['Id'])" 2>/dev/null)
        echo "  Stack ID: $NEW_STACK_ID"
    else
        echo -e "${RED}✗ Failed to create stack${NC}"
        echo "$CREATE_RESPONSE"
        exit 1
    fi
fi

# Step 4: Wait for container to start
echo ""
echo -e "${BLUE}Step 4: Waiting for container to start...${NC}"
sleep 5

# Step 5: Verify container is running
echo ""
echo -e "${BLUE}Step 5: Verifying deployment...${NC}"
CONTAINERS=$(curl -s -X GET "$PORTAINER_URL/api/endpoints/$ENDPOINT_ID/docker/containers/json" \
    -H "Authorization: Bearer $JWT_TOKEN")

MOJO_CONTAINER=$(echo "$CONTAINERS" | python3 -c "
import sys, json
containers = json.load(sys.stdin)
for c in containers:
    if 'mojo-embedding' in c['Names'][0]:
        print(json.dumps({
            'name': c['Names'][0],
            'state': c['State'],
            'status': c['Status'],
            'id': c['Id'][:12]
        }))
        break
" 2>/dev/null)

if [ -n "$MOJO_CONTAINER" ]; then
    echo -e "${GREEN}✓ Container found and running${NC}"
    echo "$MOJO_CONTAINER" | python3 -c "
import sys, json
c = json.load(sys.stdin)
print(f\"  Name: {c['name']}\")
print(f\"  State: {c['state']}\")
print(f\"  Status: {c['status']}\")
print(f\"  ID: {c['id']}\")
"
else
    echo -e "${YELLOW}⚠ Container not found yet, may still be starting...${NC}"
fi

# Step 6: Test the service
echo ""
echo -e "${BLUE}Step 6: Testing service health...${NC}"
sleep 3

if curl -s -f http://localhost:8007/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Service is healthy${NC}"
    curl -s http://localhost:8007/health | python3 -c "
import sys, json
h = json.load(sys.stdin)
print(f\"  Service: {h['service']}\")
print(f\"  Version: {h['version']}\")
print(f\"  Status: {h['status']}\")
"
else
    echo -e "${YELLOW}⚠ Service health check pending...${NC}"
    echo "  The service may still be initializing"
    echo "  Try: curl http://localhost:8007/health"
fi

# Final summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo "========================================"
echo ""
echo "📊 Deployment Details:"
echo "  • Stack Name: $STACK_NAME"
echo "  • Endpoint ID: $ENDPOINT_ID"
echo "  • Service URL: http://localhost:8007"
echo ""
echo "🔗 Quick Links:"
echo "  • Portainer: $PORTAINER_URL"
echo "  • Service Health: http://localhost:8007/health"
echo "  • API Docs: http://localhost:8007/docs"
echo "  • Metrics: http://localhost:8007/metrics"
echo ""
echo "🧪 Test the service:"
echo "  curl http://localhost:8007/health"
echo "  curl -X POST http://localhost:8007/embed/single -H 'Content-Type: application/json' -d '{\"text\":\"test\"}'"
echo ""
echo "📚 View in Portainer:"
echo "  open $PORTAINER_URL"
echo "  Navigate to: Stacks > $STACK_NAME"
echo ""
echo "🎉 Mojo Embedding Service is now managed by Portainer!"
