#!/bin/bash
# Automated Portainer Setup Script
# This script fully sets up Portainer via CLI and API

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$PROJECT_ROOT/docker/compose"

echo "🐳 PORTAINER AUTOMATED SETUP"
echo "========================================"
echo ""

# Configuration
PORTAINER_URL="http://localhost:9000"
ADMIN_USER="admin"
ADMIN_PASS="Standard2026"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Check if Portainer is running
echo -e "${BLUE}Step 1: Checking Portainer status...${NC}"
if docker ps | grep -q ai_nucleus_portainer; then
    echo -e "${GREEN}✓ Portainer container is running${NC}"
else
    echo -e "${YELLOW}⚠ Portainer not running. Starting it now...${NC}"
    cd "$COMPOSE_DIR"
    docker compose -f docker-compose.core.yml up -d portainer
    echo "⏳ Waiting for Portainer to start..."
    sleep 15
fi

# Step 2: Wait for Portainer to be ready
echo ""
echo -e "${BLUE}Step 2: Waiting for Portainer API...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "$PORTAINER_URL/api/status" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Portainer API is ready${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}✗ Portainer API failed to start${NC}"
    exit 1
fi

# Step 3: Create admin user (if not exists)
echo ""
echo -e "${BLUE}Step 3: Setting up admin user...${NC}"
INIT_RESPONSE=$(curl -s -X POST "$PORTAINER_URL/api/users/admin/init" \
    -H "Content-Type: application/json" \
    -d "{\"Username\": \"$ADMIN_USER\", \"Password\": \"$ADMIN_PASS\"}" 2>&1)

if echo "$INIT_RESPONSE" | grep -q "already exists"; then
    echo -e "${YELLOW}⚠ Admin user already exists${NC}"
else
    echo -e "${GREEN}✓ Admin user created${NC}"
fi

# Step 4: Authenticate and get JWT token
echo ""
echo -e "${BLUE}Step 4: Authenticating...${NC}"
AUTH_RESPONSE=$(curl -s -X POST "$PORTAINER_URL/api/auth" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"$ADMIN_USER\", \"password\": \"$ADMIN_PASS\"}")

JWT_TOKEN=$(echo "$AUTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['jwt'])" 2>/dev/null)

if [ -z "$JWT_TOKEN" ]; then
    echo -e "${RED}✗ Failed to get JWT token${NC}"
    echo "$AUTH_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Authentication successful${NC}"

# Step 5: Check Docker endpoint
echo ""
echo -e "${BLUE}Step 5: Checking Docker endpoint...${NC}"
ENDPOINTS=$(curl -s -X GET "$PORTAINER_URL/api/endpoints" \
    -H "Authorization: Bearer $JWT_TOKEN")

ENDPOINT_ID=$(echo "$ENDPOINTS" | python3 -c "import sys, json; eps = json.load(sys.stdin); print(eps[0]['Id'] if eps else '')" 2>/dev/null)

if [ -n "$ENDPOINT_ID" ]; then
    ENDPOINT_NAME=$(echo "$ENDPOINTS" | python3 -c "import sys, json; eps = json.load(sys.stdin); print(eps[0]['Name'])" 2>/dev/null)
    echo -e "${GREEN}✓ Docker endpoint already configured${NC}"
    echo "  ID: $ENDPOINT_ID"
    echo "  Name: $ENDPOINT_NAME"
else
    echo -e "${YELLOW}⚠ No endpoint found, this shouldn't happen...${NC}"
    echo "  Portainer should auto-detect local Docker"
fi

# Step 6: Get container statistics
echo ""
echo -e "${BLUE}Step 6: Getting container statistics...${NC}"
CONTAINERS=$(curl -s -X GET "$PORTAINER_URL/api/endpoints/$ENDPOINT_ID/docker/containers/json?all=true" \
    -H "Authorization: Bearer $JWT_TOKEN")

CONTAINER_STATS=$(echo "$CONTAINERS" | python3 -c "
import sys, json
containers = json.load(sys.stdin)
running = sum(1 for c in containers if c['State'] == 'running')
stopped = sum(1 for c in containers if c['State'] in ['exited', 'created'])
total = len(containers)
print(f'{total}|{running}|{stopped}')
" 2>/dev/null)

TOTAL=$(echo "$CONTAINER_STATS" | cut -d'|' -f1)
RUNNING=$(echo "$CONTAINER_STATS" | cut -d'|' -f2)
STOPPED=$(echo "$CONTAINER_STATS" | cut -d'|' -f3)

echo -e "${GREEN}✓ Container statistics retrieved${NC}"
echo "  Total: $TOTAL containers"
echo "  Running: $RUNNING containers"
echo "  Stopped: $STOPPED containers"

# Step 7: List some containers
echo ""
echo -e "${BLUE}Step 7: Sample containers...${NC}"
echo "$CONTAINERS" | python3 -c "
import sys, json
containers = json.load(sys.stdin)
for c in containers[:10]:
    name = c['Names'][0]
    state = c['State']
    status = c['Status']
    print(f'  • {name} - {state} - {status}')
" 2>/dev/null

# Step 8: Get system info
echo ""
echo -e "${BLUE}Step 8: Docker system information...${NC}"
SYSTEM_INFO=$(curl -s -X GET "$PORTAINER_URL/api/endpoints/$ENDPOINT_ID/docker/info" \
    -H "Authorization: Bearer $JWT_TOKEN")

echo "$SYSTEM_INFO" | python3 -c "
import sys, json
info = json.load(sys.stdin)
print(f\"  Docker Version: {info.get('ServerVersion', 'N/A')}\")
print(f\"  Operating System: {info.get('OperatingSystem', 'N/A')}\")
print(f\"  Total CPUs: {info.get('NCPU', 'N/A')}\")
print(f\"  Total Memory: {info.get('MemTotal', 0) / 1024 / 1024 / 1024:.2f} GB\")
print(f\"  Images: {info.get('Images', 0)}\")
" 2>/dev/null

# Step 9: Verify key services
echo ""
echo -e "${BLUE}Step 9: Verifying key AI Nucleus services...${NC}"
KEY_SERVICES=(
    "ai_nucleus_portainer"
    "ai_nucleus_memgraph"
    "ai_nucleus_shimmy"
)

for service in "${KEY_SERVICES[@]}"; do
    if echo "$CONTAINERS" | python3 -c "import sys, json; containers = json.load(sys.stdin); found = any(c['Names'][0] == '/$service' for c in containers); sys.exit(0 if found else 1)" 2>/dev/null; then
        echo -e "${GREEN}  ✓ $service - detected${NC}"
    else
        echo -e "${YELLOW}  ⚠ $service - not found${NC}"
    fi
done

# Final summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ PORTAINER SETUP COMPLETE!${NC}"
echo "========================================"
echo ""
echo "📊 Summary:"
echo "  • Portainer URL: $PORTAINER_URL"
echo "  • Admin Username: $ADMIN_USER"
echo "  • Admin Password: $ADMIN_PASS"
echo "  • Docker Endpoint: local (ID: $ENDPOINT_ID)"
echo "  • Total Containers: $TOTAL"
echo "  • Running: $RUNNING"
echo "  • Stopped: $STOPPED"
echo ""
echo "🚀 Access Portainer:"
echo "  open $PORTAINER_URL"
echo ""
echo "📚 Documentation:"
echo "  • Full Guide: docs/PORTAINER.md"
echo "  • Setup Guide: docs/PORTAINER_SETUP_GUIDE.md"
echo ""
echo "🎉 Everything is ready!"
