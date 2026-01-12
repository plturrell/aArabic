#!/bin/bash

set -e  # Exit on error

echo "======================================================================"
echo "🚀 MOJO EMBEDDING STACK DEPLOYMENT"
echo "======================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
STACK_NAME="mojo-embedding-stack"
PORTAINER_URL="http://localhost:9000"

# Step 1: Prerequisites check
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ Docker installed${NC}"

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ Docker Compose installed${NC}"

# Check if Portainer is running
if curl -s $PORTAINER_URL/api/status > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Portainer detected at $PORTAINER_URL${NC}"
    PORTAINER_AVAILABLE=true
else
    echo -e "${YELLOW}  ⚠️  Portainer not detected (optional)${NC}"
    PORTAINER_AVAILABLE=false
fi

# Step 2: Environment setup
echo ""
echo -e "${BLUE}Step 2: Setting up environment...${NC}"

if [ ! -f .env ]; then
    echo -e "${YELLOW}  ⚠️  No .env file found, creating from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}  ✅ Created .env file${NC}"
    echo -e "${YELLOW}  📝 Review .env file and adjust settings if needed${NC}"
else
    echo -e "${GREEN}  ✅ Using existing .env file${NC}"
fi

# Load environment variables
set -a
source .env
set +a

# Step 3: Check port availability
echo ""
echo -e "${BLUE}Step 3: Checking port availability...${NC}"

check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}  ⚠️  Port $port already in use ($service)${NC}"
        echo -e "${YELLOW}     Existing service will be used or you may need to stop it${NC}"
        return 1
    else
        echo -e "${GREEN}  ✅ Port $port available ($service)${NC}"
        return 0
    fi
}

check_port ${EMBEDDING_PORT:-8007} "Mojo Embedding"
check_port ${REDIS_PORT:-6379} "Redis Cache"
check_port ${QDRANT_PORT:-6333} "Qdrant Vector DB"

# Step 4: Create logs directory
echo ""
echo -e "${BLUE}Step 4: Creating directories...${NC}"
mkdir -p logs
echo -e "${GREEN}  ✅ Created logs directory${NC}"

# Step 5: Build and start services
echo ""
echo -e "${BLUE}Step 5: Building and starting services...${NC}"
echo -e "${YELLOW}  ⏳ This may take 5-10 minutes for first-time setup...${NC}"

# Use docker compose (new) or docker-compose (legacy)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

$DOCKER_COMPOSE up -d --build

echo -e "${GREEN}  ✅ Services started${NC}"

# Step 6: Wait for services to be healthy
echo ""
echo -e "${BLUE}Step 6: Waiting for services to become healthy...${NC}"

wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "  Waiting for $service_name"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f $url > /dev/null 2>&1; then
            echo -e " ${GREEN}✅${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}❌ Timeout${NC}"
    return 1
}

wait_for_service "http://localhost:${REDIS_PORT:-6379}" "Redis" || true
wait_for_service "http://localhost:${QDRANT_PORT:-6333}/readyz" "Qdrant"
wait_for_service "http://localhost:${EMBEDDING_PORT:-8007}/health" "Mojo Embedding"

# Step 7: Run verification tests
echo ""
echo -e "${BLUE}Step 7: Running verification tests...${NC}"

if [ -f scripts/test-deployment.sh ]; then
    ./scripts/test-deployment.sh
else
    echo -e "${YELLOW}  ⚠️  Test script not found, running basic tests...${NC}"
    
    # Basic health check
    if curl -s http://localhost:${EMBEDDING_PORT:-8007}/health | grep -q "healthy"; then
        echo -e "${GREEN}  ✅ Embedding service health check passed${NC}"
    fi
    
    # Basic embedding test
    RESULT=$(curl -s -X POST http://localhost:${EMBEDDING_PORT:-8007}/embed/single \
        -H "Content-Type: application/json" \
        -d '{"text":"test"}')
    
    if echo "$RESULT" | grep -q "embedding"; then
        echo -e "${GREEN}  ✅ Embedding generation test passed${NC}"
    fi
fi

# Step 8: Display service information
echo ""
echo "======================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo "======================================================================"
echo ""
echo "📊 Services Running:"
echo "  • Mojo Embedding Service: http://localhost:${EMBEDDING_PORT:-8007}"
echo "  • Redis Cache:            localhost:${REDIS_PORT:-6379}"
echo "  • Qdrant Vector DB:       http://localhost:${QDRANT_PORT:-6333}"
echo ""
echo "📚 Documentation:"
echo "  • API Docs:    http://localhost:${EMBEDDING_PORT:-8007}/docs"
echo "  • Health:      http://localhost:${EMBEDDING_PORT:-8007}/health"
echo "  • Metrics:     http://localhost:${EMBEDDING_PORT:-8007}/metrics"
echo "  • Qdrant UI:   http://localhost:${QDRANT_PORT:-6333}/dashboard"
echo ""

if [ "$PORTAINER_AVAILABLE" = true ]; then
    echo "🎛️  Portainer:"
    echo "  • Dashboard:   $PORTAINER_URL"
    echo "  • Stack:       $PORTAINER_URL/#/docker/stacks"
    echo ""
fi

echo "🔧 Management Commands:"
echo "  • View logs:    $DOCKER_COMPOSE logs -f"
echo "  • Stop all:     $DOCKER_COMPOSE down"
echo "  • Restart:      $DOCKER_COMPOSE restart"
echo "  • Status:       $DOCKER_COMPOSE ps"
echo ""
echo "📊 Quick Test:"
echo "  curl -X POST http://localhost:${EMBEDDING_PORT:-8007}/embed/single \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\":\"Hello world\"}'"
echo ""
echo "======================================================================"
echo -e "${GREEN}🎉 Ready to use!${NC}"
echo "======================================================================"
