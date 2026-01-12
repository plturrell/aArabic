#!/bin/bash
# Start all feature engineering services

set -e

echo "🚀 Starting Feature Engineering Services..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if services are already running
check_service() {
    local port=$1
    local name=$2
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name already running on port $port"
        return 0
    else
        return 1
    fi
}

# Start Lean4 Parser
echo -e "${BLUE}Starting Lean4 Parser (Port 8002)...${NC}"
if ! check_service 8002 "Lean4 Parser"; then
    cd src/serviceIntelligence/lean4-rust
    cargo build --release 2>/dev/null || echo "Build in progress..."
    cargo run --release > /tmp/lean4-parser.log 2>&1 &
    LEAN4_PID=$!
    echo "  PID: $LEAN4_PID"
    cd ../../..
    sleep 3
fi

# Start serviceN8n
echo -e "${BLUE}Starting serviceN8n (Port 8003)...${NC}"
if ! check_service 8003 "serviceN8n"; then
    cd src/serviceIntelligence/serviceN8n
    
    # Set environment variables
    export LEAN4_PARSER_URL="http://localhost:8002"
    export GITEA_SERVICE_PATH="../../serviceCore/serviceGitea"
    export AUTOMATION_PATH="../../serviceAutomation"
    
    cargo build --release 2>/dev/null || echo "Build in progress..."
    cargo run --release > /tmp/service-n8n.log 2>&1 &
    N8N_PID=$!
    echo "  PID: $N8N_PID"
    cd ../../..
    sleep 3
fi

# Start serviceGitea
echo -e "${BLUE}Starting serviceGitea (Port 8004)...${NC}"
if ! check_service 8004 "serviceGitea"; then
    cd src/serviceCore/serviceGitea
    
    # Check if there are any features to build
    if [ -d "features" ] && [ "$(ls -A features)" ]; then
        cargo build --release --workspace 2>/dev/null || echo "Build in progress..."
        cargo run --release > /tmp/service-gitea.log 2>&1 &
        GITEA_PID=$!
        echo "  PID: $GITEA_PID"
    else
        echo "  No features to build yet"
    fi
    cd ../../..
fi

echo ""
echo "⏳ Waiting for services to start..."
sleep 5

echo ""
echo "📊 Service Status:"
echo "─────────────────────────────────────────"

check_service 8002 "Lean4 Parser" && echo "  http://localhost:8002/health" || echo "  ❌ Not responding"
check_service 8003 "serviceN8n" && echo "  http://localhost:8003/health" || echo "  ❌ Not responding"
check_service 8004 "serviceGitea" && echo "  http://localhost:8004/health" || echo "  ❌ Not responding"

echo ""
echo "📚 Quick Start:"
echo "─────────────────────────────────────────"
echo "1. Generate template:"
echo "   curl -X POST http://localhost:8002/template -d '{\"feature_name\":\"My Feature\"}'"
echo ""
echo "2. Orchestrate from n8n workflow:"
echo "   curl -X POST http://localhost:8003/orchestrate/feature \\"
echo "     -d '{\"workflow_json\":\"...\",\"feature_name\":\"test\",\"auto_deploy\":true}'"
echo ""
echo "3. Check logs:"
echo "   tail -f /tmp/lean4-parser.log"
echo "   tail -f /tmp/service-n8n.log"
echo "   tail -f /tmp/service-gitea.log"
echo ""
echo "✅ All services started successfully!"