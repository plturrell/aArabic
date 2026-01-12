#!/bin/bash
# AI Nucleus Platform - Open All Working UIs
# Opens all accessible UI services in your default browser

echo "🚀 AI NUCLEUS PLATFORM - OPENING ALL UIS"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Opening External UIs (Browser Accessible)...${NC}"
echo ""

# Open Kafka UI
echo -e "${GREEN}1. Opening Kafka UI${NC} - http://localhost:8090"
open http://localhost:8090 2>/dev/null || xdg-open http://localhost:8090 2>/dev/null || echo "   → Please open manually: http://localhost:8090"
sleep 2

# Open Memgraph Lab
echo -e "${GREEN}2. Opening Memgraph Lab${NC} - http://localhost:3001"
open http://localhost:3001 2>/dev/null || xdg-open http://localhost:3001 2>/dev/null || echo "   → Please open manually: http://localhost:3001"
sleep 2

# Open Portainer
echo -e "${GREEN}3. Opening Portainer (Container Management)${NC} - http://localhost:9000"
open http://localhost:9000 2>/dev/null || xdg-open http://localhost:9000 2>/dev/null || echo "   → Please open manually: http://localhost:9000"
sleep 2

echo ""
echo -e "${BLUE}Internal UIs (Accessible via Gateway or Docker Exec):${NC}"
echo ""

# List internal UIs with their access methods
echo -e "${YELLOW}Backend API (Swagger):${NC}"
echo "   Internal: http://localhost:8000/docs"
echo "   Access: docker exec ai_nucleus_backend curl http://localhost:8000/docs"
echo ""

echo -e "${YELLOW}N8N Workflow Automation:${NC}"
echo "   Internal: http://localhost:5678"
echo "   Access: docker exec ai_nucleus_n8n wget -qO- http://localhost:5678"
echo ""

echo -e "${YELLOW}Langflow AI Builder:${NC}"
echo "   Internal: http://localhost:7860"
echo "   Access: docker exec ai_nucleus_langflow curl http://localhost:7860"
echo ""

echo -e "${YELLOW}HyperbookLM:${NC}"
echo "   Internal: http://localhost:3002"
echo "   Access: docker exec ai_nucleus_hyperbooklm wget -qO- http://localhost:3002"
echo ""

echo -e "${YELLOW}Gitea Git Server:${NC}"
echo "   Internal: http://localhost:3000"
echo "   Access: docker exec ai_nucleus_gitea wget -qO- http://localhost:3000"
echo ""

echo -e "${YELLOW}Keycloak Admin:${NC}"
echo "   Internal: http://localhost:8080"
echo "   Access: docker exec ai_nucleus_keycloak curl http://localhost:8080"
echo ""

echo -e "${YELLOW}Marquez Data Lineage:${NC}"
echo "   Internal: http://localhost:3000"
echo "   Access: docker exec ai_nucleus_marquez_web curl http://localhost:3000"
echo ""

echo -e "${YELLOW}Nucleus Graph Visualizer:${NC}"
echo "   Internal: http://localhost:5000"
echo "   Access: docker exec ai_nucleus_graph curl http://localhost:5000"
echo ""

echo -e "${RED}OpenCanvas (Build Issue):${NC}"
echo "   Status: Not available (TypeScript build error)"
echo "   Note: Vendor code needs LangChain compatibility fix"
echo ""

echo "✅ Browser-accessible UIs have been opened!"
echo ""
echo "💡 TIP: To access internal UIs, use the Gateway (once port mapping is configured)"
echo "        or use docker exec commands shown above."
echo ""
echo "📖 Full documentation: ./access_services.sh"
