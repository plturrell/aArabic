#!/bin/bash

echo "🚀 Starting Complete Local Inference Stack"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Start Shimmy-AI
echo -e "${BLUE}[1/3] Starting Shimmy-AI (Port 11435)...${NC}"
cd vendor/layerIntelligence/shimmy-ai
if [ ! -f "target/release/shimmy" ]; then
    echo "Building Shimmy (first time only)..."
    cargo build --release
fi
./target/release/shimmy serve > /tmp/shimmy.log 2>&1 &
SHIMMY_PID=$!
echo "  PID: $SHIMMY_PID"
sleep 2
if curl -s http://localhost:11435/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Shimmy-AI ready${NC}"
else
    echo -e "${YELLOW}⏳ Shimmy-AI starting...${NC}"
fi
cd - > /dev/null
echo ""

# 2. Start Local LLM Service
echo -e "${BLUE}[2/3] Starting Local LLM Service (Port 8006)...${NC}"
cd src/serviceCore/serviceLocalLLM
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q fastapi uvicorn pydantic
    # RLM will be imported from vendor directory
else
    source venv/bin/activate
fi
python main.py > /tmp/local-llm.log 2>&1 &
LLM_PID=$!
echo "  PID: $LLM_PID"
sleep 2
if curl -s http://localhost:8006/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Local LLM Service ready${NC}"
else
    echo -e "${YELLOW}⏳ Local LLM Service starting...${NC}"
fi
cd - > /dev/null
echo ""

# 3. Check HyperBookLM
echo -e "${BLUE}[3/3] Checking HyperBookLM (Port 3000)...${NC}"
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ HyperBookLM already running${NC}"
else
    echo -e "${YELLOW}⚠ HyperBookLM not running. Start it with:${NC}"
    echo "  cd vendor/layerIntelligence/hyperbooklm && yarn dev"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}📊 Service Status:${NC}"
echo "------------------------------------------"
printf "%-25s %-10s %s\n" "Service" "Port" "Status"
printf "%-25s %-10s %s\n" "-------" "----" "------"

check_service() {
    if curl -s "$2" > /dev/null 2>&1; then
        echo -e "$(printf "%-25s %-10s" "$1" "$3") ${GREEN}✓ Running${NC}"
    else
        echo -e "$(printf "%-25s %-10s" "$1" "$3") ${YELLOW}⏳ Starting...${NC}"
    fi
}

check_service "Shimmy-AI" "http://localhost:11435/health" "11435"
check_service "Local LLM" "http://localhost:8006/health" "8006"
check_service "HyperBookLM" "http://localhost:3000" "3000"

echo ""
echo "=========================================="
echo -e "${GREEN}📚 Quick Test:${NC}"
echo "------------------------------------------"
echo "1. Test Shimmy-AI:"
echo "   curl http://localhost:11435/v1/models"
echo ""
echo "2. Test Complete Pipeline:"
echo "   curl -X POST http://localhost:3000/api/doc-to-n8n \\"
echo "     -F \"file=@docs/documentation/Invoice_Process.md\" \\"
echo "     -o workflow.json"
echo ""
echo "=========================================="
echo -e "${GREEN}📖 Logs:${NC}"
echo "   tail -f /tmp/shimmy.log"
echo "   tail -f /tmp/local-llm.log"
echo ""
echo -e "${GREEN}✅ Local inference stack ready!${NC}"
echo -e "${YELLOW}💰 Zero external API costs!${NC}"