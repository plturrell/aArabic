#!/bin/bash
# Run Mojo Embedding Service
# Usage: ./scripts/run_mojo_embedding.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🔥 Mojo Embedding Service Launcher${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${YELLOW}📂 Project root: ${PROJECT_ROOT}${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if mojo is available
if ! command -v mojo &> /dev/null; then
    echo -e "${RED}❌ Error: mojo command not found${NC}"
    echo -e "${YELLOW}💡 Make sure Mojo is installed and in your PATH${NC}"
    echo -e "${YELLOW}   Try: export PATH=\"\$HOME/.pixi/envs/max/bin:\$PATH\"${NC}"
    exit 1
fi

# Check mojo version
MOJO_VERSION=$(mojo --version 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ Mojo version: ${MOJO_VERSION}${NC}"
echo ""

# Check if Python dependencies are installed
echo -e "${YELLOW}🔍 Checking Python dependencies...${NC}"
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Missing Python dependencies (fastapi, uvicorn)${NC}"
    echo -e "${YELLOW}💡 Installing dependencies...${NC}"
    pip install fastapi uvicorn[standard]
fi
echo -e "${GREEN}✓ Python dependencies OK${NC}"
echo ""

# Check if the main.mojo file exists
MAIN_FILE="$PROJECT_ROOT/src/serviceCore/serviceEmbedding-mojo/main.mojo"
if [ ! -f "$MAIN_FILE" ]; then
    echo -e "${RED}❌ Error: main.mojo not found at ${MAIN_FILE}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found main.mojo${NC}"
echo ""

# Run the service
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 Starting Mojo Embedding Service...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

mojo run "$MAIN_FILE"
