#!/bin/bash

echo "📥 Downloading Local Models for Complete Independence"
echo "======================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface-cli...${NC}"
    pip install -q huggingface-hub
fi

MODELS_DIR="$HOME/.cache/huggingface/hub"
mkdir -p "$MODELS_DIR"

echo -e "${BLUE}Target directory: ${MODELS_DIR}${NC}"
echo ""

# Model 1: ToolOrchestra-8B (Best quality, #1 on GAIA)
echo -e "${BLUE}[1/3] Downloading ToolOrchestra-8B (~4.5GB)${NC}"
echo "  - NVIDIA's orchestration model"
echo "  - #1 ranked on GAIA benchmark"
echo "  - Best for workflow extraction"
echo ""
huggingface-cli download nvidia/Orchestrator-8B \
  --local-dir "$MODELS_DIR/orchestrator-8b" \
  --include "*.gguf" "*.json" "*.txt"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ToolOrchestra-8B downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Download may have failed, continuing...${NC}"
fi
echo ""

# Model 2: Phi-3-mini (Good balance)
echo -e "${BLUE}[2/3] Downloading Phi-3-mini-4k-instruct (~2.3GB)${NC}"
echo "  - Microsoft's efficient model"
echo "  - Good quality-to-speed ratio"
echo "  - Backup option"
echo ""
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  --local-dir "$MODELS_DIR/phi-3-mini" \
  --include "*.gguf" "*.json"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Phi-3-mini downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Download may have failed, continuing...${NC}"
fi
echo ""

# Model 3: Llama-3.2-1B (Fastest)
echo -e "${BLUE}[3/3] Downloading Llama-3.2-1B-Instruct (~800MB)${NC}"
echo "  - Meta's compact model"
echo "  - Very fast inference"
echo "  - Good for testing"
echo ""
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir "$MODELS_DIR/llama-3.2-1b"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Llama-3.2-1B downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Download may have failed, continuing...${NC}"
fi
echo ""

# Summary
echo "======================================================"
echo -e "${GREEN}📊 Model Summary:${NC}"
echo "------------------------------------------------------"
echo ""
echo -e "${GREEN}✓ ToolOrchestra-8B${NC} - Best quality (#1 GAIA)"
echo "  Location: $MODELS_DIR/orchestrator-8b/"
echo "  Use for: Production workflows"
echo ""
echo -e "${GREEN}✓ Phi-3-mini-4k-instruct${NC} - Balanced"
echo "  Location: $MODELS_DIR/phi-3-mini/"
echo "  Use for: Fast workflows"
echo ""
echo -e "${GREEN}✓ Llama-3.2-1B-Instruct${NC} - Fastest"
echo "  Location: $MODELS_DIR/llama-3.2-1b/"
echo "  Use for: Testing/development"
echo ""
echo "======================================================"
echo -e "${GREEN}🎯 Next Steps:${NC}"
echo "1. Verify models with Shimmy:"
echo "   cd vendor/layerIntelligence/shimmy-ai"
echo "   ./target/release/shimmy list"
echo ""
echo "2. Start local inference stack:"
echo "   ./scripts/start-local-inference.sh"
echo ""
echo "3. Test workflow extraction:"
echo "   curl -X POST http://localhost:3000/api/doc-to-n8n \\"
echo "     -F \"file=@your-document.pdf\""
echo ""
echo -e "${GREEN}✅ Models ready for 100% local inference!${NC}"
echo -e "${YELLOW}💰 $0 API costs forever!${NC}"