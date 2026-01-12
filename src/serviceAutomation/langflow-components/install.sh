#!/bin/bash

# Installation Script for Langflow Custom Components
# Installs all 17 Rust API client components into Langflow

set -e

echo "=================================="
echo "Langflow Components Installation"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine Langflow components directory
if [ -d "$HOME/.langflow/components" ]; then
    LANGFLOW_DIR="$HOME/.langflow/components"
elif [ -d "$HOME/.local/share/langflow/components" ]; then
    LANGFLOW_DIR="$HOME/.local/share/langflow/components"
else
    echo -e "${YELLOW}Langflow components directory not found. Creating...${NC}"
    LANGFLOW_DIR="$HOME/.langflow/components"
    mkdir -p "$LANGFLOW_DIR"
fi

echo "Langflow components directory: $LANGFLOW_DIR"
echo ""

# Check if rust_clients.py exists
if [ ! -f "rust_clients.py" ]; then
    echo -e "${RED}Error: rust_clients.py not found in current directory${NC}"
    echo "Please run this script from the langflow-components directory"
    exit 1
fi

# Copy the components file
echo "Installing components..."
cp rust_clients.py "$LANGFLOW_DIR/"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully installed rust_clients.py${NC}"
else
    echo -e "${RED}✗ Failed to install components${NC}"
    exit 1
fi

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Restart Langflow:"
echo "   ${YELLOW}langflow run${NC}"
echo ""
echo "2. Open Langflow UI (usually http://localhost:7860)"
echo ""
echo "3. Look for these 17 components in the Custom category:"
echo "   • Langflow Operations"
echo "   • Gitea Git Hosting"
echo "   • Git Operations"
echo "   • Filesystem Operations"
echo "   • Memory Cache"
echo "   • APISIX API Gateway"
echo "   • Keycloak Authentication"
echo "   • Glean Code Intelligence"
echo "   • MarkItDown Converter"
echo "   • Marquez Data Lineage"
echo "   • PostgreSQL Database"
echo "   • Hyperbook Documentation"
echo "   • n8n Workflow Automation"
echo "   • OpenCanvas Collaboration"
echo "   • Kafka Messaging"
echo "   • Shimmy-AI Local Inference"
echo "   • Lean4 Theorem Prover"
echo ""
echo "4. Drag and drop components to build workflows!"
echo ""

# Check if Rust CLIs are installed
echo "Checking for Rust CLI installations..."
echo ""

MISSING_CLIS=()
for cli in langflow-cli gitea-cli git-cli fs-cli memory-cli apisix-cli keycloak-cli \
           glean-cli markitdown-cli marquez-cli postgres-cli hyperbook-cli \
           n8n-cli opencanvas-cli kafka-cli shimmy-cli lean4-cli; do
    if command -v "$cli" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $cli found"
    else
        echo -e "${YELLOW}✗${NC} $cli not found"
        MISSING_CLIS+=("$cli")
    fi
done

if [ ${#MISSING_CLIS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Warning: ${#MISSING_CLIS[@]} CLI(s) not found${NC}"
    echo "To use all components, build and install the Rust CLIs:"
    echo ""
    echo "cd ../../"
    echo "for dir in *-api-client; do"
    echo "  (cd \$dir && cargo build --release)"
    echo "done"
    echo ""
    echo "Then copy binaries to /usr/local/bin or add to PATH"
fi

echo ""
echo "Installation complete! Happy workflow building! 🎉"
