#!/bin/bash

# Installation Script for n8n Rust Clients Node
# Installs unified node for all 17 Rust API clients

set -e

echo "=================================="
echo "n8n Node Installation"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine n8n custom nodes directory
if [ -d "$HOME/.n8n/custom" ]; then
    N8N_DIR="$HOME/.n8n/custom"
elif [ -d "$HOME/.n8n/nodes" ]; then
    N8N_DIR="$HOME/.n8n/nodes"
else
    echo -e "${YELLOW}n8n custom directory not found. Creating...${NC}"
    N8N_DIR="$HOME/.n8n/custom"
    mkdir -p "$N8N_DIR"
fi

echo "n8n custom directory: $N8N_DIR"
echo ""

# Check if RustClients.node.ts exists
if [ ! -f "RustClients.node.ts" ]; then
    echo -e "${RED}Error: RustClients.node.ts not found in current directory${NC}"
    echo "Please run this script from the n8n-nodes directory"
    exit 1
fi

# Copy the node file
echo "Installing n8n node..."
cp RustClients.node.ts "$N8N_DIR/"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully installed RustClients.node.ts${NC}"
else
    echo -e "${RED}✗ Failed to install node${NC}"
    exit 1
fi

# Install npm dependencies if package.json exists
if [ -f "package.json" ]; then
    echo ""
    echo "Installing npm dependencies..."
    cd "$N8N_DIR"
    npm install n8n-workflow 2>/dev/null || echo -e "${YELLOW}Note: Install n8n-workflow manually if needed${NC}"
    cd - > /dev/null
fi

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Restart n8n:"
echo "   ${YELLOW}n8n start${NC}"
echo ""
echo "2. Open n8n UI (usually http://localhost:5678)"
echo ""
echo "3. Look for 'Rust API Clients' node in the nodes panel"
echo ""
echo "4. Node supports all 17 Rust clients:"
echo "   • Langflow       • Gitea          • Git"
echo "   • Glean          • MarkItDown     • Marquez"
echo "   • PostgreSQL     • Hyperbook      • n8n"
echo "   • OpenCanvas     • Kafka          • Shimmy AI"
echo "   • APISIX         • Keycloak       • Filesystem"
echo "   • Memory         • Lean4"
echo ""
echo "5. Configure node with:"
echo "   - Client dropdown (select which CLI)"
echo "   - Operation (what to execute)"
echo "   - Arguments (optional parameters)"
echo "   - URL (optional, uses defaults)"
echo ""

# Check if Rust CLIs are installed
echo "Checking for Rust CLI installations..."
echo ""

MISSING_CLIS=()
for cli in langflow-cli gitea-cli git-cli glean-cli markitdown-cli marquez-cli \
           postgres-cli hyperbook-cli n8n-cli opencanvas-cli kafka-cli shimmy-cli \
           apisix-cli keycloak-cli fs-cli memory-cli lean4-cli; do
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
    echo "To use all clients, build and install the Rust CLIs:"
    echo ""
    echo "cd ../../"
    echo "for dir in *-api-client; do"
    echo "  (cd \$dir && cargo build --release)"
    echo "  sudo cp \$dir/target/release/*-cli /usr/local/bin/"
    echo "done"
fi

echo ""
echo "Installation complete! 🎉"
echo ""
echo "Documentation: See README.md for usage examples"
