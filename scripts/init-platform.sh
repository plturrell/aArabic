#!/bin/bash
# =============================================================================
# AI NUCLEUS PLATFORM - Initialization Script
# =============================================================================
# This script initializes the platform for first-time deployment
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "AI Nucleus Platform - Initialization"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"

    # Generate random secrets
    echo -e "${YELLOW}Generating secure secrets...${NC}"

    # Generate random strings for secrets
    generate_secret() {
        openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32
    }

    # Replace placeholder secrets
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/change_me_admin_password_123/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_user_password_123/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_gateway_client_secret/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_service_secret_xyz789/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_32_char_session_secret_/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_backend_secret_key_32chars/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_keycloak_db_password/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_marquez_db_password/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_postgres_password/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_gitea_secret/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/change_me_api_key_123/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i '' "s/ai_nucleus_admin_key_change_me/$(generate_secret)/g" "$PROJECT_DIR/.env"
    else
        # Linux
        sed -i "s/change_me_admin_password_123/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_user_password_123/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_gateway_client_secret/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_service_secret_xyz789/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_32_char_session_secret_/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_backend_secret_key_32chars/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_keycloak_db_password/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_marquez_db_password/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_postgres_password/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_gitea_secret/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/change_me_api_key_123/$(generate_secret)/g" "$PROJECT_DIR/.env"
        sed -i "s/ai_nucleus_admin_key_change_me/$(generate_secret)/g" "$PROJECT_DIR/.env"
    fi

    echo -e "${GREEN}Secrets generated successfully!${NC}"
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

# Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p "$PROJECT_DIR/data/qdrant"
mkdir -p "$PROJECT_DIR/data/memgraph"
mkdir -p "$PROJECT_DIR/data/dragonfly"
mkdir -p "$PROJECT_DIR/data/keycloak_db"
mkdir -p "$PROJECT_DIR/data/gitea"
mkdir -p "$PROJECT_DIR/data/marquez_db"
mkdir -p "$PROJECT_DIR/data/langflow"
mkdir -p "$PROJECT_DIR/data/apisix_logs"
mkdir -p "$PROJECT_DIR/data/models"

echo -e "${GREEN}Data directories created${NC}"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker is running${NC}"

# Pull images
echo -e "${YELLOW}Pulling Docker images (this may take a while)...${NC}"
cd "$PROJECT_DIR"
docker-compose pull

echo ""
echo "=============================================="
echo -e "${GREEN}Initialization Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Review and customize .env file if needed"
echo "  2. Start the platform with: docker-compose up -d"
echo "  3. Wait for services to become healthy: docker-compose ps"
echo "  4. Access the platform at: http://localhost"
echo ""
echo "Default admin credentials (change after first login):"
echo "  Keycloak Admin: admin / (see KEYCLOAK_ADMIN_PASSWORD in .env)"
echo "  Nucleus Admin:  admin / (see NUCLEUS_ADMIN_PASSWORD in .env)"
echo ""
echo "Service URLs (after authentication):"
echo "  Frontend:    http://localhost/"
echo "  API:         http://localhost/api/"
echo "  Langflow:    http://localhost/langflow/"
echo "  Canvas:      http://localhost/canvas/"
echo "  Graph:       http://localhost/graph/"
echo "  Lineage:     http://localhost/lineage-ui/"
echo "  Git:         http://localhost/git/"
echo ""
