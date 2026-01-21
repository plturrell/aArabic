#!/bin/bash
# ============================================================================
# HANA Table Deployment Helper Script
# ============================================================================
# Automates the deployment of NUCLEUS schema tables to SAP BTP HANA Cloud
# Uses the Zig deployment tool with proper error handling
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HANA_CLIENT_DIR="$PROJECT_ROOT/sap-toolkit-mojo/lib/clients/hana"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SAP HANA Table Deployment Tool                          ║${NC}"
echo -e "${BLUE}║   For NUCLEUS Schema - nOpenaiServer                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command_exists zig; then
    echo -e "${RED}❌ Zig compiler not found!${NC}"
    echo -e "${YELLOW}   Install from: https://ziglang.org/download/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Zig compiler found${NC}"

# Check environment variables
echo ""
echo -e "${YELLOW}🔍 Checking environment variables...${NC}"

REQUIRED_VARS=("HANA_HOST" "HANA_DATABASE" "HANA_USER" "HANA_PASSWORD")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
        echo -e "${RED}✗ $var not set${NC}"
    else
        # Mask password
        if [ "$var" == "HANA_PASSWORD" ]; then
            echo -e "${GREEN}✓ $var: *****${NC}"
        else
            echo -e "${GREEN}✓ $var: ${!var}${NC}"
        fi
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Missing required environment variables: ${MISSING_VARS[*]}${NC}"
    echo ""
    echo -e "${YELLOW}Please set them in your .env file or export them:${NC}"
    echo "  export HANA_HOST=\"your-instance.hana.prod-us10.hanacloud.ondemand.com\""
    echo "  export HANA_PORT=443"
    echo "  export HANA_DATABASE=\"your_database\""
    echo "  export HANA_SCHEMA=\"NUCLEUS\""
    echo "  export HANA_USER=\"NUCLEUS_APP\""
    echo "  export HANA_PASSWORD=\"your_password\""
    echo ""
    exit 1
fi

# Set optional variables with defaults
export HANA_PORT="${HANA_PORT:-443}"
export HANA_SCHEMA="${HANA_SCHEMA:-NUCLEUS}"
export HANA_ENCRYPT="${HANA_ENCRYPT:-true}"
export HANA_POOL_MIN="${HANA_POOL_MIN:-2}"
export HANA_POOL_MAX="${HANA_POOL_MAX:-10}"

echo ""
echo -e "${GREEN}✅ All prerequisites met!${NC}"
echo ""

# Ask for confirmation
echo -e "${YELLOW}⚠️  This will deploy/update tables in schema: ${HANA_SCHEMA}${NC}"
echo -e "${YELLOW}   on host: ${HANA_HOST}${NC}"
echo ""
read -p "Continue? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 0
fi

# Build the deployment tool
echo -e "${BLUE}🔨 Building deployment tool...${NC}"
cd "$HANA_CLIENT_DIR"

if zig build-exe deploy_tables.zig -O ReleaseSafe 2>&1 | tee /tmp/zig_build.log; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed. Check /tmp/zig_build.log for details${NC}"
    exit 1
fi

# Run deployment
echo ""
echo -e "${BLUE}🚀 Running table deployment...${NC}"
echo ""

if ./deploy_tables 2>&1 | tee /tmp/hana_deploy.log; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✅ DEPLOYMENT SUCCESSFUL!                                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Tables have been deployed to:${NC}"
    echo -e "${GREEN}  Schema: ${HANA_SCHEMA}${NC}"
    echo -e "${GREEN}  Host: ${HANA_HOST}${NC}"
    echo ""
    echo -e "${YELLOW}📝 Deployment log saved to: /tmp/hana_deploy.log${NC}"
    echo ""
    
    # Cleanup
    rm -f deploy_tables deploy_tables.o
    
    exit 0
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ❌ DEPLOYMENT FAILED!                                    ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Deployment encountered errors. Check:${NC}"
    echo -e "${RED}  1. /tmp/hana_deploy.log for deployment errors${NC}"
    echo -e "${RED}  2. HANA connection settings${NC}"
    echo -e "${RED}  3. User permissions (NUCLEUS_APP needs CREATE TABLE rights)${NC}"
    echo ""
    
    # Cleanup
    rm -f deploy_tables deploy_tables.o
    
    exit 1
fi
