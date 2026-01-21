#!/bin/bash
# Build all Rust API clients
# Part of the final integration step

set -e  # Exit on error

echo "=========================================="
echo "Building All Rust API Clients"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BUILD_DIR="src/serviceAutomation"
FAILED_BUILDS=()
SUCCESSFUL_BUILDS=()

# Function to build a client
build_client() {
    local client_name=$1
    local client_path="${BUILD_DIR}/${client_name}"
    
    if [ ! -d "$client_path" ]; then
        echo -e "${YELLOW}⚠️  Skipping ${client_name}: directory not found${NC}"
        return
    fi
    
    if [ ! -f "$client_path/Cargo.toml" ]; then
        echo -e "${YELLOW}⚠️  Skipping ${client_name}: no Cargo.toml found${NC}"
        return
    fi
    
    echo "----------------------------------------"
    echo "Building: ${client_name}"
    echo "----------------------------------------"
    
    cd "$client_path"
    
    if cargo build --release 2>&1 | tee build.log; then
        # Find the binary
        BINARY=$(find target/release -maxdepth 1 -type f -executable ! -name "*.so" ! -name "*.dylib" ! -name "*.d" 2>/dev/null | head -1)
        if [ -n "$BINARY" ]; then
            SIZE=$(ls -lh "$BINARY" | awk '{print $5}')
            echo -e "${GREEN}✅ ${client_name} built successfully (${SIZE})${NC}"
            SUCCESSFUL_BUILDS+=("${client_name} (${SIZE})")
        else
            echo -e "${GREEN}✅ ${client_name} library built successfully${NC}"
            SUCCESSFUL_BUILDS+=("${client_name} (lib)")
        fi
    else
        echo -e "${RED}❌ ${client_name} build failed${NC}"
        FAILED_BUILDS+=("${client_name}")
    fi
    
    cd - > /dev/null
    echo ""
}

# Build all clients
echo "Starting build process..."
echo ""

# Core clients
build_client "qdrant-api-client"
build_client "memgraph-api-client"
build_client "dragonflydb-api-client"

# Other potential clients
build_client "postgres-api-client"
build_client "kafka-api-client"
build_client "gitea-api-client"
build_client "keycloak-api-client"
build_client "n8n-api-client"
build_client "langflow-api-client"

echo "=========================================="
echo "Build Summary"
echo "=========================================="
echo ""

if [ ${#SUCCESSFUL_BUILDS[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Successful builds (${#SUCCESSFUL_BUILDS[@]}):${NC}"
    for build in "${SUCCESSFUL_BUILDS[@]}"; do
        echo "   - $build"
    done
    echo ""
fi

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed builds (${#FAILED_BUILDS[@]}):${NC}"
    for build in "${FAILED_BUILDS[@]}"; do
        echo "   - $build"
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}🎉 All builds completed successfully!${NC}"
    echo ""
fi
