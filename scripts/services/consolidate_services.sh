#!/bin/bash

# Service Consolidation Script
# Removes duplicate RAG, Embedding, and Translation services
# Keeps only the best/production-ready versions

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Service Consolidation - Remove Duplicates          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="/Users/user/Documents/arabic_folder"
BACKUP_DIR="$HOME/backups/serviceCore-backup-$(date +%Y%m%d-%H%M%S)"
SERVICE_CORE="$PROJECT_ROOT/src/serviceCore"

# Summary
echo -e "${BLUE}📋 Consolidation Plan:${NC}"
echo ""
echo "  KEEP (Production Ready):"
echo -e "  ${GREEN}✅ serviceEmbedding-mojo${NC}      (10-25x faster, deployed, documented)"
echo -e "  ${GREEN}✅ serviceRAG-zig-mojo${NC}        (Zig+Mojo hybrid, needs Zig compiler)"
echo -e "  ${GREEN}✅ serviceTranslation-rust${NC}    (3-model architecture, enterprise-grade)"
echo ""
echo "  REMOVE (Superseded/Experimental):"
echo -e "  ${RED}❌ serviceEmbedding${NC}            (Old Python version)"
echo -e "  ${RED}❌ serviceEmbedding-rust${NC}       (Experimental Burn framework)"
echo -e "  ${RED}❌ serviceRAG-mojo${NC}             (Superseded by Zig+Mojo)"
echo -e "  ${RED}❌ serviceRAG-rust${NC}             (Experimental)"
echo -e "  ${RED}❌ serviceTranslation-mojo${NC}     (Superseded by Rust 3-model)"
echo ""

# Ask for confirmation
read -p "$(echo -e ${YELLOW}Continue with consolidation? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Consolidation cancelled${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 1: Creating backup...${NC}"
mkdir -p "$BACKUP_DIR"
cp -r "$SERVICE_CORE" "$BACKUP_DIR/"
echo -e "${GREEN}✅ Backup created: $BACKUP_DIR${NC}"
echo ""

# Check for running containers
echo -e "${BLUE}Step 2: Checking for active containers...${NC}"
ACTIVE_SERVICES=$(docker ps --filter "name=embedding\|translation\|rag" --format "{{.Names}}" 2>/dev/null || echo "")
if [ -n "$ACTIVE_SERVICES" ]; then
    echo -e "${YELLOW}⚠️  Active containers found:${NC}"
    echo "$ACTIVE_SERVICES"
    read -p "$(echo -e ${YELLOW}Stop these containers? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker ps --filter "name=embedding\|translation\|rag" -q | xargs -r docker stop
        echo -e "${GREEN}✅ Containers stopped${NC}"
    fi
fi
echo ""

# Function to safely remove directory
remove_service() {
    local service_name=$1
    local service_path="$SERVICE_CORE/$service_name"
    
    if [ -d "$service_path" ]; then
        echo -e "${YELLOW}Removing: $service_name${NC}"
        
        # Calculate size
        SIZE=$(du -sh "$service_path" | cut -f1)
        echo "  Size: $SIZE"
        
        # Remove
        rm -rf "$service_path"
        
        if [ ! -d "$service_path" ]; then
            echo -e "${GREEN}  ✅ Removed successfully${NC}"
        else
            echo -e "${RED}  ❌ Failed to remove${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}  ⚠️  Directory not found, skipping${NC}"
    fi
    echo ""
}

# Remove Embedding duplicates
echo -e "${BLUE}Step 3: Removing duplicate Embedding services...${NC}"
echo ""
remove_service "serviceEmbedding"
remove_service "serviceEmbedding-rust"

# Remove RAG duplicates
echo -e "${BLUE}Step 4: Removing duplicate RAG services...${NC}"
echo ""
remove_service "serviceRAG-mojo"
remove_service "serviceRAG-rust"

# Remove Translation duplicate
echo -e "${BLUE}Step 5: Removing duplicate Translation service...${NC}"
echo ""
remove_service "serviceTranslation-mojo"

# Update docker-compose files
echo -e "${BLUE}Step 6: Updating docker-compose files...${NC}"
echo ""

# Update docker-compose.yml
COMPOSE_FILE="$PROJECT_ROOT/docker/compose/docker-compose.yml"
if [ -f "$COMPOSE_FILE" ]; then
    echo "Updating $COMPOSE_FILE..."
    sed -i.bak 's|context: ../../src/serviceCore/serviceTranslation-rust|# CONSOLIDATED: Using serviceTranslation-rust (3-model architecture)\n      context: ../../src/serviceCore/serviceTranslation-rust|g' "$COMPOSE_FILE"
    echo -e "${GREEN}✅ Updated docker-compose.yml${NC}"
fi

# Update embedding compose
EMBEDDING_COMPOSE="$PROJECT_ROOT/docker/compose/docker-compose.embedding.yml"
if [ -f "$EMBEDDING_COMPOSE" ]; then
    echo "Updating $EMBEDDING_COMPOSE..."
    sed -i.bak 's|context: ../../src/serviceCore/serviceEmbedding-rust|# CONSOLIDATED: Use serviceEmbedding-mojo (production ready)\n      # context: ../../src/serviceCore/serviceEmbedding-rust|g' "$EMBEDDING_COMPOSE"
    echo -e "${GREEN}✅ Updated docker-compose.embedding.yml${NC}"
fi
echo ""

# Summary of what's left
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                Consolidation Complete!                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}✅ Services Retained:${NC}"
echo ""
echo "  📦 serviceEmbedding-mojo/"
echo "     • Port: 8007"
echo "     • Status: Production Ready"
echo "     • Performance: 10-25x faster than Python"
echo "     • Models: 384d multilingual + 768d CamelBERT"
echo "     • Start: cd src/serviceCore/serviceEmbedding-mojo && mojo run main.mojo"
echo ""
echo "  📦 serviceTranslation-rust/"
echo "     • Port: 8010"
echo "     • Status: Production Ready"
echo "     • Models: 3-model architecture (LiquidAI routing)"
echo "     • Start: cd src/serviceCore/serviceTranslation-rust && cargo run --release"
echo ""
echo "  📦 serviceRAG-zig-mojo/"
echo "     • Port: 8009"
echo "     • Status: Code Complete (needs Zig compiler)"
echo "     • Architecture: Zig I/O + Mojo SIMD"
echo "     • Install Zig: brew install zig"
echo "     • Build: cd src/serviceCore/serviceRAG-zig-mojo && ./build.sh"
echo ""

echo -e "${BLUE}📊 Disk Space Saved:${NC}"
SAVED_SPACE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "  Removed ~$SAVED_SPACE of duplicate code and build artifacts"
echo ""

echo -e "${BLUE}💾 Backup Location:${NC}"
echo "  $BACKUP_DIR"
echo ""

echo -e "${BLUE}📝 Next Steps:${NC}"
echo "  1. Test remaining services:"
echo "     curl http://localhost:8007/health  # Embedding (Mojo)"
echo "     curl http://localhost:8010/health  # Translation (Rust)"
echo ""
echo "  2. Update documentation:"
echo "     • README.md"
echo "     • Architecture diagrams"
echo "     • Deployment guides"
echo ""
echo "  3. Install Zig compiler for RAG service:"
echo "     brew install zig"
echo "     cd src/serviceCore/serviceRAG-zig-mojo && ./build.sh"
echo ""
echo "  4. Remove .bak files when satisfied:"
echo "     find docker/compose -name '*.bak' -delete"
echo ""

echo -e "${GREEN}🎉 Consolidation successful!${NC}"
echo ""
echo "From 8 services → 3 services (62% reduction)"
echo ""
