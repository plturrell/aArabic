#!/bin/bash
# nCode Docker Cleanup Script
# 
# Safely stops and removes nCode Docker containers and volumes
# 
# Usage:
#   ./scripts/docker-cleanup.sh              # Safe cleanup (preserves data)
#   ./scripts/docker-cleanup.sh --full       # Full cleanup (removes volumes)
#   ./scripts/docker-cleanup.sh --reset      # Complete reset (removes everything)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🧹 nCode Docker Cleanup${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Parse arguments
FULL_CLEANUP=false
RESET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_CLEANUP=true
            shift
            ;;
        --reset)
            RESET=true
            FULL_CLEANUP=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Usage: $0 [--full] [--reset]"
            exit 1
            ;;
    esac
done

# Navigate to project root
cd "$PROJECT_ROOT"

# Confirm destructive operations
if [ "$FULL_CLEANUP" = true ]; then
    echo -e "${YELLOW}⚠️  WARNING: This will remove Docker volumes and delete all data!${NC}"
    echo -e "${YELLOW}⚠️  Databases (Qdrant, Memgraph, Marquez) will lose all stored data.${NC}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo -e "${YELLOW}Cleanup cancelled.${NC}"
        exit 0
    fi
fi

# Stop containers
echo -e "${YELLOW}🛑 Stopping nCode containers...${NC}"
if docker-compose ps -q > /dev/null 2>&1; then
    docker-compose down
    echo -e "${GREEN}✓ Containers stopped${NC}"
else
    echo -e "${YELLOW}⚠️  No running containers found${NC}"
fi

# Remove containers
echo ""
echo -e "${YELLOW}🗑️  Removing containers...${NC}"
CONTAINERS=$(docker ps -a --filter "name=ncode-" -q)
if [ -n "$CONTAINERS" ]; then
    docker rm -f $CONTAINERS
    echo -e "${GREEN}✓ Containers removed${NC}"
else
    echo -e "${YELLOW}⚠️  No containers to remove${NC}"
fi

# Remove volumes if full cleanup
if [ "$FULL_CLEANUP" = true ]; then
    echo ""
    echo -e "${YELLOW}💾 Removing volumes...${NC}"
    VOLUMES=$(docker volume ls --filter "name=ncode-" -q)
    if [ -n "$VOLUMES" ]; then
        docker volume rm $VOLUMES 2>/dev/null || true
        echo -e "${GREEN}✓ Volumes removed${NC}"
    else
        echo -e "${YELLOW}⚠️  No volumes to remove${NC}"
    fi
fi

# Remove network
echo ""
echo -e "${YELLOW}🌐 Removing network...${NC}"
if docker network ls | grep -q "ncode-network"; then
    docker network rm ncode-network 2>/dev/null || true
    echo -e "${GREEN}✓ Network removed${NC}"
else
    echo -e "${YELLOW}⚠️  No network to remove${NC}"
fi

# Remove images if reset
if [ "$RESET" = true ]; then
    echo ""
    echo -e "${YELLOW}🖼️  Removing images...${NC}"
    IMAGES=$(docker images --filter "reference=ncode*" -q)
    if [ -n "$IMAGES" ]; then
        docker rmi -f $IMAGES
        echo -e "${GREEN}✓ Images removed${NC}"
    else
        echo -e "${YELLOW}⚠️  No images to remove${NC}"
    fi
    
    # Clean build cache
    echo ""
    echo -e "${YELLOW}🧹 Cleaning build cache...${NC}"
    docker builder prune -f
    echo -e "${GREEN}✓ Build cache cleaned${NC}"
fi

# Clean up local directories (if full cleanup)
if [ "$FULL_CLEANUP" = true ]; then
    echo ""
    echo -e "${YELLOW}📁 Cleaning local directories...${NC}"
    
    if [ -d "$PROJECT_ROOT/logs" ]; then
        rm -rf "$PROJECT_ROOT/logs"/*
        echo -e "${GREEN}✓ Logs cleared${NC}"
    fi
    
    if [ -d "$PROJECT_ROOT/indexes" ]; then
        echo -e "${YELLOW}⚠️  Indexes directory preserved (contains project data)${NC}"
    fi
fi

# System cleanup
echo ""
echo -e "${YELLOW}🧽 Running system cleanup...${NC}"
docker system prune -f > /dev/null 2>&1
echo -e "${GREEN}✓ System cleanup complete${NC}"

# Summary
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}✅ Cleanup Complete${NC}"
echo -e "${BLUE}======================================================================${NC}"

if [ "$FULL_CLEANUP" = true ]; then
    echo -e "${YELLOW}⚠️  All data has been removed${NC}"
    echo -e "${YELLOW}⚠️  You will need to re-index projects and reload databases${NC}"
else
    echo -e "${GREEN}✓ Containers stopped (data volumes preserved)${NC}"
fi

echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo -e "  • Start fresh: ${GREEN}docker-compose up -d${NC}"
echo -e "  • View status: ${GREEN}docker-compose ps${NC}"
if [ "$FULL_CLEANUP" = true ]; then
    echo -e "  • Re-index project: ${GREEN}./scripts/index_project.sh${NC}"
fi
echo ""
