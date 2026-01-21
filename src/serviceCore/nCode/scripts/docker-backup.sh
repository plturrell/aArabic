#!/bin/bash
# nCode Docker Backup Script
# 
# Creates backups of all nCode database volumes and configurations
# 
# Usage:
#   ./scripts/docker-backup.sh                    # Create backup
#   ./scripts/docker-backup.sh --restore latest   # Restore latest backup
#   ./scripts/docker-backup.sh --list             # List backups

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

# Backup configuration
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ncode_backup_${TIMESTAMP}"
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}💾 nCode Docker Backup${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Parse arguments
ACTION="backup"
RESTORE_TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --restore)
            ACTION="restore"
            RESTORE_TARGET="$2"
            shift 2
            ;;
        --list)
            ACTION="list"
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Usage: $0 [--restore <backup>] [--list]"
            exit 1
            ;;
    esac
done

# Create backup directory
mkdir -p "$BACKUP_DIR"

# List backups
if [ "$ACTION" = "list" ]; then
    echo -e "${YELLOW}📋 Available Backups:${NC}"
    echo ""
    
    if [ ! "$(ls -A $BACKUP_DIR 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠️  No backups found${NC}"
        exit 0
    fi
    
    for backup in "$BACKUP_DIR"/ncode_backup_*.tar.gz; do
        if [ -f "$backup" ]; then
            SIZE=$(du -h "$backup" | cut -f1)
            NAME=$(basename "$backup")
            echo -e "  ${GREEN}•${NC} $NAME (${SIZE})"
        fi
    done
    
    echo ""
    echo -e "${BLUE}💡 Restore with: ./scripts/docker-backup.sh --restore <backup_name>${NC}"
    exit 0
fi

# Restore backup
if [ "$ACTION" = "restore" ]; then
    if [ "$RESTORE_TARGET" = "latest" ]; then
        BACKUP_FILE=$(ls -t "$BACKUP_DIR"/ncode_backup_*.tar.gz 2>/dev/null | head -1)
    else
        BACKUP_FILE="$BACKUP_DIR/$RESTORE_TARGET"
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        echo -e "${RED}❌ Backup not found: $BACKUP_FILE${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}📦 Restoring from: $(basename $BACKUP_FILE)${NC}"
    echo ""
    
    # Stop containers
    echo -e "${YELLOW}🛑 Stopping containers...${NC}"
    cd "$PROJECT_ROOT"
    docker-compose down
    echo -e "${GREEN}✓ Containers stopped${NC}"
    
    # Extract backup
    echo ""
    echo -e "${YELLOW}📂 Extracting backup...${NC}"
    TEMP_DIR=$(mktemp -d)
    tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
    echo -e "${GREEN}✓ Backup extracted${NC}"
    
    # Restore volumes
    echo ""
    echo -e "${YELLOW}💾 Restoring volumes...${NC}"
    
    # Qdrant
    if [ -d "$TEMP_DIR/qdrant" ]; then
        docker run --rm -v ncode-qdrant-data:/data -v "$TEMP_DIR/qdrant":/backup alpine sh -c "rm -rf /data/* && cp -a /backup/. /data/"
        echo -e "${GREEN}✓ Qdrant data restored${NC}"
    fi
    
    # Memgraph
    if [ -d "$TEMP_DIR/memgraph" ]; then
        docker run --rm -v ncode-memgraph-data:/data -v "$TEMP_DIR/memgraph":/backup alpine sh -c "rm -rf /data/* && cp -a /backup/. /data/"
        echo -e "${GREEN}✓ Memgraph data restored${NC}"
    fi
    
    # Marquez DB
    if [ -d "$TEMP_DIR/marquez-db" ]; then
        docker run --rm -v ncode-marquez-db-data:/data -v "$TEMP_DIR/marquez-db":/backup alpine sh -c "rm -rf /data/* && cp -a /backup/. /data/"
        echo -e "${GREEN}✓ Marquez database restored${NC}"
    fi
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    # Start containers
    echo ""
    echo -e "${YELLOW}🚀 Starting containers...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✓ Containers started${NC}"
    
    echo ""
    echo -e "${GREEN}✅ Restore Complete${NC}"
    exit 0
fi

# Create backup
echo -e "${YELLOW}📦 Creating backup: $BACKUP_NAME${NC}"
echo ""

# Check if containers are running
cd "$PROJECT_ROOT"
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠️  Containers not running. Start with: docker-compose up -d${NC}"
    echo -e "${YELLOW}Backup will include volumes but not running state.${NC}"
    echo ""
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo -e "${YELLOW}📁 Using temporary directory: $TEMP_DIR${NC}"

# Backup volumes
echo ""
echo -e "${YELLOW}💾 Backing up volumes...${NC}"

# Qdrant
echo -e "  • Backing up Qdrant..."
docker run --rm -v ncode-qdrant-data:/data -v "$TEMP_DIR":/backup alpine tar czf /backup/qdrant.tar.gz -C /data . 2>/dev/null || true
if [ -f "$TEMP_DIR/qdrant.tar.gz" ]; then
    mkdir -p "$TEMP_DIR/qdrant"
    tar -xzf "$TEMP_DIR/qdrant.tar.gz" -C "$TEMP_DIR/qdrant"
    rm "$TEMP_DIR/qdrant.tar.gz"
    echo -e "    ${GREEN}✓ Qdrant backed up${NC}"
else
    echo -e "    ${YELLOW}⚠️  Qdrant volume empty or not found${NC}"
fi

# Memgraph
echo -e "  • Backing up Memgraph..."
docker run --rm -v ncode-memgraph-data:/data -v "$TEMP_DIR":/backup alpine tar czf /backup/memgraph.tar.gz -C /data . 2>/dev/null || true
if [ -f "$TEMP_DIR/memgraph.tar.gz" ]; then
    mkdir -p "$TEMP_DIR/memgraph"
    tar -xzf "$TEMP_DIR/memgraph.tar.gz" -C "$TEMP_DIR/memgraph"
    rm "$TEMP_DIR/memgraph.tar.gz"
    echo -e "    ${GREEN}✓ Memgraph backed up${NC}"
else
    echo -e "    ${YELLOW}⚠️  Memgraph volume empty or not found${NC}"
fi

# Marquez DB
echo -e "  • Backing up Marquez database..."
docker run --rm -v ncode-marquez-db-data:/data -v "$TEMP_DIR":/backup alpine tar czf /backup/marquez-db.tar.gz -C /data . 2>/dev/null || true
if [ -f "$TEMP_DIR/marquez-db.tar.gz" ]; then
    mkdir -p "$TEMP_DIR/marquez-db"
    tar -xzf "$TEMP_DIR/marquez-db.tar.gz" -C "$TEMP_DIR/marquez-db"
    rm "$TEMP_DIR/marquez-db.tar.gz"
    echo -e "    ${GREEN}✓ Marquez database backed up${NC}"
else
    echo -e "    ${YELLOW}⚠️  Marquez database volume empty or not found${NC}"
fi

# Backup configurations
echo ""
echo -e "${YELLOW}⚙️  Backing up configurations...${NC}"
mkdir -p "$TEMP_DIR/config"
cp "$PROJECT_ROOT/docker-compose.yml" "$TEMP_DIR/config/" 2>/dev/null || true
cp "$PROJECT_ROOT/.env" "$TEMP_DIR/config/" 2>/dev/null || true
cp "$PROJECT_ROOT/.env.example" "$TEMP_DIR/config/" 2>/dev/null || true
echo -e "${GREEN}✓ Configurations backed up${NC}"

# Create backup metadata
echo ""
echo -e "${YELLOW}📝 Creating metadata...${NC}"
cat > "$TEMP_DIR/backup_info.txt" << EOF
nCode Backup Information
========================

Backup Name: $BACKUP_NAME
Timestamp: $(date)
Created By: $(whoami)@$(hostname)

Volumes:
- Qdrant: $([ -d "$TEMP_DIR/qdrant" ] && du -sh "$TEMP_DIR/qdrant" | cut -f1 || echo "empty")
- Memgraph: $([ -d "$TEMP_DIR/memgraph" ] && du -sh "$TEMP_DIR/memgraph" | cut -f1 || echo "empty")
- Marquez DB: $([ -d "$TEMP_DIR/marquez-db" ] && du -sh "$TEMP_DIR/marquez-db" | cut -f1 || echo "empty")

Docker Compose Version: $(docker-compose --version)
Docker Version: $(docker --version)

To restore this backup:
./scripts/docker-backup.sh --restore $BACKUP_NAME.tar.gz
EOF
echo -e "${GREEN}✓ Metadata created${NC}"

# Compress backup
echo ""
echo -e "${YELLOW}🗜️  Compressing backup...${NC}"
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$TEMP_DIR" .
BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)
echo -e "${GREEN}✓ Backup compressed (${BACKUP_SIZE})${NC}"

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

# Clean old backups
echo ""
echo -e "${YELLOW}🧹 Cleaning old backups (retention: ${RETENTION_DAYS} days)...${NC}"
find "$BACKUP_DIR" -name "ncode_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete 2>/dev/null || true
REMAINING=$(ls -1 "$BACKUP_DIR"/ncode_backup_*.tar.gz 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Old backups cleaned (${REMAINING} backups remaining)${NC}"

# Summary
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}✅ Backup Complete${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "${GREEN}📦 Backup Location: $BACKUP_DIR/${BACKUP_NAME}.tar.gz${NC}"
echo -e "${GREEN}📊 Backup Size: ${BACKUP_SIZE}${NC}"
echo ""
echo -e "${BLUE}💡 To restore this backup:${NC}"
echo -e "   ./scripts/docker-backup.sh --restore ${BACKUP_NAME}.tar.gz"
echo ""
echo -e "${BLUE}💡 To list all backups:${NC}"
echo -e "   ./scripts/docker-backup.sh --list"
echo ""
