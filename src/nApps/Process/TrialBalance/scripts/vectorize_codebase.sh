#!/bin/bash
# ==============================================================================
# TrialBalance Codebase Vectorization Script
# ==============================================================================
#
# Vectorizes the TrialBalance codebase and ODPS files using nLocalModels
# embedding service for semantic code search and RAG capabilities.
#
# Usage: ./vectorize_codebase.sh [--api-url URL] [--collection NAME]
#
# Requirements:
#   - nLocalModels service running on port 8006
#   - Qdrant or HANA Vector store available
#
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NLOCALMODELS_URL="${NLOCALMODELS_URL:-http://localhost:8006}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
COLLECTION_NAME="${COLLECTION_NAME:-trial-balance-codebase}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-200}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TB_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$TB_ROOT/backend"
WEBAPP_DIR="$TB_ROOT/webapp"
BUSDOCS_DIR="$TB_ROOT/BusDocs"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       TrialBalance Codebase Vectorization                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            NLOCALMODELS_URL="$2"
            shift 2
            ;;
        --collection)
            COLLECTION_NAME="$2"
            shift 2
            ;;
        --model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  API URL:    $NLOCALMODELS_URL"
echo "  Collection: $COLLECTION_NAME"
echo "  Model:      $EMBEDDING_MODEL"
echo "  Chunk Size: $CHUNK_SIZE"
echo ""

# Check if nLocalModels is running
echo "═══ Checking nLocalModels Service ═══"
HEALTH_CHECK=$(curl -s --connect-timeout 5 "$NLOCALMODELS_URL/health" 2>/dev/null || echo "")
if [ -z "$HEALTH_CHECK" ]; then
    echo -e "${YELLOW}⚠️  nLocalModels not reachable at $NLOCALMODELS_URL${NC}"
    echo "   Starting local processing mode..."
    LOCAL_MODE=true
else
    echo -e "${GREEN}✅ nLocalModels service is running${NC}"
    LOCAL_MODE=false
fi
echo ""

# Create output directory for processed files
OUTPUT_DIR="$TB_ROOT/.vectorized"
mkdir -p "$OUTPUT_DIR"

# Function to process a file
process_file() {
    local file_path="$1"
    local file_type="$2"
    local relative_path="${file_path#$TB_ROOT/}"
    
    # Skip binary files and node_modules
    if [[ "$relative_path" == *"node_modules"* ]] || \
       [[ "$relative_path" == *".git"* ]] || \
       [[ "$relative_path" == *"zig-out"* ]] || \
       [[ "$relative_path" == *"zig-cache"* ]]; then
        return
    fi
    
    # Get file extension
    local ext="${file_path##*.}"
    
    # Determine metadata
    local module=""
    case "$relative_path" in
        backend/src/*) module="backend" ;;
        webapp/controller/*) module="controller" ;;
        webapp/view/*) module="view" ;;
        webapp/service/*) module="service" ;;
        BusDocs/models/odps/*) module="odps" ;;
        BusDocs/models/*) module="model" ;;
        *) module="other" ;;
    esac
    
    # Create document JSON
    local content=$(cat "$file_path" | jq -Rs .)
    local doc_json=$(cat <<EOF
{
    "id": "$(echo -n "$relative_path" | md5sum | cut -d' ' -f1)",
    "content": $content,
    "metadata": {
        "file_path": "$relative_path",
        "file_type": "$file_type",
        "module": "$module",
        "extension": "$ext",
        "collection": "$COLLECTION_NAME"
    }
}
EOF
)
    
    # Save to output
    echo "$doc_json" >> "$OUTPUT_DIR/documents.jsonl"
}

# Function to embed documents via API
embed_documents() {
    local input_file="$1"
    
    echo "Embedding documents via nLocalModels..."
    
    # Read documents and send to embedding API
    while IFS= read -r doc; do
        local content=$(echo "$doc" | jq -r '.content')
        local file_path=$(echo "$doc" | jq -r '.metadata.file_path')
        
        # Call embedding endpoint
        local response=$(curl -s -X POST "$NLOCALMODELS_URL/api/v1/embeddings" \
            -H "Content-Type: application/json" \
            -d "{\"input\": $(echo "$content" | jq -Rs .), \"model\": \"$EMBEDDING_MODEL\"}" 2>/dev/null)
        
        if echo "$response" | jq -e '.data[0].embedding' > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $file_path"
        else
            echo -e "${RED}✗${NC} $file_path (embedding failed)"
        fi
    done < "$input_file"
}

# Clear previous output
rm -f "$OUTPUT_DIR/documents.jsonl"

# ==============================================================================
# Process Backend (Zig files)
# ==============================================================================
echo "═══ Processing Backend (Zig) ═══"
ZIG_COUNT=0
find "$BACKEND_DIR" -name "*.zig" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "zig"
    ZIG_COUNT=$((ZIG_COUNT + 1))
done
echo -e "Found Zig files in backend directory"

# ==============================================================================
# Process Frontend (JavaScript/XML files)
# ==============================================================================
echo ""
echo "═══ Processing Frontend (JS/XML) ═══"
find "$WEBAPP_DIR" -name "*.js" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "javascript"
done
find "$WEBAPP_DIR" -name "*.xml" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "xml"
done
find "$WEBAPP_DIR" -name "*.json" -type f ! -path "*node_modules*" 2>/dev/null | while read -r file; do
    process_file "$file" "json"
done
echo "Processed frontend files"

# ==============================================================================
# Process ODPS Files
# ==============================================================================
echo ""
echo "═══ Processing ODPS Files ═══"
find "$BUSDOCS_DIR" -name "*.yaml" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "yaml"
done
find "$BUSDOCS_DIR" -name "*.json" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "json"
done
find "$BUSDOCS_DIR" -name "*.md" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "markdown"
done
echo "Processed ODPS and documentation files"

# ==============================================================================
# Process Config Files
# ==============================================================================
echo ""
echo "═══ Processing Config Files ═══"
find "$TB_ROOT/config" -type f 2>/dev/null | while read -r file; do
    process_file "$file" "config"
done
echo "Processed config files"

# Count documents
DOC_COUNT=$(wc -l < "$OUTPUT_DIR/documents.jsonl" 2>/dev/null || echo "0")

echo ""
echo "═══ Summary ═══"
echo ""
echo "Documents prepared: $DOC_COUNT"
echo "Output file: $OUTPUT_DIR/documents.jsonl"
echo ""

# If nLocalModels is available, embed documents
if [ "$LOCAL_MODE" = false ]; then
    echo "═══ Embedding Documents ═══"
    echo ""
    
    # Check for embedding endpoint
    EMBED_CHECK=$(curl -s "$NLOCALMODELS_URL/api/v1/models" 2>/dev/null || echo "")
    if echo "$EMBED_CHECK" | grep -q "embed"; then
        embed_documents "$OUTPUT_DIR/documents.jsonl"
    else
        echo -e "${YELLOW}⚠️  Embedding model not available${NC}"
        echo "   Documents saved to: $OUTPUT_DIR/documents.jsonl"
        echo "   Load manually when embedding service is available"
    fi
else
    echo "═══ Local Mode ═══"
    echo ""
    echo "Documents prepared for later embedding:"
    echo "  $OUTPUT_DIR/documents.jsonl"
    echo ""
    echo "To embed when nLocalModels is available:"
    echo "  export NLOCALMODELS_URL=http://localhost:8006"
    echo "  ./vectorize_codebase.sh"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Vectorization Complete                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""