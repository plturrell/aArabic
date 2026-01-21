#!/bin/bash

# TypeScript Project Example - Automated Demo Script
# This script demonstrates the complete workflow of indexing a TypeScript
# project and loading it into Qdrant for semantic search

set -e  # Exit on error

echo "🚀 TypeScript Example - Qdrant Integration"
echo "=========================================="
echo ""

# Check if nCode server is running
echo "Step 1: Checking nCode server..."
if curl -s http://localhost:18003/health > /dev/null 2>&1; then
    echo "✅ nCode server is running"
else
    echo "❌ nCode server is not running!"
    echo "   Please start it with: cd ../.. && ./scripts/start.sh"
    exit 1
fi
echo ""

# Install dependencies
echo "Step 2: Installing dependencies..."
if [ ! -d "node_modules" ]; then
    npm install --silent
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi
echo ""

# Check for scip-typescript
echo "Step 3: Checking for scip-typescript..."
if command -v scip-typescript > /dev/null 2>&1; then
    echo "✅ scip-typescript found"
elif npx --version > /dev/null 2>&1; then
    echo "✅ Will use npx @sourcegraph/scip-typescript"
else
    echo "❌ scip-typescript not found!"
    echo "   Install with: npm install -g @sourcegraph/scip-typescript"
    exit 1
fi
echo ""

# Build TypeScript project
echo "Step 4: Building TypeScript project..."
npm run build --silent
echo "✅ Build complete"
echo ""

# Generate SCIP index
echo "Step 5: Generating SCIP index..."
if command -v scip-typescript > /dev/null 2>&1; then
    scip-typescript index --output=index.scip
else
    npx @sourcegraph/scip-typescript index --output=index.scip
fi
echo "✅ Generated index.scip"
echo ""

# Get index stats
if [ -f "index.scip" ]; then
    INDEX_SIZE=$(du -h index.scip | cut -f1)
    echo "   Index size: $INDEX_SIZE"
fi
echo ""

# Load index into nCode server
echo "Step 6: Loading index into nCode server..."
FULL_PATH="$(pwd)/index.scip"
RESPONSE=$(curl -s -X POST http://localhost:18003/v1/index/load \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"$FULL_PATH\"}")

if echo "$RESPONSE" | grep -q "success\|loaded" > /dev/null 2>&1; then
    echo "✅ Index loaded successfully"
else
    echo "⚠️  Index load response: $RESPONSE"
fi
echo ""

# Check if Qdrant is running
echo "Step 7: Checking Qdrant availability..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "✅ Qdrant is running"
    
    # Export to Qdrant
    echo ""
    echo "Step 8: Exporting to Qdrant..."
    cd ../..
    python3 scripts/load_to_databases.py \
        examples/typescript_project/index.scip \
        --qdrant \
        --qdrant-host localhost \
        --qdrant-port 6333 \
        --qdrant-collection typescript_example 2>&1 | grep -E "✅|Loaded|Created" || true
    cd examples/typescript_project
    echo "✅ Exported to Qdrant"
else
    echo "⚠️  Qdrant is not running (skipping export)"
    echo "   Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant"
fi
echo ""

# Query examples
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "Step 9: Running semantic search queries..."
    if [ -f "query_qdrant.py" ]; then
        python3 query_qdrant.py
    else
        echo "⚠️  query_qdrant.py not found (skipping)"
    fi
else
    echo "Step 9: Semantic search queries (skipped - Qdrant not available)"
fi
echo ""

echo "✅ Example completed successfully!"
echo ""
echo "📚 What was indexed:"
echo "   - User and Product models with interfaces"
echo "   - Authentication service with token management"
echo "   - Database connection with repository pattern"
echo "   - Utility helper functions"
echo "   - Main application entry point"
echo ""
echo "🔍 Try querying nCode API:"
echo "   # Find all class constructors"
echo "   curl -X POST http://localhost:18003/v1/symbols \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"file\": \"src/models/user.ts\"}'"
echo ""
echo "   # Find definition of User class"
echo "   curl -X POST http://localhost:18003/v1/definition \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"file\": \"src/models/user.ts\", \"line\": 23, \"character\": 13}'"
echo ""
echo "🎯 Next steps:"
echo "   1. Modify the code and re-run: ./run_example.sh"
echo "   2. Try semantic search queries (if Qdrant is running)"
echo "   3. Explore other examples: cd ../python_project"
echo ""
