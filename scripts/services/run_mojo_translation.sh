#!/bin/bash

echo "🌐 MOJO TRANSLATION SERVICE"
echo "======================================"
echo ""

# Check if embedding service is running
echo "📡 Checking embedding service..."
if curl -s http://localhost:8007/health > /dev/null 2>&1; then
    echo "   ✅ Embedding service running (port 8007)"
else
    echo "   ⚠️  Embedding service not running (optional)"
    echo "   💡 Start with: python3 src/serviceCore/serviceEmbedding-mojo/server.py"
fi

# Check if Qdrant is running
echo "📡 Checking Qdrant..."
if curl -s http://localhost:6333/readyz > /dev/null 2>&1; then
    echo "   ✅ Qdrant running (port 6333)"
    
    # Create translations collection if it doesn't exist
    echo "   🔧 Creating translations collection..."
    curl -X PUT http://localhost:6333/collections/translations \
        -H "Content-Type: application/json" \
        -d '{
            "vectors": {
                "size": 384,
                "distance": "Cosine"
            }
        }' > /dev/null 2>&1
    echo "   ✅ Collection ready"
else
    echo "   ⚠️  Qdrant not running (optional for RAG)"
    echo "   💡 Start with: docker-compose -f docker/compose/docker-compose.qdrant.yml up -d"
fi

echo ""
echo "🚀 Starting translation service..."
echo ""

cd /Users/user/Documents/arabic_folder

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -q -r src/serviceCore/serviceTranslation-mojo/requirements.txt
fi

# Start the service
python3 src/serviceCore/serviceTranslation-mojo/server.py
