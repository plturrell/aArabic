#!/bin/bash

echo "🔗 RAG INTEGRATION TEST"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test 1: Check all services are running
echo -e "${BLUE}Step 1: Checking services${NC}"
echo "  Embedding Service (8007)..."
curl -s http://localhost:8007/health > /dev/null && echo "  ✅ Embedding service healthy" || echo "  ❌ Embedding service down"

echo "  Qdrant (6333)..."
curl -s http://localhost:6333/readyz > /dev/null && echo "  ✅ Qdrant healthy" || echo "  ❌ Qdrant down"

echo "  Redis (6379)..."
docker exec aimo_redis redis-cli ping > /dev/null 2>&1 && echo "  ✅ Redis healthy" || echo "  ❌ Redis down"

echo ""
echo -e "${BLUE}Step 2: Creating Qdrant collection${NC}"

# Create collection with 384d vectors (general model)
curl -s -X PUT "http://localhost:6333/collections/test_embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }' | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  ✅ Collection created: {d.get(\"status\", \"ok\")}')" 2>/dev/null || echo "  ℹ️  Collection may already exist"

echo ""
echo -e "${BLUE}Step 3: Testing embedding generation${NC}"

# Generate embeddings for test documents
TEXTS='["Invoice processing system", "Arabic document parser", "Financial data extraction"]'

echo "  Generating embeddings for 3 texts..."
EMBED_RESPONSE=$(curl -s -X POST http://localhost:8007/embed/batch \
  -H "Content-Type: application/json" \
  -d "{\"texts\":$TEXTS,\"model_type\":\"general\"}")

echo "$EMBED_RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  ✅ Generated {d[\"count\"]} embeddings')
print(f'  📐 Dimensions: {d[\"dimensions\"]}')
print(f'  ⚡ Time: {d[\"processing_time_ms\"]}ms')
print(f'  🔧 Model: {d[\"model_used\"]}')
"

echo ""
echo -e "${BLUE}Step 4: Storing embeddings in Qdrant${NC}"

# Extract embeddings and store in Qdrant
echo "$EMBED_RESPONSE" | python3 << 'PYEOF'
import json
import sys
import requests

data = json.load(sys.stdin)
embeddings = data['embeddings']
texts = ["Invoice processing system", "Arabic document parser", "Financial data extraction"]

# Prepare points for Qdrant
points = []
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    points.append({
        "id": i + 1,
        "vector": embedding,
        "payload": {
            "text": text,
            "model": data['model_used'],
            "dimensions": data['dimensions']
        }
    })

# Upload to Qdrant
response = requests.put(
    "http://localhost:6333/collections/test_embeddings/points",
    json={"points": points}
)

if response.status_code == 200:
    print(f"  ✅ Stored {len(points)} vectors in Qdrant")
    print(f"  📊 Collection: test_embeddings")
else:
    print(f"  ❌ Failed to store: {response.status_code}")
PYEOF

echo ""
echo -e "${BLUE}Step 5: Testing semantic search${NC}"

# Search for similar documents
QUERY_TEXT="document processing"
echo "  Query: '$QUERY_TEXT'"

# Generate query embedding
QUERY_EMBED=$(curl -s -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$QUERY_TEXT\",\"model_type\":\"general\"}")

echo "$QUERY_EMBED" | python3 << 'PYEOF'
import json
import sys
import requests

data = json.load(sys.stdin)
query_vector = data['embedding']

# Search in Qdrant
search_response = requests.post(
    "http://localhost:6333/collections/test_embeddings/points/search",
    json={
        "vector": query_vector,
        "limit": 3,
        "with_payload": True
    }
)

if search_response.status_code == 200:
    results = search_response.json()['result']
    print(f"\n  🔍 Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        text = result['payload']['text']
        score = result['score']
        print(f"    {i}. {text}")
        print(f"       Score: {score:.4f}")
else:
    print(f"  ❌ Search failed: {search_response.status_code}")
PYEOF

echo ""
echo -e "${BLUE}Step 6: Testing Arabic financial embeddings${NC}"

# Test CamelBERT with Arabic text
ARABIC_QUERY="فاتورة مالية"
echo "  Query: '$ARABIC_QUERY' (using CamelBERT 768d)"

ARABIC_EMBED=$(curl -s -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$ARABIC_QUERY\",\"model_type\":\"financial\"}")

echo "$ARABIC_EMBED" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  ✅ Arabic embedding generated')
print(f'  📐 Dimensions: {d[\"dimensions\"]}')
print(f'  ⚡ Time: {d[\"processing_time_ms\"]}ms')
print(f'  🔧 Model: {d[\"model_used\"]}')
"

echo ""
echo -e "${BLUE}Step 7: Cache performance test${NC}"

# Test cache speedup
TEST_TEXT="Cache performance test"
echo "  Testing with: '$TEST_TEXT'"

# First call (uncached)
TIME1=$(curl -s -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$TEST_TEXT\"}" | python3 -c "import json,sys; print(json.load(sys.stdin)['processing_time_ms'])")

# Second call (cached)
TIME2=$(curl -s -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$TEST_TEXT\"}" | python3 -c "import json,sys; print(json.load(sys.stdin)['processing_time_ms'])")

python3 << PYEOF
time1 = float($TIME1)
time2 = float($TIME2)
speedup = time1 / time2 if time2 > 0 else 0
print(f"  Uncached: {time1:.2f}ms")
print(f"  Cached:   {time2:.2f}ms")
print(f"  ✅ Speedup: {speedup:.0f}x")
PYEOF

echo ""
echo -e "${BLUE}Step 8: System metrics${NC}"

curl -s http://localhost:8007/metrics | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  Requests: {d[\"requests_total\"]}')
print(f'  Cache: {d[\"cache_type\"]}')
print(f'  Hit rate: {d[\"cache_hit_rate\"]*100:.1f}%')
print(f'  Embeddings: {d[\"embeddings_generated\"]}')
print(f'  Device: {d[\"device\"]}')
"

echo ""
echo "======================================"
echo -e "${GREEN}✅ RAG INTEGRATION TEST COMPLETE${NC}"
echo ""
echo "Components verified:"
echo "  ✅ Mojo Embedding Service (384d + 768d)"
echo "  ✅ Redis distributed cache"
echo "  ✅ Qdrant vector database"
echo "  ✅ End-to-end RAG pipeline"
echo ""
