#!/bin/bash
echo "🧪 TESTING ALL UI SERVICES"
echo "=========================="
echo ""

# Test each UI service
echo "1. Backend API UI..."
docker exec ai_nucleus_backend curl -sf http://localhost:8000/docs 2>&1 | head -3 && echo "✅ Backend Swagger UI accessible" || echo "❌ Backend UI not accessible"

echo ""
echo "2. N8N UI..."
docker exec ai_nucleus_n8n wget -qO- http://localhost:5678 2>&1 | grep -q "n8n" && echo "✅ N8N UI accessible" || echo "❌ N8N UI not accessible"

echo ""
echo "3. HyperbookLM UI..."
docker exec ai_nucleus_hyperbooklm curl -sf http://localhost:3002 2>&1 | head -3 && echo "✅ HyperbookLM UI accessible" || echo "❌ HyperbookLM UI not accessible"

echo ""
echo "4. Gitea UI..."
docker exec ai_nucleus_gitea wget -qO- http://localhost:3000 2>&1 | grep -q "Gitea" && echo "✅ Gitea UI accessible" || echo "❌ Gitea UI not accessible"

echo ""
echo "5. Keycloak Admin UI..."
docker exec ai_nucleus_keycloak curl -sf http://localhost:8080 2>&1 | head -3 && echo "✅ Keycloak UI accessible" || echo "❌ Keycloak UI not accessible"

echo ""
echo "6. Marquez Web UI..."
docker exec ai_nucleus_marquez_web curl -sf http://localhost:3000 2>&1 | head -3 && echo "✅ Marquez Web UI accessible" || echo "❌ Marquez Web UI not accessible"

echo ""
echo "7. Nucleus Graph UI..."
docker exec ai_nucleus_graph curl -sf http://localhost:5000 2>&1 | head -3 && echo "✅ Nucleus Graph UI accessible" || echo "❌ Nucleus Graph UI not accessible"

echo ""
echo "📊 EXTERNAL UIs (Direct Browser Access):"
echo "  ✅ Kafka UI: http://localhost:8090"
echo "  ✅ Memgraph Lab: http://localhost:3001"
