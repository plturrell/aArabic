#!/bin/bash
# Rebuild Langflow with Memgraph AI Toolkit integration

set -e

echo "🔄 Rebuilding Langflow with Memgraph AI Toolkit..."

# Stop current Langflow container
echo "📦 Stopping current Langflow container..."
docker stop ai_nucleus_langflow 2>/dev/null || true
docker rm ai_nucleus_langflow 2>/dev/null || true

# Build new image
echo "🏗️  Building new Langflow image..."
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile.langflow -t ai_nucleus/langflow:latest .

# Start services
echo "🚀 Starting services..."
cd docker/compose
docker-compose -f docker-compose.core.yml up -d langflow

# Wait for health check
echo "⏳ Waiting for Langflow to be healthy..."
timeout 120 bash -c 'until docker exec ai_nucleus_langflow curl -sf http://localhost:7860/health > /dev/null 2>&1; do sleep 2; done' || {
    echo "❌ Langflow failed to start"
    docker logs ai_nucleus_langflow --tail 50
    exit 1
}

echo ""
echo "✅ Langflow rebuilt successfully!"
echo ""
echo "📊 Access Points:"
echo "  - Langflow UI: http://localhost:9080/langflow"
echo "  - Direct (internal): http://ai_nucleus_langflow:7860"
echo ""
echo "🔗 Integrated Services:"
echo "  - Memgraph: bolt://ai_nucleus_memgraph:7687"
echo "  - Qdrant: http://ai_nucleus_qdrant:6333"
echo "  - Marquez: http://ai_nucleus_marquez:5000"
echo ""
echo "📦 Installed Packages:"
echo "  - langchain-memgraph (LangChain integration)"
echo "  - mcp-memgraph (MCP server)"
echo "  - memgraph-toolbox (Core utilities)"
echo "  - unstructured2graph (Data to graph conversion)"
echo ""
echo "📁 Custom Components:"
echo "  - SCIP Indexer: /app/custom_components/scip_indexer.py"
echo "  - Memgraph Graph: /app/custom_components/memgraph_graph.py"
echo ""
echo "🎯 Next Steps:"
echo "  1. Access Langflow UI via APIxIS: http://localhost:9080/langflow"
echo "  2. Import workflow: src/serviceAutomation/workflows/vendor_code_to_n8n_orchestrated.json"
echo "  3. Process HyperBookLM: vendor/layerIntelligence/hyperbooklm"
echo ""