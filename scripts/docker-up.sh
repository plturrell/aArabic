#!/bin/bash
# Unified Docker Compose Launcher
# Starts all AI Nucleus services using organized compose files

set -e

COMPOSE_DIR="docker/compose"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🚀 Starting AI Nucleus Platform..."
echo ""

# Create network if it doesn't exist
docker network create ai_nucleus_network 2>/dev/null || true

# Start services in order
services=(
    "core"
    "qdrant"
    "embedding"
)

for service in "${services[@]}"; do
    compose_file="$COMPOSE_DIR/docker-compose.$service.yml"
    
    if [ -f "$compose_file" ]; then
        echo "📦 Starting $service services..."
        docker-compose -f "$compose_file" up -d
        echo "✅ $service started"
        echo ""
    else
        echo "⚠️  Compose file not found: $compose_file"
    fi
done

echo ""
echo "🎉 All services started!"
echo ""
echo "📊 Service Status:"
docker-compose \
    -f "$COMPOSE_DIR/docker-compose.core.yml" \
    -f "$COMPOSE_DIR/docker-compose.qdrant.yml" \
    -f "$COMPOSE_DIR/docker-compose.embedding.yml" \
    ps

echo ""
echo "🔗 Service URLs:"
echo "  Local LLM:   http://localhost:8006"
echo "  Embedding:   http://localhost:8007"
echo "  Qdrant:      http://localhost:6333"
echo "  Memgraph:    bolt://localhost:7687"
echo "  Keycloak:    http://localhost:8080"
echo ""
echo "💡 To stop all services: ./scripts/docker-down.sh"