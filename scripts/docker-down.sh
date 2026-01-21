#!/bin/bash
# Stop all AI Nucleus services

set -e

COMPOSE_DIR="docker/compose"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🛑 Stopping AI Nucleus Platform..."
echo ""

# Stop services in reverse order
services=(
    "embedding"
    "qdrant"
    "core"
)

for service in "${services[@]}"; do
    compose_file="$COMPOSE_DIR/docker-compose.$service.yml"
    
    if [ -f "$compose_file" ]; then
        echo "📦 Stopping $service services..."
        docker-compose -f "$compose_file" down
        echo "✅ $service stopped"
        echo ""
    fi
done

echo "🎉 All services stopped!"
echo ""
echo "💡 To start services again: ./scripts/docker-up.sh"