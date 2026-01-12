#!/bin/bash
# Start all services using Docker Compose

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$PROJECT_ROOT/docker/compose"

echo "🚀 Starting AI Nucleus Platform..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "📦 Building and starting containers..."
docker compose --project-directory "$PROJECT_ROOT" \
    -f "$COMPOSE_DIR/docker-compose.yml" \
    -f "$COMPOSE_DIR/docker-compose.services.yml" \
    -f "$COMPOSE_DIR/docker-compose.wrappers.yml" \
    up -d --build

echo ""
echo "✅ Services started!"
echo ""
echo "┌───────────────────────────────────────────────────────────────────┐"
echo "│                      AI Nucleus Platform                          │"
echo "├───────────────────────────────────────────────────────────────────┤"
echo "│  🌐 MAIN GATEWAY (ONLY EXTERNAL ACCESS)                           │"
echo "│  └─ All Services:        http://localhost                         │"
echo "│                          (Port 80 via APISIX Gateway)             │"
echo "├───────────────────────────────────────────────────────────────────┤"
echo "│  📍 SERVICE ROUTES (Access via Gateway)                           │"
echo "│  ├─ Backend API:         http://localhost/api                     │"
echo "│  ├─ Lean4 Runtime:       http://localhost/lean4                   │"
echo "│  ├─ Shimmy AI:           http://localhost/shimmy                  │"
echo "│  ├─ Langflow:            http://localhost/langflow                │"
echo "│  ├─ Open Canvas:         http://localhost/canvas                  │"
echo "│  ├─ HyperbookLM:         http://localhost/hyperbooklm             │"
echo "│  ├─ NucleusGraph:        http://localhost/graph                   │"
echo "│  ├─ Gitea:               http://localhost/git                     │"
echo "│  ├─ Marquez Lineage:     http://localhost/lineage                 │"
echo "│  ├─ Keycloak Auth:       http://localhost/auth                    │"
echo "│  └─ Portainer (Direct):  http://localhost:9000                    │"
echo "├───────────────────────────────────────────────────────────────────┤"
echo "│  🔒 INTERNAL SERVICES (Not directly accessible)                   │"
echo "│  All services run on private network - access via gateway only    │"
echo "└───────────────────────────────────────────────────────────────────┘"
echo ""

# Wait for services to be healthy
echo "⏳ Waiting for services to become healthy..."
sleep 10

# Check health of critical services
check_health() {
    local service=$1
    local url=$2
    if curl -sf "$url" > /dev/null 2>&1; then
        echo "   ✅ $service is healthy"
    else
        echo "   ⏳ $service is starting..."
    fi
}

echo ""
echo "🔍 Checking service health..."
check_health "Gateway (APISIX)" "http://localhost/apisix/status"
check_health "Translation Service" "http://localhost/translate/health"
check_health "Backend API" "http://localhost/api/health"
echo ""
echo "ℹ️  Note: Individual services are on internal network."
echo "   Access them via gateway routes listed above."
echo ""
echo "📊 Service Map:"
echo "   Gateway:      http://localhost (APISIX on port 80)"
echo "   Translation:  http://localhost/translate"
echo "   Backend API:  http://localhost/api"
echo "   Langflow:     http://localhost/langflow"
echo "   OpenCanvas:   http://localhost/canvas"
echo "   NucleusGraph: http://localhost/graph"
echo "   Portainer:    http://localhost:9000 (Container Management UI)"
echo ""
echo "   To check container status:"
echo "   docker compose -f $COMPOSE_DIR/docker-compose.yml \\"
echo "                  -f $COMPOSE_DIR/docker-compose.services.yml \\"
echo "                  -f $COMPOSE_DIR/docker-compose.wrappers.yml ps"

echo ""
echo "📜 Tailing logs (Ctrl+C to stop following logs, services will remain running)..."
docker compose --project-directory "$PROJECT_ROOT" \
    -f "$COMPOSE_DIR/docker-compose.yml" \
    -f "$COMPOSE_DIR/docker-compose.services.yml" \
    -f "$COMPOSE_DIR/docker-compose.wrappers.yml" \
    logs -f
