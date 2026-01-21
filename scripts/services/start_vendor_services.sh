#!/bin/sh
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$PROJECT_ROOT/docker/compose"

echo "🚀 Starting vendor-only stack (internal services only)..."

if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker is not running. Please start Docker and try again."
  exit 1
fi

docker compose --project-directory "$PROJECT_ROOT" \
  -f "$COMPOSE_DIR/docker-compose.vendor-services.yml" \
  -f "$COMPOSE_DIR/docker-compose.wrappers.yml" \
  up -d

echo "✅ Vendor services + wrappers started."
