#!/bin/bash
# HANA Bridge Deployment Script
# Usage: ./scripts/deploy.sh [mode]
# Modes: dev, prod, scaled, monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; exit 1; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

MODE=${1:-dev}

# Check for .env file
if [[ ! -f .env ]]; then
    warn ".env file not found"
    if [[ -f .env.example ]]; then
        log "Copying .env.example to .env"
        cp .env.example .env
        warn "Please edit .env with your HANA credentials"
        exit 1
    fi
fi

case $MODE in
    dev)
        log "Starting in development mode..."
        npm install
        node server.js
        ;;

    prod)
        log "Starting in production mode..."
        npm ci --only=production
        node server.prod.js
        ;;

    pm2)
        log "Starting with PM2..."
        npm ci --only=production
        mkdir -p logs

        if ! command -v pm2 &> /dev/null; then
            warn "PM2 not installed, installing globally..."
            npm install -g pm2
        fi

        pm2 start ecosystem.config.js --env production
        pm2 save
        log "PM2 started. Use 'pm2 logs hana-bridge' to view logs"
        ;;

    docker)
        log "Building and starting Docker container..."
        docker build -t hana-bridge .
        docker run -d \
            --name hana-bridge \
            --env-file .env \
            -p 3001:3001 \
            --restart unless-stopped \
            hana-bridge
        log "Docker container started on port 3001"
        ;;

    compose)
        log "Starting with Docker Compose..."
        docker-compose up -d hana-bridge
        log "Docker Compose started. Use 'docker-compose logs -f' to view logs"
        ;;

    scaled)
        log "Starting scaled deployment with load balancer..."
        docker-compose --profile scaled up -d
        log "Scaled deployment started:"
        info "  - 3x HANA Bridge replicas"
        info "  - Nginx load balancer on port 3000"
        info "Use 'docker-compose --profile scaled logs -f' to view logs"
        ;;

    monitoring)
        log "Starting with full monitoring stack..."
        docker-compose --profile monitoring up -d
        log "Monitoring stack started:"
        info "  - HANA Bridge on port 3001"
        info "  - Prometheus on port 9090"
        info "  - Grafana on port 3030"
        ;;

    full)
        log "Starting full production stack..."
        docker-compose --profile scaled --profile monitoring up -d
        log "Full stack started:"
        info "  - 3x HANA Bridge replicas"
        info "  - Nginx load balancer on port 3000"
        info "  - Prometheus on port 9090"
        info "  - Grafana on port 3030"
        ;;

    stop)
        log "Stopping all services..."
        docker-compose --profile scaled --profile monitoring down 2>/dev/null || true
        pm2 delete hana-bridge 2>/dev/null || true
        docker stop hana-bridge 2>/dev/null || true
        docker rm hana-bridge 2>/dev/null || true
        log "All services stopped"
        ;;

    status)
        log "Checking service status..."
        echo ""

        # Check Docker
        if docker ps --format '{{.Names}}' | grep -q hana; then
            info "Docker containers:"
            docker ps --filter "name=hana" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        fi

        # Check PM2
        if command -v pm2 &> /dev/null && pm2 list | grep -q hana-bridge; then
            echo ""
            info "PM2 processes:"
            pm2 list | grep hana-bridge
        fi

        # Health check
        echo ""
        if curl -s http://localhost:3001/health > /dev/null 2>&1; then
            log "Health check: $(curl -s http://localhost:3001/health)"
        else
            warn "Health check failed - service may not be running"
        fi
        ;;

    *)
        echo "HANA Bridge Deployment Script"
        echo ""
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  dev        - Development mode (single instance)"
        echo "  prod       - Production mode (single instance)"
        echo "  pm2        - PM2 cluster mode"
        echo "  docker     - Single Docker container"
        echo "  compose    - Docker Compose (single instance)"
        echo "  scaled     - Scaled deployment (3 replicas + nginx)"
        echo "  monitoring - With Prometheus + Grafana"
        echo "  full       - Full stack (scaled + monitoring)"
        echo "  stop       - Stop all services"
        echo "  status     - Check service status"
        exit 1
        ;;
esac
