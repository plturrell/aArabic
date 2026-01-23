#!/bin/bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Production Shimmy Dashboard (Pure Zig)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BRIDGE_DIR="$SCRIPT_DIR/src/hana_bridge"
BRIDGE_LOG="/tmp/hana_bridge.log"
BRIDGE_PORT="${BRIDGE_PORT:-3001}"

# Load .env if present to pick up HANA credentials
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

# Normalize HANA env vars for the Zig OData client
export HANA_HOST="${HANA_HOST:-$HANA_SQL_HOST}"
export HANA_PORT="${HANA_PORT:-${HANA_SQL_PORT:-443}}"
export HANA_DATABASE="${HANA_DATABASE:-${HANA_SQL_DATABASE:-servicecore}}"
export HANA_USER="${HANA_USER:-${HANA_USERNAME:-NUCLEUS_APP}}"
export HANA_PASSWORD="${HANA_PASSWORD:-${HANA_PASSWORD:-}}"
export HANA_SCHEMA="${HANA_SCHEMA:-DBADMIN}"
export HANA_BRIDGE_URL="${HANA_BRIDGE_URL:-http://localhost:${BRIDGE_PORT}/sql}"

# Check if binaries exist
if [ ! -f "$BIN_DIR/openai_http_server" ]; then
    echo "❌ Error: openai_http_server not found!"
    echo "Please compile: zig build-exe src/openai_http_server.zig -O ReleaseFast -femit-bin=bin/openai_http_server"
    exit 1
fi

if [ ! -f "$BIN_DIR/production_server" ]; then
    echo "❌ Error: production_server not found!"
    echo "Please compile: zig build-exe src/production_server.zig -O ReleaseFast -femit-bin=bin/production_server"
    exit 1
fi

# Kill existing processes
echo "🧹 Cleaning up old processes..."
pkill -f production_server 2>/dev/null || true
pkill -f unified_server 2>/dev/null || true
pkill -f openai_http_server 2>/dev/null || true
pkill -f "nWebServe.*3000" 2>/dev/null || true
pkill -f "hana_bridge/server.js" 2>/dev/null || true
sleep 2

# Start HANA bridge (Node) for SQL → HANA Cloud
if lsof -i :"$BRIDGE_PORT" > /dev/null 2>&1; then
    echo "🔗 HANA bridge already running on port $BRIDGE_PORT"
else
    if [ ! -d "$BRIDGE_DIR" ]; then
        echo "❌ HANA bridge directory not found: $BRIDGE_DIR"
        exit 1
    fi
    echo "🔗 Starting HANA bridge (port $BRIDGE_PORT)..."
    (cd "$BRIDGE_DIR" && \
        HANA_HOST="$HANA_HOST" \
        HANA_PORT="$HANA_PORT" \
        HANA_USER="$HANA_USER" \
        HANA_PASSWORD="$HANA_PASSWORD" \
        HANA_SCHEMA="$HANA_SCHEMA" \
        BRIDGE_PORT="$BRIDGE_PORT" \
        node server.js > "$BRIDGE_LOG" 2>&1 &)
    sleep 3
    if ! lsof -i :"$BRIDGE_PORT" > /dev/null 2>&1; then
        echo "❌ HANA bridge failed to start; see $BRIDGE_LOG"
        exit 1
    fi
    echo "✅ HANA bridge running (PID $(lsof -i :"$BRIDGE_PORT" -t | head -n1))"
fi

# Start OpenAI API server (background) with model directory
echo "🦙 Starting OpenAI API Server (port 11434)..."
SHIMMY_MODEL_DIR="/Users/user/Documents/arabic_folder/vendor/layerModels" \
$BIN_DIR/openai_http_server > /tmp/openai_server.log 2>&1 &
OPENAI_PID=$!
sleep 4

# Check if OpenAI server started
if ! lsof -i :11434 > /dev/null 2>&1; then
    echo "❌ OpenAI server failed to start!"
    echo ""
    echo "Checking logs:"
    tail -30 /tmp/openai_server.log
    exit 1
fi
echo "✅ OpenAI API running (PID: $OPENAI_PID)"

# Start production proxy server (background)
echo "🌐 Starting Production Server (port 8080)..."
$BIN_DIR/production_server > /tmp/production_server.log 2>&1 &
PROD_PID=$!
sleep 3

if ! lsof -i :8080 > /dev/null 2>&1; then
    echo "❌ Production server failed to start!"
    echo ""
    echo "Checking logs:"
    tail -20 /tmp/production_server.log
    kill $OPENAI_PID 2>/dev/null || true
    exit 1
fi
echo "✅ Production server running (PID: $PROD_PID)"

# Health checks
echo ""
echo "🏥 Running health checks..."
sleep 2

if curl -s http://localhost:11434/health > /dev/null; then
    echo "✅ OpenAI API responding"
else
    echo "⚠️  OpenAI API health check failed"
fi

if curl -s http://localhost:8080/ > /dev/null; then
    echo "✅ Frontend responding"
else
    echo "⚠️  Frontend health check failed"
fi

# Test API proxy
echo ""
echo "🧪 Testing API proxy..."
API_TEST=$(curl -s http://localhost:8080/api/v1/models | jq -r '.data[0].id' 2>/dev/null || echo "failed")
if [ "$API_TEST" != "failed" ]; then
    echo "✅ API proxy working (model: $API_TEST)"
else
    echo "⚠️  API proxy test failed"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All services started successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Dashboard:  http://localhost:8080"
echo "🦙 OpenAI API: http://localhost:11434 (proxied)"
echo ""
echo "📋 Architecture:"
echo "   Browser → production_server (8080) → Static Files (webapp/*)"
echo "                         ↓"
echo "                    OpenAI API (11434) [LLM Inference]"
echo ""
echo "🔧 Process IDs:"
echo "   OpenAI Server:     $OPENAI_PID"
echo "   Production Server: $PROD_PID"
echo ""
echo "📝 Logs:"
echo "   tail -f /tmp/openai_server.log"
echo "   tail -f /tmp/production_server.log"
echo ""
echo "🛑 To stop:"
echo "   pkill -f production_server"
echo "   pkill -f openai_http_server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
