#!/bin/bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Production Shimmy Dashboard (Pure Zig)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if binaries exist
if [ ! -f "./openai_http_server" ]; then
    echo "❌ Error: openai_http_server not found!"
    echo "Please compile: zig build-exe openai_http_server.zig -O ReleaseFast"
    exit 1
fi

if [ ! -f "./production_server" ]; then
    echo "❌ Error: production_server not found!"
    echo "Please compile: zig build-exe production_server.zig -O ReleaseFast"
    exit 1
fi

# Kill existing processes
echo "🧹 Cleaning up old processes..."
pkill -f production_server 2>/dev/null || true
pkill -f unified_server 2>/dev/null || true
pkill -f openai_http_server 2>/dev/null || true
pkill -f "nWebServe.*3000" 2>/dev/null || true
sleep 2

# Start OpenAI API server (background) with model directory
echo "🦙 Starting OpenAI API Server (port 11434)..."
SHIMMY_MODEL_DIR="/Users/user/Documents/arabic_folder/vendor/layerModels" \
./openai_http_server > /tmp/openai_server.log 2>&1 &
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
./production_server > /tmp/production_server.log 2>&1 &
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
