#!/bin/bash
# Build and run Zig dashboard API server with real data connections

set -e

cd "$(dirname "$0")"

echo "🔨 Building dashboard API server..."
zig build-exe dashboard_api_server.zig \
    --name dashboard_api_server \
    -O ReleaseSafe \
    2>&1 | head -20

if [ -f dashboard_api_server ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting dashboard API server..."
    ./dashboard_api_server
else
    echo "❌ Build failed"
    exit 1
fi
