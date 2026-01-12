#!/bin/bash
# Start Shimmy-Mojo HTTP Server

set -e

cd "$(dirname "$0")"

echo "================================================================================"
echo "🚀 Starting Shimmy-Mojo HTTP Server"
echo "================================================================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check for Mojo
if ! command -v mojo &> /dev/null; then
    echo "⚠️  Warning: Mojo is not installed"
    echo "   Server will run in mock mode"
fi

# Install Python dependencies
echo "📦 Checking Python dependencies..."
pip3 install -q fastapi uvicorn pydantic 2>/dev/null || echo "   Dependencies already installed"
echo ""

# Check if server directory exists
if [ ! -d "server" ]; then
    echo "❌ Server directory not found"
    exit 1
fi

# Make sure Python files are executable
chmod +x server/http_server.py
chmod +x server/mojo_bridge.py

echo "🔧 Configuration:"
echo "   Host: 0.0.0.0"
echo "   Port: 11434"
echo "   Models dir: ./models"
echo ""

echo "🌐 Starting server..."
echo ""

# Start the server
python3 server/http_server.py
