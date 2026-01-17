#!/bin/bash

# Start HyperShimmy Server

set -e  # Exit on error

echo "======================================================================"
echo "🚀 Starting HyperShimmy Server"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if executable exists
EXECUTABLE="zig-out/bin/hypershimmy"

if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ ERROR: Server executable not found!"
    echo ""
    echo "Expected location: $EXECUTABLE"
    echo ""
    echo "Please build the project first:"
    echo "  ./scripts/build_all.sh"
    echo ""
    exit 1
fi

echo "✓ Found server executable"
echo ""

# Display configuration
echo "Configuration:"
echo "  • Server:    http://localhost:11434"
echo "  • OData:     http://localhost:11434/odata/v4/research/"
echo "  • UI:        http://localhost:11434/"
echo "  • Health:    http://localhost:11434/health"
echo ""

# Check if port is already in use
if lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  WARNING: Port 11434 is already in use"
    echo ""
    echo "Please stop the existing process or choose a different port."
    echo ""
    
    # Show what's using the port
    echo "Process using port 11434:"
    lsof -Pi :11434 -sTCP:LISTEN 2>/dev/null || echo "  (unable to determine)"
    echo ""
    exit 1
fi

echo "======================================================================"
echo "🎯 Starting Server..."
echo "======================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server
# Log both stdout and stderr to file while also displaying to console
"$EXECUTABLE" 2>&1 | tee logs/server.log

# Note: Script will remain running until server is stopped with Ctrl+C
