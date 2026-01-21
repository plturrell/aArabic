#!/bin/bash
# Unified Dashboard Stack Startup Script
# Starts: nLaunchpad (3000) + Dashboard API (8080) + nWebServe for UI (8081)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Starting Unified Dashboard Stack"
echo "════════════════════════════════════════════════════════════════"

# Check if services are already running
check_port() {
    lsof -ti:$1 > /dev/null 2>&1
}

# Kill existing processes on ports
cleanup_ports() {
    for port in 3000 8080 8081; do
        if check_port $port; then
            echo "⚠️  Port $port is in use, killing existing process..."
            lsof -ti:$port | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
    done
}

cleanup_ports

# Build nWebServe if needed
echo ""
echo "📦 Building nWebServe..."
cd "$PROJECT_ROOT/src/serviceCore/nWebServe"
if [ ! -f "zig-out/bin/nWebServe" ]; then
    zig build
fi

# Build nLaunchpad WebServe if needed
echo ""
echo "📦 Checking nLaunchpad..."
cd "$PROJECT_ROOT/src/serviceCore/nLaunchpad"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Starting Services..."
echo "════════════════════════════════════════════════════════════════"

# Start nWebServe for Launchpad (Port 3000)
echo ""
echo "1️⃣  Starting nLaunchpad on http://localhost:3000"
cd "$PROJECT_ROOT/src/serviceCore/nWebServe"
./zig-out/bin/nWebServe 3000 "$PROJECT_ROOT/src/serviceCore/nLaunchpad/webapp" > /tmp/nlaunchpad.log 2>&1 &
LAUNCHPAD_PID=$!
sleep 2

# Start nWebServe for Dashboard UI (Port 8081)
echo "2️⃣  Starting Dashboard UI on http://localhost:8081"
./zig-out/bin/nWebServe 8081 "$PROJECT_ROOT/src/serviceCore/nOpenaiServer/webapp" > /tmp/dashboard-ui.log 2>&1 &
DASHBOARD_UI_PID=$!
sleep 2

# Start Dashboard API Server (Port 8080) - Mock for now since compilation has issues
echo "3️⃣  Starting Dashboard API (Mock) on http://localhost:8080"
cd "$PROJECT_ROOT"

# Create a simple Python mock API server for now
cat > /tmp/mock_dashboard_api.py << 'PYEOF'
#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time

class MockDashboardAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        if self.path == '/api/v1/models':
            response = {
                "models": [
                    {
                        "id": "llama-3.3-70b",
                        "name": "Llama 3.3 70B",
                        "version": "1.0.0",
                        "architecture": "llama",
                        "quantization": "Q4_K_M"
                    }
                ]
            }
        elif self.path == '/api/v1/metrics':
            response = {
                "type": "metrics_update",
                "timestamp": int(time.time() * 1000),
                "data": {
                    "connected": True,
                    "models": [{
                        "id": "llama-3.3-70b",
                        "name": "Llama 3.3 70B",
                        "health": "healthy",
                        "requests": 1250,
                        "latency": 2,
                        "throughput": 15000
                    }],
                    "tiers": {
                        "gpu": {"used": 0, "total": 80, "hitRate": 0.85},
                        "ram": {"used": 24.5, "total": 128, "hitRate": 0.72},
                        "dragonfly": {"used": 12.1, "total": 64, "hitRate": 0.65},
                        "postgres": {"used": 5, "total": 100, "hitRate": 0.45},
                        "ssd": {"used": 250, "total": 1000, "hitRate": 0.30}
                    },
                    "cache": {
                        "totalHitRate": 0.65,
                        "sharingRatio": 0.42,
                        "compressionRatio": 2.5,
                        "evictions": 125
                    }
                }
            }
        else:
            response = {"error": "Not found"}
        
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), MockDashboardAPI)
    print('Mock Dashboard API running on port 8080')
    server.serve_forever()
PYEOF

chmod +x /tmp/mock_dashboard_api.py
python3 /tmp/mock_dashboard_api.py > /tmp/dashboard-api.log 2>&1 &
API_PID=$!
sleep 2

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ All Services Started!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📍 Service URLs:"
echo "   • nLaunchpad:    http://localhost:3000"
echo "   • Dashboard UI:  http://localhost:8081"
echo "   • Dashboard API: http://localhost:8080"
echo ""
echo "🎯 Open http://localhost:3000 and click 'Shimmy AI' tile"
echo ""
echo "📊 Process IDs:"
echo "   • Launchpad:     $LAUNCHPAD_PID"
echo "   • Dashboard UI:  $DASHBOARD_UI_PID"
echo "   • Dashboard API: $API_PID"
echo ""
echo "🛑 To stop all services:"
echo "   kill $LAUNCHPAD_PID $DASHBOARD_UI_PID $API_PID"
echo ""
echo "📝 Logs:"
echo "   • Launchpad:     tail -f /tmp/nlaunchpad.log"
echo "   • Dashboard UI:  tail -f /tmp/dashboard-ui.log"
echo "   • Dashboard API: tail -f /tmp/dashboard-api.log"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Press Ctrl+C to stop all services"
echo "════════════════════════════════════════════════════════════════"

# Save PIDs for cleanup
echo "$LAUNCHPAD_PID $DASHBOARD_UI_PID $API_PID" > /tmp/dashboard_stack.pids

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Shutting down...'; kill $LAUNCHPAD_PID $DASHBOARD_UI_PID $API_PID 2>/dev/null; rm /tmp/dashboard_stack.pids; echo '✅ All services stopped'; exit 0" INT

# Keep script running
wait
