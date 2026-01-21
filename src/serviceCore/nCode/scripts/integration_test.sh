#!/bin/bash

# Run integration tests against the nCode server.
# Starts the server, runs tests, then shuts down.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PORT="${NCODE_PORT:-18003}"
BIN="$PROJECT_ROOT/zig-out/bin/ncode-server"
API_TEST="$PROJECT_ROOT/zig-out/bin/api-test"
LOG="/tmp/ncode-integration.log"

cd "$PROJECT_ROOT"

# Build everything
echo "Building nCode server and integration tests..."
zig build

# Build integration test binary
echo "Building integration test binary..."
zig build-exe tests/integration/api_test.zig -femit-bin="$API_TEST"

# Start server
echo "Starting nCode server on port $PORT..."
NCODE_PORT="$PORT" "$BIN" >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Server failed to start. Log:"
        cat "$LOG"
        exit 1
    fi
    sleep 1
done

# Run integration tests
echo ""
echo "Running integration tests..."
"$API_TEST" "$PORT"
RESULT=$?

# Cleanup
kill $PID 2>/dev/null || true

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "Integration tests PASSED"
else
    echo ""
    echo "Integration tests FAILED"
    echo "Server log:"
    cat "$LOG"
fi

exit $RESULT

