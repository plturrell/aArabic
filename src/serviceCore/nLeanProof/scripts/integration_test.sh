#!/bin/bash

# Run integration tests against the leanshimmy server.
# Starts the server, runs tests, then shuts down.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PORT="${LEANSHIMMY_PORT:-18002}"
BIN="$PROJECT_ROOT/zig-out/bin/leanshimmy"
LOG="/tmp/leanshimmy-integration.log"
MODEL_PATH="${LEANSHIMMY_MODEL_PATH:-stub-model}"

UNAME="$(uname -s)"
LIB_EXT="dylib"
if [[ "$UNAME" == "Linux" ]]; then
    LIB_EXT="so"
fi
LIB_PATH="${LEANSHIMMY_INFERENCE_LIB:-$PROJECT_ROOT/zig-out/lib/libinference_fixture.${LIB_EXT}}"

cd "$PROJECT_ROOT"

# Build everything
echo "Building leanshimmy and integration tests..."
zig build

# Ensure fixture library exists
if [[ ! -f "$LIB_PATH" ]]; then
    echo "Building inference fixture..."
    mkdir -p "$PROJECT_ROOT/zig-out/lib"
    zig build-lib -dynamic server/inference_fixture.zig -femit-bin="$LIB_PATH"
fi

# Start server
echo "Starting leanshimmy on port $PORT..."
LEANSHIMMY_PORT="$PORT" LEANSHIMMY_MODEL_PATH="$MODEL_PATH" LEANSHIMMY_INFERENCE_LIB="$LIB_PATH" "$BIN" >"$LOG" 2>&1 &
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
./zig-out/bin/api-test "$PORT"
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
