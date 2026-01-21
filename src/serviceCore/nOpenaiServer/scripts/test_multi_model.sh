#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_BIN="$PROJECT_DIR/shimmy_openai_server"
CONFIG_JSON="$PROJECT_DIR/config.multimodel.json"
LOG_FILE="$(mktemp /tmp/multimodel-log.XXXXXX)"
PORT="${SHIMMY_TEST_PORT:-11439}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$LOG_FILE"
}
trap cleanup EXIT

echo "== Shimmy multi-model integration test =="

if [[ -z "${SHIMMY_TEST_MODEL_A:-}" || -z "${SHIMMY_TEST_MODEL_B:-}" ]]; then
  echo "Please set SHIMMY_TEST_MODEL_A and SHIMMY_TEST_MODEL_B to lightweight model paths."
  exit 1
fi

cat > "$CONFIG_JSON" <<JSON
{
  "host": "127.0.0.1",
  "port": $PORT,
  "num_workers": 4,
  "models": [
    { "id": "model-a", "path": "${SHIMMY_TEST_MODEL_A}", "preload": true },
    { "id": "model-b", "path": "${SHIMMY_TEST_MODEL_B}", "preload": true }
  ],
  "default_model_id": "model-a"
}
JSON

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Server binary not found; build with ./scripts/build_zig.sh"
  exit 1
fi

echo "Starting server with $CONFIG_JSON ..."
(
  cd "$PROJECT_DIR"
  SHIMMY_DEBUG=1 SHIMMY_CONFIG="$CONFIG_JSON" ./shimmy_openai_server > "$LOG_FILE" 2>&1 &
  SERVER_PID=$!
)

echo "Waiting for server..."
for _ in {1..30}; do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 1
done

if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  echo "Server failed to start. Logs:"
  cat "$LOG_FILE"
  exit 1
fi

test_chat() {
  local model_id="$1"
  curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\":\"$model_id\",
      \"messages\":[{\"role\":\"user\",\"content\":\"Hello from $model_id\"}]
    }" | jq .
}

echo "Testing model-a..."
test_chat "model-a" >/tmp/mm_a.json
echo "Testing model-b..."
test_chat "model-b" >/tmp/mm_b.json

echo "Responses stored in /tmp/mm_a.json and /tmp/mm_b.json"
echo "Multi-model integration test complete."
