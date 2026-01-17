#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/zig-out/bin/leanshimmy"
PORT="${LEANSHIMMY_PORT:-18001}"
LOG="/tmp/leanshimmy.log"
MODEL_PATH="${LEANSHIMMY_MODEL_PATH:-stub-model}"
UNAME="$(uname -s)"
LIB_EXT="dylib"
if [[ "$UNAME" == "Linux" ]]; then
  LIB_EXT="so"
fi
LIB_PATH="${LEANSHIMMY_INFERENCE_LIB:-$ROOT/zig-out/lib/libinference_fixture.${LIB_EXT}}"

ensure_fixture() {
  if [[ -f "$LIB_PATH" ]]; then
    return
  fi
  echo "Building inference fixture at $LIB_PATH..."
  mkdir -p "$ROOT/zig-out/lib"
  (cd "$ROOT" && zig build-lib -dynamic server/inference_fixture.zig -femit-bin="$LIB_PATH")
}

if [ ! -x "$BIN" ]; then
  echo "Building leanshimmy..."
  (cd "$ROOT" && zig build)
fi

echo "Starting leanshimmy on port $PORT with lib $LIB_PATH..."
ensure_fixture
LEANSHIMMY_PORT="$PORT" LEANSHIMMY_MODEL_PATH="$MODEL_PATH" LEANSHIMMY_INFERENCE_LIB="$LIB_PATH" "$BIN" >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

sleep 1

chat_body='{"model":"stub","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"temperature":0.7,"stream":false}'
completion_body='{"model":"stub","prompt":"hi","max_tokens":32,"temperature":0.5}'
embedding_body='{"model":"stub","input":"hello"}'
chat_stream_body='{"model":"stub","messages":[{"role":"user","content":"stream please"}],"max_tokens":16,"temperature":0.6,"stream":true}'
chat_bad_body='{}'

attempt() {
  local url="$1"; shift
  local data="$1"; shift
  local tries=5
  local delay=1
  local resp=""
  for _ in $(seq 1 $tries); do
    if resp=$(curl -sf -X POST "$url" -H "Content-Type: application/json" -d "$data"); then
      echo "$resp"
      return 0
    fi
    sleep $delay
  done
  return 22
}

attempt_stream() {
  local url="$1"; shift
  local data="$1"; shift
  local tries=5
  local delay=1
  local resp=""
  for _ in $(seq 1 $tries); do
    if resp=$(curl -sfN -X POST "$url" -H "Content-Type: application/json" -d "$data"); then
      echo "$resp"
      return 0
    fi
    sleep $delay
  done
  return 22
}

attempt_error_400() {
  local url="$1"; shift
  local data="$1"; shift
  local tries=5
  local delay=1
  for _ in $(seq 1 $tries); do
    local tmp
    tmp=$(mktemp)
    local code
    code=$(curl -s -o "$tmp" -w "%{http_code}" -X POST "$url" -H "Content-Type: application/json" -d "$data")
    if [ "$code" = "400" ]; then
      cat "$tmp"
      rm -f "$tmp"
      return 0
    fi
    rm -f "$tmp"
    sleep $delay
  done
  return 22
}

chat_resp=$(attempt "http://127.0.0.1:${PORT}/v1/chat/completions" "$chat_body")
completion_resp=$(attempt "http://127.0.0.1:${PORT}/v1/completions" "$completion_body")
embedding_resp=$(attempt "http://127.0.0.1:${PORT}/v1/embeddings" "$embedding_body")
chat_stream_resp=$(attempt_stream "http://127.0.0.1:${PORT}/v1/chat/completions" "$chat_stream_body")
chat_error_resp=$(attempt_error_400 "http://127.0.0.1:${PORT}/v1/chat/completions" "$chat_bad_body")

CHAT_RESP="$chat_resp" COMP_RESP="$completion_resp" EMB_RESP="$embedding_resp" CHAT_STREAM="$chat_stream_resp" CHAT_ERR="$chat_error_resp" python - <<'PY'
import json, sys, os
responses = {
    "chat": os.environ["CHAT_RESP"],
    "completion": os.environ["COMP_RESP"],
    "embedding": os.environ["EMB_RESP"],
}
for name, raw in responses.items():
    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"{name} response not JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if name == "chat":
        if data.get("object") != "chat.completion" or not data.get("choices"):
            print("chat response shape invalid", file=sys.stderr); sys.exit(1)
        if not data["choices"][0]["message"]["content"]:
            print("chat content empty", file=sys.stderr); sys.exit(1)
    if name == "completion":
        if data.get("object") not in ("text_completion", "completion"):
            print("completion response shape invalid", file=sys.stderr); sys.exit(1)
        if not data["choices"][0].get("text"):
            print("completion text empty", file=sys.stderr); sys.exit(1)
    if name == "embedding":
        if data.get("object") != "list":
            print("embedding response shape invalid", file=sys.stderr); sys.exit(1)
        emb = data["data"][0].get("embedding")
        if not emb or len(emb) == 0:
            print("embedding empty", file=sys.stderr); sys.exit(1)

stream_raw = os.environ["CHAT_STREAM"]
lines = [ln.strip() for ln in stream_raw.splitlines() if ln.strip()]
data_lines = [ln[len("data: "):] for ln in lines if ln.startswith("data: ")]
if not data_lines:
    print("stream response missing data lines", file=sys.stderr); sys.exit(1)
if "[DONE]" not in stream_raw:
    print("stream response missing DONE sentinel", file=sys.stderr); sys.exit(1)
first_payload = data_lines[0]
try:
    first = json.loads(first_payload)
except Exception as e:
    print(f"stream first chunk not JSON: {e}", file=sys.stderr); sys.exit(1)
if first.get("object") != "chat.completion.chunk":
    print("stream chunk object invalid", file=sys.stderr); sys.exit(1)
choices = first.get("choices") or []
if not choices:
    print("stream chunk choices empty", file=sys.stderr); sys.exit(1)
delta = choices[0].get("delta", {})
if not delta.get("role") and not delta.get("content"):
    print("stream delta missing role/content", file=sys.stderr); sys.exit(1)

err_raw = os.environ["CHAT_ERR"]
try:
    err = json.loads(err_raw)
except Exception as e:
    print(f"error response not JSON: {e}", file=sys.stderr); sys.exit(1)
if err.get("error", {}).get("code") != "prompt_required":
    print("error code mismatch for missing prompt", file=sys.stderr); sys.exit(1)
print("ok")
PY

echo "Smoke test passed."
