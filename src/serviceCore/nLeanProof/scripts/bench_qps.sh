#!/usr/bin/env bash
set -euo pipefail

# Simple QPS/latency sanity harness for chat completions.
# Usage: bench_qps.sh [-n requests] [-c concurrency] [--min-qps threshold] [--max-seconds sec] [--max-ttft sec] [--no-start]
# Measures aggregate QPS and elapsed; TTFT/latency sampling is approximated via curl timings.

REQUESTS=50
CONCURRENCY=4
START_SERVER=1
MIN_QPS=""
MAX_SECONDS=""
SAMPLE_LATENCY=5
MAX_TTFT=""
PROMPT_BYTES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) REQUESTS="$2"; shift 2 ;;
    -c) CONCURRENCY="$2"; shift 2 ;;
    --min-qps) MIN_QPS="$2"; shift 2 ;;
    --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
    --max-ttft) MAX_TTFT="$2"; shift 2 ;;
    --no-start) START_SERVER=0; shift ;;
    --latency-samples) SAMPLE_LATENCY="$2"; shift 2 ;;
    --prompt-bytes) PROMPT_BYTES="$2"; shift 2 ;;
    *) echo "Usage: $0 [-n requests] [-c concurrency] [--min-qps qps] [--max-seconds sec] [--max-ttft sec] [--prompt-bytes N] [--no-start]" >&2; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/zig-out/bin/leanshimmy"
PORT="${LEANSHIMMY_PORT:-18001}"
LOG="/tmp/leanshimmy-bench.log"
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

if [ "$START_SERVER" -eq 1 ]; then
  echo "Starting leanshimmy on port $PORT for bench (requests=$REQUESTS, concurrency=$CONCURRENCY)..."
  ensure_fixture
  LEANSHIMMY_PORT="$PORT" LEANSHIMMY_MODEL_PATH="$MODEL_PATH" LEANSHIMMY_INFERENCE_LIB="$LIB_PATH" "$BIN" >"$LOG" 2>&1 &
  PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT
  # Wait for server to be ready
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
else
  echo "Using existing server on port $PORT for bench (requests=$REQUESTS, concurrency=$CONCURRENCY)..."
fi

PROMPT_TEXT="hello"
if [ -n "$PROMPT_BYTES" ] && [ "$PROMPT_BYTES" -gt 0 ]; then
  PROMPT_TEXT=$(head -c "$PROMPT_BYTES" /dev/zero | tr '\0' 'a')
fi
PROMPT_LEN=$(printf "%s" "$PROMPT_TEXT" | wc -c | tr -d ' ')
echo "Prompt bytes: $PROMPT_LEN"

CHAT_BODY=$(cat <<EOF
{"model":"stub","messages":[{"role":"user","content":"$PROMPT_TEXT"}],"max_tokens":32,"temperature":0.7,"stream":false}
EOF
)
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

start_ts=$(perl -MTime::HiRes=time -E 'say time')
codes=$(seq 1 "$REQUESTS" | xargs -n1 -P"$CONCURRENCY" -I{} curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL" -H "Content-Type: application/json" -d "$CHAT_BODY")
end_ts=$(perl -MTime::HiRes=time -E 'say time')

elapsed=$(perl -e "printf \"%.3f\", $end_ts - $start_ts")
success=$(printf "%s\n" "$codes" | grep -c '^200$' || true)

qps=$(perl -e "printf \"%.2f\", $REQUESTS / ($elapsed == 0 ? 1 : $elapsed)")
echo "Requests: $REQUESTS, Concurrency: $CONCURRENCY, Success: $success, Elapsed: ${elapsed}s, QPS: $qps"

if [ "$SAMPLE_LATENCY" -gt 0 ]; then
  echo "Sampling latency on $SAMPLE_LATENCY requests..."
  tmpfile=$(mktemp)
  for _ in $(seq 1 "$SAMPLE_LATENCY"); do
    curl -s -w "%{time_starttransfer}\n" -o /dev/null -X POST "$URL" -H "Content-Type: application/json" -d "$CHAT_BODY" >> "$tmpfile"
  done
  ttfts=()
  while IFS= read -r line; do
    ttfts+=("$line")
  done < "$tmpfile"
  rm -f "$tmpfile"
  ttft_sum=0
  for t in "${ttfts[@]}"; do
    ttft_sum=$(perl -e "print $ttft_sum + $t" || echo "$ttft_sum")
  done
  count="${#ttfts[@]}"
  if [ "$count" -gt 0 ]; then
    ttft_avg=$(perl -e "printf \"%.4f\", $ttft_sum / $count")
    echo "Sampled TTFT avg (s): $ttft_avg"
    if [ -n "$MAX_TTFT" ]; then
      comp=$(perl -e "print ($ttft_avg > $MAX_TTFT) ? 1 : 0")
      if [ "$comp" = "1" ]; then
        echo "Bench failed: TTFT avg ${ttft_avg}s exceeds max ${MAX_TTFT}s" >&2
        exit 1
      fi
    fi
  fi
fi

if [ "$success" -ne "$REQUESTS" ]; then
  echo "Bench failed: not all requests succeeded" >&2
  exit 1
fi

if [ -n "$MIN_QPS" ]; then
  comp=$(perl -e "print ($qps < $MIN_QPS) ? 1 : 0")
  if [ "$comp" = "1" ]; then
    echo "Bench failed: QPS $qps below threshold $MIN_QPS" >&2
    exit 1
  fi
fi

if [ -n "$MAX_SECONDS" ]; then
  comp=$(perl -e "print ($elapsed > $MAX_SECONDS) ? 1 : 0")
  if [ "$comp" = "1" ]; then
    echo "Bench failed: elapsed ${elapsed}s exceeds max ${MAX_SECONDS}s" >&2
    exit 1
  fi
fi
