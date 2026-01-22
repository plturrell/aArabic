#!/usr/bin/env bash
set -euo pipefail

# Iterate over model directories and run bench_qps.sh for each.
# Usage: bench_models.sh --lib /path/to/libinference.so|dylib --models-root /path/to/models [--prompt-bytes N] [--min-qps Q] [--max-ttft S] [--max-seconds S] [--concurrency C] [--requests N]

LIB=""
MODELS_ROOT=""
PROMPT_BYTES=0
MIN_QPS=0
MAX_TTFT=0
MAX_SECONDS=0
CONCURRENCY=4
REQUESTS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib) LIB="$2"; shift 2 ;;
    --models-root) MODELS_ROOT="$2"; shift 2 ;;
    --prompt-bytes) PROMPT_BYTES="$2"; shift 2 ;;
    --min-qps) MIN_QPS="$2"; shift 2 ;;
    --max-ttft) MAX_TTFT="$2"; shift 2 ;;
    --max-seconds) MAX_SECONDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    *)
      echo "Usage: $0 --lib LIB --models-root ROOT [--prompt-bytes N] [--min-qps Q] [--max-ttft S] [--max-seconds S] [--concurrency C] [--requests N]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$LIB" || -z "$MODELS_ROOT" ]]; then
  echo "--lib and --models-root are required" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/zig-out/bin/leanshimmy"
PORT="${LEANSHIMMY_PORT:-18001}"

if [ ! -x "$BIN" ]; then
  echo "Building leanshimmy..."
  (cd "$ROOT" && zig build)
fi

models=()
while IFS= read -r -d '' path; do
  if [ -f "$path/config.json" ]; then
    models+=("$path")
  fi
done < <(find "$MODELS_ROOT" -maxdepth 1 -mindepth 1 -type d -print0)

for model in "${models[@]}"; do
  echo "=== Benchmarking model: $model ==="
  LEANSHIMMY_INFERENCE_LIB="$LIB" LEANSHIMMY_MODEL_PATH="$model" \
    ./scripts/bench_qps.sh \
      -n "$REQUESTS" \
      -c "$CONCURRENCY" \
      ${PROMPT_BYTES:+--prompt-bytes "$PROMPT_BYTES"} \
      ${MIN_QPS:+--min-qps "$MIN_QPS"} \
      ${MAX_TTFT:+--max-ttft "$MAX_TTFT"} \
      ${MAX_SECONDS:+--max-seconds "$MAX_SECONDS"}
done
