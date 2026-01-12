#!/bin/sh
set -e

BOOTSTRAP_DIR="/bootstrap/workflows"
MARKER_FILE="/home/node/.n8n/.bootstrapped"

if [ "${N8N_BOOTSTRAP_WORKFLOWS:-true}" = "true" ] && [ -d "$BOOTSTRAP_DIR" ] && [ ! -f "$MARKER_FILE" ]; then
  echo "[n8n] Importing bootstrap workflows from $BOOTSTRAP_DIR"
  n8n import:workflow --separate --input="$BOOTSTRAP_DIR" || true
  touch "$MARKER_FILE"
fi

exec n8n
