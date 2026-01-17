#!/bin/bash

# Capture parser outputs and snapshot a baseline directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

"$PROJECT_ROOT/scripts/conformance_parser.sh" "$@"

current_dir="$PROJECT_ROOT/tmp/conformance/parser"
baseline_dir="$PROJECT_ROOT/tmp/conformance/parser_baseline"

if [[ ! -d "$current_dir" ]]; then
    echo "Parser output dir not found: $current_dir" >&2
    exit 1
fi

rm -rf "$baseline_dir"
cp -R "$current_dir" "$baseline_dir"

echo "Baseline captured: $baseline_dir"
