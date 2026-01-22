#!/bin/bash

# Diff parser outputs against a baseline directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

baseline_dir="$PROJECT_ROOT/tmp/conformance/parser_baseline"
current_dir="$PROJECT_ROOT/tmp/conformance/parser"
diff_dir="$PROJECT_ROOT/tmp/conformance/parser_diffs"

usage() {
    echo "Usage: $0 [--baseline DIR] [--current DIR]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)
            baseline_dir="$2"
            shift 2
            ;;
        --current)
            current_dir="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "$baseline_dir" ]]; then
    echo "Baseline dir not found: $baseline_dir" >&2
    exit 1
fi

if [[ ! -d "$current_dir" ]]; then
    echo "Current dir not found: $current_dir" >&2
    exit 1
fi

mkdir -p "$diff_dir"

total=0
diffs=0
missing=0

while IFS= read -r -d '' file; do
    total=$((total + 1))
    rel="${file#$current_dir/}"
    base="$baseline_dir/$rel"
    diff_path="$diff_dir/$rel.diff"
    mkdir -p "$(dirname "$diff_path")"

    if [[ ! -f "$base" ]]; then
        missing=$((missing + 1))
        echo "Missing baseline: $base" > "$diff_path"
        continue
    fi

    if diff -u "$base" "$file" > "$diff_path"; then
        rm -f "$diff_path"
    else
        diffs=$((diffs + 1))
    fi
done < <(find "$current_dir" -type f -name "*.syntax" -print0)

echo "Parser outputs: $total"
echo "Diffs: $diffs"
echo "Missing baseline: $missing"
echo "Diff output: $diff_dir"
