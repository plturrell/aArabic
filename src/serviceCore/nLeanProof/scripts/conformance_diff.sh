#!/bin/bash

# Diff oracle outputs against upstream expected outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_ROOT/../../.." && pwd)"
TESTS_ROOT="$PROJECT_ROOT/tests/lean4"

ORACLE_DIR="$PROJECT_ROOT/tmp/conformance/oracle"
DIFF_DIR="$PROJECT_ROOT/tmp/conformance/diffs"

mkdir -p "$DIFF_DIR"

total=0
diffs=0
missing=0

while IFS= read -r -d '' file; do
    total=$((total + 1))
    rel="${file#$ORACLE_DIR/}"
    expected="$TESTS_ROOT/${rel%.out}.expected.out"
    diff_path="$DIFF_DIR/${rel}.diff"
    diff_dir="$(dirname "$diff_path")"
    mkdir -p "$diff_dir"

    if [[ ! -f "$expected" ]]; then
        missing=$((missing + 1))
        echo "Missing expected: $expected" > "$diff_path"
        continue
    fi

    if diff -au --strip-trailing-cr -I "executing external script" "$expected" "$file" > "$diff_path"; then
        rm -f "$diff_path"
    else
        diffs=$((diffs + 1))
    fi
done < <(find "$ORACLE_DIR" -type f -name "*.out" -print0)

echo "Oracle outputs: $total"
echo "Diffs: $diffs"
echo "Missing expected: $missing"
echo "Diff output: $DIFF_DIR"
