#!/bin/bash
# Fix ArrayList initialization in tests for Zig 0.15.2

cd "$(dirname "$0")"

# Use Python to properly fix the ArrayList initialization
python3 << 'PYTHON_SCRIPT'
import re

with open('io/pdf_parser.zig', 'r') as f:
    content = f.read()

# Replace the problematic initialization pattern in tests
# From: var buffer = std.ArrayList(u8).init(std.testing.allocator);
# To: var buffer: std.ArrayList(u8) = std.ArrayList(u8).init(std.testing.allocator);

content = re.sub(
    r'(\s+)var buffer = std\.ArrayList\(u8\)\.init\(std\.testing\.allocator\);',
    r'\1var buffer: std.ArrayList(u8) = std.ArrayList(u8).init(std.testing.allocator);',
    content
)

with open('io/pdf_parser.zig', 'w') as f:
    f.write(content)

print("Fixed ArrayList initializations")
PYTHON_SCRIPT

echo "Running tests..."
zig build test --summary all
