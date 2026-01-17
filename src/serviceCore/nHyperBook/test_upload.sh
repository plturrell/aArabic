#!/bin/bash
# Test file upload endpoint

cd "$(dirname "$0")"

echo "Creating test files..."

# Create a simple text file
cat > /tmp/test.txt << 'EOF'
This is a test document.
It contains multiple lines.
Testing file upload functionality.
EOF

# Create a simple HTML file
cat > /tmp/test.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<h1>Test Document</h1>
<p>This is a test paragraph.</p>
<p>Another paragraph for testing.</p>
</body>
</html>
EOF

echo "Test files created:"
echo "  - /tmp/test.txt ($(wc -c < /tmp/test.txt) bytes)"
echo "  - /tmp/test.html ($(wc -c < /tmp/test.html) bytes)"
echo ""

# Test upload endpoint
echo "Testing file upload endpoint..."
echo "================================"
echo ""

# Test 1: Upload text file
echo "Test 1: Upload text file"
curl -X POST \
  -F "file=@/tmp/test.txt" \
  http://localhost:11434/api/upload \
  2>/dev/null | jq '.'
echo ""

# Test 2: Upload HTML file
echo "Test 2: Upload HTML file"
curl -X POST \
  -F "file=@/tmp/test.html" \
  http://localhost:11434/api/upload \
  2>/dev/null | jq '.'
echo ""

# Test 3: Check uploaded files
echo "Test 3: Checking uploaded files..."
if [ -d "uploads" ]; then
    echo "Upload directory exists:"
    ls -lh uploads/ | tail -5
    echo ""
    echo "Extracted text files:"
    for f in uploads/*.txt; do
        if [ -f "$f" ]; then
            echo "  - $f ($(wc -c < "$f") bytes)"
            echo "    Preview: $(head -c 100 "$f")..."
        fi
    done
else
    echo "  ⚠️  Upload directory not created yet"
fi

echo ""
echo "================================"
echo "✅ Upload tests complete"
