#!/bin/bash

# Test script for OData Slides endpoints
# Tests all CRUD operations and actions for Presentation entity

set -e  # Exit on error

echo "=========================================="
echo "OData Slides Endpoints Test"
echo "=========================================="
echo ""

# Create test output directory
TEST_DIR="test_output/odata_slides"
mkdir -p "$TEST_DIR"

echo "Test directory: $TEST_DIR"
echo ""

# ==========================================
# Test 1: GenerateSlides Action
# ==========================================
echo "Test 1: GenerateSlides Action"
echo "------------------------------"
echo ""

cat > "$TEST_DIR/generate_request.json" << 'EOF'
{
  "SourceId": "source_001",
  "Title": "AI Research Overview",
  "Theme": "professional",
  "TargetAudience": "technical",
  "DetailLevel": "high",
  "NumSlides": 7
}
EOF

echo "Request body:"
cat "$TEST_DIR/generate_request.json"
echo ""

echo "Expected response:"
cat > "$TEST_DIR/generate_response.json" << 'EOF'
{
  "PresentationId": "pres_20260116_200000",
  "Status": "completed",
  "FilePath": "output/slides/pres_20260116_200000.html",
  "NumSlides": 7,
  "Message": "Presentation generated successfully with 7 slides"
}
EOF
cat "$TEST_DIR/generate_response.json"
echo ""

# Test with curl (when server is running)
cat > "$TEST_DIR/test_generate_curl.sh" << 'EOF'
#!/bin/bash
# POST /odata/v4/research/GenerateSlides
curl -X POST http://localhost:8080/odata/v4/research/GenerateSlides \
  -H "Content-Type: application/json" \
  -d @test_output/odata_slides/generate_request.json
EOF
chmod +x "$TEST_DIR/test_generate_curl.sh"

echo "✅ GenerateSlides test prepared"
echo ""

# ==========================================
# Test 2: ExportPresentation Action
# ==========================================
echo "Test 2: ExportPresentation Action"
echo "----------------------------------"
echo ""

cat > "$TEST_DIR/export_request.json" << 'EOF'
{
  "PresentationId": "pres_20260116_200000",
  "Format": "html",
  "IncludeNotes": true,
  "Standalone": true,
  "Compress": false
}
EOF

echo "Request body:"
cat "$TEST_DIR/export_request.json"
echo ""

echo "Expected response:"
cat > "$TEST_DIR/export_response.json" << 'EOF'
{
  "PresentationId": "pres_20260116_200000",
  "ExportPath": "output/exports/pres_20260116_200000_notes.html",
  "Format": "html",
  "FileSize": 45678,
  "Message": "Presentation exported successfully in html format"
}
EOF
cat "$TEST_DIR/export_response.json"
echo ""

# Test with curl
cat > "$TEST_DIR/test_export_curl.sh" << 'EOF'
#!/bin/bash
# POST /odata/v4/research/ExportPresentation
curl -X POST http://localhost:8080/odata/v4/research/ExportPresentation \
  -H "Content-Type: application/json" \
  -d @test_output/odata_slides/export_request.json
EOF
chmod +x "$TEST_DIR/test_export_curl.sh"

echo "✅ ExportPresentation test prepared"
echo ""

# ==========================================
# Test 3: GET Presentation Collection
# ==========================================
echo "Test 3: GET Presentation Collection"
echo "------------------------------------"
echo ""

echo "Expected response:"
cat > "$TEST_DIR/list_response.json" << 'EOF'
{
  "@odata.context": "$metadata#Presentation",
  "value": [
    {
      "PresentationId": "pres_20260116_200000",
      "SourceId": "source_001",
      "Title": "AI Research Overview",
      "Author": "HyperShimmy",
      "Theme": "professional",
      "FilePath": "output/slides/pres_20260116_200000.html",
      "FileSize": 42568,
      "NumSlides": 7,
      "TargetAudience": "technical",
      "DetailLevel": "high",
      "GeneratedAt": 1737028800,
      "ProcessingTimeMs": 3420,
      "Status": "completed",
      "ErrorMessage": null,
      "Version": 1,
      "ExportFormat": "html"
    }
  ]
}
EOF
cat "$TEST_DIR/list_response.json"
echo ""

# Test with curl
cat > "$TEST_DIR/test_list_curl.sh" << 'EOF'
#!/bin/bash
# GET /odata/v4/research/Presentation
curl -X GET http://localhost:8080/odata/v4/research/Presentation \
  -H "Accept: application/json"
EOF
chmod +x "$TEST_DIR/test_list_curl.sh"

echo "✅ GET collection test prepared"
echo ""

# ==========================================
# Test 4: GET Presentation by ID
# ==========================================
echo "Test 4: GET Presentation by ID"
echo "-------------------------------"
echo ""

echo "Expected response:"
cat > "$TEST_DIR/get_response.json" << 'EOF'
{
  "@odata.context": "$metadata#Presentation/$entity",
  "PresentationId": "pres_20260116_200000",
  "SourceId": "source_001",
  "Title": "AI Research Overview",
  "Author": "HyperShimmy",
  "Theme": "professional",
  "FilePath": "output/slides/pres_20260116_200000.html",
  "FileSize": 42568,
  "NumSlides": 7,
  "TargetAudience": "technical",
  "DetailLevel": "high",
  "GeneratedAt": 1737028800,
  "ProcessingTimeMs": 3420,
  "Status": "completed",
  "ErrorMessage": null,
  "Version": 1,
  "ExportFormat": "html"
}
EOF
cat "$TEST_DIR/get_response.json"
echo ""

# Test with curl
cat > "$TEST_DIR/test_get_curl.sh" << 'EOF'
#!/bin/bash
# GET /odata/v4/research/Presentation('pres_20260116_200000')
curl -X GET "http://localhost:8080/odata/v4/research/Presentation('pres_20260116_200000')" \
  -H "Accept: application/json"
EOF
chmod +x "$TEST_DIR/test_get_curl.sh"

echo "✅ GET by ID test prepared"
echo ""

# ==========================================
# Test 5: GET Slides for Presentation
# ==========================================
echo "Test 5: GET Slides for Presentation"
echo "------------------------------------"
echo ""

echo "Expected response:"
cat > "$TEST_DIR/slides_response.json" << 'EOF'
{
  "@odata.context": "$metadata#Slide",
  "value": [
    {
      "SlideId": "slide_001",
      "PresentationId": "pres_20260116_200000",
      "SlideNumber": 1,
      "Layout": "title",
      "Title": "AI Research Overview",
      "Content": "A comprehensive analysis of artificial intelligence research",
      "Subtitle": "Technical Deep Dive",
      "Notes": "Welcome the audience and introduce the topic"
    },
    {
      "SlideId": "slide_002",
      "PresentationId": "pres_20260116_200000",
      "SlideNumber": 2,
      "Layout": "content",
      "Title": "Key Findings",
      "Content": "• Machine learning advances\n• Neural network architectures\n• Practical applications",
      "Subtitle": null,
      "Notes": "Discuss each point with examples"
    }
  ]
}
EOF
cat "$TEST_DIR/slides_response.json"
echo ""

# Test with curl
cat > "$TEST_DIR/test_slides_curl.sh" << 'EOF'
#!/bin/bash
# GET /odata/v4/research/Presentation('pres_20260116_200000')/Slides
curl -X GET "http://localhost:8080/odata/v4/research/Presentation('pres_20260116_200000')/Slides" \
  -H "Accept: application/json"
EOF
chmod +x "$TEST_DIR/test_slides_curl.sh"

echo "✅ GET Slides test prepared"
echo ""

# ==========================================
# Test 6: DELETE Presentation
# ==========================================
echo "Test 6: DELETE Presentation"
echo "----------------------------"
echo ""

echo "Expected response: 204 No Content"
echo ""

# Test with curl
cat > "$TEST_DIR/test_delete_curl.sh" << 'EOF'
#!/bin/bash
# DELETE /odata/v4/research/Presentation('pres_20260116_200000')
curl -X DELETE "http://localhost:8080/odata/v4/research/Presentation('pres_20260116_200000')" \
  -v
EOF
chmod +x "$TEST_DIR/test_delete_curl.sh"

echo "✅ DELETE test prepared"
echo ""

# ==========================================
# Test 7: Filter by SourceId
# ==========================================
echo "Test 7: Filter Presentations by SourceId"
echo "-----------------------------------------"
echo ""

echo "Expected response:"
cat > "$TEST_DIR/filter_response.json" << 'EOF'
{
  "@odata.context": "$metadata#Presentation",
  "value": [
    {
      "PresentationId": "pres_20260116_200000",
      "SourceId": "source_001",
      "Title": "AI Research Overview",
      "Version": 1,
      "GeneratedAt": 1737028800
    },
    {
      "PresentationId": "pres_20260116_210000",
      "SourceId": "source_001",
      "Title": "AI Research Overview",
      "Version": 2,
      "GeneratedAt": 1737032400
    }
  ]
}
EOF
cat "$TEST_DIR/filter_response.json"
echo ""

# Test with curl
cat > "$TEST_DIR/test_filter_curl.sh" << 'EOF'
#!/bin/bash
# GET /odata/v4/research/Presentation?$filter=SourceId eq 'source_001'
curl -X GET "http://localhost:8080/odata/v4/research/Presentation?\$filter=SourceId%20eq%20'source_001'" \
  -H "Accept: application/json"
EOF
chmod +x "$TEST_DIR/test_filter_curl.sh"

echo "✅ Filter test prepared"
echo ""

# ==========================================
# Test Summary
# ==========================================
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

echo "Test files created in: $TEST_DIR/"
echo ""
echo "Generated files:"
ls -lh "$TEST_DIR"
echo ""

echo "OData V4 Endpoints to implement:"
echo ""
echo "1. Actions:"
echo "   POST /odata/v4/research/GenerateSlides"
echo "   POST /odata/v4/research/ExportPresentation"
echo ""
echo "2. CRUD Operations:"
echo "   GET    /odata/v4/research/Presentation"
echo "   GET    /odata/v4/research/Presentation('{id}')"
echo "   GET    /odata/v4/research/Presentation('{id}')/Slides"
echo "   DELETE /odata/v4/research/Presentation('{id}')"
echo ""
echo "3. Query Options:"
echo "   \$filter - Filter by SourceId"
echo "   \$orderby - Sort by GeneratedAt, Version"
echo "   \$top - Limit results"
echo "   \$skip - Pagination"
echo "   \$expand - Expand Slides navigation"
echo ""

echo "To test with a running server:"
echo "  cd $TEST_DIR"
echo "  ./test_generate_curl.sh"
echo "  ./test_export_curl.sh"
echo "  ./test_list_curl.sh"
echo "  ./test_get_curl.sh"
echo "  ./test_slides_curl.sh"
echo "  ./test_filter_curl.sh"
echo "  ./test_delete_curl.sh"
echo ""

# ==========================================
# Create integration test documentation
# ==========================================
cat > "$TEST_DIR/README.md" << 'EOF'
# OData Slides Endpoints Test Suite

## Overview

Test suite for HyperShimmy OData V4 Presentation/Slides endpoints.

## Endpoints

### Actions

#### 1. GenerateSlides
**POST** `/odata/v4/research/GenerateSlides`

Generate a new presentation from a source document.

**Request:**
```json
{
  "SourceId": "source_001",
  "Title": "AI Research Overview",
  "Theme": "professional",
  "TargetAudience": "technical",
  "DetailLevel": "high",
  "NumSlides": 7
}
```

**Response:**
```json
{
  "PresentationId": "pres_20260116_200000",
  "Status": "completed",
  "FilePath": "output/slides/pres_20260116_200000.html",
  "NumSlides": 7,
  "Message": "Presentation generated successfully with 7 slides"
}
```

#### 2. ExportPresentation
**POST** `/odata/v4/research/ExportPresentation`

Export a presentation with custom options.

**Request:**
```json
{
  "PresentationId": "pres_20260116_200000",
  "Format": "html",
  "IncludeNotes": true,
  "Standalone": true,
  "Compress": false
}
```

**Response:**
```json
{
  "PresentationId": "pres_20260116_200000",
  "ExportPath": "output/exports/pres_20260116_200000_notes.html",
  "Format": "html",
  "FileSize": 45678,
  "Message": "Presentation exported successfully in html format"
}
```

### CRUD Operations

#### 3. List Presentations
**GET** `/odata/v4/research/Presentation`

Get all presentations.

**Query Options:**
- `$filter=SourceId eq 'source_001'` - Filter by source
- `$orderby=GeneratedAt desc` - Sort by date
- `$top=10` - Limit results
- `$skip=20` - Pagination

#### 4. Get Presentation
**GET** `/odata/v4/research/Presentation('{id}')`

Get a single presentation by ID.

#### 5. Get Slides
**GET** `/odata/v4/research/Presentation('{id}')/Slides`

Get all slides for a presentation.

#### 6. Delete Presentation
**DELETE** `/odata/v4/research/Presentation('{id}')`

Delete a presentation (cascades to slides and file).

## Running Tests

### Prerequisites
- HyperShimmy server running on `http://localhost:8080`
- Test data loaded in database

### Execute Tests

```bash
# Generate presentation
./test_generate_curl.sh

# Export with notes
./test_export_curl.sh

# List all presentations
./test_list_curl.sh

# Get specific presentation
./test_get_curl.sh

# Get slides
./test_slides_curl.sh

# Filter by source
./test_filter_curl.sh

# Delete presentation
./test_delete_curl.sh
```

## Test Data

### Sample Presentation
```json
{
  "PresentationId": "pres_20260116_200000",
  "SourceId": "source_001",
  "Title": "AI Research Overview",
  "Author": "HyperShimmy",
  "Theme": "professional",
  "NumSlides": 7,
  "TargetAudience": "technical",
  "DetailLevel": "high",
  "Status": "completed",
  "Version": 1
}
```

### Sample Slide
```json
{
  "SlideId": "slide_001",
  "PresentationId": "pres_20260116_200000",
  "SlideNumber": 1,
  "Layout": "title",
  "Title": "AI Research Overview",
  "Content": "A comprehensive analysis",
  "Notes": "Welcome the audience"
}
```

## Integration Points

### Database
- Presentation table
- Slide table
- Foreign key constraints
- Cascade deletes

### File System
- HTML files in `output/slides/`
- Exports in `output/exports/`

### Handler Integration
- `slide_handler.zig`
- `slide_generator.mojo`
- `slide_template.zig`

## Success Criteria

- ✅ GenerateSlides creates presentation + HTML file
- ✅ ExportPresentation generates export variants
- ✅ GET operations return proper OData format
- ✅ DELETE cascades to slides and removes file
- ✅ Query options work ($filter, $orderby, etc.)
- ✅ Navigation properties work (Slides)
- ✅ Error handling (404, 400, 500)

## Notes

- All timestamps are Unix epoch (seconds)
- File sizes are in bytes
- Version numbers auto-increment per source
- Theme options: professional, minimal, dark, colorful
- Detail levels: low, medium, high
- Target audiences: general, technical, executive, academic
EOF

echo "✅ Test documentation created: $TEST_DIR/README.md"
echo ""

echo "=========================================="
echo "✅ All test files prepared successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Integrate odata_slides.zig into main OData router"
echo "2. Start HyperShimmy server"
echo "3. Run test scripts to verify endpoints"
echo "4. Test with SAPUI5 frontend"
echo ""
