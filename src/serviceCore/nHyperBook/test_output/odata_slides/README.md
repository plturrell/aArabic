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
