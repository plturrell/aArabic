# HyperShimmy API Reference

**Version**: 1.0.0  
**Protocol**: OData V4  
**Base URL**: `http://localhost:8080/odata/v4/research`

## Overview

HyperShimmy exposes a RESTful OData V4 API for managing research sources and generating AI-powered content. All endpoints follow OData conventions for querying, filtering, and pagination.

## Table of Contents

- [Authentication](#authentication)
- [Entity Sets](#entity-sets)
- [Query Options](#query-options)
- [Actions](#actions)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Authentication

Currently, HyperShimmy operates without authentication. For production deployments, consider implementing:
- OAuth 2.0 / OpenID Connect
- API Keys
- JWT tokens

---

## Entity Sets

### Sources

Represents documents, articles, PDFs, or other content sources.

**Endpoint**: `/Sources`

#### Properties

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| ID | String | Unique identifier | Yes (auto-generated) |
| Title | String | Display title | Yes |
| SourceType | Enum | Type of source (URL, PDF, Text, YouTube, File) | Yes |
| Url | String | Source URL (if applicable) | Conditional |
| Content | String | Extracted text content | No |
| Status | Enum | Processing status (Pending, Processing, Ready, Error) | Yes |
| CreatedAt | DateTime | Creation timestamp | Yes (auto-generated) |
| ModifiedAt | DateTime | Last modification timestamp | Yes (auto-generated) |

#### SourceType Enum
- `URL` - Web page
- `PDF` - PDF document
- `Text` - Plain text
- `YouTube` - YouTube video
- `File` - Uploaded file

#### Status Enum
- `Pending` - Awaiting processing
- `Processing` - Currently being processed
- `Ready` - Ready for use
- `Error` - Processing failed

---

## Query Options

HyperShimmy supports standard OData V4 query options:

### $filter

Filter collections based on criteria.

**Syntax**: `$filter=<expression>`

**Operators**:
- `eq` - Equals
- `ne` - Not equals
- `gt` - Greater than
- `ge` - Greater than or equal
- `lt` - Less than
- `le` - Less than or equal
- `and` - Logical AND
- `or` - Logical OR
- `not` - Logical NOT

**Example**:
```
GET /Sources?$filter=SourceType eq 'PDF' and Status eq 'Ready'
```

### $select

Select specific properties to return.

**Syntax**: `$select=<property1>,<property2>,...`

**Example**:
```
GET /Sources?$select=ID,Title,Status
```

### $top

Limit the number of results returned.

**Syntax**: `$top=<number>`

**Example**:
```
GET /Sources?$top=10
```

### $skip

Skip a specified number of results.

**Syntax**: `$skip=<number>`

**Example**:
```
GET /Sources?$skip=20
```

### $orderby

Sort results by one or more properties.

**Syntax**: `$orderby=<property> [asc|desc]`

**Example**:
```
GET /Sources?$orderby=CreatedAt desc
```

### $count

Include total count of matching records.

**Syntax**: `$count=true`

**Example**:
```
GET /Sources?$count=true
```

**Response includes**:
```json
{
  "@odata.context": "...",
  "@odata.count": 42,
  "value": [...]
}
```

### Combining Options

Query options can be combined:

```
GET /Sources?$filter=Status eq 'Ready'&$select=ID,Title&$top=5&$orderby=CreatedAt desc
```

---

## Actions

Actions perform operations that go beyond simple CRUD.

### Chat

Generate AI-powered chat responses using RAG (Retrieval Augmented Generation).

**Endpoint**: `POST /Chat`

**Request Body**:
```json
{
  "query": "What are the main topics discussed?",
  "sourceIds": ["src-001", "src-002"],
  "conversationHistory": [
    {
      "role": "user",
      "content": "Previous question"
    },
    {
      "role": "assistant",
      "content": "Previous answer"
    }
  ]
}
```

**Parameters**:
- `query` (required): User's question or prompt
- `sourceIds` (required): Array of source IDs to use as context
- `conversationHistory` (optional): Previous conversation for context
- `maxTokens` (optional): Maximum response length (default: 1000)
- `temperature` (optional): Response creativity (0.0-1.0, default: 0.7)

**Response**:
```json
{
  "response": "The main topics include...",
  "sources": ["src-001", "src-002"],
  "relevantChunks": [
    {
      "sourceId": "src-001",
      "content": "...",
      "similarity": 0.89
    }
  ]
}
```

### Summary

Generate a comprehensive summary from multiple sources.

**Endpoint**: `POST /Summary`

**Request Body**:
```json
{
  "sourceIds": ["src-001", "src-002", "src-003"],
  "format": "detailed",
  "maxLength": 500
}
```

**Parameters**:
- `sourceIds` (required): Array of source IDs to summarize
- `format` (optional): "brief" or "detailed" (default: "detailed")
- `maxLength` (optional): Maximum summary length in words

**Response**:
```json
{
  "summary": "Comprehensive summary text...",
  "keyPoints": [
    "Key point 1",
    "Key point 2",
    "Key point 3"
  ],
  "wordCount": 487
}
```

### GenerateAudio

Create an audio narration of content (Text-to-Speech).

**Endpoint**: `POST /GenerateAudio`

**Request Body**:
```json
{
  "sourceIds": ["src-001"],
  "voice": "alloy",
  "speed": 1.0,
  "format": "mp3"
}
```

**Parameters**:
- `sourceIds` (required): Sources to narrate
- `voice` (optional): Voice to use (alloy, echo, fable, onyx, nova, shimmer)
- `speed` (optional): Speech speed (0.25-4.0, default: 1.0)
- `format` (optional): Audio format (mp3, wav, opus)

**Response**:
```json
{
  "audioUrl": "/audio/abc123.mp3",
  "duration": 180,
  "format": "mp3",
  "sizeBytes": 2457600
}
```

### GenerateSlides

Create a presentation from content.

**Endpoint**: `POST /GenerateSlides`

**Request Body**:
```json
{
  "sourceIds": ["src-001", "src-002"],
  "title": "Research Presentation",
  "slideCount": 10,
  "style": "professional"
}
```

**Parameters**:
- `sourceIds` (required): Sources to create slides from
- `title` (required): Presentation title
- `slideCount` (optional): Target number of slides (default: 10)
- `style` (optional): Visual style (professional, minimal, creative)

**Response**:
```json
{
  "slides": [
    {
      "number": 1,
      "type": "title",
      "title": "Research Presentation",
      "content": ""
    },
    {
      "number": 2,
      "type": "content",
      "title": "Overview",
      "content": "Key points..."
    }
  ],
  "exportUrl": "/slides/presentation-xyz.html"
}
```

### GenerateMindmap

Create a visual knowledge graph and mindmap.

**Endpoint**: `POST /GenerateMindmap`

**Request Body**:
```json
{
  "sourceIds": ["src-001", "src-002"],
  "maxDepth": 3
}
```

**Parameters**:
- `sourceIds` (required): Sources to analyze
- `maxDepth` (optional): Maximum tree depth (default: 3)

**Response**:
```json
{
  "root": {
    "id": "root",
    "label": "Main Topic",
    "children": [
      {
        "id": "node1",
        "label": "Subtopic 1",
        "children": []
      }
    ]
  },
  "entities": 45,
  "relationships": 78
}
```

---

## CRUD Operations

### Create Source

**Request**: `POST /Sources`

```json
{
  "Title": "My Research Paper",
  "SourceType": "PDF",
  "Url": "https://example.com/paper.pdf"
}
```

**Response**: `201 Created`
```json
{
  "@odata.context": "...",
  "ID": "src-001",
  "Title": "My Research Paper",
  "SourceType": "PDF",
  "Url": "https://example.com/paper.pdf",
  "Status": "Pending",
  "CreatedAt": "2026-01-16T21:00:00Z",
  "ModifiedAt": "2026-01-16T21:00:00Z"
}
```

### Read Source

**Request**: `GET /Sources('src-001')`

**Response**: `200 OK`
```json
{
  "@odata.context": "...",
  "ID": "src-001",
  "Title": "My Research Paper",
  ...
}
```

### Update Source

**Request**: `PATCH /Sources('src-001')`

```json
{
  "Title": "Updated Title"
}
```

**Response**: `200 OK`

### Delete Source

**Request**: `DELETE /Sources('src-001')`

**Response**: `204 No Content`

### List Sources

**Request**: `GET /Sources`

**Response**: `200 OK`
```json
{
  "@odata.context": "...",
  "value": [
    {
      "ID": "src-001",
      "Title": "Source 1",
      ...
    },
    {
      "ID": "src-002",
      "Title": "Source 2",
      ...
    }
  ]
}
```

---

## Error Handling

HyperShimmy follows OData error format:

### Error Response Format

```json
{
  "error": {
    "code": "InvalidInput",
    "message": "The provided input is invalid",
    "target": "SourceType",
    "details": [
      {
        "code": "ValueOutOfRange",
        "message": "SourceType must be one of: URL, PDF, Text, YouTube, File"
      }
    ]
  }
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 204 | No Content | Resource deleted successfully |
| 400 | Bad Request | Invalid request syntax |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error occurred |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

- `InvalidInput` - Request contains invalid data
- `ResourceNotFound` - Requested resource doesn't exist
- `ValidationFailed` - Data validation failed
- `ProcessingError` - Error during content processing
- `RateLimitExceeded` - Too many requests
- `ServiceUnavailable` - External service unavailable

---

## Examples

### Example 1: Upload and Process PDF

```bash
# 1. Create source
curl -X POST http://localhost:8080/odata/v4/research/Sources \
  -H "Content-Type: application/json" \
  -d '{
    "Title": "Machine Learning Paper",
    "SourceType": "PDF",
    "Url": "https://example.com/ml-paper.pdf"
  }'

# Response: { "ID": "src-123", "Status": "Processing", ... }

# 2. Check status
curl http://localhost:8080/odata/v4/research/Sources('src-123')

# Response: { "ID": "src-123", "Status": "Ready", ... }
```

### Example 2: Chat with Multiple Sources

```bash
curl -X POST http://localhost:8080/odata/v4/research/Chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key findings",
    "sourceIds": ["src-123", "src-456"]
  }'

# Response: { "response": "The key findings indicate...", ... }
```

### Example 3: Generate Audio Summary

```bash
curl -X POST http://localhost:8080/odata/v4/research/GenerateAudio \
  -H "Content-Type: application/json" \
  -d '{
    "sourceIds": ["src-123"],
    "voice": "alloy",
    "speed": 1.2
  }'

# Response: { "audioUrl": "/audio/summary-xyz.mp3", ... }
```

### Example 4: Advanced Filtering

```bash
# Get all PDF sources that are ready, ordered by date
curl "http://localhost:8080/odata/v4/research/Sources?\$filter=SourceType%20eq%20'PDF'%20and%20Status%20eq%20'Ready'&\$orderby=CreatedAt%20desc&\$top=10"
```

---

## Rate Limiting

- **Default limit**: 100 requests per minute per client
- **Rate limit headers** (in responses):
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

When rate limit is exceeded, the API returns `429 Too Many Requests`.

---

## CORS

HyperShimmy supports Cross-Origin Resource Sharing (CORS):

- **Allowed Origins**: Configurable (default: all)
- **Allowed Methods**: GET, POST, PUT, PATCH, DELETE, OPTIONS
- **Allowed Headers**: Content-Type, Authorization, Accept
- **Max Age**: 86400 seconds (24 hours)

---

## Pagination

For large collections, use `$top` and `$skip`:

```bash
# Page 1 (items 0-9)
GET /Sources?$top=10&$skip=0

# Page 2 (items 10-19)
GET /Sources?$top=10&$skip=10

# Page 3 (items 20-29)
GET /Sources?$top=10&$skip=20
```

Responses include `@odata.nextLink` when more results are available:

```json
{
  "@odata.context": "...",
  "@odata.nextLink": "/Sources?$skip=10",
  "value": [...]
}
```

---

## Metadata

Get service metadata:

```bash
GET /$metadata
```

Returns XML metadata document describing all entity sets, types, and operations.

---

## Best Practices

1. **Use $select** to request only needed properties
2. **Use $top** to limit result sizes
3. **Use $filter** instead of client-side filtering
4. **Check Status** before using processed sources
5. **Handle errors** gracefully with retry logic
6. **Respect rate limits** to avoid throttling
7. **Use appropriate Content-Type** headers
8. **Include error handling** for all requests

---

## SDK and Client Libraries

Currently, HyperShimmy provides raw HTTP/OData access. Client libraries coming soon:

- JavaScript/TypeScript SDK
- Python SDK  
- Zig bindings
- Mojo bindings

---

## Support

- **Documentation**: [docs.hypershimmy.dev](https://docs.hypershimmy.dev)
- **Issues**: [github.com/hypershimmy/issues](https://github.com/hypershimmy/issues)
- **Community**: [discord.gg/hypershimmy](https://discord.gg/hypershimmy)

---

**Last Updated**: January 16, 2026  
**API Version**: 1.0.0
