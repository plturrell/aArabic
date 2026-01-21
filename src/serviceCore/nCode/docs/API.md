# nCode API Reference

Complete HTTP API documentation for nCode code intelligence server.

**Base URL**: `http://localhost:18003`

**Protocol**: HTTP/1.1

**Content-Type**: `application/json`

---

## Table of Contents

- [Overview](#overview)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [GET /health](#get-health)
  - [POST /v1/index/load](#post-v1indexload)
  - [POST /v1/definition](#post-v1definition)
  - [POST /v1/references](#post-v1references)
  - [POST /v1/hover](#post-v1hover)
  - [POST /v1/symbols](#post-v1symbols)
  - [POST /v1/document-symbols](#post-v1document-symbols)
- [Database Export Endpoints](#database-export-endpoints)
- [Response Formats](#response-formats)
- [Examples](#examples)

---

## Overview

The nCode API provides code intelligence operations on SCIP (Source Code Intelligence Protocol) indexes. All endpoints except `/health` require a SCIP index to be loaded first via `/v1/index/load`.

### Request Format

All POST requests must include:
```http
Content-Type: application/json
```

Request body must be valid JSON.

### Response Format

All responses are JSON with this structure:

**Success (200 OK)**:
```json
{
  "status": "ok",
  "data": { ... }
}
```

**Error (4xx/5xx)**:
```json
{
  "status": "error",
  "message": "Error description"
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid JSON or missing required fields |
| 404 | Not Found | Resource not found (endpoint or symbol) |
| 500 | Internal Server Error | Server error (parsing failure, etc.) |

### Common Errors

**No Index Loaded**:
```json
{
  "status": "error",
  "message": "No index loaded"
}
```

**Invalid Request**:
```json
{
  "status": "error",
  "message": "Invalid JSON"
}
```

**Symbol Not Found**:
```json
{
  "status": "error",
  "message": "Symbol not found"
}
```

---

## Endpoints

### GET /health

Health check endpoint to verify server is running.

**Request**: None

**Response**:
```json
{
  "status": "ok",
  "server": "ncode",
  "version": "1.0"
}
```

**cURL Example**:
```bash
curl http://localhost:18003/health
```

---

### POST /v1/index/load

Load a SCIP index file into memory. This must be called before other endpoints can be used.

**Request Body**:
```json
{
  "path": "/path/to/index.scip"
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| path | string | Yes | Absolute or relative path to SCIP file |

**Response**:
```json
{
  "status": "ok",
  "documents": 42,
  "symbols": 1337,
  "external_symbols": 523,
  "metadata": {
    "version": 1,
    "tool_name": "scip-typescript",
    "tool_version": "0.3.0",
    "project_root": "/Users/user/project"
  }
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| documents | integer | Number of indexed documents |
| symbols | integer | Number of symbols defined |
| external_symbols | integer | Number of external symbols referenced |
| metadata | object | Index metadata (tool info, project root) |

**cURL Example**:
```bash
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d '{
    "path": "./index.scip"
  }'
```

**Errors**:
- `400`: Invalid path or file doesn't exist
- `500`: Failed to parse SCIP file (corrupt/invalid format)

---

### POST /v1/definition

Find the definition location of a symbol at a given position.

**Request Body**:
```json
{
  "file": "src/utils.ts",
  "line": 42,
  "character": 15
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | string | Yes | File path (relative to project root) |
| line | integer | Yes | Line number (0-indexed) |
| character | integer | Yes | Character offset in line (0-indexed) |

**Response**:
```json
{
  "status": "ok",
  "symbol": "scip-typescript npm my-pkg src/utils.ts formatDate().",
  "file": "src/utils.ts",
  "start_line": 10,
  "start_char": 9,
  "end_line": 10,
  "end_char": 19,
  "documentation": "Format a date to ISO 8601 string",
  "kind": "Function"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| symbol | string | SCIP symbol identifier |
| file | string | File containing definition |
| start_line | integer | Definition start line (0-indexed) |
| start_char | integer | Definition start character (0-indexed) |
| end_line | integer | Definition end line |
| end_char | integer | Definition end character |
| documentation | string | Symbol documentation (if available) |
| kind | string | Symbol kind (Function, Type, Variable, etc.) |

**cURL Example**:
```bash
curl -X POST http://localhost:18003/v1/definition \
  -H "Content-Type: application/json" \
  -d '{
    "file": "src/utils.ts",
    "line": 42,
    "character": 15
  }'
```

**Errors**:
- `400`: No index loaded
- `404`: No symbol at position
- `404`: File not in index

---

### POST /v1/references

Find all references to a symbol across the codebase.

**Request Body**:
```json
{
  "symbol": "scip-typescript npm my-pkg src/utils.ts formatDate()."
}
```

Or find by position:
```json
{
  "file": "src/utils.ts",
  "line": 10,
  "character": 9
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| symbol | string | No* | SCIP symbol identifier |
| file | string | No* | File path (alternative to symbol) |
| line | integer | No | Line number (used with file) |
| character | integer | No | Character offset (used with file) |

*Either `symbol` OR (`file`, `line`, `character`) must be provided.

**Response**:
```json
{
  "status": "ok",
  "symbol": "scip-typescript npm my-pkg src/utils.ts formatDate().",
  "references": [
    {
      "file": "src/main.ts",
      "start_line": 5,
      "start_char": 15,
      "end_line": 5,
      "end_char": 25,
      "role": "Reference"
    },
    {
      "file": "src/helpers.ts",
      "start_line": 12,
      "start_char": 8,
      "end_line": 12,
      "end_char": 18,
      "role": "Reference"
    }
  ],
  "count": 2
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| symbol | string | SCIP symbol identifier |
| references | array | List of reference locations |
| count | integer | Total number of references |

**Reference Object**:
| Field | Type | Description |
|-------|------|-------------|
| file | string | File containing reference |
| start_line | integer | Reference start line (0-indexed) |
| start_char | integer | Reference start character |
| end_line | integer | Reference end line |
| end_char | integer | Reference end character |
| role | string | "Definition" or "Reference" |

**cURL Example**:
```bash
# By symbol
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "scip-typescript npm my-pkg src/utils.ts formatDate()."
  }'

# By position
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{
    "file": "src/utils.ts",
    "line": 10,
    "character": 9
  }'
```

**Errors**:
- `400`: No index loaded or invalid request
- `404`: Symbol not found

---

### POST /v1/hover

Get hover information (documentation, type) for a symbol at a position.

**Request Body**:
```json
{
  "file": "src/api.ts",
  "line": 25,
  "character": 10
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | string | Yes | File path (relative to project root) |
| line | integer | Yes | Line number (0-indexed) |
| character | integer | Yes | Character offset (0-indexed) |

**Response**:
```json
{
  "status": "ok",
  "symbol": "scip-typescript npm my-pkg src/api.ts ApiClient#",
  "documentation": "HTTP client for API requests\n\n@example\nconst client = new ApiClient();\nawait client.get('/users');",
  "kind": "Type",
  "signature": "class ApiClient"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| symbol | string | SCIP symbol identifier |
| documentation | string | Symbol documentation (markdown) |
| kind | string | Symbol kind |
| signature | string | Symbol signature (if available) |

**cURL Example**:
```bash
curl -X POST http://localhost:18003/v1/hover \
  -H "Content-Type: application/json" \
  -d '{
    "file": "src/api.ts",
    "line": 25,
    "character": 10
  }'
```

**Errors**:
- `400`: No index loaded
- `404`: No symbol at position

---

### POST /v1/symbols

List all symbols in a specific file.

**Request Body**:
```json
{
  "file": "src/models.ts"
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | string | Yes | File path (relative to project root) |

**Response**:
```json
{
  "status": "ok",
  "file": "src/models.ts",
  "symbols": [
    {
      "symbol": "scip-typescript npm my-pkg src/models.ts User#",
      "name": "User",
      "kind": "Type",
      "line": 5,
      "character": 13
    },
    {
      "symbol": "scip-typescript npm my-pkg src/models.ts User#id.",
      "name": "id",
      "kind": "Property",
      "line": 6,
      "character": 2
    },
    {
      "symbol": "scip-typescript npm my-pkg src/models.ts User#name.",
      "name": "name",
      "kind": "Property",
      "line": 7,
      "character": 2
    }
  ],
  "count": 3
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| file | string | Requested file path |
| symbols | array | List of symbols in file |
| count | integer | Total number of symbols |

**Symbol Object**:
| Field | Type | Description |
|-------|------|-------------|
| symbol | string | SCIP symbol identifier |
| name | string | Symbol display name |
| kind | string | Symbol kind |
| line | integer | Definition line (0-indexed) |
| character | integer | Definition character (0-indexed) |

**cURL Example**:
```bash
curl -X POST http://localhost:18003/v1/symbols \
  -H "Content-Type: application/json" \
  -d '{
    "file": "src/models.ts"
  }'
```

**Errors**:
- `400`: No index loaded
- `404`: File not in index

---

### POST /v1/document-symbols

Get hierarchical document outline (symbols with nesting).

**Request Body**:
```json
{
  "file": "src/services/auth.ts"
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | string | Yes | File path (relative to project root) |

**Response**:
```json
{
  "status": "ok",
  "file": "src/services/auth.ts",
  "symbols": [
    {
      "symbol": "scip-typescript npm my-pkg src/services/auth.ts AuthService#",
      "name": "AuthService",
      "kind": "Type",
      "range": {
        "start_line": 10,
        "start_char": 13,
        "end_line": 50,
        "end_char": 1
      },
      "children": [
        {
          "symbol": "scip-typescript npm my-pkg src/services/auth.ts AuthService#login().",
          "name": "login",
          "kind": "Method",
          "range": {
            "start_line": 15,
            "start_char": 2,
            "end_line": 20,
            "end_char": 3
          }
        },
        {
          "symbol": "scip-typescript npm my-pkg src/services/auth.ts AuthService#logout().",
          "name": "logout",
          "kind": "Method",
          "range": {
            "start_line": 22,
            "start_char": 2,
            "end_line": 25,
            "end_char": 3
          }
        }
      ]
    }
  ]
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| file | string | Requested file path |
| symbols | array | Hierarchical symbol tree |

**Symbol Object**:
| Field | Type | Description |
|-------|------|-------------|
| symbol | string | SCIP symbol identifier |
| name | string | Symbol display name |
| kind | string | Symbol kind |
| range | object | Source range (start/end line/char) |
| children | array | Nested symbols (optional) |

**cURL Example**:
```bash
curl -X POST http://localhost:18003/v1/document-symbols \
  -H "Content-Type: application/json" \
  -d '{
    "file": "src/services/auth.ts"
  }'
```

**Errors**:
- `400`: No index loaded
- `404`: File not in index

---

## Database Export Endpoints

These endpoints provide instructions for exporting SCIP data to external databases using Python loaders.

### POST /v1/index/export/qdrant

Get command to export index to Qdrant vector database.

**Request Body**:
```json
{
  "host": "localhost",
  "port": 6333,
  "collection": "code_symbols"
}
```

**Response**:
```json
{
  "status": "pending",
  "message": "Use Python loader for Qdrant export",
  "command": "python scripts/load_to_databases.py index.scip --qdrant --qdrant-host localhost --qdrant-port 6333 --qdrant-collection code_symbols",
  "host": "localhost",
  "port": 6333,
  "collection": "code_symbols"
}
```

### POST /v1/index/export/memgraph

Get command to export index to Memgraph graph database.

**Request Body**:
```json
{
  "host": "localhost",
  "port": 7687
}
```

**Response**:
```json
{
  "status": "pending",
  "message": "Use Python loader for Memgraph export",
  "command": "python scripts/load_to_databases.py index.scip --memgraph --memgraph-host localhost --memgraph-port 7687",
  "host": "localhost",
  "port": 7687
}
```

### POST /v1/index/export/marquez

Get command to export index to Marquez lineage tracker.

**Request Body**:
```json
{
  "url": "http://localhost:5000",
  "project": "my-project"
}
```

**Response**:
```json
{
  "status": "pending",
  "message": "Use Python loader for Marquez export",
  "command": "python scripts/load_to_databases.py index.scip --marquez --marquez-url http://localhost:5000 --project my-project",
  "url": "http://localhost:5000",
  "project": "my-project"
}
```

### POST /v1/index/export/all

Get command to export to all databases at once.

**Response**:
```json
{
  "status": "pending",
  "message": "Use Python loader for all exports",
  "command": "python scripts/load_to_databases.py index.scip --all",
  "databases": ["qdrant", "memgraph", "marquez"]
}
```

---

## Response Formats

### Symbol Kinds

Possible values for `kind` field:

- `UnspecifiedKind`
- `Macro`
- `Type`
- `Parameter`
- `Variable`
- `Property`
- `Enum`
- `EnumMember`
- `Function`
- `Method`
- `Constructor`
- `Interface`
- `Namespace`
- `TypeParameter`
- `Trait`

### Symbol Roles

Possible values for `role` field in references:

- `Definition` - Symbol definition
- `Reference` - Symbol usage/reference

---

## Examples

### Complete Workflow Example

```bash
# 1. Index a TypeScript project
npx @sourcegraph/scip-typescript index
# Creates index.scip

# 2. Start nCode server
./scripts/start.sh
# Server running on http://localhost:18003

# 3. Load the index
curl -X POST http://localhost:18003/v1/index/load \
  -H "Content-Type: application/json" \
  -d '{"path": "index.scip"}'

# 4. Find definition
curl -X POST http://localhost:18003/v1/definition \
  -H "Content-Type: application/json" \
  -d '{"file": "src/main.ts", "line": 10, "character": 5}'

# 5. Find all references
curl -X POST http://localhost:18003/v1/references \
  -H "Content-Type: application/json" \
  -d '{"symbol": "scip-typescript npm my-pkg src/main.ts main()."}'

# 6. Get hover info
curl -X POST http://localhost:18003/v1/hover \
  -H "Content-Type: application/json" \
  -d '{"file": "src/utils.ts", "line": 15, "character": 8}'

# 7. List file symbols
curl -X POST http://localhost:18003/v1/symbols \
  -H "Content-Type: application/json" \
  -d '{"file": "src/models.ts"}'

# 8. Get document outline
curl -X POST http://localhost:18003/v1/document-symbols \
  -H "Content-Type: application/json" \
  -d '{"file": "src/services/api.ts"}'

# 9. Export to databases
python scripts/load_to_databases.py index.scip --all
```

### Python Client Example

```python
import requests

# Load index
response = requests.post(
    "http://localhost:18003/v1/index/load",
    json={"path": "index.scip"}
)
print(f"Loaded {response.json()['documents']} documents")

# Find definition
response = requests.post(
    "http://localhost:18003/v1/definition",
    json={
        "file": "src/main.ts",
        "line": 10,
        "character": 5
    }
)
definition = response.json()
print(f"Symbol: {definition['symbol']}")
print(f"Location: {definition['file']}:{definition['start_line']}")
```

### JavaScript Client Example

```javascript
// Load index
const loadResponse = await fetch('http://localhost:18003/v1/index/load', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({path: 'index.scip'})
});
const loadData = await loadResponse.json();
console.log(`Loaded ${loadData.documents} documents`);

// Find references
const refsResponse = await fetch('http://localhost:18003/v1/references', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    file: 'src/utils.ts',
    line: 10,
    character: 9
  })
});
const refs = await refsResponse.json();
console.log(`Found ${refs.count} references`);
refs.references.forEach(ref => {
  console.log(`  ${ref.file}:${ref.start_line}:${ref.start_char}`);
});
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider:
- Adding rate limiting middleware
- Implementing request queuing
- Setting up load balancing

---

## Versioning

API Version: 1.0

The API follows semantic versioning. Breaking changes will increment the major version.

---

**Last Updated**: 2026-01-17  
**Version**: 1.0
