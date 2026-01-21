# nMetaData API Reference

**Version:** v1  
**Base URL:** `http://localhost:8080/api/v1`  
**Last Updated:** January 20, 2026

---

## Table of Contents

1. [Authentication](#authentication)
2. [Response Format](#response-format)
3. [Error Handling](#error-handling)
4. [Endpoints](#endpoints)
   - [System](#system-endpoints)
   - [Datasets](#dataset-endpoints)
   - [Lineage](#lineage-endpoints)
5. [Examples](#examples)

---

## Authentication

Currently, the API is open for development. Authentication will be added in Day 32.

**Future (Day 32):**
```http
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <API_KEY>
```

---

## Response Format

All responses follow a consistent JSON format:

### Success Response
```json
{
  "success": true,
  "data": { ... }
}
```

### Error Response
```json
{
  "error": "Error message",
  "status": 400
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., dependencies exist) |
| 500 | Internal Server Error | Server error |

---

## System Endpoints

### Health Check

Check if the service is running.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1705734120,
  "uptime_seconds": 3600,
  "version": "0.1.0"
}
```

---

### System Status

Get detailed system status and metrics.

```http
GET /api/v1/status
```

**Response:**
```json
{
  "status": "operational",
  "timestamp": 1705734120,
  "components": {
    "api": "healthy",
    "database": "healthy",
    "cache": "not_configured"
  },
  "metrics": {
    "total_datasets": 3,
    "total_edges": 5,
    "requests_per_minute": 42,
    "avg_response_time_ms": 12.5
  },
  "version": {
    "api": "v1",
    "server": "0.1.0"
  }
}
```

---

### API Info

Get API information and available endpoints.

```http
GET /api/v1/info
```

**Response:**
```json
{
  "name": "nMetaData API",
  "version": "0.1.0",
  "description": "Metadata Management System REST API",
  "endpoints": {
    "datasets": "/api/v1/datasets",
    "lineage": "/api/v1/lineage",
    "health": "/api/v1/health",
    "status": "/api/v1/status"
  },
  "documentation": "/api/v1/docs",
  "support": "https://github.com/nmetadata/api"
}
```

---

## Dataset Endpoints

### List Datasets

Retrieve a paginated list of datasets.

```http
GET /api/v1/datasets?page=1&limit=10
```

**Query Parameters:**

| Parameter | Type | Default | Max | Description |
|-----------|------|---------|-----|-------------|
| page | integer | 1 | - | Page number |
| limit | integer | 10 | 100 | Items per page |

**Response:**
```json
{
  "success": true,
  "data": {
    "datasets": [
      {
        "id": "ds-001",
        "name": "users_table",
        "type": "table",
        "schema": "public",
        "created_at": "2026-01-15T10:00:00Z",
        "updated_at": "2026-01-20T08:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "offset": 0,
      "total": 3,
      "total_pages": 1
    }
  }
}
```

---

### Create Dataset

Create a new dataset.

```http
POST /api/v1/datasets
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "new_table",
  "type": "table",
  "schema": "public",
  "description": "Optional description",
  "metadata": {
    "owner": "data_team",
    "tags": ["important", "production"]
  }
}
```

**Required Fields:**
- `name` (string): Dataset name
- `type` (string): One of: `table`, `view`, `pipeline`, `stream`, `file`

**Optional Fields:**
- `schema` (string): Schema name (default: "public")
- `description` (string): Dataset description
- `metadata` (object): Additional metadata

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": "ds-004",
    "name": "new_table",
    "type": "table",
    "schema": "public",
    "description": "Optional description",
    "created_at": "2026-01-20T08:21:00Z",
    "updated_at": "2026-01-20T08:21:00Z"
  }
}
```

**Errors:**
- `400`: Invalid type or missing required fields
- `409`: Dataset name already exists

---

### Get Dataset

Retrieve a specific dataset by ID.

```http
GET /api/v1/datasets/:id
```

**Path Parameters:**
- `id` (string): Dataset ID

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "ds-001",
    "name": "users_table",
    "type": "table",
    "schema": "public",
    "description": "User information table",
    "columns": [
      {
        "name": "id",
        "type": "integer",
        "nullable": false
      },
      {
        "name": "username",
        "type": "varchar",
        "nullable": false
      }
    ],
    "created_at": "2026-01-15T10:00:00Z",
    "updated_at": "2026-01-20T08:00:00Z"
  }
}
```

**Errors:**
- `400`: Invalid dataset ID
- `404`: Dataset not found

---

### Update Dataset

Update an existing dataset.

```http
PUT /api/v1/datasets/:id
Content-Type: application/json
```

**Path Parameters:**
- `id` (string): Dataset ID

**Request Body:**
```json
{
  "name": "updated_name",
  "description": "Updated description",
  "metadata": {
    "owner": "new_owner"
  }
}
```

**Note:** At least one field must be provided.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "ds-001",
    "name": "updated_name",
    "description": "Updated description",
    "updated_at": "2026-01-20T08:21:30Z"
  }
}
```

**Errors:**
- `400`: No fields provided or invalid ID
- `404`: Dataset not found

---

### Delete Dataset

Delete a dataset.

```http
DELETE /api/v1/datasets/:id?force=false
```

**Path Parameters:**
- `id` (string): Dataset ID

**Query Parameters:**
- `force` (boolean): Force delete even if dataset has downstream dependencies

**Response:**
```json
{
  "success": true,
  "message": "Dataset deleted successfully",
  "id": "ds-002"
}
```

**Errors:**
- `400`: Invalid dataset ID
- `404`: Dataset not found
- `409`: Dataset has dependencies (use `force=true` to override)

---

## Lineage Endpoints

### Get Upstream Lineage

Retrieve upstream dependencies (sources) for a dataset.

```http
GET /api/v1/lineage/upstream/:id?depth=5
```

**Path Parameters:**
- `id` (string): Dataset ID

**Query Parameters:**
- `depth` (integer): Maximum traversal depth (default: 5, max: 10)

**Response:**
```json
{
  "success": true,
  "data": {
    "dataset_id": "ds-001",
    "depth": 5,
    "upstream": [
      {
        "id": "ds-100",
        "name": "source_table_1",
        "type": "table",
        "level": 1
      },
      {
        "id": "ds-200",
        "name": "raw_data",
        "type": "file",
        "level": 2
      }
    ],
    "edges": [
      {
        "source": "ds-100",
        "target": "ds-001",
        "type": "direct"
      },
      {
        "source": "ds-200",
        "target": "ds-100",
        "type": "direct"
      }
    ]
  }
}
```

**Errors:**
- `400`: Invalid ID or depth > 10

---

### Get Downstream Lineage

Retrieve downstream consumers for a dataset.

```http
GET /api/v1/lineage/downstream/:id?depth=5
```

**Path Parameters:**
- `id` (string): Dataset ID

**Query Parameters:**
- `depth` (integer): Maximum traversal depth (default: 5, max: 10)

**Response:**
```json
{
  "success": true,
  "data": {
    "dataset_id": "ds-001",
    "depth": 5,
    "downstream": [
      {
        "id": "ds-300",
        "name": "analytics_view",
        "type": "view",
        "level": 1
      },
      {
        "id": "ds-400",
        "name": "dashboard",
        "type": "dashboard",
        "level": 2
      }
    ],
    "edges": [
      {
        "source": "ds-001",
        "target": "ds-300",
        "type": "direct"
      },
      {
        "source": "ds-300",
        "target": "ds-400",
        "type": "direct"
      }
    ]
  }
}
```

---

### Create Lineage Edge

Create a lineage relationship between two datasets.

```http
POST /api/v1/lineage/edges
Content-Type: application/json
```

**Request Body:**
```json
{
  "source_id": "ds-001",
  "target_id": "ds-002",
  "edge_type": "direct",
  "metadata": {
    "transformation": "SELECT * FROM source",
    "confidence": 1.0
  }
}
```

**Required Fields:**
- `source_id` (string): Source dataset ID
- `target_id` (string): Target dataset ID

**Optional Fields:**
- `edge_type` (string): Edge type (default: "direct")
- `metadata` (object): Additional edge metadata

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": "edge-001",
    "source_id": "ds-001",
    "target_id": "ds-002",
    "edge_type": "direct",
    "created_at": "2026-01-20T08:22:00Z"
  }
}
```

**Errors:**
- `400`: Missing IDs, self-loop detected, or invalid edge
- `404`: Source or target dataset not found
- `409`: Edge already exists

---

## Examples

### Example 1: Create and Query Dataset

```bash
# 1. Create a new dataset
curl -X POST http://localhost:8080/api/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "customer_orders",
    "type": "table",
    "schema": "sales",
    "description": "Customer order transactions"
  }'

# Response: {"success":true,"data":{"id":"ds-new-001",...}}

# 2. Get the created dataset
curl http://localhost:8080/api/v1/datasets/ds-new-001

# 3. Update the dataset
curl -X PUT http://localhost:8080/api/v1/datasets/ds-new-001 \
  -H 'Content-Type: application/json' \
  -d '{"description": "Updated: Customer order transactions with details"}'
```

---

### Example 2: Build Lineage Graph

```bash
# 1. Create source dataset
curl -X POST http://localhost:8080/api/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{"name": "raw_events", "type": "stream"}'

# 2. Create target dataset
curl -X POST http://localhost:8080/api/v1/datasets \
  -H 'Content-Type: application/json' \
  -d '{"name": "processed_events", "type": "table"}'

# 3. Create lineage edge
curl -X POST http://localhost:8080/api/v1/lineage/edges \
  -H 'Content-Type: application/json' \
  -d '{
    "source_id": "ds-raw-001",
    "target_id": "ds-proc-001",
    "edge_type": "pipeline"
  }'

# 4. Query downstream lineage
curl http://localhost:8080/api/v1/lineage/downstream/ds-raw-001?depth=3
```

---

### Example 3: Pagination

```bash
# Get first page (10 items)
curl http://localhost:8080/api/v1/datasets

# Get second page with 20 items
curl http://localhost:8080/api/v1/datasets?page=2&limit=20

# Get all (up to 100)
curl http://localhost:8080/api/v1/datasets?limit=100
```

---

### Example 4: Delete with Dependencies

```bash
# Try to delete dataset with dependencies
curl -X DELETE http://localhost:8080/api/v1/datasets/ds-001
# Response: 409 Conflict - has dependencies

# Force delete
curl -X DELETE http://localhost:8080/api/v1/datasets/ds-001?force=true
# Response: 200 OK - deleted
```

---

## Rate Limits

Currently no rate limits. Will be added in Day 32.

---

## Changelog

### v0.1.0 (Day 30) - January 20, 2026
- Initial API release
- Dataset CRUD operations
- Lineage tracking
- System health endpoints
- Mock data responses (database integration pending)

---

## Support

- **Documentation:** https://docs.nmetadata.io
- **GitHub:** https://github.com/nmetadata/api
- **Issues:** https://github.com/nmetadata/api/issues

---

**Last Updated:** January 20, 2026  
**API Version:** v1  
**Server Version:** 0.1.0
