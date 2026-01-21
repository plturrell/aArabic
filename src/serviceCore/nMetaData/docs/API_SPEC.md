# nMetaData API Specification

**OpenAPI 3.0 compliant, OpenAI-compatible design**

---

## Overview

nMetaData provides an OpenAI-compatible API for metadata management and lineage tracking. All endpoints follow REST principles with JSON payloads.

---

## Base URL

```
http://localhost:8080
```

---

## Authentication

All `/v1/*` endpoints require Bearer token authentication:

```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/v1/namespaces
```

Set API key via environment or config:
```bash
export API_KEY="your-secret-key"
```

---

## Common Response Format

### Success Response
```json
{
  "object": "list|namespace|dataset|job|run",
  "data": { ... } or [ ... ]
}
```

### Error Response
```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error|authentication_error|not_found_error|server_error",
    "code": "BAD_REQUEST|UNAUTHORIZED|NOT_FOUND|INTERNAL_ERROR"
  }
}
```

---

## Endpoints

### 1. Health & Observability

#### GET /health
Check service health status.

**Response:**
```json
{
  "status": "ready",
  "database_connected": true,
  "version": "1.0.0"
}
```

**Status Values:**
- `cold` - Server started, database not connected
- `degraded` - Database connected but issues detected
- `ready` - Fully operational

---

#### GET /metrics
Get JSON-formatted metrics.

**Response:**
```json
{
  "total_requests": 12345,
  "failed_requests": 10,
  "avg_response_time_ms": 15.3,
  "database_connected": true,
  "events_ingested": 5000,
  "datasets_tracked": 247,
  "jobs_tracked": 89,
  "runs_tracked": 1523
}
```

---

#### GET /metrics/prometheus
Get Prometheus-formatted metrics.

**Response:**
```
# HELP nmetadata_requests_total Total requests
# TYPE nmetadata_requests_total counter
nmetadata_requests_total{endpoint="/v1/lineage/events"} 5000
nmetadata_requests_total{endpoint="/v1/datasets"} 247

# HELP nmetadata_response_time_ms Response time
# TYPE nmetadata_response_time_ms histogram
nmetadata_response_time_ms_bucket{le="10"} 8500
nmetadata_response_time_ms_bucket{le="50"} 11000
```

---

### 2. Namespaces

#### POST /v1/namespaces
Create a new namespace.

**Request:**
```json
{
  "name": "production",
  "owner": "data-team",
  "description": "Production data pipelines"
}
```

**Response (201):**
```json
{
  "object": "namespace",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "production",
  "owner": "data-team",
  "description": "Production data pipelines",
  "created_at": "2024-01-20T10:30:00Z",
  "updated_at": "2024-01-20T10:30:00Z"
}
```

---

#### GET /v1/namespaces
List all namespaces.

**Query Parameters:**
- `limit` (integer, default: 100, max: 1000)
- `offset` (integer, default: 0)

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "production",
      "owner": "data-team",
      "description": "Production data pipelines",
      "created_at": "2024-01-20T10:30:00Z",
      "updated_at": "2024-01-20T10:30:00Z"
    }
  ],
  "has_more": false,
  "total": 1
}
```

---

#### GET /v1/namespaces/:namespace
Get a specific namespace.

**Response:**
```json
{
  "object": "namespace",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "production",
  "owner": "data-team",
  "description": "Production data pipelines",
  "created_at": "2024-01-20T10:30:00Z",
  "updated_at": "2024-01-20T10:30:00Z",
  "stats": {
    "datasets": 247,
    "jobs": 89,
    "runs": 1523
  }
}
```

---

### 3. Datasets

#### POST /v1/datasets
Register a new dataset.

**Request:**
```json
{
  "namespace": "production",
  "name": "users",
  "type": "DB_TABLE",
  "source_name": "postgres_prod",
  "description": "User master table",
  "schema": {
    "fields": [
      {
        "name": "id",
        "type": "INTEGER",
        "nullable": false,
        "description": "Primary key"
      },
      {
        "name": "email",
        "type": "STRING",
        "nullable": false
      },
      {
        "name": "created_at",
        "type": "TIMESTAMP",
        "nullable": false
      }
    ]
  },
  "tags": ["pii", "critical"]
}
```

**Response (201):**
```json
{
  "object": "dataset",
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "namespace": "production",
  "name": "users",
  "type": "DB_TABLE",
  "source_name": "postgres_prod",
  "description": "User master table",
  "schema": {
    "fields": [ ... ]
  },
  "tags": ["pii", "critical"],
  "version": 1,
  "created_at": "2024-01-20T10:35:00Z",
  "updated_at": "2024-01-20T10:35:00Z"
}
```

---

#### GET /v1/datasets
List datasets.

**Query Parameters:**
- `namespace` (string, optional) - Filter by namespace
- `tags` (string, optional) - Comma-separated tags
- `limit` (integer, default: 100)
- `offset` (integer, default: 0)

**Response:**
```json
{
  "object": "list",
  "data": [ ... ],
  "has_more": false,
  "total": 247
}
```

---

#### GET /v1/datasets/:namespace/:name
Get a specific dataset.

**Response:**
```json
{
  "object": "dataset",
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "namespace": "production",
  "name": "users",
  "type": "DB_TABLE",
  "schema": { ... },
  "tags": ["pii", "critical"],
  "version": 3,
  "upstream_count": 2,
  "downstream_count": 5,
  "last_modified": "2024-01-20T10:35:00Z",
  "created_at": "2024-01-15T08:00:00Z",
  "updated_at": "2024-01-20T10:35:00Z"
}
```

---

### 4. Jobs

#### POST /v1/jobs
Register a new job.

**Request:**
```json
{
  "namespace": "production",
  "name": "extract_users",
  "type": "BATCH",
  "description": "Extract users from source system",
  "location": "s3://pipelines/extract_users.py"
}
```

**Response (201):**
```json
{
  "object": "job",
  "id": "750e8400-e29b-41d4-a716-446655440002",
  "namespace": "production",
  "name": "extract_users",
  "type": "BATCH",
  "description": "Extract users from source system",
  "location": "s3://pipelines/extract_users.py",
  "version": 1,
  "created_at": "2024-01-20T10:40:00Z",
  "updated_at": "2024-01-20T10:40:00Z"
}
```

---

#### GET /v1/jobs
List jobs.

**Query Parameters:**
- `namespace` (string, optional)
- `type` (string, optional) - BATCH|STREAMING|SERVICE
- `limit` (integer, default: 100)
- `offset` (integer, default: 0)

---

#### GET /v1/jobs/:namespace/:name
Get a specific job with lineage information.

**Response:**
```json
{
  "object": "job",
  "id": "750e8400-e29b-41d4-a716-446655440002",
  "namespace": "production",
  "name": "extract_users",
  "type": "BATCH",
  "inputs": [
    {
      "namespace": "production",
      "name": "raw_users",
      "type": "DB_TABLE"
    }
  ],
  "outputs": [
    {
      "namespace": "staging",
      "name": "clean_users",
      "type": "DB_TABLE"
    }
  ],
  "latest_run": {
    "id": "850e8400-e29b-41d4-a716-446655440003",
    "state": "COMPLETE",
    "started_at": "2024-01-20T10:00:00Z",
    "completed_at": "2024-01-20T10:05:00Z",
    "duration_seconds": 300
  },
  "run_stats": {
    "total_runs": 145,
    "success_rate": 0.97,
    "avg_duration_seconds": 285
  }
}
```

---

### 5. Runs

#### POST /v1/runs
Create a new run record.

**Request:**
```json
{
  "run_id": "850e8400-e29b-41d4-a716-446655440003",
  "job_namespace": "production",
  "job_name": "extract_users",
  "state": "START",
  "nominal_start_time": "2024-01-20T10:00:00Z",
  "run_args": {
    "date": "2024-01-20",
    "mode": "full"
  }
}
```

**Response (201):**
```json
{
  "object": "run",
  "id": "850e8400-e29b-41d4-a716-446655440003",
  "run_id": "850e8400-e29b-41d4-a716-446655440003",
  "job": {
    "namespace": "production",
    "name": "extract_users"
  },
  "state": "START",
  "nominal_start_time": "2024-01-20T10:00:00Z",
  "created_at": "2024-01-20T10:00:00Z"
}
```

---

#### GET /v1/runs/:id
Get run details.

**Response:**
```json
{
  "object": "run",
  "id": "850e8400-e29b-41d4-a716-446655440003",
  "run_id": "850e8400-e29b-41d4-a716-446655440003",
  "job": {
    "namespace": "production",
    "name": "extract_users"
  },
  "state": "COMPLETE",
  "nominal_start_time": "2024-01-20T10:00:00Z",
  "nominal_end_time": "2024-01-20T10:05:00Z",
  "state_history": [
    {
      "state": "START",
      "transitioned_at": "2024-01-20T10:00:00Z"
    },
    {
      "state": "RUNNING",
      "transitioned_at": "2024-01-20T10:00:05Z"
    },
    {
      "state": "COMPLETE",
      "transitioned_at": "2024-01-20T10:05:00Z"
    }
  ],
  "duration_seconds": 300,
  "facets": { ... }
}
```

---

#### GET /v1/runs
List runs with filtering.

**Query Parameters:**
- `job` (string) - Format: "namespace/name"
- `state` (string) - START|RUNNING|COMPLETE|FAIL|ABORT
- `since` (timestamp) - ISO 8601 format
- `limit` (integer, default: 100)
- `offset` (integer, default: 0)

---

### 6. Lineage Events (OpenLineage)

#### POST /v1/lineage/events
Ingest OpenLineage event.

**Request:**
```json
{
  "eventType": "COMPLETE",
  "eventTime": "2024-01-20T10:05:00Z",
  "run": {
    "runId": "850e8400-e29b-41d4-a716-446655440003",
    "facets": {
      "nominalTime": {
        "_producer": "https://github.com/OpenLineage/OpenLineage/tree/1.0.0",
        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/NominalTimeRunFacet.json",
        "nominalStartTime": "2024-01-20T10:00:00Z",
        "nominalEndTime": "2024-01-20T10:05:00Z"
      }
    }
  },
  "job": {
    "namespace": "production",
    "name": "extract_users",
    "facets": {
      "sourceCodeLocation": {
        "_producer": "https://github.com/OpenLineage/OpenLineage/tree/1.0.0",
        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SourceCodeLocationJobFacet.json",
        "type": "git",
        "url": "https://github.com/example/pipelines",
        "path": "extract_users.py"
      }
    }
  },
  "inputs": [
    {
      "namespace": "production",
      "name": "raw_users",
      "facets": {
        "schema": {
          "_producer": "https://github.com/OpenLineage/OpenLineage/tree/1.0.0",
          "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
          "fields": [
            {
              "name": "id",
              "type": "INTEGER",
              "description": "User ID"
            },
            {
              "name": "email",
              "type": "STRING"
            }
          ]
        }
      }
    }
  ],
  "outputs": [
    {
      "namespace": "staging",
      "name": "clean_users",
      "facets": {
        "schema": {
          "_producer": "https://github.com/OpenLineage/OpenLineage/tree/1.0.0",
          "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
          "fields": [
            {
              "name": "id",
              "type": "INTEGER"
            },
            {
              "name": "email",
              "type": "STRING"
            },
            {
              "name": "email_verified",
              "type": "BOOLEAN"
            }
          ]
        },
        "columnLineage": {
          "_producer": "https://github.com/OpenLineage/OpenLineage/tree/1.0.0",
          "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ColumnLineageDatasetFacet.json",
          "fields": {
            "id": {
              "inputFields": [
                {
                  "namespace": "production",
                  "name": "raw_users",
                  "field": "id"
                }
              ]
            },
            "email": {
              "inputFields": [
                {
                  "namespace": "production",
                  "name": "raw_users",
                  "field": "email",
                  "transformations": [
                    {
                      "type": "DIRECT",
                      "description": "lower(email)"
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

**Response (200):**
```json
{
  "status": "accepted",
  "run_id": "850e8400-e29b-41d4-a716-446655440003"
}
```

---

### 7. Lineage Queries

#### GET /v1/lineage/upstream
Get upstream lineage for a dataset.

**Query Parameters:**
- `dataset` (string, required) - Format: "namespace.name"
- `depth` (integer, default: 5, max: 20)
- `include_columns` (boolean, default: false)

**Example:**
```bash
curl "http://localhost:8080/v1/lineage/upstream?dataset=production.users&depth=3&include_columns=true"
```

**Response:**
```json
{
  "object": "lineage.graph",
  "dataset": {
    "namespace": "production",
    "name": "users"
  },
  "depth": 3,
  "nodes": [
    {
      "id": "production.raw_users",
      "type": "dataset",
      "namespace": "production",
      "name": "raw_users",
      "distance": 1
    },
    {
      "id": "production.source_db",
      "type": "dataset",
      "namespace": "production",
      "name": "source_db",
      "distance": 2
    }
  ],
  "edges": [
    {
      "source": "production.raw_users",
      "target": "production.users",
      "job": "production.extract_users",
      "type": "CONSUMES"
    },
    {
      "source": "production.source_db",
      "target": "production.raw_users",
      "job": "production.ingest_raw",
      "type": "CONSUMES"
    }
  ],
  "column_lineage": [
    {
      "output_field": "production.users.id",
      "input_fields": [
        "production.raw_users.id"
      ],
      "transformation": "DIRECT"
    }
  ]
}
```

---

#### GET /v1/lineage/downstream
Get downstream lineage for a dataset.

**Query Parameters:**
- `dataset` (string, required) - Format: "namespace.name"
- `depth` (integer, default: 5, max: 20)
- `include_impact` (boolean, default: false)

**Response:**
```json
{
  "object": "lineage.graph",
  "dataset": {
    "namespace": "production",
    "name": "users"
  },
  "depth": 3,
  "nodes": [ ... ],
  "edges": [ ... ],
  "impact_analysis": {
    "affected_datasets": 5,
    "affected_jobs": 3,
    "critical_paths": [
      {
        "path": ["production.users", "staging.user_profiles", "production.customer_360"],
        "risk_score": 0.8,
        "reason": "Changes may break customer_360 dashboard"
      }
    ]
  }
}
```

---

### 8. Natural Language Queries (🌟 Unique Feature)

#### POST /v1/lineage/query
Query metadata using natural language.

**Request:**
```json
{
  "query": "Show me all datasets that depend on raw_users and were modified in the last 7 days",
  "model": "qwen2-72b-instruct",
  "temperature": 0.0,
  "include_graph": true,
  "max_results": 100
}
```

**Response:**
```json
{
  "object": "lineage.query.result",
  "query": "Show me all datasets that depend on raw_users and were modified in the last 7 days",
  "interpretation": "Finding downstream datasets of 'raw_users' with modifications in the past week",
  "results": {
    "datasets": [
      {
        "namespace": "production",
        "name": "clean_users",
        "modified_at": "2024-01-15T10:30:00Z",
        "distance": 1,
        "modification_type": "schema_change"
      },
      {
        "namespace": "production",
        "name": "user_profiles",
        "modified_at": "2024-01-14T08:20:00Z",
        "distance": 2,
        "modification_type": "data_update"
      }
    ],
    "count": 2
  },
  "graph": {
    "nodes": [
      {
        "id": "production.raw_users",
        "type": "dataset",
        "namespace": "production",
        "name": "raw_users"
      },
      {
        "id": "production.clean_users",
        "type": "dataset",
        "namespace": "production",
        "name": "clean_users"
      }
    ],
    "edges": [
      {
        "source": "production.raw_users",
        "target": "production.clean_users",
        "job": "production.cleanse_users",
        "type": "CONSUMES"
      }
    ]
  },
  "confidence": 0.95,
  "sql_generated": "SELECT d.* FROM datasets d JOIN lineage_edges e ON e.target_dataset_id = d.id WHERE e.source_dataset_id IN (SELECT id FROM datasets WHERE namespace = 'production' AND name = 'raw_users') AND d.updated_at > NOW() - INTERVAL '7 days'",
  "execution_time_ms": 45
}
```

**Example Queries:**
- "What datasets does the extract_users job consume?"
- "Show me all jobs that write to users_table"
- "Find breaking changes in customer_profiles schema"
- "What's the upstream lineage of sales_report with depth 3?"
- "Which datasets were modified today?"
- "Show me all failed runs in the last 24 hours"

---

### 9. Schema Evolution

#### GET /v1/datasets/:namespace/:name/schema-history
Get schema version history.

**Response:**
```json
{
  "object": "schema.history",
  "dataset": "production.users",
  "versions": [
    {
      "version": 3,
      "schema": {
        "fields": [ ... ]
      },
      "created_at": "2024-01-20T10:00:00Z",
      "changes": [
        "Added column: phone_verified (BOOLEAN)",
        "Changed type: email (VARCHAR(255) → VARCHAR(500))"
      ]
    },
    {
      "version": 2,
      "schema": {
        "fields": [ ... ]
      },
      "created_at": "2024-01-15T08:00:00Z",
      "changes": [
        "Added column: email_verified (BOOLEAN)"
      ]
    },
    {
      "version": 1,
      "schema": {
        "fields": [ ... ]
      },
      "created_at": "2024-01-10T12:00:00Z",
      "changes": [
        "Initial schema"
      ]
    }
  ]
}
```

---

#### GET /v1/datasets/:namespace/:name/breaking-changes
Detect breaking schema changes.

**Query Parameters:**
- `from_version` (integer, optional) - Default: previous version
- `to_version` (integer, optional) - Default: current version

**Response:**
```json
{
  "object": "schema.breaking_changes",
  "dataset": "production.users",
  "from_version": 2,
  "to_version": 3,
  "breaking_changes": [
    {
      "type": "TYPE_CHANGED",
      "field": "email",
      "old_value": "VARCHAR(255)",
      "new_value": "VARCHAR(500)",
      "severity": "MEDIUM",
      "reason": "Increased column size may cause issues in consuming systems with fixed buffer sizes",
      "affected_jobs": [
        "production.export_users",
        "staging.validate_emails"
      ],
      "recommendation": "Update consuming jobs to handle larger email values"
    }
  ],
  "non_breaking_changes": [
    {
      "type": "COLUMN_ADDED",
      "field": "phone_verified",
      "new_value": "BOOLEAN",
      "severity": "LOW"
    }
  ],
  "impact_score": 0.4
}
```

---

### 10. Data Quality

#### POST /v1/quality/tests
Define a data quality test.

**Request:**
```json
{
  "dataset_namespace": "production",
  "dataset_name": "users",
  "test_name": "email_format_validation",
  "test_type": "PATTERN_MATCH",
  "config": {
    "column": "email",
    "pattern": "^[\\w.-]+@[\\w.-]+\\.\\w+$"
  },
  "severity": "HIGH"
}
```

**Response (201):**
```json
{
  "object": "quality.test",
  "id": "950e8400-e29b-41d4-a716-446655440004",
  "dataset": "production.users",
  "test_name": "email_format_validation",
  "test_type": "PATTERN_MATCH",
  "config": { ... },
  "severity": "HIGH",
  "created_at": "2024-01-20T11:00:00Z"
}
```

---

#### POST /v1/quality/tests/:id/run
Execute a quality test.

**Response:**
```json
{
  "object": "quality.test.result",
  "test_id": "950e8400-e29b-41d4-a716-446655440004",
  "status": "PASS",
  "executed_at": "2024-01-20T11:05:00Z",
  "results": {
    "total_rows": 10000,
    "passed_rows": 9985,
    "failed_rows": 15,
    "pass_rate": 0.9985
  },
  "sample_failures": [
    {
      "row_id": "12345",
      "field": "email",
      "value": "invalid-email",
      "reason": "Does not match pattern"
    }
  ]
}
```

---

#### GET /v1/datasets/:namespace/:name/quality
Get quality score for a dataset.

**Response:**
```json
{
  "object": "quality.score",
  "dataset": "production.users",
  "overall_score": 0.92,
  "dimensions": {
    "completeness": 0.98,
    "accuracy": 0.95,
    "freshness": 0.88,
    "consistency": 0.92
  },
  "tests_run": 8,
  "tests_passed": 7,
  "tests_failed": 1,
  "last_evaluated": "2024-01-20T11:00:00Z"
}
```

---

## Rate Limiting

Requests are rate limited per API key:
- Default: 100 requests/second
- Burst: 200 requests
- Returns `429 Too Many Requests` when exceeded

**Response:**
```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "retry_after": 5
  }
}
```

---

## Error Codes

| Code | Status | Description |
|------|--------|-------------|
| BAD_REQUEST | 400 | Invalid request payload |
| UNAUTHORIZED | 401 | Missing or invalid API key |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| CONFLICT | 409 | Resource already exists |
| RATE_LIMIT | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

---

## Pagination

List endpoints support pagination:

**Request:**
```bash
curl "http://localhost:8080/v1/datasets?limit=50&offset=100"
```

**Response:**
```json
{
  "object": "list",
  "data": [ ... ],
  "has_more": true,
  "total": 247,
  "limit": 50,
  "offset": 100
}
```

---

## Timestamps

All timestamps are in ISO 8601 format with UTC timezone:
```
2024-01-20T10:30:00Z
```

---

## Versioning

API version is included in the path:
- Current: `/v1/...`
- Beta features: `/v1-beta/...`

---

Last Updated: January 20, 2026
