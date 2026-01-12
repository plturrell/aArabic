# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

When `ENABLE_AUTH=true`, include API key in headers:

```
X-API-Key: your-api-key-here
```

## Endpoints

### Health Check

**GET** `/api/v1/health`

Check system health and model status.

**Response:**
```json
{
  "status": "online",
  "models": {
    "camelbert": true,
    "m2m100": true
  },
  "ready": true
}
```

### Translate Text

**POST** `/api/v1/translate`

Translate Arabic text to English.

**Request:**
```json
{
  "text": "مرحبا بك"
}
```

**Response:**
```json
{
  "original": "مرحبا بك",
  "translated_text": "Welcome",
  "model": "m2m100-418M",
  "source_lang": "ar",
  "target_lang": "en"
}
```

### Analyze Invoice

**POST** `/api/v1/analyze`

Analyze invoice text for compliance.

**Request:**
```json
{
  "text": "Invoice text here..."
}
```

**Response:**
```json
{
  "classification": "Full Tax Invoice",
  "confidence": 0.95,
  "raw_logits": [[-0.5, 0.8]],
  "model": "camelbert-dialect-financial"
}
```

### Execute Workflow

**POST** `/orchestrate/execute`

Execute an orchestrated workflow.

**Request:**
```json
{
  "workflow_id": "default-invoice-processing",
  "inputs": {
    "invoice_document": "invoice.pdf"
  }
}
```

**Response:**
```json
{
  "execution_id": "exec-123",
  "workflow_id": "default-invoice-processing",
  "status": "completed",
  "outputs": {
    "result": "success"
  }
}
```

### List Workflows

**GET** `/orchestrate/workflows`

List all available workflows.

**Response:**
```json
{
  "workflows": [
    {
      "id": "default-invoice-processing",
      "name": "Default Invoice Processing",
      "description": "Standard invoice processing pipeline",
      "node_count": 8,
      "edge_count": 7,
      "version": "1.0.0"
    }
  ]
}
```

### Get Workflow

**GET** `/orchestrate/workflows/{workflow_id}`

Get a specific workflow definition.

**Response:**
```json
{
  "id": "default-invoice-processing",
  "name": "Default Invoice Processing",
  "nodes": [...],
  "edges": [...]
}
```

### Save Workflow

**POST** `/orchestrate/workflows`

Save or update a workflow definition.

**Request:**
```json
{
  "id": "my-workflow",
  "name": "My Workflow",
  "nodes": [...],
  "edges": [...]
}
```

**Response:**
```json
{
  "success": true,
  "workflow_id": "my-workflow",
  "message": "Workflow saved"
}
```

### Delete Workflow

**DELETE** `/orchestrate/workflows/{workflow_id}`

Delete a workflow.

**Response:**
```json
{
  "success": true,
  "message": "Workflow my-workflow deleted"
}
```

### Generate A2UI

**POST** `/a2ui/generate`

Generate A2UI interface from data.

**Request:**
```json
{
  "type": "invoice",
  "invoice_data": {
    "amount": 1000,
    "vendor": "ABC Corp"
  }
}
```

**Response:**
```json
{
  "surface": "form",
  "components": [
    {
      "id": "amount-field",
      "type": "text-field",
      "properties": {
        "label": "Amount",
        "value": "1000"
      }
    }
  ],
  "data": {}
}
```

### Workflow Result to A2UI

**POST** `/a2ui/workflow-result`

Convert workflow execution result to A2UI format.

**Request:**
```json
{
  "execution_id": "exec-123",
  "status": "completed",
  "outputs": {...}
}
```

**Response:**
```json
{
  "surface": "chat",
  "components": [...],
  "data": {...}
}
```

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "additional context"
    }
  }
}
```

### Error Codes

- `VALIDATION_ERROR` (400) - Invalid input
- `NOT_FOUND` (404) - Resource not found
- `SERVICE_UNAVAILABLE` (503) - Service unavailable
- `INTERNAL_ERROR` (500) - Internal server error
- `MISSING_API_KEY` (401) - API key required
- `INVALID_API_KEY` (401) - Invalid API key

## Rate Limiting

When authentication is enabled, rate limiting applies:
- Default: 60 requests per minute per IP
- Configurable via `RATE_LIMIT_PER_MINUTE`

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1234567890
```

## Examples

### Python

```python
import requests

# Translate text
response = requests.post(
    "http://localhost:8000/api/v1/translate",
    json={"text": "مرحبا"}
)
print(response.json())

# Execute workflow
response = requests.post(
    "http://localhost:8000/orchestrate/execute",
    json={
        "workflow_id": "default-invoice-processing",
        "inputs": {"invoice_document": "invoice.pdf"}
    }
)
print(response.json())
```

### JavaScript

```javascript
// Translate text
const response = await fetch('http://localhost:8000/api/v1/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'مرحبا' })
});
const data = await response.json();
console.log(data);
```

### cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Translate
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا"}'
```

