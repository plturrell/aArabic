# Dashboard API Endpoints Specification

**Server**: Zig HTTP Server (openai_http_server.zig or dashboard_api_server.zig)  
**Base URL**: http://localhost:11434  
**Authentication**: Bearer token (Keycloak OAuth2)  
**Database**: SAP HANA (existing schema: PROMPT_HISTORY, MODEL_PERFORMANCE, PROMPT_MODE_CONFIGS)

---

## 1. MODEL MANAGEMENT

### GET /api/v1/models
List all available models from config.json

**Response:**
```json
{
  "models": [
    {
      "id": "lfm2.5-1.2b-q4_0",
      "name": "LFM2.5 1.2B Q4_0",
      "architecture": "lfm2",
      "quantization": "Q4_0",
      "size_mb": 664,
      "status": "active",
      "path": "LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf",
      "avg_latency": 52,
      "avg_throughput": 65,
      "total_requests": 1247,
      "last_used": "2026-01-20T15:20:00Z"
    }
  ]
}
```

### GET /api/v1/models/{id}
Get specific model details

### POST /api/v1/models/{id}/load
Load/switch to a different model

**Request:**
```json
{
  "tier_config": {
    "max_ram_mb": 1024,
    "kv_cache_ram_mb": 256
  }
}
```

**Response:**
```json
{
  "success": true,
  "model_id": "lfm2.5-1.2b-q4_0",
  "loaded_at": "2026-01-20T15:20:00Z",
  "memory_used_mb": 950
}
```

### GET /api/v1/models/{id}/status
Check if model is loaded and ready

---

## 2. METRICS & PERFORMANCE

### GET /api/v1/metrics/current?model={id}
Get current real-time metrics for selected model

**Response:**
```json
{
  "model_id": "lfm2.5-1.2b-q4_0",
  "timestamp": "2026-01-20T15:20:00Z",
  "metrics": {
    "latency_p50": 52.5,
    "latency_p95": 89.2,
    "latency_p99": 156.7,
    "throughput": 65.3,
    "ttft": 35.2,
    "cache_hit_rate": 0.82,
    "queue_depth": 3,
    "tokens_total": 245,
    "tokens_input": 73,
    "tokens_output": 172
  }
}
```

### GET /api/v1/metrics/history?model={id}&range={1h|24h|7d}
Get time-series metrics history

**Response:**
```json
{
  "model_id": "lfm2.5-1.2b-q4_0",
  "range": "1h",
  "data_points": [
    {
      "timestamp": "2026-01-20T14:20:00Z",
      "latency_p50": 52.5,
      "latency_p95": 89.2,
      "latency_p99": 156.7,
      "throughput": 65.3,
      "ttft": 35.2,
      "cache_hit_rate": 0.82,
      "queue_depth": 3,
      "tokens_input": 73,
      "tokens_output": 172
    }
  ]
}
```

**Implementation:** Query HANA MODEL_METRICS_TIMESERIES table

### GET /api/v1/tiers/stats
Get current tier statistics (GPU/RAM/SSD/Cache)

**Response:**
```json
{
  "tiers": {
    "gpu": {
      "used": 0,
      "total": 0,
      "hitRate": 0,
      "available": false
    },
    "ram": {
      "used": 2.2,
      "total": 16,
      "hitRate": 0.15,
      "available": true
    },
    "dragonfly": {
      "used": 0.5,
      "total": 2,
      "hitRate": 0.82,
      "available": true,
      "connection": "127.0.0.1:6379"
    },
    "postgres": {
      "used": 1.2,
      "total": 10,
      "hitRate": 0.45,
      "available": true
    },
    "ssd": {
      "used": 5.6,
      "total": 50,
      "hitRate": 0.12,
      "available": true
    }
  }
}
```

---

## 3. CHAT / PROMPTS

### POST /v1/chat/completions (OpenAI Compatible)
Send prompt and get completion

**Request:**
```json
{
  "model": "lfm2.5-1.2b-q4_0",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1705765200,
  "model": "lfm2.5-1.2b-q4_0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2+2 equals 4..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "ttft_ms": 35,
  "generation_time": 850
}
```

### POST /api/v1/prompts
Save prompt/response to HANA PROMPT_HISTORY

**Request:**
```json
{
  "user_id": "demo-user",
  "model_id": "lfm2.5-1.2b-q4_0",
  "mode_name": "Normal",
  "prompt_text": "What is 2+2?",
  "response_text": "2+2 equals 4...",
  "latency_ms": 850,
  "ttft_ms": 35,
  "tokens_generated": 50,
  "tokens_per_second": 58.8,
  "prompt_tokens": 10
}
```

### GET /api/v1/prompts/history?user={id}&limit={n}
Get user's prompt history from HANA

**Response:**
```json
{
  "history": [
    {
      "prompt_id": "uuid-123",
      "timestamp": "2026-01-20T15:20:00Z",
      "model_id": "lfm2.5-1.2b-q4_0",
      "mode_name": "Normal",
      "prompt_text": "What is 2+2?",
      "response_text": "2+2 equals 4...",
      "latency_ms": 850,
      "tokens_generated": 50,
      "user_rating": null
    }
  ],
  "total": 125,
  "limit": 50
}
```

### GET /api/v1/prompts/saved?user={id}
Get user's saved prompt templates

---

## 4. MODE MANAGEMENT

### GET /api/v1/modes
Get all mode presets (Fast/Normal/Expert/Research)

**Response:**
```json
{
  "modes": [
    {
      "name": "Fast",
      "display_name": "Fast Mode",
      "description": "Optimized for lowest latency...",
      "icon": "sap-icon://accelerated",
      "color": "#00A600",
      "compatible_models": ["lfm2.5-1.2b-q4_0"],
      "resource_allocation": {
        "gpu": 65,
        "ram": 25,
        "ssd": 10
      }
    }
  ]
}
```

**Implementation:** Query HANA MODE_PRESETS table

### POST /api/v1/modes/{name}/activate
Activate a preset mode and reconfigure server

---

## 5. MHC FINE-TUNING

### GET /api/v1/mhc/config?model={id}
Get MHC tuning configuration

### POST /api/v1/mhc/train
Start MHC training job

### GET /api/v1/mhc/jobs
List all training jobs

---

## 6. ORCHESTRATION

### GET /api/v1/agents
List configured agents

### POST /api/v1/workflows/{id}/execute
Execute multi-agent workflow

---

## 7. WEBSOCKET PROTOCOL

### WS ws://localhost:8080/ws

**Client → Server (Auth):**
```json
{
  "type": "auth",
  "token": "Bearer eyJhbG..."
}
```

**Server → Client (Metrics Update):**
```json
{
  "type": "metrics_update",
  "model_id": "lfm2.5-1.2b-q4_0",
  "timestamp": "2026-01-20T15:20:00Z",
  "metrics": {
    "latency_p50": 52.5,
    "throughput": 65.3,
    "ttft": 35.2,
    "cache_hit_rate": 0.82,
    "queue_depth": 3,
    "tokens_total": 245
  },
  "tiers": {
    "gpu": {"used": 0, "total": 0, "hit_rate": 0},
    "ram": {"used": 2.2, "total": 16, "hit_rate": 0.15}
  }
}
```

**Frequency:** Every 5 seconds

---

## IMPLEMENTATION PRIORITY

1. **Critical (Phase 4A):**
   - GET /api/v1/models
   - GET /api/v1/metrics/current
   - POST /v1/chat/completions (already exists)
   - GET /api/v1/tiers/stats

2. **High (Phase 4B):**
   - POST /api/v1/prompts (save to HANA)
   - GET /api/v1/prompts/history
   - GET /api/v1/metrics/history
   - WebSocket implementation

3. **Medium (Phase 4C):**
   - POST /api/v1/models/{id}/load
   - GET /api/v1/modes
   - POST /api/v1/modes/{name}/activate

4. **Lower (Phase 4D):**
   - MHC endpoints
   - Orchestration endpoints
   - Saved prompts

---

## HANA SQL QUERIES

### Save Prompt to History
```sql
INSERT INTO PROMPT_HISTORY (
    PROMPT_ID, USER_ID, MODEL_ID, MODE_NAME,
    PROMPT_TEXT, RESPONSE_TEXT,
    LATENCY_MS, TTFT_MS, TOKENS_GENERATED, 
    TOKENS_PER_SECOND, PROMPT_TOKENS,
    TIMESTAMP
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
```

### Get Metrics History
```sql
SELECT 
    TIMESTAMP, MODEL_ID,
    LATENCY_P50, LATENCY_P95, LATENCY_P99,
    THROUGHPUT, TTFT, CACHE_HIT_RATE,
    QUEUE_DEPTH, TOKENS_INPUT, TOKENS_OUTPUT
FROM MODEL_METRICS_TIMESERIES
WHERE MODEL_ID = ?
  AND TIMESTAMP > ADD_HOURS(CURRENT_TIMESTAMP, -1)
ORDER BY TIMESTAMP ASC
```

### Get Prompt History
```sql
SELECT 
    PROMPT_ID, TIMESTAMP, MODEL_ID, MODE_NAME,
    PROMPT_TEXT, RESPONSE_TEXT,
    LATENCY_MS, TOKENS_GENERATED, USER_RATING
FROM PROMPT_HISTORY
WHERE USER_ID = ?
ORDER BY TIMESTAMP DESC
LIMIT ?
```

---

## ERROR HANDLING

All endpoints should return consistent error format:

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model lfm2.5-1.2b-q4_0 not found",
    "details": "..."
  }
}
```

**HTTP Status Codes:**
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Server Error
- 503: Service Unavailable (model not loaded)

---

## NEXT STEPS

1. ✅ Frontend ApiService.js created
2. ✅ Dashboard controller updated
3. ⏳ Implement Zig endpoints (see openai_http_server.zig)
4. ⏳ Add HANA connection to Zig server
5. ⏳ Implement WebSocket server
6. ⏳ Test end-to-end
