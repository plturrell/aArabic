# Three-Tab Dashboard Implementation Plan

**Created:** 2026-01-20  
**Status:** In Progress  
**Priority:** High - Production UI for LLM Platform

## Overview

Transform the nOpenaiServer dashboard into a comprehensive three-tab system:
1. **Prompt Testing** - Interactive testing with 4 modes
2. **mHC Fine-Tuning** - Geometric intelligence configuration
3. **Agent Orchestration** - Multi-service workflows

---

## 🎯 Tab 1: Prompt Testing (PRIORITY 1)

### User Stories
- As a developer, I want to test prompts with different modes to optimize performance
- As a researcher, I want to compare responses across all 4 modes side-by-side
- As an operator, I want to track prompt history and performance metrics

### UI Components

**1.1 Quick Test Panel**
```
┌─────────────────────────────────────────────────────┐
│ Prompt Testing                                       │
├─────────────────────────────────────────────────────┤
│ Mode: [Fast] [Normal] [Expert] [Research]           │
│                                                      │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Enter your prompt here...                       │ │
│ │                                                 │ │
│ │                                                 │ │
│ └─────────────────────────────────────────────────┘ │
│                                                      │
│ Model: LFM2.5 1.2B Q4_0 (auto-selected) ✓          │
│                                                      │
│ [Test Prompt] [Batch Test All Modes] [Clear]       │
├─────────────────────────────────────────────────────┤
│ Response:                                            │
│ ┌─────────────────────────────────────────────────┐ │
│ │ [Streaming response appears here...]            │ │
│ └─────────────────────────────────────────────────┘ │
│                                                      │
│ Metrics:                                             │
│ • Latency: 85ms (TTFT: 12ms)                        │
│ • Throughput: 58 tok/s                              │
│ • Cache Hit: 79%                                     │
│ • Tokens: 150 (prompt: 10, response: 140)          │
└─────────────────────────────────────────────────────┘
```

**1.2 Batch Test Panel**
- Test same prompt across all 4 modes simultaneously
- Side-by-side comparison view
- Aggregate metrics (avg latency, cost, quality scores)

**1.3 History Panel**
- Table of previous prompts with filters
- Export to CSV/JSON
- Re-run previous tests
- View full prompt/response details

### Backend APIs Needed

```
POST /api/v1/prompts/test
{
  "prompt": "What is 2+2?",
  "mode": "Fast",
  "max_tokens": 100,
  "temperature": 0.7,
  "system_prompt": "You are a helpful assistant"
}

Response:
{
  "prompt_id": "uuid",
  "response": "2+2 equals 4...",
  "metrics": {
    "latency_ms": 85,
    "ttft_ms": 12,
    "tokens_per_second": 58,
    "tokens_generated": 140,
    "cache_hit_rate": 0.79,
    "model_used": "lfm2.5-1.2b-q4_0"
  }
}

POST /api/v1/prompts/batch-test
{
  "prompt": "What is 2+2?",
  "modes": ["Fast", "Normal", "Expert", "Research"],
  "max_tokens": 100
}

Response:
{
  "batch_id": "uuid",
  "results": [
    { "mode": "Fast", "response": "...", "metrics": {...} },
    { "mode": "Normal", "response": "...", "metrics": {...} },
    ...
  ],
  "comparison": {
    "avg_latency": 250,
    "best_mode": "Fast",
    "total_cost": 0.0025
  }
}

GET /api/v1/prompts/history?limit=50&mode=Fast
Response:
{
  "prompts": [
    {
      "prompt_id": "uuid",
      "prompt_text": "What is 2+2?",
      "mode": "Fast",
      "timestamp": "2026-01-20T12:00:00Z",
      "metrics": {...}
    },
    ...
  ]
}
```

### HANA Integration
- Save all prompts to `PROMPT_HISTORY` table
- Query history with filters (mode, date range, user)
- Analytics: popular prompts, mode usage, performance trends

---

## ⚙️ Tab 2: mHC Fine-Tuning (PRIORITY 2)

### User Stories
- As a researcher, I want to configure mHC geometric constraints for different languages
- As an ML engineer, I want to monitor stability metrics in real-time
- As an Arabic NLP specialist, I want to validate improvements in morphology/dialects

### UI Components

**2.1 mHC Configuration Panel**
```
┌─────────────────────────────────────────────────────┐
│ mHC Configuration                                    │
├─────────────────────────────────────────────────────┤
│ Enable mHC: [✓] Enabled                             │
│                                                      │
│ Core Settings:                                       │
│ • Sinkhorn Iterations: [10] ──────────● (1-50)     │
│ • Stability Threshold: [1e-4] ──●──── (1e-6-1e-3)  │
│ • Manifold Beta: [10.0] ───────●──── (0.1-100)     │
│                                                      │
│ Manifold Type:                                       │
│ ○ Euclidean (default)                               │
│ ◉ Hyperbolic (Arabic morphology +35%)               │
│ ○ Spherical (cross-dialectal +28%)                  │
│ ○ Product (code-switching +20%)                     │
│ ○ Auto-detect                                        │
│                                                      │
│ Layer Range:                                         │
│ Apply to layers: [0] to [79] (all 80 layers)       │
│ Or specific: [30-50] (middle layers only)           │
│                                                      │
│ [Apply Configuration] [Reset to Defaults] [Export]  │
└─────────────────────────────────────────────────────┘
```

**2.2 Stability Monitoring Panel**
- Real-time line chart: α factor per layer (should be ≈1.0)
- Convergence iterations histogram
- Alert indicators (red/yellow/green status)
- Failure detection panel (over-constraint, geo-stat conflict, energy spike)

**2.3 Geometric Intelligence Panel**
- Curvature detection visualization
- Manifold type confidence scores (bar chart)
- Auto-detection results table
- Uncertainty quantification (bootstrap confidence intervals)

**2.4 Arabic NLP Validation Panel**
- Morphology accuracy: 65% → 100% (+35% improvement chart)
- Dialect similarity: 72% → 100% (+28% improvement chart)
- Code-switching: 80% → 100% (+20% improvement chart)
- Long document translation quality (distortion reduction graph)

**2.5 Performance Profiling Panel**
- mHC overhead: 4.2% (target <5%) ✓ green indicator
- SIMD speedup: 3.5x (ARM NEON)
- Memory usage: breakdown by component
- Speculation acceptance rate: 75% (target 70-85%) ✓

### Backend APIs Needed

```
GET /api/v1/mhc/config
Response:
{
  "enabled": true,
  "sinkhorn_iterations": 10,
  "stability_threshold": 1e-4,
  "manifold_type": "hyperbolic",
  "layer_range": { "start": 0, "end": 79 },
  ...
}

PUT /api/v1/mhc/config
{
  "enabled": true,
  "manifold_type": "spherical",
  "sinkhorn_iterations": 15
}

GET /api/v1/mhc/metrics/stability
Response:
{
  "timestamp": "2026-01-20T12:00:00Z",
  "layers": [
    { "layer_id": 0, "alpha_factor": 1.02, "is_stable": true },
    { "layer_id": 1, "alpha_factor": 0.98, "is_stable": true },
    ...
  ],
  "global_stats": {
    "avg_alpha": 1.00,
    "stable_layers": 78,
    "unstable_layers": 2
  }
}

GET /api/v1/mhc/geometry/detection
Response:
{
  "detected_type": "hyperbolic",
  "confidence": 0.92,
  "curvature": -0.15,
  "alternatives": [
    { "type": "spherical", "confidence": 0.05 },
    { "type": "euclidean", "confidence": 0.03 }
  ]
}

GET /api/v1/mhc/arabic/validation
Response:
{
  "morphology": {
    "baseline": 0.65,
    "with_mhc": 1.00,
    "improvement": 0.35
  },
  "dialects": {
    "baseline": 0.72,
    "with_mhc": 1.00,
    "improvement": 0.28
  },
  "code_switching": {
    "baseline": 0.80,
    "with_mhc": 1.00,
    "improvement": 0.20
  }
}
```

### Zig Module Integration
Connect 21 mHC Zig modules to HTTP endpoints:
- `mhc_configuration.zig` → `/api/v1/mhc/config`
- `mhc_constraints.zig` → `/api/v1/mhc/metrics/stability`
- `mhc_geometry_detector.zig` → `/api/v1/mhc/geometry/detection`
- `mhc_arabic_nlp_validation.zig` → `/api/v1/mhc/arabic/validation`
- `mhc_monitor.zig` → `/api/v1/mhc/alerts`

---

## 🤖 Tab 3: Agent Orchestration (PRIORITY 3)

### User Stories
- As a system architect, I want to chain multiple AI services together
- As a product manager, I want to build complex workflows without coding
- As an operations engineer, I want to monitor multi-service performance

### UI Components

**3.1 Service Status Panel**
```
┌─────────────────────────────────────────────────────┐
│ Service Health                                       │
├─────────────────────────────────────────────────────┤
│ Translation Service    ✓ Healthy  99.9% uptime     │
│ Embedding Service      ✓ Healthy  99.8% uptime     │
│ RAG Service            ✓ Healthy  99.5% uptime     │
│ KTO Policy             ⚠ Degraded 95.0% uptime     │
│ Recursive LLM          ✓ Healthy  99.7% uptime     │
│ TAU2-Bench             ✓ Healthy  100% uptime      │
└─────────────────────────────────────────────────────┘
```

**3.2 Workflow Builder Panel**
- Visual drag-drop editor (nodes = services, edges = data flow)
- Pre-built templates (e.g., "Translate → Embed → RAG → Generate")
- Conditional branching (if/else logic)
- Parallel execution support
- Fallback strategies (retry, alternate service)

**3.3 Multi-Agent Coordination Panel**
- Agent roster with capabilities
- Task delegation rules
- Consensus mechanisms (voting, averaging)
- Conflict resolution strategies

**3.4 Orchestration Metrics Panel**
- Total workflow latency (end-to-end)
- Per-service breakdown
- Error rates and retry attempts
- Resource utilization (across all services)
- Cost tracking ($ per workflow execution)

### Backend APIs Needed

```
GET /api/v1/orchestration/services
Response:
{
  "services": [
    {
      "name": "TranslationService",
      "status": "healthy",
      "uptime": 0.999,
      "mhc_enabled": true
    },
    ...
  ]
}

POST /api/v1/orchestration/workflow
{
  "name": "Arabic Translation Pipeline",
  "steps": [
    {
      "service": "TranslationService",
      "input": "{{ prompt }}",
      "output": "translation"
    },
    {
      "service": "EmbeddingService",
      "input": "{{ translation }}",
      "output": "embedding"
    },
    {
      "service": "RAGService",
      "input": { "query": "{{ translation }}", "embedding": "{{ embedding }}" },
      "output": "context"
    }
  ]
}

Response:
{
  "workflow_id": "uuid",
  "status": "completed",
  "results": { "context": "..." },
  "metrics": {
    "total_latency_ms": 450,
    "steps": [
      { "service": "TranslationService", "latency_ms": 120 },
      { "service": "EmbeddingService", "latency_ms": 80 },
      { "service": "RAGService", "latency_ms": 250 }
    ]
  }
}

GET /api/v1/orchestration/metrics?workflow_id=uuid
Response:
{
  "total_executions": 1250,
  "avg_latency_ms": 435,
  "error_rate": 0.02,
  "total_cost": 125.50
}
```

---

## 📋 Implementation Roadmap

### Phase 1: Fix UI & Add Tab 1 (2 days)
- [x] Fix XML validation errors
- [ ] Debug blank page rendering
- [ ] Create PromptTesting.view.xml
- [ ] Create PromptTesting.controller.js
- [ ] Add backend API endpoints (`/api/v1/prompts/*`)
- [ ] Connect to HANA PROMPT_HISTORY table
- [ ] Test end-to-end flow

### Phase 2: Add Tab 2 (3 days)
- [ ] Create MHCFineTuning.view.xml (4 sub-panels)
- [ ] Create MHCFineTuning.controller.js
- [ ] Add backend API endpoints (`/api/v1/mhc/*`)
- [ ] Connect Zig mHC modules to HTTP layer
- [ ] Create real-time WebSocket for metrics
- [ ] Test Arabic NLP validation

### Phase 3: Add Tab 3 (4 days)
- [ ] Create AgentOrchestration.view.xml
- [ ] Create AgentOrchestration.controller.js
- [ ] Build visual workflow editor (drag-drop)
- [ ] Add backend API endpoints (`/api/v1/orchestration/*`)
- [ ] Integrate Mojo service discovery
- [ ] Test multi-service workflows

### Phase 4: Polish & Documentation (1 day)
- [ ] Create user guide with screenshots
- [ ] Add tooltips and help text
- [ ] Performance optimization
- [ ] Security audit (auth, input validation)
- [ ] Production deployment checklist

---

## 🎨 Navigation Structure

```
webapp/
├── view/
│   ├── Main.view.xml (existing dashboard)
│   ├── ModelConfigurator.view.xml (existing, updated)
│   ├── PromptTesting.view.xml (NEW)
│   ├── MHCFineTuning.view.xml (NEW)
│   └── AgentOrchestration.view.xml (NEW)
├── controller/
│   ├── Main.controller.js
│   ├── ModelConfigurator.controller.js
│   ├── PromptTesting.controller.js (NEW)
│   ├── MHCFineTuning.controller.js (NEW)
│   └── AgentOrchestration.controller.js (NEW)
└── i18n/
    └── i18n.properties (add new labels)
```

**Main Navigation Bar:**
```
[Nucleus Openaiserver]
┌─────────────────────────────────────────────────────┐
│ [Dashboard] [Prompt Testing] [mHC Tuning] [Agents] │
└─────────────────────────────────────────────────────┘
```

---

## 🔒 Security Considerations

1. **Authentication**: All API endpoints require valid session
2. **Authorization**: Role-based access (admin, operator, viewer)
3. **Input Validation**: Sanitize all prompts, prevent injection
4. **Rate Limiting**: Max 100 requests/min per user
5. **Audit Logging**: Track all configuration changes
6. **Data Privacy**: Mask sensitive prompts in logs

---

## 📊 Success Metrics

**Tab 1: Prompt Testing**
- [ ] <200ms average response time for Fast mode
- [ ] 100+ prompts tested per day
- [ ] 4-mode comparison takes <10 seconds

**Tab 2: mHC Fine-Tuning**
- [ ] mHC overhead <5% confirmed
- [ ] Arabic NLP targets met (+35%, +28%, +20%)
- [ ] Zero stability failures in production

**Tab 3: Agent Orchestration**
- [ ] 10+ workflows created
- [ ] <500ms average workflow latency
- [ ] 99.9% workflow success rate

---

**Status:** Ready for implementation  
**Next Step:** Toggle to Act mode and start Phase 1
