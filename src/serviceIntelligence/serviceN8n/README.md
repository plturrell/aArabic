# serviceN8n - n8n Workflow Orchestration Service

## Overview

This service orchestrates the complete forward/reverse engineering workflow using n8n:

```
┌──────────────────────────────────────────────────────────┐
│                 n8n META-WORKFLOW                         │
│  (Orchestrates the entire feature generation process)    │
└────────┬────────────────────────────────────────┬────────┘
         │                                        │
         ↓                                        ↓
   Design Phase                            Generation Phase
         │                                        │
    ┌────┴────┐                            ┌─────┴─────┐
    │ n8n UI  │                            │ Generators │
    │ Design  │                            │  - Gitea   │
    └────┬────┘                            │  - SCIP    │
         │                                 │  - Langflow│
         ↓                                 └─────┬─────┘
    Export JSON                                  │
         │                                       ↓
         ↓                              ┌────────────────┐
   ┌─────────────┐                     │   Deploy to:   │
   │ Lean4 Rust  │                     │ - serviceGitea │
   │   Parser    │                     │ - serviceAuto  │
   │ POST /n8n   │                     │ - serviceN8n   │
   └──────┬──────┘                     └────────────────┘
          │
          ↓
    MD Specification
          │
          ↓
    POST /generate
          │
          ↓
    ┌─────────────────────┐
    │  Generated Outputs: │
    │  1. Go Code         │
    │  2. Lean4 Proofs    │
    │  3. SCIP Specs      │
    │  4. Langflow JSON   │
    └─────────────────────┘
```

## Service Structure

```
src/serviceIntelligence/serviceN8n/
├── README.md                    # This file
├── Cargo.toml                   # Rust dependencies
├── n8n-workflows/               # n8n workflow definitions
│   ├── meta-orchestrator.json  # Main orchestration workflow
│   ├── design-to-spec.json     # n8n → MD → SCIP
│   └── spec-to-code.json       # MD → Go + Lean4 + Langflow
├── src/
│   ├── main.rs                 # Service entry point
│   ├── lib.rs                  # Library exports
│   ├── orchestrator.rs         # Workflow orchestration engine
│   ├── generators/
│   │   ├── mod.rs
│   │   ├── gitea_generator.rs  # Generate Gitea services
│   │   ├── langflow_generator.rs # Generate Langflow workflows
│   │   └── scip_generator.rs   # Generate SCIP compliance
│   └── deployers/
│       ├── mod.rs
│       ├── gitea_deployer.rs   # Deploy to serviceGitea/
│       ├── automation_deployer.rs # Deploy to serviceAutomation/
│       └── n8n_deployer.rs     # Deploy n8n workflows
└── tests/
    └── integration_tests.rs
```

## Workflow Definition

### Meta-Orchestrator Workflow (n8n)

This n8n workflow orchestrates the entire process:

**Nodes:**
1. **Webhook Trigger** - Receives feature design request
2. **Load Template** - Calls `/template` endpoint
3. **Design Phase** - Human designs in n8n UI
4. **Export JSON** - n8n exports workflow JSON
5. **Convert to MD** - POST to `/n8n-to-md`
6. **Generate Feature** - POST to `/generate`
7. **Split Outputs** - Separate Go, Lean4, SCIP, Langflow
8. **Deploy Gitea** - Write to `src/serviceCore/serviceGitea/`
9. **Deploy Automation** - Write to `src/serviceAutomation/`
10. **Deploy N8n** - Write to `src/serviceIntelligence/serviceN8n/`
11. **Generate Langflow** - Convert SCIP → Langflow JSON
12. **Notify Complete** - Send completion notification

## Integration Points

### With Lean4 Parser Service
```bash
POST http://localhost:8002/n8n-to-md
POST http://localhost:8002/generate
POST http://localhost:8002/template
```

### With Gitea Service
```bash
# Generated services deployed here
src/serviceCore/serviceGitea/
├── features/
│   └── {feature-name}/
│       ├── feature.go
│       ├── feature_proofs.lean
│       └── scip_spec.json
```

### With Automation Service
```bash
# Langflow workflows deployed here
src/serviceAutomation/
├── workflows/
│   └── {feature-name}/
│       ├── workflow.json
│       └── README.md
```

### With N8n Service (Self)
```bash
# n8n workflows stored here
src/serviceIntelligence/serviceN8n/n8n-workflows/
└── {feature-name}-workflow.json
```

## Example Meta-Workflow

```json
{
  "name": "Feature Engineering Meta-Orchestrator",
  "nodes": [
    {
      "name": "Feature Request",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "httpMethod": "POST",
        "path": "/orchestrate/feature"
      }
    },
    {
      "name": "Call Lean4 Parser",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8002/n8n-to-md",
        "bodyParameters": {
          "parameters": [
            {
              "name": "workflow_json",
              "value": "={{$json.workflow}}"
            }
          ]
        }
      }
    },
    {
      "name": "Generate Code",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8002/generate",
        "bodyParameters": {
          "parameters": [
            {
              "name": "markdown",
              "value": "={{$json.markdown}}"
            }
          ]
        }
      }
    },
    {
      "name": "Deploy to Gitea",
      "type": "n8n-nodes-base.function",
      "parameters": {
        "functionCode": "const fs = require('fs');\nfs.writeFileSync('serviceGitea/feature.go', items[0].json.go_code);\nreturn items;"
      }
    },
    {
      "name": "Generate Langflow",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8003/scip-to-langflow",
        "bodyParameters": {
          "parameters": [
            {
              "name": "scip_json",
              "value": "={{$json.scip_spec}}"
            }
          ]
        }
      }
    }
  ],
  "connections": {
    "Feature Request": {
      "main": [[{"node": "Call Lean4 Parser"}]]
    },
    "Call Lean4 Parser": {
      "main": [[{"node": "Generate Code"}]]
    },
    "Generate Code": {
      "main": [
        [
          {"node": "Deploy to Gitea"},
          {"node": "Generate Langflow"}
        ]
      ]
    }
  }
}
```

## Deployment Strategy

### Phase 1: Service Setup
1. Create serviceN8n Rust service
2. Create serviceGitea Rust service structure
3. Set up deployment directories

### Phase 2: Generator Development
1. Implement Gitea generator (Rust)
2. Implement Langflow generator (Rust)
3. Implement SCIP generator (Rust)

### Phase 3: Orchestration
1. Deploy n8n meta-workflow
2. Connect to Lean4 Parser service
3. Test end-to-end flow

### Phase 4: Automation
1. Auto-deploy generated services
2. Auto-commit to Git
3. Auto-test generated code
4. Auto-verify Lean4 proofs

## Development Roadmap

- [x] Design architecture
- [ ] Create serviceN8n Rust service
- [ ] Create serviceGitea structure
- [ ] Implement Gitea generator
- [ ] Implement Langflow generator  
- [ ] Create meta-orchestrator n8n workflow
- [ ] Implement auto-deployment
- [ ] Add CI/CD integration
- [ ] Add monitoring and logging

## API Endpoints (serviceN8n)

### POST /orchestrate/feature
Trigger complete feature generation from n8n workflow

### POST /deploy/gitea
Deploy generated code to serviceGitea

### POST /deploy/automation
Deploy Langflow workflow to serviceAutomation

### POST /scip-to-langflow
Convert SCIP specification to Langflow workflow JSON

### GET /status/:feature-id
Get status of feature generation process

## Environment Variables

```bash
LEAN4_PARSER_URL=http://localhost:8002
GITEA_SERVICE_PATH=../serviceCore/serviceGitea
AUTOMATION_PATH=../serviceAutomation
N8N_WORKFLOW_PATH=./n8n-workflows
LANGFLOW_URL=http://localhost:7860
```

## Next Steps

1. **Create serviceN8n Rust service** with orchestration engine
2. **Set up serviceGitea** directory structure
3. **Implement Langflow generator** (SCIP → Langflow JSON)
4. **Create deployment automation** for all services
5. **Build meta-workflow** in n8n that ties everything together

---

**Status:** Architecture designed, ready for implementation
**Priority:** High - Core infrastructure for feature engineering
**Complexity:** High - Multi-service orchestration