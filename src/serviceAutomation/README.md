# serviceAutomation - Langflow Workflow Automation

## Overview

This directory contains auto-generated Langflow workflows created from SCIP specifications.

```
SCIP Spec → Langflow JSON → Deploy Here
```

## Directory Structure

```
src/serviceAutomation/
├── README.md                    # This file
├── workflows/                   # Generated Langflow workflows
│   ├── {feature-name}/
│   │   ├── workflow.json       # Langflow workflow definition
│   │   ├── nodes/              # Custom node implementations
│   │   └── README.md           # Workflow documentation
│   └── .gitkeep
└── shared/                      # Shared workflow components
    ├── nodes/                  # Reusable custom nodes
    └── templates/              # Workflow templates
```

## Generated Workflow Structure

Each workflow contains:
- **Start Node** - Entry point
- **Requirement Nodes** - Validate requirements
- **Verification Nodes** - Check compliance
- **Control Nodes** - Business logic
- **End Node** - Completion

## Integration with Langflow

1. **Import**: Copy `workflow.json` into Langflow
2. **Configure**: Set API keys and parameters
3. **Test**: Run workflow in Langflow UI
4. **Deploy**: Export and deploy to production

## Workflow Types

### From SCIP Requirements
- **LLMChain** nodes for requirements
- **Validator** nodes for verification
- **Condition** nodes for control flow

### Example:
```json
{
  "name": "Feature Workflow",
  "nodes": [
    {
      "type": "Start",
      "data": {"label": "Start"}
    },
    {
      "type": "LLMChain",
      "data": {"label": "MUST validate input"}
    },
    {
      "type": "End",
      "data": {"label": "End"}
    }
  ]
}
```

## Deployment

Workflows are automatically deployed by serviceN8n when:
- Feature is generated from n8n workflow
- SCIP specification is created
- Auto-deploy is enabled

## Monitoring

- Langflow UI: http://localhost:7860
- View workflow execution logs
- Track success/failure rates
- Monitor performance metrics

---

**Status:** Ready for workflow generation
**Integration:** Langflow (Python/FastAPI)
**Port:** 7860