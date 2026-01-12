"""
Local LLM Inference Service
Uses RLM (Recursive Language Models) and ToolOrchestra for workflow extraction
Replaces external API dependencies (GPT-4, etc.)
Port: 8006
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
import os
from typing import Optional, List, Dict
from datetime import datetime

# Add RLM to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../vendor/layerModels/folderRepos/rlm'))

try:
    from rlm import RLM
    RLM_AVAILABLE = True
except ImportError:
    RLM_AVAILABLE = False
    logging.warning("RLM not available, falling back to mock responses")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local LLM Service",
    description="Local inference using RLM and ToolOrchestra",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RLM with local model
# Using vLLM backend for ToolOrchestra/Orchestrator-8B
rlm_client = None
if RLM_AVAILABLE:
    try:
        rlm_client = RLM(
            backend="vllm",  # Using vLLM for local inference
            backend_kwargs={
                "model_name": "nvidia/Orchestrator-8B",  # ToolOrchestra model
                "base_url": "http://localhost:8000/v1",  # vLLM server
            },
            environment="local",  # Use local REPL environment
            verbose=True,
        )
        logger.info("RLM initialized with ToolOrchestra-8B")
    except Exception as e:
        logger.warning(f"Could not initialize RLM: {e}")
        rlm_client = None

class WorkflowExtractionRequest(BaseModel):
    markdown: str
    temperature: float = 0.3

class WorkflowStep(BaseModel):
    id: str
    type: str  # trigger, action, condition, transform, integration
    name: str
    description: str
    parameters: Optional[Dict] = None

class WorkflowSpec(BaseModel):
    name: str
    description: str
    steps: List[WorkflowStep]
    connections: List[Dict[str, str]]

class WorkflowExtractionResponse(BaseModel):
    success: bool
    workflow: WorkflowSpec
    reasoning: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    rlm_available: bool
    backend: str
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="local-llm",
        version="1.0.0",
        rlm_available=RLM_AVAILABLE and rlm_client is not None,
        backend="RLM + ToolOrchestra-8B" if rlm_client else "Mock",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/extract-workflow", response_model=WorkflowExtractionResponse)
async def extract_workflow(request: WorkflowExtractionRequest):
    """
    Extract workflow specification from markdown using local models
    
    Uses RLM with ToolOrchestra-8B for intelligent workflow extraction
    """
    try:
        logger.info(f"Extracting workflow from {len(request.markdown)} chars of markdown")
        
        # Create prompt for workflow extraction
        prompt = f"""You are a workflow extraction expert. Analyze this document and extract a structured business process workflow.

Document:
{request.markdown}

Extract and return a JSON workflow specification with:
1. Workflow name and description
2. List of steps with these types: trigger, action, condition, transform, integration
3. Each step needs: id, type, name, description
4. Connections between steps (from/to relationships)

Focus on:
- Sequential process steps
- Decision points (IF/THEN)
- Data transformations
- External integrations (APIs, databases)
- Loops and iterations

Return ONLY valid JSON in this exact format:
{{
  "name": "Workflow Name",
  "description": "Brief description",
  "steps": [
    {{
      "id": "step1",
      "type": "trigger",
      "name": "Step Name",
      "description": "What this step does"
    }}
  ],
  "connections": [
    {{"from": "step1", "to": "step2"}}
  ]
}}"""

        if rlm_client:
            # Use RLM for intelligent extraction
            logger.info("Using RLM + ToolOrchestra for extraction")
            result = rlm_client.completion(prompt)
            response_text = result.response
            reasoning = result.logs if hasattr(result, 'logs') else None
        else:
            # Fallback: Simple pattern matching
            logger.info("Using fallback extraction (RLM not available)")
            response_text, reasoning = fallback_extraction(request.markdown)
        
        # Parse JSON response
        import json
        try:
            workflow_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                workflow_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse workflow JSON from response")
        
        # Convert to WorkflowSpec
        workflow = WorkflowSpec(
            name=workflow_data.get("name", "Extracted Workflow"),
            description=workflow_data.get("description", ""),
            steps=[
                WorkflowStep(
                    id=step["id"],
                    type=step["type"],
                    name=step["name"],
                    description=step.get("description", ""),
                    parameters=step.get("parameters")
                )
                for step in workflow_data.get("steps", [])
            ],
            connections=workflow_data.get("connections", [])
        )
        
        logger.info(f"Extracted workflow '{workflow.name}' with {len(workflow.steps)} steps")
        
        return WorkflowExtractionResponse(
            success=True,
            workflow=workflow,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error(f"Error extracting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def fallback_extraction(markdown: str) -> tuple[str, str]:
    """Simple fallback extraction when RLM is not available"""
    import re
    
    # Extract title
    title_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
    title = title_match.group(1) if title_match else "Extracted Workflow"
    
    # Extract numbered/bulleted steps
    steps = []
    step_patterns = [
        r'^\d+\.\s+(.+)$',  # Numbered lists
        r'^[-*]\s+(.+)$',   # Bulleted lists
    ]
    
    for pattern in step_patterns:
        matches = re.findall(pattern, markdown, re.MULTILINE)
        if matches:
            steps.extend(matches)
            break
    
    # Create workflow structure
    workflow_steps = []
    connections = []
    
    for i, step_text in enumerate(steps[:10]):  # Limit to 10 steps
        step_id = f"step{i+1}"
        
        # Determine step type from keywords
        step_lower = step_text.lower()
        if any(kw in step_lower for kw in ['receive', 'trigger', 'webhook', 'start']):
            step_type = 'trigger'
        elif any(kw in step_lower for kw in ['if', 'check', 'when', 'condition']):
            step_type = 'condition'
        elif any(kw in step_lower for kw in ['transform', 'extract', 'parse', 'convert']):
            step_type = 'transform'
        elif any(kw in step_lower for kw in ['api', 'call', 'request', 'post', 'get']):
            step_type = 'integration'
        else:
            step_type = 'action'
        
        workflow_steps.append({
            "id": step_id,
            "type": step_type,
            "name": step_text[:50],  # Truncate long names
            "description": step_text
        })
        
        # Create sequential connections
        if i > 0:
            connections.append({
                "from": f"step{i}",
                "to": step_id
            })
    
    workflow = {
        "name": title,
        "description": f"Workflow extracted from document: {title}",
        "steps": workflow_steps,
        "connections": connections
    }
    
    import json
    return json.dumps(workflow, indent=2), "Fallback extraction used (pattern matching)"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")