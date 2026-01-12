"""
Lean4 to n8n Component Generator
Converts Lean4 formal specifications into n8n workflow components
Port: 8010
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import json
import re
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lean4 to n8n Generator",
    description="Generate n8n workflow components from Lean4 specifications",
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

class N8nGenerationRequest(BaseModel):
    lean4_file_path: str
    component_type: str = "custom"  # custom, trigger, action
    generate_credentials: bool = True

class N8nGenerationResponse(BaseModel):
    success: bool
    n8n_components: List[Dict]
    components_count: int
    output_directory: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="lean4-to-n8n",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/generate-n8n", response_model=N8nGenerationResponse)
async def generate_n8n_from_lean4(request: N8nGenerationRequest):
    """
    Generate n8n components from Lean4 specifications
    
    Example:
    {
        "lean4_file_path": "/app/lean4-output/ShimmyAI.lean",
        "component_type": "custom",
        "generate_credentials": true
    }
    """
    try:
        lean4_file_path = Path(request.lean4_file_path)
        if not lean4_file_path.exists():
            raise HTTPException(status_code=404, detail="Lean4 file not found")
        
        # Read Lean4 file
        with open(lean4_file_path) as f:
            lean4_content = f.read()
        
        logger.info(f"Generating n8n components from: {lean4_file_path.name}")
        
        # Parse Lean4 and extract specifications
        specs = parse_lean4_specifications(lean4_content)
        
        # Generate n8n components
        n8n_components = []
        
        for spec in specs:
            component = generate_n8n_component(
                spec=spec,
                component_type=request.component_type,
                generate_credentials=request.generate_credentials
            )
            n8n_components.append(component)
        
        # Save n8n components
        output_dir = Path("/app/n8n-components")
        output_dir.mkdir(exist_ok=True)
        
        project_name = lean4_file_path.stem
        component_dir = output_dir / project_name
        component_dir.mkdir(exist_ok=True)
        
        for i, component in enumerate(n8n_components):
            component_file = component_dir / f"{component['name']}.node.json"
            with open(component_file, 'w') as f:
                json.dump(component, f, indent=2)
            logger.info(f"Saved n8n component: {component_file}")
        
        return N8nGenerationResponse(
            success=True,
            n8n_components=n8n_components,
            components_count=len(n8n_components),
            output_directory=str(component_dir)
        )
        
    except Exception as e:
        logger.error(f"Error generating n8n components: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_lean4_specifications(lean4_content: str) -> List[Dict]:
    """Parse Lean4 content and extract specifications"""
    specs = []
    
    # Extract namespace
    namespace_match = re.search(r'namespace\s+(\w+)', lean4_content)
    namespace = namespace_match.group(1) if namespace_match else "Unknown"
    
    # Extract structures (specifications)
    structure_pattern = r'structure\s+(\w+)\s+where(.*?)(?=\n\n|structure|theorem|def|end)'
    
    for match in re.finditer(structure_pattern, lean4_content, re.DOTALL):
        structure_name = match.group(1)
        structure_body = match.group(2)
        
        # Extract fields
        fields = []
        field_pattern = r'(\w+)\s*:\s*(\w+)(?:\s*:=\s*(.+?))?(?=\n|$)'
        for field_match in re.finditer(field_pattern, structure_body):
            fields.append({
                "name": field_match.group(1),
                "type": field_match.group(2),
                "default": field_match.group(3).strip() if field_match.group(3) else None
            })
        
        specs.append({
            "namespace": namespace,
            "name": structure_name,
            "fields": fields,
            "kind": "structure"
        })
    
    # Extract API endpoints
    endpoint_pattern = r'def\s+(\w+_endpoints)\s*:.*?\[(.*?)\]'
    for match in re.finditer(endpoint_pattern, lean4_content, re.DOTALL):
        endpoint_name = match.group(1)
        endpoints_body = match.group(2)
        
        specs.append({
            "namespace": namespace,
            "name": endpoint_name,
            "endpoints_definition": endpoints_body,
            "kind": "api_endpoints"
        })
    
    return specs

def generate_n8n_component(spec: Dict, component_type: str, generate_credentials: bool) -> Dict:
    """Generate n8n node component from specification"""
    
    name = spec["name"]
    namespace = spec["namespace"]
    
    # Base n8n node structure
    component = {
        "name": f"{namespace}.{name}",
        "displayName": name.replace("_", " ").title(),
        "description": f"Generated from Lean4 specification: {name}",
        "version": 1.0,
        "defaults": {
            "name": name.replace("_", " ").title()
        },
        "inputs": ["main"],
        "outputs": ["main"],
        "properties": []
    }
    
    if spec["kind"] == "structure":
        # Generate properties from Lean4 structure fields
        for field in spec["fields"]:
            field_name = field["name"]
            field_type = field["type"]
            
            # Map Lean4 types to n8n types
            n8n_type = "string"
            if field_type in ["Int", "Nat"]:
                n8n_type = "number"
            elif field_type == "Bool":
                n8n_type = "boolean"
            
            property_def = {
                "displayName": field_name.replace("_", " ").title(),
                "name": field_name,
                "type": n8n_type,
                "default": field.get("default", ""),
                "description": f"{field_name} ({field_type})"
            }
            
            component["properties"].append(property_def)
    
    elif spec["kind"] == "api_endpoints":
        # Generate HTTP request node configuration
        component["properties"] = [
            {
                "displayName": "Endpoint",
                "name": "endpoint",
                "type": "options",
                "options": [
                    {"name": "Health Check", "value": "/health"},
                    {"name": "Generate", "value": "/generate"}
                ],
                "default": "/health",
                "description": "API endpoint to call"
            },
            {
                "displayName": "Method",
                "name": "method",
                "type": "options",
                "options": [
                    {"name": "GET", "value": "GET"},
                    {"name": "POST", "value": "POST"}
                ],
                "default": "GET"
            },
            {
                "displayName": "Request Body",
                "name": "body",
                "type": "json",
                "default": "{}",
                "description": "Request body for POST requests"
            }
        ]
    
    # Add credentials if requested
    if generate_credentials:
        component["credentials"] = [
            {
                "name": f"{namespace}Api",
                "required": False
            }
        ]
    
    # Add execute function template
    component["codex"] = {
        "categories": ["Custom"],
        "resources": {
            "primaryDocumentation": [
                {
                    "url": f"https://example.com/{namespace}/docs"
                }
            ]
        }
    }
    
    return component

@app.get("/n8n-components")
async def list_n8n_components():
    """List all generated n8n components"""
    try:
        output_dir = Path("/app/n8n-components")
        if not output_dir.exists():
            return {"components": [], "count": 0}
        
        components = []
        for component_file in output_dir.rglob("*.node.json"):
            with open(component_file) as f:
                component_data = json.load(f)
            
            components.append({
                "name": component_data.get("name", ""),
                "displayName": component_data.get("displayName", ""),
                "path": str(component_file),
                "version": component_data.get("version", 1.0),
                "properties_count": len(component_data.get("properties", [])),
                "created": datetime.fromtimestamp(component_file.stat().st_mtime).isoformat()
            })
        
        return {"components": components, "count": len(components)}
        
    except Exception as e:
        logger.error(f"Error listing n8n components: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-workflow")
async def generate_n8n_workflow(lean4_file_path: str, workflow_name: str):
    """Generate complete n8n workflow from Lean4 file"""
    try:
        lean4_file = Path(lean4_file_path)
        if not lean4_file.exists():
            raise HTTPException(status_code=404, detail="Lean4 file not found")
        
        with open(lean4_file) as f:
            lean4_content = f.read()
        
        specs = parse_lean4_specifications(lean4_content)
        
        # Generate n8n workflow
        workflow = {
            "name": workflow_name,
            "nodes": [],
            "connections": {},
            "settings": {},
            "staticData": None,
            "tags": ["generated", "lean4"],
            "triggerCount": 0,
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }
        
        # Add start node
        workflow["nodes"].append({
            "parameters": {},
            "name": "Start",
            "type": "n8n-nodes-base.start",
            "typeVersion": 1,
            "position": [250, 300]
        })
        
        # Add nodes for each spec
        for i, spec in enumerate(specs):
            node_name = spec["name"]
            workflow["nodes"].append({
                "parameters": {
                    "specification": spec
                },
                "name": node_name,
                "type": f"n8n-nodes-custom.{spec['namespace']}.{node_name}",
                "typeVersion": 1,
                "position": [450 + (i * 200), 300]
            })
            
            # Connect nodes
            if i == 0:
                workflow["connections"]["Start"] = {
                    "main": [[{"node": node_name, "type": "main", "index": 0}]]
                }
            else:
                prev_node = specs[i-1]["name"]
                workflow["connections"][prev_node] = {
                    "main": [[{"node": node_name, "type": "main", "index": 0}]]
                }
        
        # Save workflow
        output_dir = Path("/app/n8n-workflows")
        output_dir.mkdir(exist_ok=True)
        workflow_file = output_dir / f"{workflow_name}.json"
        
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "workflow": workflow,
            "file_path": str(workflow_file),
            "nodes_count": len(workflow["nodes"])
        }
        
    except Exception as e:
        logger.error(f"Error generating workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")