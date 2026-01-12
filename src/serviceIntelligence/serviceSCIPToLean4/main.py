"""
SCIP to Lean4 Generator Service
Converts SCIP code indexes into Lean4 formal specifications
Port: 8009
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SCIP to Lean4 Generator",
    description="Generate Lean4 formal specifications from SCIP code indexes",
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

class Lean4GenerationRequest(BaseModel):
    scip_index_path: str
    project_name: str
    generate_proofs: bool = False
    include_tests: bool = True

class Lean4GenerationResponse(BaseModel):
    success: bool
    lean4_code: str
    file_path: str
    specifications_count: int
    theorems_count: int

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
        service="scip-to-lean4",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/generate-lean4", response_model=Lean4GenerationResponse)
async def generate_lean4_from_scip(request: Lean4GenerationRequest):
    """
    Generate Lean4 formal specifications from SCIP index
    
    Example:
    {
        "scip_index_path": "/app/scip-indexes/shimmy-ai.scip",
        "project_name": "ShimmyAI",
        "generate_proofs": false,
        "include_tests": true
    }
    """
    try:
        scip_index_path = Path(request.scip_index_path)
        if not scip_index_path.exists():
            raise HTTPException(status_code=404, detail="SCIP index not found")
        
        # Load SCIP index
        with open(scip_index_path) as f:
            scip_data = json.load(f)
        
        logger.info(f"Generating Lean4 from SCIP index: {scip_index_path.name}")
        
        # Extract metadata
        metadata = scip_data.get("metadata", {})
        documents = scip_data.get("documents", [])
        
        # Generate Lean4 code
        lean4_code = generate_lean4_specifications(
            project_name=request.project_name,
            scip_data=scip_data,
            generate_proofs=request.generate_proofs,
            include_tests=request.include_tests
        )
        
        # Save Lean4 file
        output_dir = Path("/app/lean4-output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{request.project_name}.lean"
        
        with open(output_file, 'w') as f:
            f.write(lean4_code)
        
        # Count specifications and theorems
        specs_count = lean4_code.count("def ") + lean4_code.count("structure ")
        theorems_count = lean4_code.count("theorem ")
        
        logger.info(f"Generated Lean4: {specs_count} specs, {theorems_count} theorems")
        
        return Lean4GenerationResponse(
            success=True,
            lean4_code=lean4_code,
            file_path=str(output_file),
            specifications_count=specs_count,
            theorems_count=theorems_count
        )
        
    except Exception as e:
        logger.error(f"Error generating Lean4: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_lean4_specifications(
    project_name: str,
    scip_data: Dict,
    generate_proofs: bool,
    include_tests: bool
) -> str:
    """Generate Lean4 code from SCIP data"""
    
    documents = scip_data.get("documents", [])
    metadata = scip_data.get("metadata", {})
    
    # Start Lean4 file
    lean4_lines = [
        f"-- Lean4 Formal Specification for {project_name}",
        f"-- Generated from SCIP index",
        f"-- Project: {metadata.get('project_root', 'unknown')}",
        "",
        "import Lean",
        "import Std",
        "",
        f"namespace {project_name.replace('-', '')}",
        ""
    ]
    
    # Generate structures for each module/symbol
    for doc in documents:
        doc_path = doc.get("relative_path", "")
        symbols = doc.get("symbols", [])
        
        if not symbols:
            continue
        
        lean4_lines.append(f"-- From: {doc_path}")
        lean4_lines.append("")
        
        for symbol in symbols:
            symbol_name = symbol.get("display_name", "unknown")
            symbol_kind = symbol.get("kind", "unknown")
            
            # Clean symbol name for Lean4
            clean_name = symbol_name.replace("-", "_").replace(".", "_")
            
            if symbol_kind in ["function", "method"]:
                # Generate function specification
                lean4_lines.extend([
                    f"-- Function: {symbol_name}",
                    f"structure {clean_name}Spec where",
                    f"  name : String := \"{symbol_name}\"",
                    f"  kind : String := \"{symbol_kind}\"",
                    f"  -- Add preconditions",
                    f"  precondition : Bool := true",
                    f"  -- Add postconditions",
                    f"  postcondition : Bool := true",
                    ""
                ])
                
                if generate_proofs:
                    lean4_lines.extend([
                        f"-- Correctness theorem for {symbol_name}",
                        f"theorem {clean_name}_correct : ",
                        f"  ∀ (spec : {clean_name}Spec),",
                        f"    spec.precondition → spec.postcondition := by",
                        f"  sorry  -- Proof to be completed",
                        ""
                    ])
            
            elif symbol_kind in ["module", "class", "struct"]:
                # Generate module/type specification
                lean4_lines.extend([
                    f"-- Module: {symbol_name}",
                    f"structure {clean_name}Type where",
                    f"  name : String := \"{symbol_name}\"",
                    f"  kind : String := \"{symbol_kind}\"",
                    f"  invariant : Bool := true",
                    ""
                ])
    
    # Add integration API specification
    lean4_lines.extend([
        "-- API Integration Specification",
        "structure APIEndpoint where",
        "  path : String",
        "  method : String",
        "  request_type : String",
        "  response_type : String",
        "",
        f"-- {project_name} API endpoints",
        f"def {project_name.replace('-', '_')}_endpoints : List APIEndpoint := [",
        "  { path := \"/health\", method := \"GET\", request_type := \"None\", response_type := \"HealthResponse\" },",
        "  { path := \"/generate\", method := \"POST\", request_type := \"GenerateRequest\", response_type := \"GenerateResponse\" }",
        "]",
        ""
    ])
    
    if include_tests:
        lean4_lines.extend([
            "-- Test specifications",
            "#check APIEndpoint",
            f"#eval {project_name.replace('-', '_')}_endpoints.length",
            ""
        ])
    
    # Close namespace
    lean4_lines.extend([
        f"end {project_name.replace('-', '')}",
        ""
    ])
    
    return "\n".join(lean4_lines)

@app.get("/lean4-files")
async def list_lean4_files():
    """List all generated Lean4 files"""
    try:
        output_dir = Path("/app/lean4-output")
        if not output_dir.exists():
            return {"files": [], "count": 0}
        
        files = []
        for lean_file in output_dir.glob("*.lean"):
            with open(lean_file) as f:
                content = f.read()
            
            files.append({
                "name": lean_file.stem,
                "path": str(lean_file),
                "size_bytes": lean_file.stat().st_size,
                "lines": len(content.split('\n')),
                "created": datetime.fromtimestamp(lean_file.stat().st_mtime).isoformat()
            })
        
        return {"files": files, "count": len(files)}
        
    except Exception as e:
        logger.error(f"Error listing Lean4 files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009, log_level="info")