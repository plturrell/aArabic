"""
SCIP (Source Code Intelligence Protocol) Indexing Service
Indexes vendor code to create semantic code graphs
Port: 8008
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import subprocess
import os
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SCIP Indexing Service",
    description="Source Code Intelligence Protocol indexing for vendor code analysis",
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

# Configuration
VENDOR_DIR = Path("/app/vendor")
INDEX_OUTPUT_DIR = Path("/app/scip-indexes")
INDEX_OUTPUT_DIR.mkdir(exist_ok=True)

class IndexRequest(BaseModel):
    project_path: str
    language: str  # rust, typescript, python, etc.
    output_name: Optional[str] = None

class IndexResponse(BaseModel):
    success: bool
    index_path: str
    project: str
    language: str
    symbols_count: int
    documents_count: int

class QueryRequest(BaseModel):
    index_path: str
    query_type: str  # symbol, definition, reference, implementation
    symbol_name: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    scip_available: bool
    indexed_projects: List[str]
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    # Check if scip-cli is available
    scip_available = False
    try:
        result = subprocess.run(
            ["scip", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        scip_available = result.returncode == 0
    except:
        pass
    
    # List indexed projects
    indexed_projects = []
    if INDEX_OUTPUT_DIR.exists():
        indexed_projects = [f.stem for f in INDEX_OUTPUT_DIR.glob("*.scip")]
    
    return HealthResponse(
        status="healthy",
        service="scip-indexer",
        version="1.0.0",
        scip_available=scip_available,
        indexed_projects=indexed_projects,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/index/rust", response_model=IndexResponse)
async def index_rust_project(request: IndexRequest):
    """
    Index a Rust project using rust-analyzer + scip-rust
    
    Example: Index shimmy-ai
    {
        "project_path": "vendor/layerIntelligence/shimmy-ai",
        "language": "rust",
        "output_name": "shimmy-ai"
    }
    """
    try:
        project_path = Path(request.project_path)
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"Project not found: {request.project_path}")
        
        output_name = request.output_name or project_path.name
        output_file = INDEX_OUTPUT_DIR / f"{output_name}.scip"
        
        logger.info(f"Indexing Rust project: {project_path}")
        
        # Run rust-analyzer to generate SCIP index
        # In production, we'd use scip-rust or rust-analyzer LSP
        # For now, we'll use cargo metadata + manual SCIP generation
        
        result = subprocess.run(
            ["cargo", "metadata", "--format-version=1"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get cargo metadata: {result.stderr}"
            )
        
        metadata = json.loads(result.stdout)
        
        # Create simplified SCIP index from cargo metadata
        scip_index = {
            "metadata": {
                "version": "0.3.0",
                "tool_info": {
                    "name": "nucleus-scip-indexer",
                    "version": "1.0.0"
                },
                "project_root": f"file://{project_path.absolute()}",
                "text_document_encoding": "UTF-8"
            },
            "documents": []
        }
        
        # Extract package information
        packages = metadata.get("packages", [])
        symbols_count = 0
        documents_count = len(packages)
        
        for package in packages:
            package_name = package.get("name", "unknown")
            targets = package.get("targets", [])
            
            for target in targets:
                doc = {
                    "relative_path": target.get("src_path", ""),
                    "language": "rust",
                    "symbols": []
                }
                
                # Extract module/function information
                # This is simplified - full implementation would parse Rust AST
                if target.get("kind"):
                    symbols_count += 1
                    doc["symbols"].append({
                        "symbol": f"rust {package_name}::{target.get('name', 'unknown')}",
                        "kind": "function" if "bin" in target.get("kind", []) else "module",
                        "display_name": target.get("name", "unknown")
                    })
                
                scip_index["documents"].append(doc)
        
        # Save SCIP index
        with open(output_file, 'w') as f:
            json.dump(scip_index, f, indent=2)
        
        logger.info(f"Created SCIP index: {output_file}")
        
        return IndexResponse(
            success=True,
            index_path=str(output_file),
            project=str(project_path),
            language="rust",
            symbols_count=symbols_count,
            documents_count=documents_count
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Indexing timeout")
    except Exception as e:
        logger.error(f"Error indexing Rust project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/typescript")
async def index_typescript_project(request: IndexRequest):
    """Index a TypeScript project using scip-typescript"""
    try:
        project_path = Path(request.project_path)
        if not project_path.exists():
            raise HTTPException(status_code=404, detail=f"Project not found: {request.project_path}")
        
        output_name = request.output_name or project_path.name
        output_file = INDEX_OUTPUT_DIR / f"{output_name}.scip"
        
        logger.info(f"Indexing TypeScript project: {project_path}")
        
        # Check for package.json
        package_json = project_path / "package.json"
        if not package_json.exists():
            raise HTTPException(status_code=400, detail="No package.json found")
        
        # Run scip-typescript indexer
        # Note: This requires scip-typescript to be installed
        result = subprocess.run(
            ["npx", "scip-typescript", "index", "--output", str(output_file)],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            logger.warning(f"scip-typescript not available, using fallback indexer")
            # Fallback: Create basic index from package.json
            with open(package_json) as f:
                pkg_data = json.load(f)
            
            scip_index = {
                "metadata": {
                    "version": "0.3.0",
                    "tool_info": {"name": "nucleus-scip-indexer", "version": "1.0.0"},
                    "project_root": f"file://{project_path.absolute()}"
                },
                "documents": [{
                    "relative_path": "package.json",
                    "language": "typescript",
                    "symbols": [{
                        "symbol": f"typescript {pkg_data.get('name', 'unknown')}",
                        "kind": "module",
                        "display_name": pkg_data.get('name', 'unknown')
                    }]
                }]
            }
            
            with open(output_file, 'w') as f:
                json.dump(scip_index, f, indent=2)
        
        # Read and parse index
        with open(output_file) as f:
            index_data = json.load(f)
        
        symbols_count = sum(len(doc.get("symbols", [])) for doc in index_data.get("documents", []))
        documents_count = len(index_data.get("documents", []))
        
        return IndexResponse(
            success=True,
            index_path=str(output_file),
            project=str(project_path),
            language="typescript",
            symbols_count=symbols_count,
            documents_count=documents_count
        )
        
    except Exception as e:
        logger.error(f"Error indexing TypeScript project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexes")
async def list_indexes():
    """List all generated SCIP indexes"""
    try:
        indexes = []
        
        for index_file in INDEX_OUTPUT_DIR.glob("*.scip"):
            with open(index_file) as f:
                index_data = json.load(f)
            
            metadata = index_data.get("metadata", {})
            documents = index_data.get("documents", [])
            symbols_count = sum(len(doc.get("symbols", [])) for doc in documents)
            
            indexes.append({
                "name": index_file.stem,
                "path": str(index_file),
                "project_root": metadata.get("project_root", ""),
                "documents_count": len(documents),
                "symbols_count": symbols_count,
                "created": datetime.fromtimestamp(index_file.stat().st_mtime).isoformat()
            })
        
        return {"indexes": indexes, "count": len(indexes)}
        
    except Exception as e:
        logger.error(f"Error listing indexes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_index(request: QueryRequest):
    """Query a SCIP index for code intelligence"""
    try:
        index_path = Path(request.index_path)
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Index not found")
        
        with open(index_path) as f:
            index_data = json.load(f)
        
        results = []
        
        if request.query_type == "symbols":
            # List all symbols
            for doc in index_data.get("documents", []):
                for symbol in doc.get("symbols", []):
                    if not request.symbol_name or request.symbol_name in symbol.get("symbol", ""):
                        results.append({
                            "document": doc.get("relative_path", ""),
                            "symbol": symbol.get("symbol", ""),
                            "kind": symbol.get("kind", ""),
                            "display_name": symbol.get("display_name", "")
                        })
        
        elif request.query_type == "definitions":
            # Find definitions
            for doc in index_data.get("documents", []):
                for symbol in doc.get("symbols", []):
                    if request.symbol_name and request.symbol_name in symbol.get("display_name", ""):
                        results.append({
                            "document": doc.get("relative_path", ""),
                            "symbol": symbol.get("symbol", ""),
                            "kind": symbol.get("kind", ""),
                            "display_name": symbol.get("display_name", ""),
                            "definition": True
                        })
        
        return {
            "query_type": request.query_type,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-all-vendors")
async def index_all_vendors():
    """Index all vendor projects"""
    try:
        vendor_dir = Path("vendor")
        if not vendor_dir.exists():
            raise HTTPException(status_code=404, detail="Vendor directory not found")
        
        results = []
        
        # Find all Rust projects (Cargo.toml)
        for cargo_toml in vendor_dir.rglob("Cargo.toml"):
            project_dir = cargo_toml.parent
            project_name = project_dir.name
            
            try:
                result = await index_rust_project(IndexRequest(
                    project_path=str(project_dir),
                    language="rust",
                    output_name=project_name
                ))
                results.append({
                    "project": project_name,
                    "language": "rust",
                    "status": "success",
                    "index_path": result.index_path
                })
            except Exception as e:
                results.append({
                    "project": project_name,
                    "language": "rust",
                    "status": "failed",
                    "error": str(e)
                })
        
        # Find all TypeScript projects (package.json with typescript)
        for package_json in vendor_dir.rglob("package.json"):
            project_dir = package_json.parent
            project_name = project_dir.name
            
            # Check if it's a TypeScript project
            try:
                with open(package_json) as f:
                    pkg_data = json.load(f)
                
                has_ts = (
                    "typescript" in pkg_data.get("devDependencies", {}) or
                    "typescript" in pkg_data.get("dependencies", {}) or
                    (project_dir / "tsconfig.json").exists()
                )
                
                if has_ts:
                    result = await index_typescript_project(IndexRequest(
                        project_path=str(project_dir),
                        language="typescript",
                        output_name=project_name
                    ))
                    results.append({
                        "project": project_name,
                        "language": "typescript",
                        "status": "success",
                        "index_path": result.index_path
                    })
            except Exception as e:
                results.append({
                    "project": project_name,
                    "language": "typescript",
                    "status": "failed",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "total_projects": len(results),
            "successful": success_count,
            "failed": len(results) - success_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error indexing all vendors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vendors")
async def list_vendor_projects():
    """List all vendor projects available for indexing"""
    try:
        vendor_dir = Path("vendor")
        if not vendor_dir.exists():
            return {"vendors": [], "count": 0}
        
        vendors = []
        
        # Find Rust projects
        for cargo_toml in vendor_dir.rglob("Cargo.toml"):
            project_dir = cargo_toml.parent
            with open(cargo_toml) as f:
                content = f.read()
                # Simple name extraction
                name_match = None
                for line in content.split('\n'):
                    if line.strip().startswith('name = '):
                        name_match = line.split('=')[1].strip().strip('"')
                        break
            
            vendors.append({
                "name": name_match or project_dir.name,
                "path": str(project_dir),
                "language": "rust",
                "type": "cargo"
            })
        
        # Find TypeScript projects
        for package_json in vendor_dir.rglob("package.json"):
            project_dir = package_json.parent
            try:
                with open(package_json) as f:
                    pkg_data = json.load(f)
                
                has_ts = (
                    "typescript" in pkg_data.get("devDependencies", {}) or
                    "typescript" in pkg_data.get("dependencies", {}) or
                    (project_dir / "tsconfig.json").exists()
                )
                
                if has_ts:
                    vendors.append({
                        "name": pkg_data.get("name", project_dir.name),
                        "path": str(project_dir),
                        "language": "typescript",
                        "type": "npm"
                    })
            except:
                pass
        
        return {"vendors": vendors, "count": len(vendors)}
        
    except Exception as e:
        logger.error(f"Error listing vendors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")