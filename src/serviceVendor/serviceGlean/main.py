"""
Glean Integration Service
Indexes SCIP data into Glean for powerful code querying
Port: 8011
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import subprocess
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Glean Integration Service",
    description="Index SCIP data into Glean for advanced code queries",
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
GLEAN_DB_DIR = Path("/app/glean-db")
SCIP_INDEXES_DIR = Path("/app/scip-indexes")
GLEAN_DB_DIR.mkdir(exist_ok=True)

class GleanIndexRequest(BaseModel):
    scip_index_path: str
    project_name: str
    repo_name: Optional[str] = None

class GleanQueryRequest(BaseModel):
    project_name: str
    query: str  # Angle query language
    limit: int = 100

class GleanIndexResponse(BaseModel):
    success: bool
    project_name: str
    glean_db_path: str
    facts_count: int
    indexed_at: str

class GleanQueryResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    count: int
    query: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    glean_available: bool
    indexed_projects: List[str]
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    # Check if glean is available
    glean_available = False
    try:
        result = subprocess.run(
            ["glean", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        glean_available = result.returncode == 0
    except:
        logger.warning("Glean binary not found - using fallback mode")
        glean_available = False
    
    # List indexed projects
    indexed_projects = []
    if GLEAN_DB_DIR.exists():
        indexed_projects = [d.name for d in GLEAN_DB_DIR.iterdir() if d.is_dir()]
    
    return HealthResponse(
        status="healthy",
        service="glean-integration",
        version="1.0.0",
        glean_available=glean_available,
        indexed_projects=indexed_projects,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/index", response_model=GleanIndexResponse)
async def index_scip_in_glean(request: GleanIndexRequest):
    """
    Index SCIP data into Glean database
    
    Example:
    {
        "scip_index_path": "/app/scip-indexes/shimmy-ai.scip",
        "project_name": "shimmy-ai",
        "repo_name": "github.com/shimmy-ai/shimmy"
    }
    """
    try:
        scip_path = Path(request.scip_index_path)
        if not scip_path.exists():
            raise HTTPException(status_code=404, detail="SCIP index not found")
        
        logger.info(f"Indexing SCIP data into Glean: {request.project_name}")
        
        # Create Glean DB directory for this project
        project_db_dir = GLEAN_DB_DIR / request.project_name
        project_db_dir.mkdir(exist_ok=True)
        
        # Load SCIP index
        with open(scip_path) as f:
            scip_data = json.load(f)
        
        # Convert SCIP to Glean facts
        glean_facts = convert_scip_to_glean_facts(
            scip_data=scip_data,
            project_name=request.project_name,
            repo_name=request.repo_name or request.project_name
        )
        
        # Save Glean facts
        facts_file = project_db_dir / "facts.json"
        with open(facts_file, 'w') as f:
            json.dump(glean_facts, f, indent=2)
        
        logger.info(f"Created Glean database at: {project_db_dir}")
        
        return GleanIndexResponse(
            success=True,
            project_name=request.project_name,
            glean_db_path=str(project_db_dir),
            facts_count=len(glean_facts.get("facts", [])),
            indexed_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error indexing in Glean: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_scip_to_glean_facts(scip_data: Dict, project_name: str, repo_name: str) -> Dict:
    """Convert SCIP index to Glean facts format"""
    
    facts = []
    metadata = scip_data.get("metadata", {})
    documents = scip_data.get("documents", [])
    
    # Create repo fact
    repo_fact = {
        "predicate": "code.Repo",
        "key": repo_name,
        "value": {
            "name": repo_name,
            "project": project_name
        }
    }
    facts.append(repo_fact)
    
    # Process each document
    for doc_idx, doc in enumerate(documents):
        doc_path = doc.get("relative_path", f"unknown_{doc_idx}")
        language = doc.get("language", "unknown")
        symbols = doc.get("symbols", [])
        
        # Create file fact
        file_fact = {
            "predicate": "code.File",
            "key": f"{repo_name}:{doc_path}",
            "value": {
                "path": doc_path,
                "repo": repo_name,
                "language": language
            }
        }
        facts.append(file_fact)
        
        # Create symbol facts
        for symbol in symbols:
            symbol_name = symbol.get("symbol", "")
            symbol_kind = symbol.get("kind", "unknown")
            display_name = symbol.get("display_name", symbol_name)
            
            symbol_fact = {
                "predicate": "code.Symbol",
                "key": symbol_name,
                "value": {
                    "name": display_name,
                    "kind": symbol_kind,
                    "file": f"{repo_name}:{doc_path}",
                    "repo": repo_name
                }
            }
            facts.append(symbol_fact)
            
            # Create definition fact
            definition_fact = {
                "predicate": "code.Definition",
                "key": f"{symbol_name}:def",
                "value": {
                    "symbol": symbol_name,
                    "file": f"{repo_name}:{doc_path}",
                    "kind": "definition"
                }
            }
            facts.append(definition_fact)
    
    return {
        "version": "1.0",
        "repo": repo_name,
        "project": project_name,
        "facts": facts,
        "predicates": [
            "code.Repo",
            "code.File",
            "code.Symbol",
            "code.Definition",
            "code.Reference"
        ]
    }

@app.post("/query", response_model=GleanQueryResponse)
async def query_glean(request: GleanQueryRequest):
    """
    Query Glean database using Angle query language
    
    Example queries:
    - Find all symbols: code.Symbol _
    - Find functions: code.Symbol { kind = "function" }
    - Find by name: code.Symbol { name = "shimmy" }
    """
    try:
        project_db_dir = GLEAN_DB_DIR / request.project_name
        if not project_db_dir.exists():
            raise HTTPException(status_code=404, detail="Project not indexed in Glean")
        
        logger.info(f"Querying Glean: {request.query}")
        
        # Load Glean facts
        facts_file = project_db_dir / "facts.json"
        with open(facts_file) as f:
            glean_data = json.load(f)
        
        # Parse and execute query (simplified Angle interpreter)
        results = execute_angle_query(
            query=request.query,
            facts=glean_data.get("facts", []),
            limit=request.limit
        )
        
        return GleanQueryResponse(
            success=True,
            results=results,
            count=len(results),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error querying Glean: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def execute_angle_query(query: str, facts: List[Dict], limit: int) -> List[Dict]:
    """
    Simplified Angle query executor
    Supports basic queries like:
    - code.Symbol _ (all symbols)
    - code.Symbol { kind = "function" } (filtered)
    """
    
    results = []
    
    # Parse query
    query = query.strip()
    
    # Extract predicate
    if " " in query:
        predicate = query.split()[0]
        filter_part = query.split("{")[1].split("}")[0].strip() if "{" in query else ""
    else:
        predicate = query
        filter_part = ""
    
    # Find matching facts
    for fact in facts:
        if fact.get("predicate") == predicate:
            # Apply filters
            if filter_part:
                # Parse filter: kind = "function"
                if "=" in filter_part:
                    key, value = filter_part.split("=")
                    key = key.strip()
                    value = value.strip().strip('"')
                    
                    if fact.get("value", {}).get(key) == value:
                        results.append(fact)
                else:
                    results.append(fact)
            else:
                results.append(fact)
            
            if len(results) >= limit:
                break
    
    return results

@app.get("/projects")
async def list_indexed_projects():
    """List all projects indexed in Glean"""
    try:
        projects = []
        
        for project_dir in GLEAN_DB_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            
            facts_file = project_dir / "facts.json"
            if facts_file.exists():
                with open(facts_file) as f:
                    data = json.load(f)
                
                projects.append({
                    "name": project_dir.name,
                    "repo": data.get("repo", ""),
                    "facts_count": len(data.get("facts", [])),
                    "predicates": data.get("predicates", []),
                    "indexed_at": datetime.fromtimestamp(facts_file.stat().st_mtime).isoformat()
                })
        
        return {"projects": projects, "count": len(projects)}
        
    except Exception as e:
        logger.error(f"Error listing projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query-examples")
async def get_query_examples():
    """Get example Angle queries"""
    return {
        "examples": [
            {
                "name": "All Symbols",
                "query": "code.Symbol _",
                "description": "Find all symbols in the codebase"
            },
            {
                "name": "Functions Only",
                "query": "code.Symbol { kind = \"function\" }",
                "description": "Find all function symbols"
            },
            {
                "name": "Modules Only",
                "query": "code.Symbol { kind = \"module\" }",
                "description": "Find all module symbols"
            },
            {
                "name": "All Files",
                "query": "code.File _",
                "description": "Find all files in the project"
            },
            {
                "name": "Rust Files",
                "query": "code.File { language = \"rust\" }",
                "description": "Find all Rust source files"
            },
            {
                "name": "All Definitions",
                "query": "code.Definition _",
                "description": "Find all code definitions"
            }
        ]
    }

@app.post("/analyze/{project_name}")
async def analyze_project(project_name: str):
    """Analyze project using Glean queries"""
    try:
        project_db_dir = GLEAN_DB_DIR / project_name
        if not project_db_dir.exists():
            raise HTTPException(status_code=404, detail="Project not indexed")
        
        # Load facts
        facts_file = project_db_dir / "facts.json"
        with open(facts_file) as f:
            data = json.load(f)
        
        facts = data.get("facts", [])
        
        # Analyze
        analysis = {
            "project": project_name,
            "total_facts": len(facts),
            "by_predicate": {},
            "by_language": {},
            "by_symbol_kind": {}
        }
        
        for fact in facts:
            predicate = fact.get("predicate", "unknown")
            analysis["by_predicate"][predicate] = analysis["by_predicate"].get(predicate, 0) + 1
            
            if predicate == "code.File":
                language = fact.get("value", {}).get("language", "unknown")
                analysis["by_language"][language] = analysis["by_language"].get(language, 0) + 1
            
            if predicate == "code.Symbol":
                kind = fact.get("value", {}).get("kind", "unknown")
                analysis["by_symbol_kind"][kind] = analysis["by_symbol_kind"].get(kind, 0) + 1
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011, log_level="info")