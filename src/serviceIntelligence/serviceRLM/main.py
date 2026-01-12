"""
RLM Code Intelligence Service
Recursive reasoning over code analysis using all intelligence tools
Port: 8012
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RLM Code Intelligence Service",
    description="Recursive reasoning for code analysis using Shimmy + all intelligence tools",
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

# RLM will be initialized when first used
rlm_agent = None

def get_rlm_agent():
    """Lazy load RLM agent"""
    global rlm_agent
    if rlm_agent is None:
        try:
            from rlm import RLM
            
            rlm_agent = RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "nvidia/Orchestrator-8B",
                    "base_url": "http://localhost:11435/v1",
                    "api_key": "not-needed"
                },
                environment="local"
            )
            
            # Register tools
            register_tools(rlm_agent)
            
            logger.info("RLM agent initialized with Shimmy backend")
        except ImportError:
            logger.error("RLM not available. Install with: pip install rlm")
            raise HTTPException(status_code=503, detail="RLM not installed")
    
    return rlm_agent

def register_tools(agent):
    """Register all intelligence tools for RLM to use"""
    
    @agent.tool()
    def query_glean(project_name: str, query: str) -> Dict:
        """Query Glean database for code facts using Angle language"""
        try:
            response = requests.post(
                "http://localhost:8011/query",
                json={"project_name": project_name, "query": query},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    @agent.tool()
    def search_similar_code(symbol_text: str, limit: int = 5) -> List[Dict]:
        """Find similar code symbols using Qdrant vector search"""
        try:
            # Get embedding
            embedding = requests.post(
                "http://localhost:8007/embed",
                json={"text": symbol_text},
                timeout=10
            ).json()["embeddings"][0]
            
            # Search Qdrant
            from qdrant_client import QdrantClient
            qdrant = QdrantClient(host="localhost", port=6333)
            
            results = qdrant.search(
                collection_name="code_symbols",
                query_vector=embedding,
                limit=limit
            )
            
            return [
                {
                    "symbol": r.payload.get("symbol_name", ""),
                    "kind": r.payload.get("kind", ""),
                    "project": r.payload.get("project", ""),
                    "similarity": r.score
                }
                for r in results
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    @agent.tool()
    def query_code_graph(cypher_query: str) -> List[Dict]:
        """Query Memgraph code graph using Cypher"""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))
            
            with driver.session() as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            return [{"error": str(e)}]
    
    @agent.tool()
    def get_lineage(dataset_name: str) -> Dict:
        """Get data lineage from Marquez"""
        try:
            response = requests.get(
                f"http://localhost:5000/api/v1/lineage",
                params={"dataset": dataset_name},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    @agent.tool()
    def analyze_scip_index(index_path: str) -> Dict:
        """Get statistics from SCIP index"""
        try:
            response = requests.get(
                f"http://localhost:8008/indexes",
                timeout=10
            )
            indexes = response.json().get("indexes", [])
            for idx in indexes:
                if index_path in idx["path"]:
                    return idx
            return {"error": "Index not found"}
        except Exception as e:
            return {"error": str(e)}

class CodeAnalysisRequest(BaseModel):
    project_name: str
    task: str
    use_tools: bool = True
    max_iterations: int = 10

class WorkflowGenerationRequest(BaseModel):
    project_name: str
    requirements: str
    max_iterations: int = 15

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    rlm_available: bool
    shimmy_available: bool
    tools_registered: int
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    
    # Check RLM
    rlm_available = False
    try:
        from rlm import RLM
        rlm_available = True
    except ImportError:
        pass
    
    # Check Shimmy
    shimmy_available = False
    try:
        response = requests.get("http://localhost:11435/health", timeout=2)
        shimmy_available = response.status_code == 200
    except:
        pass
    
    return HealthResponse(
        status="healthy" if (rlm_available and shimmy_available) else "degraded",
        service="rlm-code-intelligence",
        version="1.0.0",
        rlm_available=rlm_available,
        shimmy_available=shimmy_available,
        tools_registered=5,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/analyze")
async def analyze_with_rlm(request: CodeAnalysisRequest):
    """
    Use RLM to analyze code with recursive reasoning
    
    Example:
    {
        "project_name": "shimmy-ai",
        "task": "Find all critical functions and explain their purpose",
        "use_tools": true,
        "max_iterations": 10
    }
    """
    try:
        agent = get_rlm_agent()
        
        logger.info(f"RLM analyzing: {request.task}")
        
        prompt = f"""
        Project: {request.project_name}
        Task: {request.task}
        
        Available tools:
        - query_glean(project_name, query): Query code facts from Glean using Angle
        - search_similar_code(symbol_text, limit): Find similar code with Qdrant
        - query_code_graph(cypher_query): Query Memgraph code graph
        - get_lineage(dataset_name): Get data lineage from Marquez
        - analyze_scip_index(index_path): Get SCIP index statistics
        
        Analyze the code and provide detailed insights.
        Use multiple tools to get complete understanding.
        """
        
        result = agent.completion(
            prompt=prompt,
            max_iterations=request.max_iterations,
            use_tools=request.use_tools
        )
        
        return {
            "success": True,
            "project": request.project_name,
            "task": request.task,
            "analysis": result.text if hasattr(result, 'text') else str(result),
            "tool_calls": result.tool_calls if hasattr(result, 'tool_calls') else [],
            "iterations": result.iterations if hasattr(result, 'iterations') else 0
        }
        
    except Exception as e:
        logger.error(f"Error in RLM analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-workflow")
async def generate_workflow_with_rlm(request: WorkflowGenerationRequest):
    """
    Use RLM to generate optimal workflow with reasoning
    
    RLM will:
    1. Query Glean for code structure
    2. Search Qdrant for similar patterns
    3. Query Memgraph for dependencies
    4. Reason about optimal workflow
    5. Design n8n components
    
    Example:
    {
        "project_name": "shimmy-ai",
        "requirements": "Create workflow for local inference with caching and error handling",
        "max_iterations": 15
    }
    """
    try:
        agent = get_rlm_agent()
        
        prompt = f"""
        Generate an n8n workflow for project: {request.project_name}
        Requirements: {request.requirements}
        
        Steps to follow:
        1. Use query_glean to understand the code structure
        2. Use search_similar_code to find related patterns in other projects
        3. Use query_code_graph to understand dependencies and call relationships
        4. Use get_lineage to see how this fits in the data pipeline
        5. Design optimal workflow considering all insights
        6. Explain your reasoning for each component
        
        Return a structured workflow design with:
        - Component list
        - Connections between components
        - Rationale for design decisions
        - Potential improvements
        """
        
        result = agent.completion(
            prompt=prompt,
            max_iterations=request.max_iterations
        )
        
        return {
            "success": True,
            "project": request.project_name,
            "requirements": request.requirements,
            "workflow_design": result.text if hasattr(result, 'text') else str(result),
            "reasoning_steps": result.tool_calls if hasattr(result, 'tool_calls') else [],
            "iterations": result.iterations if hasattr(result, 'iterations') else 0
        }
        
    except Exception as e:
        logger.error(f"Error generating workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools():
    """List all tools available to RLM"""
    return {
        "tools": [
            {
                "name": "query_glean",
                "description": "Query Glean code database with Angle language",
                "parameters": ["project_name", "query"]
            },
            {
                "name": "search_similar_code",
                "description": "Find similar code using Qdrant vector search",
                "parameters": ["symbol_text", "limit"]
            },
            {
                "name": "query_code_graph",
                "description": "Query Memgraph code graph with Cypher",
                "parameters": ["cypher_query"]
            },
            {
                "name": "get_lineage",
                "description": "Get data lineage from Marquez",
                "parameters": ["dataset_name"]
            },
            {
                "name": "analyze_scip_index",
                "description": "Get SCIP index statistics",
                "parameters": ["index_path"]
            }
        ],
        "count": 5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012, log_level="info")