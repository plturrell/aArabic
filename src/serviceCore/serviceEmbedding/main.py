"""
Embedding Service - Local Text Embedding Generation
Uses sentence-transformers for vector embeddings
Port: 8007
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import List, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Embedding Service",
    description="Local text embedding generation using sentence-transformers",
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

# Initialize embedding model
embedding_model = None
if EMBEDDINGS_AVAILABLE:
    try:
        # Using all-MiniLM-L6-v2: 80MB, 768 dimensions, fast
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded embedding model: all-MiniLM-L6-v2 (768 dims)")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        embedding_model = None

class EmbeddingRequest(BaseModel):
    text: str | List[str]
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    model: str
    count: int

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_name: str
    dimensions: int
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if embedding_model else "degraded",
        service="embedding",
        version="1.0.0",
        model_loaded=embedding_model is not None,
        model_name="all-MiniLM-L6-v2" if embedding_model else "none",
        dimensions=768 if embedding_model else 0,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for text input
    
    Supports single text or batch of texts
    Returns 768-dimensional vectors
    """
    try:
        if not EMBEDDINGS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Embedding model not available. Install sentence-transformers."
            )
        
        if not embedding_model:
            raise HTTPException(
                status_code=503,
                detail="Embedding model failed to load"
            )
        
        # Convert single text to list
        texts = request.text if isinstance(request.text, list) else [request.text]
        
        logger.info(f"Generating embeddings for {len(texts)} text(s)")
        
        # Generate embeddings
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False
        )
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        logger.info(f"Generated {len(embeddings_list)} embeddings of {len(embeddings_list[0])} dimensions")
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]),
            model="all-MiniLM-L6-v2",
            count=len(embeddings_list)
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/workflow")
async def embed_workflow(workflow_text: str, workflow_metadata: Optional[dict] = None):
    """
    Generate embedding for workflow text with metadata
    Optimized for workflow similarity search
    """
    try:
        # Combine workflow text with metadata for richer embedding
        if workflow_metadata:
            workflow_name = workflow_metadata.get("name", "")
            workflow_desc = workflow_metadata.get("description", "")
            combined_text = f"{workflow_name}. {workflow_desc}. {workflow_text}"
        else:
            combined_text = workflow_text
        
        # Generate embedding
        embeddings = embedding_model.encode(
            [combined_text],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return {
            "embedding": embeddings[0].tolist(),
            "dimensions": len(embeddings[0]),
            "model": "all-MiniLM-L6-v2"
        }
        
    except Exception as e:
        logger.error(f"Error embedding workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/invoice")
async def embed_invoice(invoice_text: str, extracted_data: Optional[dict] = None):
    """
    Generate embedding for invoice with extracted data
    Optimized for invoice similarity search
    """
    try:
        # Combine invoice text with key extracted fields
        if extracted_data:
            vendor = extracted_data.get("vendor_name", "")
            amount = extracted_data.get("total_amount", "")
            currency = extracted_data.get("currency", "")
            combined_text = f"Vendor: {vendor}. Amount: {amount} {currency}. {invoice_text[:500]}"
        else:
            combined_text = invoice_text[:500]  # Truncate long invoices
        
        # Generate embedding
        embeddings = embedding_model.encode(
            [combined_text],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return {
            "embedding": embeddings[0].tolist(),
            "dimensions": len(embeddings[0]),
            "model": "all-MiniLM-L6-v2"
        }
        
    except Exception as e:
        logger.error(f"Error embedding invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/document")
async def embed_document(document_text: str, chunk_size: int = 512):
    """
    Generate embeddings for long documents (chunked)
    Returns multiple embeddings for large documents
    """
    try:
        # Split into chunks
        chunks = []
        words = document_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Generate embeddings for all chunks
        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return {
            "embeddings": embeddings.tolist(),
            "dimensions": len(embeddings[0]),
            "chunks": len(chunks),
            "model": "all-MiniLM-L6-v2"
        }
        
    except Exception as e:
        logger.error(f"Error embedding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available embedding models"""
    return {
        "current_model": {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 768,
            "size_mb": 80,
            "speed": "fast",
            "quality": "good"
        },
        "available_models": [
            {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 768,
                "size_mb": 80,
                "use_case": "General purpose, fast"
            },
            {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "size_mb": 420,
                "use_case": "Higher quality, slower"
            },
            {
                "name": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": 384,
                "size_mb": 420,
                "use_case": "Multilingual support (Arabic!)"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007, log_level="info")