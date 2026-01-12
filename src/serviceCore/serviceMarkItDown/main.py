"""
MarkItDown Service
FastAPI server that converts documents to markdown using Microsoft's MarkItDown
Supports: PDF, XLS, PPT, DOC, Images, Audio, and more
Port: 8005
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from markitdown import MarkItDown
from pydantic import BaseModel
import io
import logging
from typing import Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarkItDown Service",
    description="Multi-format document converter to Markdown",
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

# Initialize MarkItDown
md_converter = MarkItDown()

class ConversionResponse(BaseModel):
    success: bool
    markdown: str
    metadata: dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    supported_formats: list[str]
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="markitdown",
        version="1.0.0",
        supported_formats=[
            "pdf", "docx", "doc", "pptx", "ppt",
            "xlsx", "xls", "csv", "html", "txt",
            "jpg", "jpeg", "png", "gif", "bmp",
            "wav", "mp3", "zip", "epub", "json", "xml"
        ],
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/convert", response_model=ConversionResponse)
async def convert_document(
    file: UploadFile = File(...),
    use_llm_for_images: bool = False
):
    """
    Convert uploaded document to Markdown
    
    Args:
        file: Document file (PDF, Office, Image, etc.)
        use_llm_for_images: Whether to use LLM for image descriptions
        
    Returns:
        Markdown text with preserved structure
    """
    try:
        logger.info(f"Converting file: {file.filename}, type: {file.content_type}")
        
        # Read file content
        content = await file.read()
        file_stream = io.BytesIO(content)
        
        # Convert to markdown
        result = md_converter.convert_stream(
            file_stream,
            file_extension=file.filename.split('.')[-1] if '.' in file.filename else None
        )
        
        # Extract metadata
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "format": file.filename.split('.')[-1] if '.' in file.filename else "unknown"
        }
        
        # Add title if found in markdown
        lines = result.text_content.split('\n')
        if lines and lines[0].startswith('#'):
            metadata["title"] = lines[0].lstrip('#').strip()
        
        logger.info(f"Successfully converted {file.filename} to markdown ({len(result.text_content)} chars)")
        
        return ConversionResponse(
            success=True,
            markdown=result.text_content,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error converting {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert-batch")
async def convert_batch(files: list[UploadFile] = File(...)):
    """Convert multiple documents at once"""
    results = []
    
    for file in files:
        try:
            content = await file.read()
            file_stream = io.BytesIO(content)
            
            result = md_converter.convert_stream(
                file_stream,
                file_extension=file.filename.split('.')[-1] if '.' in file.filename else None
            )
            
            results.append({
                "filename": file.filename,
                "success": True,
                "markdown": result.text_content,
                "size": len(result.text_content)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results, "total": len(files), "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")