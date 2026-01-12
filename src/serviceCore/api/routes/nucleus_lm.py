"""
NucleusLM API Routes
Provides local LLM capabilities using Shimmy backend
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from backend.adapters.shimmy import ShimmyService, ShimmyServiceStatus
from backend.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nucleus-lm", tags=["nucleus-lm"])

# Initialize Shimmy service
shimmy_service = ShimmyService(
    base_url=settings.shimmy_backend_url,
    default_model=settings.default_local_model
)


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="The input prompt")
    model: Optional[str] = Field(None, description="Model to use (defaults to configured model)")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    success: bool
    response: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response model for available models"""
    success: bool
    models: list = []
    error: Optional[str] = None


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health = await shimmy_service.health_check()
        shimmy_status = health.get("status")
        is_healthy = shimmy_status == ShimmyServiceStatus.HEALTHY or shimmy_status == ShimmyServiceStatus.HEALTHY.value

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "shimmy_backend": settings.shimmy_backend_url,
            "default_model": settings.default_local_model,
            "shimmy_health": health
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "shimmy_backend": settings.shimmy_backend_url
        }


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    try:
        models = await shimmy_service.list_models(refresh_cache=True)
        return ModelsResponse(
            success=True,
            models=[{
                "name": model.name,
                "path": model.path,
                "template": model.template,
                "ctx_len": model.ctx_len,
                "loaded": model.loaded,
                "size_mb": model.size_mb
            } for model in models]
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return ModelsResponse(success=False, error=str(e))


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using local LLM"""
    try:
        result = await shimmy_service.generate_text(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        if result.get("success"):
            return GenerateResponse(
                success=True,
                response=result.get("response"),
                model=result.get("model"),
                prompt_tokens=result.get("prompt_tokens"),
                completion_tokens=result.get("completion_tokens")
            )
        else:
            return GenerateResponse(
                success=False,
                error=result.get("error", "Unknown error")
            )
            
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        return GenerateResponse(success=False, error=str(e))


@router.post("/chat")
async def chat_completion(request: GenerateRequest):
    """Chat completion endpoint (alias for generate)"""
    return await generate_text(request)


@router.get("/discover")
async def discover_models():
    """Trigger model discovery"""
    try:
        result = await shimmy_service.discover_models()
        return {
            "success": True,
            "message": "Model discovery completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
