"""
Model API routes
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from backend.services.model_service import model_service
from backend.api.errors import ServiceUnavailableError
from backend.utils.validation import sanitize_string
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["models"])


class TextRequest(BaseModel):
    text: str


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    camel_loaded, m2m_loaded = model_service.load_models()
    return {
        "status": "online",
        "models": {
            "camelbert": camel_loaded,
            "m2m100": m2m_loaded
        },
        "ready": model_service.is_ready
    }


@router.post("/translate")
async def translate_text(req: TextRequest):
    """
    Translate Arabic text to English
    
    Args:
        req: Text request with text to translate
    
    Returns:
        Translation result
    """
    # Sanitize input
    text = sanitize_string(req.text, max_length=10000)
    
    try:
        result = model_service.translate(text)
        logger.info(f"Translation completed for text length: {len(text)}")
        return result
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise ServiceUnavailableError("translation", f"Translation service error: {str(e)}")


@router.post("/analyze")
async def analyze_invoice_text(req: TextRequest):
    """
    Analyze invoice text for compliance and classification
    
    Args:
        req: Text request with invoice text to analyze
    
    Returns:
        Analysis result with classification and confidence
    """
    # Sanitize input
    text = sanitize_string(req.text, max_length=10000)
    
    try:
        result = model_service.analyze(text)
        logger.info(f"Analysis completed for text length: {len(text)}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise ServiceUnavailableError("analysis", f"Analysis service error: {str(e)}")

