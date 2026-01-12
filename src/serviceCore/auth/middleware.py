"""
Authentication and authorization middleware
"""

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
from backend.config.settings import settings
from backend.api.errors import ValidationError

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request) -> Optional[str]:
    """
    Verify API key from request header
    
    Args:
        request: FastAPI request object
    
    Returns:
        API key if valid, None otherwise
    
    Raises:
        HTTPException: If authentication is required but key is invalid
    """
    if not settings.enable_auth:
        return None
    
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "MISSING_API_KEY",
                    "message": "API key is required"
                }
            }
        )
    
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "INVALID_API_KEY",
                    "message": "Invalid API key"
                }
            }
        )
    
    return api_key


def get_api_key_dependency():
    """Dependency for API key verification"""
    if settings.enable_auth:
        return verify_api_key
    return lambda: None

