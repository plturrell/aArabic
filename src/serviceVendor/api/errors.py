"""
Custom exception classes and error handling
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any
from backend.constants import (
    HTTP_BAD_REQUEST,
    HTTP_NOT_FOUND,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_SERVICE_UNAVAILABLE,
)


class APIError(HTTPException):
    """Base API error class"""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code or f"ERR_{status_code}"
        self.details = details or {}
        super().__init__(status_code=status_code, detail=self._format_error())
    
    def _format_error(self) -> Dict[str, Any]:
        """Format error response"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.detail if isinstance(self.detail, str) else str(self.detail),
                "details": self.details
            }
        }


class ValidationError(APIError):
    """Validation error (400)"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            status_code=HTTP_BAD_REQUEST,
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class NotFoundError(APIError):
    """Resource not found error (404)"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=HTTP_NOT_FOUND,
            message=f"{resource_type} '{resource_id}' not found",
            error_code="NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ServiceUnavailableError(APIError):
    """Service unavailable error (503)"""
    
    def __init__(self, service_name: str, message: Optional[str] = None):
        super().__init__(
            status_code=HTTP_SERVICE_UNAVAILABLE,
            message=message or f"{service_name} service is not available",
            error_code="SERVICE_UNAVAILABLE",
            details={"service": service_name}
        )


class InternalServerError(APIError):
    """Internal server error (500)"""
    
    def __init__(self, message: str = "Internal server error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=HTTP_INTERNAL_SERVER_ERROR,
            message=message,
            error_code="INTERNAL_ERROR",
            details=details or {}
        )

