"""
Main FastAPI application server
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from backend.config.settings import settings
from backend.utils.logging import setup_logging
from backend.api.routes import models, orchestration, discovery, a2ui, gateway, qdrant, hybrid_orchestration, saudi_otp_vat, nucleus_lm, nucleus_graph
from backend.api.errors import APIError
from backend.api.middleware import setup_rate_limiting
from backend.services.model_service import model_service
from backend.services.discovery import initialize_service_discovery, shutdown_service_discovery
from backend.services.gateway import initialize_api_gateway, shutdown_api_gateway
from backend.adapters.qdrant import initialize_qdrant_adapter, shutdown_qdrant_adapter
from backend.adapters.hybrid_orchestration import initialize_hybrid_orchestration, shutdown_hybrid_orchestration

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI Nucleus Backend - Arabic Invoice Processing with ToolOrchestra & A2UI",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if not settings.is_production else settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Setup rate limiting
if settings.enable_auth:
    setup_rate_limiting(app)

# Request logging middleware (must be first)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None
        }
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2)
            }
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "process_time_ms": round(process_time * 1000, 2)
            },
            exc_info=True
        )
        raise


# Error handler
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle API errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error" if not settings.debug else str(exc)
            }
        }
    )


# Include routers
app.include_router(models.router)
app.include_router(orchestration.router)
app.include_router(a2ui.router)
app.include_router(discovery.router)
app.include_router(gateway.router)
app.include_router(qdrant.router)
app.include_router(hybrid_orchestration.router)
app.include_router(saudi_otp_vat.router)
app.include_router(nucleus_lm.router)
app.include_router(nucleus_graph.router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize service discovery
    try:
        await initialize_service_discovery()
        logger.info("Service discovery initialized")
    except Exception as e:
        logger.error(f"Failed to initialize service discovery: {e}")

    # Initialize API gateway
    try:
        await initialize_api_gateway()
        logger.info("API gateway initialized")
    except Exception as e:
        logger.error(f"Failed to initialize API gateway: {e}")

    # Initialize Qdrant adapter
    try:
        success = await initialize_qdrant_adapter()
        if success:
            logger.info("Qdrant adapter initialized")
        else:
            logger.warning("Qdrant adapter initialization failed - vector features may be limited")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant adapter: {e}")

    # Initialize hybrid orchestration
    try:
        success = await initialize_hybrid_orchestration()
        if success:
            logger.info("Hybrid orchestration initialized")
        else:
            logger.warning("Hybrid orchestration initialization failed - some features may be limited")
    except Exception as e:
        logger.error(f"Failed to initialize hybrid orchestration: {e}")

    # Load models
    camel_loaded, m2m_loaded = model_service.load_models()
    if not camel_loaded and not m2m_loaded:
        logger.warning("No models loaded - some features may be unavailable")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")

    # Shutdown hybrid orchestration
    try:
        await shutdown_hybrid_orchestration()
        logger.info("Hybrid orchestration shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown hybrid orchestration: {e}")

    # Shutdown Qdrant adapter
    try:
        await shutdown_qdrant_adapter()
        logger.info("Qdrant adapter shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown Qdrant adapter: {e}")

    # Shutdown API gateway
    try:
        await shutdown_api_gateway()
        logger.info("API gateway shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown API gateway: {e}")

    # Shutdown service discovery
    try:
        await shutdown_service_discovery()
        logger.info("Service discovery shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown service discovery: {e}")

    # Close Shimmy service session
    try:
        await nucleus_lm.shimmy_service.close()
        logger.info("Shimmy service session closed")
    except Exception as e:
        logger.error(f"Failed to close Shimmy service session: {e}")

    # Close orchestration adapter session if needed
    if orchestration.orchestration_adapter:
        await orchestration.orchestration_adapter.close()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "docs": "/docs" if settings.debug else "disabled"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"\n🚀 Starting {settings.app_name}...")
    logger.info(f"👉 Server will run on {settings.host}:{settings.port}")
    logger.info(f"👉 Environment: {settings.environment}\n")
    
    uvicorn.run(
        "backend.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

