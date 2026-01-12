"""
Service Discovery API Routes
Provides endpoints for service discovery and health monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from backend.services.discovery import get_service_discovery, ServiceDiscovery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["Service Discovery"])

class ServiceRegistration(BaseModel):
    """Service registration request"""
    name: str
    url: str
    health_endpoint: str
    metadata: Optional[Dict[str, Any]] = None

class ServiceStatus(BaseModel):
    """Service status response"""
    name: str
    url: str
    status: str
    last_check: Optional[str] = None
    response_time: Optional[float] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = {}

class RegistryStatus(BaseModel):
    """Registry status response"""
    healthy_services: int
    total_services: int
    last_updated: Optional[str] = None
    services: Dict[str, ServiceStatus]

@router.get("/status", response_model=RegistryStatus)
async def get_registry_status(discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """Get complete service registry status"""
    try:
        status = discovery.get_registry_status()
        
        # Convert to response model
        services_status = {}
        for service_id, service_data in status['services'].items():
            services_status[service_id] = ServiceStatus(**service_data)
        
        return RegistryStatus(
            healthy_services=status['healthy_services'],
            total_services=status['total_services'],
            last_updated=status['last_updated'],
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get registry status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get registry status")

@router.get("/services", response_model=Dict[str, ServiceStatus])
async def list_services(discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """List all registered services"""
    try:
        status = discovery.get_registry_status()
        
        services_status = {}
        for service_id, service_data in status['services'].items():
            services_status[service_id] = ServiceStatus(**service_data)
        
        return services_status
        
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(status_code=500, detail="Failed to list services")

@router.get("/services/healthy", response_model=List[str])
async def list_healthy_services(discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """List healthy service IDs"""
    try:
        healthy_services = discovery.get_healthy_services()
        return [service.name for service in healthy_services]
        
    except Exception as e:
        logger.error(f"Failed to list healthy services: {e}")
        raise HTTPException(status_code=500, detail="Failed to list healthy services")

@router.get("/services/{service_id}", response_model=ServiceStatus)
async def get_service_status(service_id: str, discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """Get status of a specific service"""
    try:
        endpoint = discovery.get_service(service_id)
        if not endpoint:
            raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")
        
        return ServiceStatus(
            name=endpoint.name,
            url=endpoint.url,
            status=endpoint.status,
            last_check=endpoint.last_check.isoformat() if endpoint.last_check else None,
            response_time=endpoint.response_time,
            version=endpoint.version,
            metadata=endpoint.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service status for {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")

@router.get("/services/{service_id}/url")
async def get_service_url(service_id: str, discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """Get URL of a healthy service"""
    try:
        url = discovery.get_service_url(service_id)
        if not url:
            raise HTTPException(status_code=404, detail=f"Healthy service '{service_id}' not found")
        
        return {"service_id": service_id, "url": url}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service URL for {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service URL")

@router.post("/services/{service_id}/register")
async def register_service(
    service_id: str, 
    registration: ServiceRegistration,
    discovery: ServiceDiscovery = Depends(get_service_discovery)
):
    """Register a new service"""
    try:
        await discovery.register_service(
            service_id=service_id,
            name=registration.name,
            url=registration.url,
            health_endpoint=registration.health_endpoint,
            metadata=registration.metadata
        )
        
        return {"message": f"Service '{service_id}' registered successfully"}
        
    except Exception as e:
        logger.error(f"Failed to register service {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to register service")

@router.delete("/services/{service_id}")
async def unregister_service(service_id: str, discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """Unregister a service"""
    try:
        await discovery.unregister_service(service_id)
        return {"message": f"Service '{service_id}' unregistered successfully"}
        
    except Exception as e:
        logger.error(f"Failed to unregister service {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to unregister service")

@router.post("/refresh")
async def refresh_services(discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """Manually refresh service discovery"""
    try:
        await discovery.discover_services()
        status = discovery.get_registry_status()
        
        return {
            "message": "Service discovery refreshed",
            "healthy_services": status['healthy_services'],
            "total_services": status['total_services']
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh services: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh services")

@router.get("/services/type/{service_type}")
async def get_services_by_type(service_type: str, discovery: ServiceDiscovery = Depends(get_service_discovery)):
    """Get services by type"""
    try:
        services = discovery.get_services_by_type(service_type)
        
        result = []
        for service in services:
            result.append({
                "name": service.name,
                "url": service.url,
                "status": service.status,
                "metadata": service.metadata
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get services by type {service_type}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get services by type")

@router.get("/health")
async def health_check():
    """Health check endpoint for the discovery service itself"""
    return {
        "status": "healthy",
        "service": "service_discovery",
        "version": "1.0.0"
    }
