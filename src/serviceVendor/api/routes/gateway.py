"""
API Gateway Routes
Provides gateway endpoints for vendor service routing
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from typing import Dict, Any
import logging
import json

from backend.services.gateway import get_api_gateway, APIGateway

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gateway", tags=["API Gateway"])

@router.api_route("/shimmy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def shimmy_proxy(
    path: str,
    request: Request,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Proxy requests to Shimmy AI service"""
    return await proxy_request(f"/api/shimmy/{path}", request, gateway)

@router.api_route("/memgraph/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def memgraph_proxy(
    path: str,
    request: Request,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Proxy requests to Memgraph service"""
    return await proxy_request(f"/api/memgraph/{path}", request, gateway)

@router.api_route("/marquez/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def marquez_proxy(
    path: str,
    request: Request,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Proxy requests to Marquez lineage service"""
    return await proxy_request(f"/api/marquez/{path}", request, gateway)

@router.api_route("/toolorchestra/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def toolorchestra_proxy(
    path: str,
    request: Request,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Proxy requests to ToolOrchestra service"""
    return await proxy_request(f"/api/toolorchestra/{path}", request, gateway)

@router.api_route("/opencanvas/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def opencanvas_proxy(
    path: str,
    request: Request,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Proxy requests to Open Canvas service"""
    return await proxy_request(f"/api/opencanvas/{path}", request, gateway)

async def proxy_request(target_path: str, request: Request, gateway: APIGateway):
    """Generic request proxy function"""
    try:
        # Extract request data
        headers = dict(request.headers)
        params = dict(request.query_params)
        
        # Get request body
        json_data = None
        data = None
        
        if request.headers.get('content-type', '').startswith('application/json'):
            try:
                json_data = await request.json()
            except:
                pass
        else:
            data = await request.body()
        
        # Forward request through gateway
        status_code, response_headers, response_data = await gateway.forward_request(
            method=request.method,
            path=target_path,
            headers=headers,
            params=params,
            json_data=json_data,
            data=data
        )
        
        # Prepare response
        if isinstance(response_data, (dict, list)):
            return JSONResponse(
                content=response_data,
                status_code=status_code,
                headers=response_headers
            )
        elif isinstance(response_data, str):
            return Response(
                content=response_data,
                status_code=status_code,
                headers=response_headers,
                media_type="text/plain"
            )
        else:
            return Response(
                content=response_data,
                status_code=status_code,
                headers=response_headers
            )
            
    except Exception as e:
        logger.error(f"Gateway proxy error for {target_path}: {e}")
        raise HTTPException(status_code=500, detail="Gateway proxy error")

@router.get("/status")
async def gateway_status(gateway: APIGateway = Depends(get_api_gateway)):
    """Get gateway status and routing information"""
    try:
        routes_info = {}
        for path_prefix, route_config in gateway.routes.items():
            service_url = gateway.select_service_endpoint(route_config)
            circuit_breaker = gateway.circuit_breakers.get(service_url, {})
            
            routes_info[path_prefix] = {
                "service_type": route_config.service_type,
                "service_id": route_config.service_id,
                "service_url": service_url,
                "timeout": route_config.timeout,
                "retries": route_config.retries,
                "load_balance": route_config.load_balance,
                "circuit_breaker": {
                    "state": circuit_breaker.get("state", "closed"),
                    "failure_count": circuit_breaker.get("failure_count", 0),
                    "last_failure": circuit_breaker.get("last_failure").isoformat() if circuit_breaker.get("last_failure") else None
                }
            }
        
        return {
            "status": "healthy",
            "routes": routes_info,
            "service_counters": gateway.service_counters
        }
        
    except Exception as e:
        logger.error(f"Failed to get gateway status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get gateway status")

@router.post("/routes/{path_prefix:path}")
async def add_route(
    path_prefix: str,
    route_data: Dict[str, Any],
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Add a new route to the gateway"""
    try:
        from backend.services.gateway import RouteConfig
        
        # Ensure path starts with /
        if not path_prefix.startswith('/'):
            path_prefix = '/' + path_prefix
        
        route_config = RouteConfig(
            path_prefix=path_prefix,
            service_type=route_data["service_type"],
            service_id=route_data.get("service_id"),
            strip_prefix=route_data.get("strip_prefix", True),
            timeout=route_data.get("timeout", 30),
            retries=route_data.get("retries", 3),
            load_balance=route_data.get("load_balance", True)
        )
        
        gateway.routes[path_prefix] = route_config
        
        return {"message": f"Route {path_prefix} added successfully"}
        
    except Exception as e:
        logger.error(f"Failed to add route {path_prefix}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add route")

@router.delete("/routes/{path_prefix:path}")
async def remove_route(
    path_prefix: str,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Remove a route from the gateway"""
    try:
        # Ensure path starts with /
        if not path_prefix.startswith('/'):
            path_prefix = '/' + path_prefix
        
        if path_prefix in gateway.routes:
            del gateway.routes[path_prefix]
            return {"message": f"Route {path_prefix} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Route not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove route {path_prefix}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove route")

@router.post("/circuit-breaker/{service_url:path}/reset")
async def reset_circuit_breaker(
    service_url: str,
    gateway: APIGateway = Depends(get_api_gateway)
):
    """Reset circuit breaker for a service"""
    try:
        if service_url in gateway.circuit_breakers:
            gateway.circuit_breakers[service_url] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': None
            }
            return {"message": f"Circuit breaker reset for {service_url}"}
        else:
            raise HTTPException(status_code=404, detail="Circuit breaker not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker for {service_url}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset circuit breaker")

@router.get("/health")
async def gateway_health():
    """Gateway health check"""
    return {
        "status": "healthy",
        "service": "api_gateway",
        "version": "1.0.0"
    }
