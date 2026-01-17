"""
API Gateway for Vendor Service Routing
Provides intelligent routing, load balancing, and request forwarding to vendor services
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from urllib.parse import urljoin, urlparse
import random

from backend.services.discovery import get_service_discovery, ServiceDiscovery

logger = logging.getLogger(__name__)

@dataclass
class RouteConfig:
    """Route configuration for API gateway"""
    path_prefix: str
    service_type: str
    service_id: Optional[str] = None
    strip_prefix: bool = True
    timeout: int = 30
    retries: int = 3
    load_balance: bool = True

class APIGateway:
    """API Gateway for vendor service routing"""
    
    def __init__(self):
        self.routes: Dict[str, RouteConfig] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.discovery: Optional[ServiceDiscovery] = None
        
        # Load balancing state
        self.service_counters: Dict[str, int] = {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict] = {}
        self.circuit_breaker_threshold = 5  # failures
        self.circuit_breaker_timeout = 60  # seconds
        
        # Default routes
        self.setup_default_routes()
    
    def setup_default_routes(self):
        """Setup default routing configuration"""
        self.routes = {
            "/api/shimmy": RouteConfig(
                path_prefix="/api/shimmy",
                service_type="workflow_engine",
                service_id="shimmy",
                strip_prefix=True,
                timeout=60  # Longer timeout for workflow execution
            ),
            "/api/memgraph": RouteConfig(
                path_prefix="/api/memgraph",
                service_type="graph_database",
                service_id="memgraph",
                strip_prefix=True,
                timeout=30
            ),
            "/api/marquez": RouteConfig(
                path_prefix="/api/marquez",
                service_type="lineage_service",
                service_id="marquez",
                strip_prefix=True,
                timeout=30
            ),
            "/api/toolorchestra": RouteConfig(
                path_prefix="/api/toolorchestra",
                service_type="rl_orchestrator",
                service_id="toolorchestra",
                strip_prefix=True,
                timeout=45
            ),
            "/api/opencanvas": RouteConfig(
                path_prefix="/api/opencanvas",
                service_type="ui_framework",
                service_id="open_canvas",
                strip_prefix=True,
                timeout=30
            )
        }
    
    async def initialize(self):
        """Initialize the API gateway"""
        self.session = aiohttp.ClientSession()
        self.discovery = await get_service_discovery()
        logger.info("API Gateway initialized")
    
    async def shutdown(self):
        """Shutdown the API gateway"""
        if self.session:
            await self.session.close()
        logger.info("API Gateway shutdown")
    
    def get_route_config(self, path: str) -> Optional[RouteConfig]:
        """Get route configuration for a path"""
        # Find the longest matching prefix
        matching_routes = [
            (prefix, config) for prefix, config in self.routes.items()
            if path.startswith(prefix)
        ]
        
        if matching_routes:
            # Sort by prefix length (longest first)
            matching_routes.sort(key=lambda x: len(x[0]), reverse=True)
            return matching_routes[0][1]
        
        return None
    
    def select_service_endpoint(self, route_config: RouteConfig) -> Optional[str]:
        """Select service endpoint using load balancing"""
        if not self.discovery:
            return None
        
        # Get available services
        if route_config.service_id:
            # Specific service requested
            endpoint = self.discovery.get_service(route_config.service_id)
            if endpoint and endpoint.status == "healthy":
                return endpoint.url
        else:
            # Load balance across services of the same type
            services = self.discovery.get_services_by_type(route_config.service_type)
            healthy_services = [s for s in services if s.status == "healthy"]
            
            if healthy_services:
                if route_config.load_balance and len(healthy_services) > 1:
                    # Round-robin load balancing
                    service_key = route_config.service_type
                    counter = self.service_counters.get(service_key, 0)
                    selected_service = healthy_services[counter % len(healthy_services)]
                    self.service_counters[service_key] = counter + 1
                    return selected_service.url
                else:
                    # Use first healthy service
                    return healthy_services[0].url
        
        return None
    
    def is_circuit_breaker_open(self, service_url: str) -> bool:
        """Check if circuit breaker is open for a service"""
        breaker = self.circuit_breakers.get(service_url)
        if not breaker:
            return False
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if datetime.now() - breaker['last_failure'] > timedelta(seconds=self.circuit_breaker_timeout):
                # Reset to half-open
                breaker['state'] = 'half-open'
                breaker['failure_count'] = 0
                return False
            return True
        
        return False
    
    def record_success(self, service_url: str):
        """Record successful request"""
        if service_url in self.circuit_breakers:
            breaker = self.circuit_breakers[service_url]
            if breaker['state'] == 'half-open':
                # Reset circuit breaker
                breaker['state'] = 'closed'
                breaker['failure_count'] = 0
    
    def record_failure(self, service_url: str):
        """Record failed request"""
        if service_url not in self.circuit_breakers:
            self.circuit_breakers[service_url] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': None
            }
        
        breaker = self.circuit_breakers[service_url]
        breaker['failure_count'] += 1
        breaker['last_failure'] = datetime.now()
        
        if breaker['failure_count'] >= self.circuit_breaker_threshold:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {service_url}")
    
    def build_target_url(self, route_config: RouteConfig, original_path: str, service_url: str) -> str:
        """Build target URL for the service"""
        if route_config.strip_prefix:
            # Remove the prefix from the path
            target_path = original_path[len(route_config.path_prefix):]
            if not target_path.startswith('/'):
                target_path = '/' + target_path
        else:
            target_path = original_path
        
        return urljoin(service_url.rstrip('/'), target_path.lstrip('/'))
    
    async def forward_request(
        self, 
        method: str, 
        path: str, 
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        json_data: Any = None,
        data: bytes = None
    ) -> Tuple[int, Dict[str, str], Any]:
        """Forward request to appropriate service"""
        
        # Get route configuration
        route_config = self.get_route_config(path)
        if not route_config:
            return 404, {}, {"error": "Route not found"}
        
        # Select service endpoint
        service_url = self.select_service_endpoint(route_config)
        if not service_url:
            return 503, {}, {"error": "Service unavailable"}
        
        # Check circuit breaker
        if self.is_circuit_breaker_open(service_url):
            return 503, {}, {"error": "Service temporarily unavailable (circuit breaker open)"}
        
        # Build target URL
        target_url = self.build_target_url(route_config, path, service_url)
        
        # Prepare request
        request_headers = headers or {}
        request_headers.pop('host', None)  # Remove host header
        
        # Add gateway headers
        request_headers['X-Forwarded-By'] = 'AI-Nucleus-Gateway'
        request_headers['X-Gateway-Route'] = route_config.path_prefix
        
        # Retry logic
        last_exception = None
        for attempt in range(route_config.retries):
            try:
                timeout = aiohttp.ClientTimeout(total=route_config.timeout)
                
                async with self.session.request(
                    method=method,
                    url=target_url,
                    headers=request_headers,
                    params=params,
                    json=json_data,
                    data=data,
                    timeout=timeout
                ) as response:
                    
                    # Read response
                    response_headers = dict(response.headers)
                    
                    try:
                        if response.content_type == 'application/json':
                            response_data = await response.json()
                        else:
                            response_data = await response.text()
                    except:
                        response_data = await response.read()
                    
                    # Record success
                    self.record_success(service_url)
                    
                    logger.debug(f"Gateway forwarded {method} {path} -> {target_url} ({response.status})")
                    
                    return response.status, response_headers, response_data
                    
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Request timeout for {target_url} (attempt {attempt + 1})")
                self.record_failure(service_url)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed for {target_url} (attempt {attempt + 1}): {e}")
                self.record_failure(service_url)
                
                # Don't retry on client errors
                if hasattr(e, 'status') and 400 <= e.status < 500:
                    break
        
        # All retries failed
        logger.error(f"All retries failed for {target_url}: {last_exception}")
        return 502, {}, {"error": "Bad gateway - service request failed"}

# Global API gateway instance
api_gateway = APIGateway()

async def get_api_gateway() -> APIGateway:
    """Get the global API gateway instance"""
    return api_gateway

async def initialize_api_gateway():
    """Initialize the global API gateway"""
    await api_gateway.initialize()

async def shutdown_api_gateway():
    """Shutdown the global API gateway"""
    await api_gateway.shutdown()
