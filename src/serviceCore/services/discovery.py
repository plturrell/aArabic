"""
Service Discovery for Vendor Components
Implements service discovery and health monitoring for all vendor integrations
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    name: str
    url: str
    health_endpoint: str
    version: Optional[str] = None
    status: str = "unknown"  # unknown, healthy, unhealthy, unreachable
    last_check: Optional[datetime] = None
    response_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceRegistry:
    """Service registry containing all discovered services"""
    services: Dict[str, ServiceEndpoint] = field(default_factory=dict)
    last_updated: Optional[datetime] = None

class ServiceDiscovery:
    """Service discovery and health monitoring"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.registry = ServiceRegistry()
        self.config_path = config_path or "backend/config/services.json"
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self.session: Optional[aiohttp.ClientSession] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Default service configurations
        self.default_services = {
            "shimmy": {
                "name": "Shimmy AI",
                "url": "http://localhost:3001",
                "health_endpoint": "/health",
                "metadata": {
                    "type": "workflow_engine",
                    "vendor": "shimmy-ai",
                    "capabilities": ["workflow_execution", "model_management", "streaming"]
                }
            },
            "memgraph": {
                "name": "Memgraph",
                "url": "bolt://localhost:7687",
                "health_endpoint": "bolt://localhost:7687",  # Special handling for Bolt protocol
                "metadata": {
                    "type": "graph_database",
                    "vendor": "memgraph",
                    "capabilities": ["graph_storage", "cypher_queries", "analytics"]
                }
            },
            "a2ui": {
                "name": "A2UI Service",
                "url": "http://localhost:8000",  # Same as main backend
                "health_endpoint": "/api/a2ui/health",
                "metadata": {
                    "type": "ui_generator",
                    "vendor": "google",
                    "capabilities": ["dynamic_ui", "component_rendering", "data_binding"]
                }
            },
            "toolorchestra": {
                "name": "ToolOrchestra",
                "url": "http://localhost:8000",  # Integrated with main backend
                "health_endpoint": "/api/orchestration/health",
                "metadata": {
                    "type": "rl_orchestrator",
                    "vendor": "nvidia",
                    "capabilities": ["reinforcement_learning", "tool_optimization", "strategy_selection"]
                }
            },
            "marquez": {
                "name": "Marquez Lineage Service",
                "url": "http://localhost:9000",
                "health_endpoint": "/healthcheck",
                "metadata": {
                    "type": "lineage_service",
                    "vendor": "marquez",
                    "capabilities": [
                        "dataset_catalog",
                        "job_run_tracking",
                        "openlineage_api",
                        "graph_visualization"
                    ],
                    "admin_endpoint": "http://localhost:9100",
                    "web_ui": "http://localhost:39000",
                    "api_docs": "http://localhost:9000/api/v1/openapi.json"
                }
            }
        }
    
    async def initialize(self):
        """Initialize service discovery"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
        )
        
        # Load service configurations
        await self.load_service_config()
        
        # Perform initial discovery
        await self.discover_services()
        
        # Start health monitoring
        self.start_health_monitoring()
        
        logger.info("Service discovery initialized")
    
    async def shutdown(self):
        """Shutdown service discovery"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("Service discovery shutdown")
    
    async def load_service_config(self):
        """Load service configuration from file"""
        registry_url = os.getenv("SERVICE_REGISTRY_URL")
        if registry_url:
            loaded = await self.load_registry_services(registry_url)
            if loaded:
                return
            logger.warning("Falling back to local config after registry load failure")

        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                # Merge with default services
                services_config = {**self.default_services, **config.get('services', {})}
                
                for service_id, service_config in services_config.items():
                    endpoint = ServiceEndpoint(
                        name=service_config['name'],
                        url=service_config['url'],
                        health_endpoint=service_config['health_endpoint'],
                        version=service_config.get('version'),
                        metadata=service_config.get('metadata', {})
                    )
                    self.registry.services[service_id] = endpoint
                    
                logger.info(f"Loaded {len(services_config)} services from config")
                
            except Exception as e:
                logger.error(f"Failed to load service config: {e}")
                # Fall back to default services
                await self.load_default_services()
        else:
            logger.info("No service config file found, using defaults")
            await self.load_default_services()

    async def load_registry_services(self, registry_url: str) -> bool:
        """Load service configuration from the Rust service registry"""
        if not self.session:
            return False

        try:
            registry_url = registry_url.rstrip("/")
            async with self.session.get(f"{registry_url}/services") as response:
                if response.status != 200:
                    logger.warning(f"Registry returned HTTP {response.status}")
                    return False

                services = await response.json()
                if not isinstance(services, list):
                    logger.warning("Registry response is not a list")
                    return False

                self.registry.services.clear()
                for service in services:
                    service_id = service.get("id") or service.get("name")
                    if not service_id:
                        continue

                    upstream_url = service.get("upstream_url") or ""
                    health_path = service.get("health_path") or ""
                    if upstream_url.startswith("http") and not health_path:
                        health_path = "/health"

                    metadata = {
                        "layer": service.get("layer"),
                        "kind": service.get("kind"),
                        "type": service.get("kind"),
                        "tags": service.get("tags") or [],
                        "dependencies": service.get("dependencies") or [],
                        "registry": True,
                    }

                    endpoint = ServiceEndpoint(
                        name=service.get("name", service_id),
                        url=upstream_url,
                        health_endpoint=health_path,
                        version=service.get("version"),
                        metadata=metadata,
                    )
                    self.registry.services[service_id] = endpoint

                self.registry.last_updated = datetime.now()
                logger.info(f"Loaded {len(self.registry.services)} services from registry")
                return True

        except Exception as e:
            logger.warning(f"Failed to load registry services: {e}")
            return False
    
    async def load_default_services(self):
        """Load default service configurations"""
        for service_id, service_config in self.default_services.items():
            endpoint = ServiceEndpoint(
                name=service_config['name'],
                url=service_config['url'],
                health_endpoint=service_config['health_endpoint'],
                metadata=service_config.get('metadata', {})
            )
            self.registry.services[service_id] = endpoint
    
    async def discover_services(self):
        """Discover and register services"""
        logger.info("Starting service discovery...")
        
        # Check all registered services
        tasks = []
        for service_id, endpoint in self.registry.services.items():
            task = asyncio.create_task(self.check_service_health(service_id, endpoint))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.registry.last_updated = datetime.now()
        logger.info(f"Service discovery completed. Found {len(self.get_healthy_services())} healthy services")
    
    async def check_service_health(self, service_id: str, endpoint: ServiceEndpoint) -> bool:
        """Check health of a specific service"""
        start_time = datetime.now()
        
        try:
            if not endpoint.url:
                endpoint.status = "unknown"
                endpoint.last_check = datetime.now()
                endpoint.response_time = None
                return False

            if endpoint.url.startswith('bolt://'):
                # Special handling for Memgraph Bolt protocol
                health_status = await self.check_memgraph_health(endpoint)
            elif endpoint.url.startswith(('redis://', 'rediss://')):
                health_status = await self.check_redis_health(endpoint)
            else:
                # HTTP health check
                health_status = await self.check_http_health(endpoint)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            endpoint.status = "healthy" if health_status else "unhealthy"
            endpoint.last_check = datetime.now()
            endpoint.response_time = response_time
            
            logger.debug(f"Service {service_id} health check: {endpoint.status} ({response_time:.3f}s)")
            return health_status
            
        except Exception as e:
            endpoint.status = "unreachable"
            endpoint.last_check = datetime.now()
            endpoint.response_time = None
            logger.warning(f"Service {service_id} health check failed: {e}")
            return False

    async def check_http_health(self, endpoint: ServiceEndpoint) -> bool:
        """Check HTTP service health"""
        try:
            if not endpoint.url:
                return False

            health_endpoint = endpoint.health_endpoint or ""
            if health_endpoint.startswith(("http://", "https://")):
                health_url = health_endpoint
            else:
                health_url = f"{endpoint.url.rstrip('/')}{health_endpoint}"

            async with self.session.get(health_url) as response:
                if response.status == 200:
                    # Try to parse response for additional info
                    try:
                        data = await response.json()
                        if isinstance(data, dict):
                            endpoint.version = data.get('version')
                            endpoint.metadata.update(data.get('metadata', {}))
                    except:
                        pass  # Ignore JSON parsing errors

                    return True
                else:
                    logger.warning(f"Health check failed for {endpoint.name}: HTTP {response.status}")
                    return False

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for {endpoint.name}")
            return False
        except Exception as e:
            logger.warning(f"Health check error for {endpoint.name}: {e}")
            return False

    async def check_memgraph_health(self, endpoint: ServiceEndpoint) -> bool:
        """Check Memgraph health using Bolt protocol"""
        try:
            # For now, we'll use a simple TCP connection check
            # In production, you'd use the neo4j driver
            import socket

            # Parse bolt://localhost:7687
            url_parts = endpoint.url.replace('bolt://', '').split(':')
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 7687

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.health_check_timeout)

            result = sock.connect_ex((host, port))
            sock.close()

            return result == 0

        except Exception as e:
            logger.warning(f"Memgraph health check error: {e}")
            return False

    async def check_redis_health(self, endpoint: ServiceEndpoint) -> bool:
        """Check Redis-compatible health via TCP connection"""
        try:
            import socket

            url = endpoint.url.replace('redis://', '').replace('rediss://', '')
            url = url.split('@')[-1]
            host_port = url.split('/')[0]
            parts = host_port.split(':')
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 6379

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.health_check_timeout)

            result = sock.connect_ex((host, port))
            sock.close()

            return result == 0

        except Exception as e:
            logger.warning(f"Redis health check error: {e}")
            return False

    def start_health_monitoring(self):
        """Start periodic health monitoring"""
        if self._health_check_task:
            self._health_check_task.cancel()

        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Health monitoring started")

    async def _health_monitor_loop(self):
        """Health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.discover_services()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    def get_service(self, service_id: str) -> Optional[ServiceEndpoint]:
        """Get service endpoint by ID"""
        return self.registry.services.get(service_id)

    def get_healthy_services(self) -> List[ServiceEndpoint]:
        """Get all healthy services"""
        return [
            endpoint for endpoint in self.registry.services.values()
            if endpoint.status == "healthy"
        ]

    def get_services_by_type(self, service_type: str) -> List[ServiceEndpoint]:
        """Get services by type"""
        return [
            endpoint for endpoint in self.registry.services.values()
            if endpoint.metadata.get('type') == service_type
        ]

    def get_service_url(self, service_id: str) -> Optional[str]:
        """Get service URL by ID"""
        endpoint = self.get_service(service_id)
        return endpoint.url if endpoint and endpoint.status == "healthy" else None

    def is_service_healthy(self, service_id: str) -> bool:
        """Check if service is healthy"""
        endpoint = self.get_service(service_id)
        return endpoint is not None and endpoint.status == "healthy"

    def get_registry_status(self) -> Dict[str, Any]:
        """Get complete registry status"""
        healthy_count = len(self.get_healthy_services())
        total_count = len(self.registry.services)

        services_status = {}
        for service_id, endpoint in self.registry.services.items():
            services_status[service_id] = {
                'name': endpoint.name,
                'url': endpoint.url,
                'status': endpoint.status,
                'last_check': endpoint.last_check.isoformat() if endpoint.last_check else None,
                'response_time': endpoint.response_time,
                'version': endpoint.version,
                'metadata': endpoint.metadata
            }

        return {
            'healthy_services': healthy_count,
            'total_services': total_count,
            'last_updated': self.registry.last_updated.isoformat() if self.registry.last_updated else None,
            'services': services_status
        }

    async def register_service(self, service_id: str, name: str, url: str,
                             health_endpoint: str, metadata: Optional[Dict] = None):
        """Dynamically register a new service"""
        endpoint = ServiceEndpoint(
            name=name,
            url=url,
            health_endpoint=health_endpoint,
            metadata=metadata or {}
        )

        self.registry.services[service_id] = endpoint

        # Immediately check health
        await self.check_service_health(service_id, endpoint)

        logger.info(f"Registered new service: {service_id} ({name})")

    async def unregister_service(self, service_id: str):
        """Unregister a service"""
        if service_id in self.registry.services:
            del self.registry.services[service_id]
            logger.info(f"Unregistered service: {service_id}")

    async def save_config(self):
        """Save current service configuration to file"""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)

            services_config = {}
            for service_id, endpoint in self.registry.services.items():
                services_config[service_id] = {
                    'name': endpoint.name,
                    'url': endpoint.url,
                    'health_endpoint': endpoint.health_endpoint,
                    'version': endpoint.version,
                    'metadata': endpoint.metadata
                }

            config = {
                'services': services_config,
                'last_updated': datetime.now().isoformat()
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Service configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save service config: {e}")

# Global service discovery instance
service_discovery = ServiceDiscovery()

async def get_service_discovery() -> ServiceDiscovery:
    """Get the global service discovery instance"""
    return service_discovery

async def initialize_service_discovery():
    """Initialize the global service discovery"""
    await service_discovery.initialize()

async def shutdown_service_discovery():
    """Shutdown the global service discovery"""
    await service_discovery.shutdown()
