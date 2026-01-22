# service-registry

Central service discovery and orchestration for the serviceCore platform.

## Overview

**Language**: Rust (Actix-web)  
**Port**: 8100  
**Status**: Production  
**Repository**: `src/serviceCore/service-registry/`

The service-registry is the heart of the serviceCore platform, managing:
- Service discovery and registration
- Health check monitoring
- Dynamic routing
- Load balancing
- Circuit breaker patterns

## Quick Start

### Running Locally

```bash
# Build
cd src/serviceCore/service-registry
cargo build --release

# Run
cargo run --release

# Or use Docker
docker run -p 8100:8100 plturrell/service-registry:latest
```

### Health Check

```bash
curl http://localhost:8100/health
```

## Architecture

### Core Components

1. **Service Registry**
   - In-memory service catalog
   - TTL-based expiration
   - Automatic cleanup

2. **Health Monitor**
   - Periodic health checks
   - Configurable intervals
   - Failure detection

3. **Load Balancer**
   - Round-robin routing
   - Least connections
   - Weighted distribution

4. **Circuit Breaker**
   - Failure thresholds
   - Automatic recovery
   - Half-open state testing

### Service Discovery Flow

```
Service Startup
    ↓
Register with service-registry
    ↓
Periodic heartbeat (health check)
    ↓
service-registry tracks status
    ↓
Other services query for available instances
    ↓
service-registry returns healthy instances
```

## API Reference

### Register Service

```bash
POST /register
Content-Type: application/json

{
  "name": "my-service",
  "host": "my-service",
  "port": 8500,
  "health_endpoint": "/health",
  "metadata": {
    "version": "1.0.0",
    "region": "us-east-1"
  }
}
```

**Response**:
```json
{
  "service_id": "uuid-here",
  "status": "registered"
}
```

### Deregister Service

```bash
DELETE /deregister/{service_id}
```

### Query Services

```bash
GET /services/{service_name}
```

**Response**:
```json
{
  "service_name": "my-service",
  "instances": [
    {
      "service_id": "uuid-1",
      "host": "my-service-1",
      "port": 8500,
      "status": "healthy",
      "last_heartbeat": "2026-01-22T23:54:00Z"
    }
  ]
}
```

### List All Services

```bash
GET /services
```

### Health Endpoint

```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "registered_services": 6,
  "healthy_services": 5
}
```

## Configuration

### Environment Variables

```bash
# Server configuration
SERVICE_REGISTRY_BIND=0.0.0.0:8100
SERVICE_REGISTRY_LOG_LEVEL=info

# Health check configuration
HEALTH_CHECK_INTERVAL=30  # seconds
HEALTH_CHECK_TIMEOUT=5    # seconds
HEALTH_CHECK_FAILURES=3   # failures before marking unhealthy

# Service TTL
SERVICE_TTL=300           # seconds

# SAP HANA Cloud
HANA_ODATA_URL=https://...
HANA_USERNAME=DBADMIN
HANA_PASSWORD=...
```

### Configuration File

`config/service_registry.json`:
```json
{
  "bind_address": "0.0.0.0:8100",
  "health_check": {
    "interval_seconds": 30,
    "timeout_seconds": 5,
    "failure_threshold": 3
  },
  "service_ttl_seconds": 300,
  "load_balancing": {
    "strategy": "round_robin",
    "enable_circuit_breaker": true
  }
}
```

## Integration

### From Your Service (Python)

```python
import requests
import time

class ServiceRegistryClient:
    def __init__(self, registry_url="http://service-registry:8100"):
        self.registry_url = registry_url
        self.service_id = None
    
    def register(self, name, host, port, health_endpoint="/health"):
        response = requests.post(
            f"{self.registry_url}/register",
            json={
                "name": name,
                "host": host,
                "port": port,
                "health_endpoint": health_endpoint
            }
        )
        self.service_id = response.json()["service_id"]
        return self.service_id
    
    def heartbeat(self):
        """Send periodic heartbeat"""
        if self.service_id:
            requests.post(
                f"{self.registry_url}/heartbeat/{self.service_id}"
            )
    
    def discover(self, service_name):
        """Discover service instances"""
        response = requests.get(
            f"{self.registry_url}/services/{service_name}"
        )
        return response.json()["instances"]
    
    def deregister(self):
        """Deregister on shutdown"""
        if self.service_id:
            requests.delete(
                f"{self.registry_url}/deregister/{self.service_id}"
            )

# Usage
registry = ServiceRegistryClient()
registry.register("my-service", "my-service", 8500)

# Keep alive with heartbeats
while True:
    registry.heartbeat()
    time.sleep(30)
```

### From Your Service (Rust)

```rust
use reqwest::Client;
use serde_json::json;

pub struct ServiceRegistry {
    client: Client,
    registry_url: String,
    service_id: Option<String>,
}

impl ServiceRegistry {
    pub fn new(registry_url: String) -> Self {
        Self {
            client: Client::new(),
            registry_url,
            service_id: None,
        }
    }
    
    pub async fn register(
        &mut self,
        name: &str,
        host: &str,
        port: u16,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let response = self.client
            .post(&format!("{}/register", self.registry_url))
            .json(&json!({
                "name": name,
                "host": host,
                "port": port,
                "health_endpoint": "/health"
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;
        
        let service_id = response["service_id"]
            .as_str()
            .unwrap()
            .to_string();
        
        self.service_id = Some(service_id.clone());
        Ok(service_id)
    }
}
```

## Monitoring

### Metrics

The service-registry exposes metrics via `/metrics` endpoint:

- `registry_services_total` - Total registered services
- `registry_services_healthy` - Healthy services count
- `registry_requests_total` - Total API requests
- `registry_health_checks_total` - Health checks performed
- `registry_health_check_failures_total` - Failed health checks

### Logging

All logs are sent to SAP HANA Cloud in structured JSON format:

```json
{
  "timestamp": "2026-01-22T23:54:00Z",
  "level": "info",
  "service": "service-registry",
  "message": "Service registered",
  "service_id": "uuid-here",
  "service_name": "my-service"
}
```

## Troubleshooting

### Service Won't Register

**Problem**: Registration fails  
**Solution**:
- Check service-registry is running
- Verify network connectivity
- Check service-registry logs
- Verify JSON payload format

### Services Marked Unhealthy

**Problem**: Healthy services show as unhealthy  
**Solution**:
- Check service health endpoint responds within timeout
- Verify health endpoint returns 200 status
- Check network latency
- Increase health check timeout if needed

### Service Discovery Returns Empty

**Problem**: No instances found  
**Solution**:
- Verify services are registered
- Check service names match exactly
- Verify services are passing health checks
- Check service TTL hasn't expired

## Development

### Building from Source

```bash
cd src/serviceCore/service-registry
cargo build
cargo test
cargo run
```

### Running Tests

```bash
cargo test --all-features
```

### Docker Build

```bash
docker build -f docker/Dockerfile.service-registry -t service-registry:dev .
```

## Production Deployment

### Docker Compose

```yaml
service-registry:
  image: plturrell/service-registry:latest
  ports:
    - "8100:8100"
  environment:
    - SERVICE_REGISTRY_BIND=0.0.0.0:8100
    - HANA_ODATA_URL=${HANA_ODATA_URL}
    - HANA_USERNAME=${HANA_USERNAME}
    - HANA_PASSWORD=${HANA_PASSWORD}
  volumes:
    - ./config/service_registry.json:/app/config/service_registry.json:ro
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8100/health"]
    interval: 30s
    timeout: 5s
    retries: 3
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: service-registry
spec:
  replicas: 3
  selector:
    matchLabels:
      app: service-registry
  template:
    metadata:
      labels:
        app: service-registry
    spec:
      containers:
      - name: service-registry
        image: plturrell/service-registry:latest
        ports:
        - containerPort: 8100
        env:
        - name: SERVICE_REGISTRY_BIND
          value: "0.0.0.0:8100"
        livenessProbe:
          httpGet:
            path: /health
            port: 8100
          initialDelaySeconds: 10
          periodSeconds: 30
```

## Related Documentation

- [Architecture Overview](../../01-architecture/)
- [nWebServe (API Gateway)](../nWebServe/)
- [Operations Runbook](../../04-operations/OPERATOR_RUNBOOK.md)

---

**Language**: Rust  
**Status**: Production  
**Port**: 8100  
**Last Updated**: January 22, 2026
