# HyperShimmy Deployment Guide

**Version**: 1.0.0  
**Last Updated**: January 16, 2026

## Deployment Options

1. **Local Development** - Single machine
2. **Docker** - Containerized deployment
3. **Kubernetes** - Production orchestration
4. **Cloud** - Managed services

## Local Deployment

### Prerequisites

- Zig 0.12.0+
- Mojo runtime
- 4GB RAM minimum
- 10GB disk space

### Steps

```bash
# 1. Build release binary
zig build -Doptimize=ReleaseFast

# 2. Set environment variables
export PORT=8080
export LOG_LEVEL=INFO

# 3. Run server
./zig-out/bin/hypershimmy
```

### Configuration

Create `.env` file:

```bash
PORT=8080
HOST=0.0.0.0
LOG_LEVEL=INFO
QDRANT_URL=http://localhost:6333
SHIMMY_URL=http://localhost:8000
```

## Docker Deployment

### Build Image

```dockerfile
# Dockerfile
FROM zigimg/zig:0.12.0 as builder
WORKDIR /app
COPY . .
RUN zig build -Doptimize=ReleaseFast

FROM ubuntu:22.04
COPY --from=builder /app/zig-out/bin/hypershimmy /usr/local/bin/
EXPOSE 8080
CMD ["hypershimmy"]
```

```bash
# Build
docker build -t hypershimmy:1.0.0 .

# Run
docker run -p 8080:8080 hypershimmy:1.0.0
```

### Docker Compose

```yaml
version: '3.8'
services:
  hypershimmy:
    build: .
    ports:
      - "8080:8080"
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
  
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

```bash
docker-compose up -d
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypershimmy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypershimmy
  template:
    metadata:
      labels:
        app: hypershimmy
    spec:
      containers:
      - name: hypershimmy
        image: hypershimmy:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: hypershimmy
spec:
  selector:
    app: hypershimmy
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

```bash
kubectl apply -f k8s/deployment.yaml
```

## Production Checklist

### Security

- [ ] Enable HTTPS/TLS
- [ ] Configure authentication
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Secure API keys
- [ ] Configure CORS properly

### Monitoring

- [ ] Set up health checks
- [ ] Configure logging
- [ ] Enable metrics collection
- [ ] Set up alerts
- [ ] Monitor resource usage

### Performance

- [ ] Enable caching
- [ ] Configure load balancer
- [ ] Set up CDN for static assets
- [ ] Optimize database queries
- [ ] Enable connection pooling

### Backup & Recovery

- [ ] Database backups
- [ ] Configuration backups
- [ ] Disaster recovery plan
- [ ] Backup testing

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8080 | HTTP port |
| HOST | 0.0.0.0 | Bind address |
| LOG_LEVEL | INFO | Logging level |
| QDRANT_URL | http://localhost:6333 | Vector DB URL |
| SHIMMY_URL | http://localhost:8000 | LLM service URL |
| MAX_UPLOAD_SIZE | 10485760 | Max file size (bytes) |
| RATE_LIMIT | 100 | Requests per minute |

## Scaling

### Horizontal Scaling

```bash
# Kubernetes
kubectl scale deployment hypershimmy --replicas=5

# Docker Swarm
docker service scale hypershimmy=5
```

### Vertical Scaling

Increase container resources:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "2000m"
  limits:
    memory: "4Gi"
    cpu: "4000m"
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600
}
```

### Metrics

Prometheus endpoint:
```bash
curl http://localhost:8080/metrics
```

## Troubleshooting

### Issue: Server won't start

Check logs:
```bash
docker logs hypershimmy
kubectl logs deployment/hypershimmy
```

### Issue: High memory usage

Monitor with:
```bash
docker stats hypershimmy
kubectl top pods
```

### Issue: Slow responses

Check metrics and scale up if needed.

## Support

- Documentation: [docs.hypershimmy.dev](https://docs.hypershimmy.dev)
- Issues: [github.com/hypershimmy/issues](https://github.com/hypershimmy/issues)

---

**For API reference**, see [API.md](API.md)  
**For architecture**, see [ARCHITECTURE.md](ARCHITECTURE.md)  
**For development**, see [DEVELOPER.md](DEVELOPER.md)
