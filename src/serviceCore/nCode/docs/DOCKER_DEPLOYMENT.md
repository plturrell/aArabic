# Docker Deployment Guide - nCode

**Complete guide for deploying nCode with Docker Compose**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Backup & Restore](#backup--restore)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)

---

## Quick Start

Get nCode running with Docker in under 5 minutes:

```bash
# 1. Clone and navigate to nCode directory
cd src/serviceCore/nCode

# 2. Copy environment template
cp .env.example .env

# 3. Start all services
docker-compose up -d

# 4. Check status
docker-compose ps

# 5. Access services
# nCode API: http://localhost:18003
# Qdrant: http://localhost:6333
# Memgraph Lab: http://localhost:3000
# Marquez UI: http://localhost:3001
```

---

## Architecture

### Service Overview

The nCode Docker Compose stack includes:

```
┌─────────────────────────────────────────────────────────┐
│                    nCode Platform                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  nCode   │  │  Qdrant  │  │ Memgraph │  │Marquez │ │
│  │  Server  │  │  Vector  │  │  Graph   │  │Lineage │ │
│  │   API    │  │   DB     │  │    DB    │  │   DB   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │             │              │             │      │
│       └─────────────┴──────────────┴─────────────┘      │
│                    nCode Network                         │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         PostgreSQL (Marquez Backend)           │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Services

| Service | Purpose | Port | Dependencies |
|---------|---------|------|--------------|
| **ncode** | Code intelligence API | 18003 | Qdrant, Memgraph, Marquez |
| **qdrant** | Vector search database | 6333, 6334 | None |
| **memgraph** | Graph database | 7687, 3000 | None |
| **marquez** | Lineage tracking | 5000, 3001 | PostgreSQL |
| **marquez-db** | PostgreSQL for Marquez | 5432 | None |

---

## Prerequisites

### Required Software

- **Docker:** Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose:** Version 2.0+ ([Install Compose](https://docs.docker.com/compose/install/))
- **Minimum System Requirements:**
  - CPU: 4 cores
  - RAM: 8 GB
  - Disk: 20 GB free space

### Verify Installation

```bash
# Check Docker
docker --version
# Expected: Docker version 20.10.0 or higher

# Check Docker Compose
docker-compose --version
# Expected: Docker Compose version 2.0.0 or higher

# Check Docker is running
docker ps
# Should show no errors
```

---

## Installation

### Step 1: Prepare Environment

```bash
# Navigate to nCode directory
cd src/serviceCore/nCode

# Copy environment template
cp .env.example .env

# Edit environment variables (optional)
nano .env
```

### Step 2: Build Images

```bash
# Build nCode image
docker-compose build

# Or build without cache
docker-compose build --no-cache
```

### Step 3: Start Services

```bash
# Start all services in background
docker-compose up -d

# Or start with logs
docker-compose up

# Start specific service
docker-compose up -d ncode
```

### Step 4: Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Should show all services as "Up"
# NAME                STATUS              PORTS
# ncode-server        Up (healthy)        0.0.0.0:18003->18003/tcp
# ncode-qdrant        Up (healthy)        0.0.0.0:6333->6333/tcp
# ncode-memgraph      Up (healthy)        0.0.0.0:7687->7687/tcp
# ncode-marquez       Up (healthy)        0.0.0.0:5000->5000/tcp
# ncode-marquez-db    Up (healthy)        0.0.0.0:5432->5432/tcp

# Test nCode API
curl http://localhost:18003/health
# Expected: {"status":"ok","version":"2.0.0"}
```

---

## Configuration

### Environment Variables

Edit `.env` to customize your deployment:

#### Core Settings

```bash
# Server Configuration
NCODE_PORT=18003              # nCode API port
LOG_LEVEL=INFO                # Logging: DEBUG, INFO, WARN, ERROR

# Database Ports
QDRANT_HTTP_PORT=6333         # Qdrant HTTP API
QDRANT_GRPC_PORT=6334         # Qdrant gRPC
MEMGRAPH_BOLT_PORT=7687       # Memgraph Bolt protocol
MEMGRAPH_WEB_PORT=3000        # Memgraph Lab UI
MARQUEZ_PORT=5000             # Marquez API
MARQUEZ_WEB_PORT=3001         # Marquez Web UI
```

#### Security (Production)

```bash
# Change default passwords
MARQUEZ_DB_PASSWORD=your-secure-password

# Enable API authentication (if implemented)
API_KEY=your-api-key

# Configure CORS
CORS_ORIGINS=https://yourdomain.com
```

#### Resource Limits

Add to `docker-compose.yml` under each service:

```yaml
services:
  ncode:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Volumes

Persistent data is stored in named Docker volumes:

```bash
# List volumes
docker volume ls | grep ncode

# Inspect volume
docker volume inspect ncode-qdrant-data

# Backup volume
docker run --rm -v ncode-qdrant-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/qdrant-backup.tar.gz -C /data .
```

---

## Usage

### Common Operations

#### Start Services

```bash
# Start all services
docker-compose up -d

# Start and follow logs
docker-compose up

# Start specific service
docker-compose up -d qdrant
```

#### Stop Services

```bash
# Stop all services (preserves data)
docker-compose stop

# Stop and remove containers (preserves data)
docker-compose down

# Stop and remove volumes (DELETES DATA!)
docker-compose down -v
```

#### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ncode

# Last 100 lines
docker-compose logs --tail=100 ncode

# Since timestamp
docker-compose logs --since 2026-01-18T06:00:00 ncode
```

#### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart ncode

# Reload after config change
docker-compose up -d --force-recreate
```

### Service Management

#### Scale Services

```bash
# Scale nCode instances (if load balancing configured)
docker-compose up -d --scale ncode=3
```

#### Update Services

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build

# Update specific service
docker-compose up -d --build ncode
```

#### Execute Commands

```bash
# Shell into container
docker-compose exec ncode sh

# Run command in container
docker-compose exec ncode ls -la /app

# Run as specific user
docker-compose exec -u root ncode sh
```

---

## Backup & Restore

### Automated Backup

Use the provided backup script:

```bash
# Create backup
./scripts/docker-backup.sh

# List backups
./scripts/docker-backup.sh --list

# Restore latest backup
./scripts/docker-backup.sh --restore latest

# Restore specific backup
./scripts/docker-backup.sh --restore ncode_backup_20260118_064500.tar.gz
```

### Manual Backup

#### Backup All Volumes

```bash
# Create backup directory
mkdir -p backups

# Backup Qdrant
docker run --rm -v ncode-qdrant-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant.tar.gz -C /data .

# Backup Memgraph
docker run --rm -v ncode-memgraph-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/memgraph.tar.gz -C /data .

# Backup Marquez DB
docker run --rm -v ncode-marquez-db-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/marquez-db.tar.gz -C /data .
```

#### Restore Volumes

```bash
# Restore Qdrant
docker run --rm -v ncode-qdrant-data:/data -v $(pwd)/backups:/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/qdrant.tar.gz -C /data"

# Restart services
docker-compose restart
```

### Backup Configuration

```bash
# Backup docker-compose.yml and .env
tar czf ncode-config-$(date +%Y%m%d).tar.gz docker-compose.yml .env

# Restore
tar xzf ncode-config-20260118.tar.gz
```

---

## Monitoring

### Health Checks

All services include health checks:

```bash
# Check health status
docker-compose ps

# View health check details
docker inspect ncode-server --format='{{.State.Health.Status}}'

# Check all services
for service in ncode-server ncode-qdrant ncode-memgraph ncode-marquez; do
  echo -n "$service: "
  docker inspect $service --format='{{.State.Health.Status}}'
done
```

### Metrics

#### nCode Metrics

```bash
# Prometheus metrics
curl http://localhost:18003/metrics

# JSON metrics
curl http://localhost:18003/metrics.json
```

#### Resource Usage

```bash
# Container stats
docker stats ncode-server

# All containers
docker-compose stats

# Disk usage
docker system df

# Detailed usage
docker system df -v
```

### Logs

#### Structured Logging

```bash
# nCode logs (JSON format)
docker-compose logs ncode | jq .

# Filter by level
docker-compose logs ncode | jq 'select(.level=="ERROR")'

# Last hour of errors
docker-compose logs --since 1h ncode | jq 'select(.level=="ERROR")'
```

#### Export Logs

```bash
# Export to file
docker-compose logs ncode > ncode-logs.txt

# Export with timestamps
docker-compose logs -t ncode > ncode-logs-$(date +%Y%m%d).txt
```

---

## Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs ncode

# Check system resources
docker system info

# Clean and restart
docker-compose down
docker system prune -f
docker-compose up -d
```

#### Port Conflicts

```bash
# Check what's using port
lsof -i :18003  # macOS/Linux
netstat -ano | findstr :18003  # Windows

# Change port in .env
NCODE_PORT=18004

# Restart
docker-compose down
docker-compose up -d
```

#### Connection Issues

```bash
# Test from host
curl http://localhost:18003/health

# Test from within network
docker-compose exec ncode curl http://qdrant:6333/health

# Check network
docker network inspect ncode-network

# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

#### Performance Issues

```bash
# Check resources
docker stats

# Increase limits in docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G

# Clean unused data
docker system prune -a --volumes
```

### Debug Mode

Enable debug logging:

```bash
# Update .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart ncode

# View debug logs
docker-compose logs -f ncode
```

### Reset Everything

```bash
# Complete reset (WARNING: DELETES ALL DATA)
./scripts/docker-cleanup.sh --reset

# Or manually
docker-compose down -v
docker system prune -a --volumes
docker-compose up -d
```

---

## Production Deployment

### Security Hardening

#### 1. Change Default Passwords

```bash
# Generate secure password
openssl rand -base64 32

# Update .env
MARQUEZ_DB_PASSWORD=your-generated-password
```

#### 2. Enable TLS

Add to `docker-compose.yml`:

```yaml
services:
  ncode:
    environment:
      - ENABLE_HTTPS=true
      - SSL_CERT_PATH=/certs/cert.pem
      - SSL_KEY_PATH=/certs/key.pem
    volumes:
      - ./certs:/certs:ro
```

#### 3. Network Isolation

```yaml
services:
  marquez-db:
    networks:
      - backend  # Don't expose to public network
networks:
  ncode-network:
    driver: bridge
  backend:
    driver: bridge
    internal: true
```

#### 4. Resource Limits

```yaml
services:
  ncode:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
      restart_policy:
        condition: on-failure
        max_attempts: 3
```

### High Availability

#### Multiple nCode Instances

```yaml
services:
  ncode:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

#### Load Balancing

Use nginx as reverse proxy:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - ncode
```

### Monitoring Setup

#### Prometheus Integration

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
```

#### Grafana Dashboard

```yaml
services:
  grafana:
    image: grafana/grafana
    ports:
      - "3002:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
```

### Backup Automation

Add to crontab:

```bash
# Daily backup at 2 AM
0 2 * * * cd /path/to/ncode && ./scripts/docker-backup.sh

# Weekly cleanup
0 0 * * 0 docker system prune -f
```

---

## Best Practices

### Development

- Use `.env` files for configuration
- Mount code volumes for live reload
- Use `docker-compose logs -f` for debugging
- Regular `docker system prune` to free space

### Production

- Use specific image tags (not `latest`)
- Enable health checks
- Set resource limits
- Implement backup strategy
- Monitor logs and metrics
- Use secrets management
- Enable TLS/HTTPS
- Regular security updates

### Maintenance

- Weekly backups
- Monthly docker system cleanup
- Quarterly security audits
- Monitor disk usage
- Update images regularly
- Test restore procedures

---

## Quick Reference

### Essential Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f ncode

# Restart
docker-compose restart ncode

# Backup
./scripts/docker-backup.sh

# Cleanup
./scripts/docker-cleanup.sh

# Status
docker-compose ps

# Shell
docker-compose exec ncode sh
```

### Service URLs

```
nCode API:       http://localhost:18003
nCode Health:    http://localhost:18003/health
nCode Metrics:   http://localhost:18003/metrics

Qdrant API:      http://localhost:6333
Qdrant Dashboard: http://localhost:6333/dashboard

Memgraph Lab:    http://localhost:3000
Memgraph Bolt:   bolt://localhost:7687

Marquez API:     http://localhost:5000
Marquez UI:      http://localhost:3001
```

---

## Support

### Getting Help

- **Documentation:** Check other guides in `docs/`
- **Troubleshooting:** See `docs/TROUBLESHOOTING.md`
- **Issues:** Report via project issue tracker
- **Logs:** Include logs when reporting issues

### Useful Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Memgraph Docs](https://memgraph.com/docs)
- [Marquez Docs](https://marquezproject.ai/docs/)

---

**Last Updated:** 2026-01-18  
**Version:** 1.0  
**Status:** Production Ready ✅
