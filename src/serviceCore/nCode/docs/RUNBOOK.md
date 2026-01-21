# nCode v1.0 Production Runbook

**Version:** 1.0.0  
**Last Updated:** 2026-01-18  
**Target Audience:** DevOps, SRE, Operations Teams

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [System Overview](#system-overview)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Common Operations](#common-operations)
6. [Troubleshooting](#troubleshooting)
7. [Incident Response](#incident-response)
8. [Maintenance](#maintenance)
9. [Backup & Recovery](#backup--recovery)
10. [Performance Tuning](#performance-tuning)

---

## Quick Reference

### Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | oncall@example.com | PagerDuty |
| Team Lead | lead@example.com | Phone |
| DevOps Manager | manager@example.com | Email |

### Critical Commands

```bash
# Service Status
docker-compose ps

# Restart All Services
docker-compose restart

# View Logs (last 100 lines)
docker-compose logs --tail=100

# Health Check
curl http://localhost:18003/health

# Emergency Stop
docker-compose down

# Full Restart
docker-compose down && docker-compose up -d
```

### Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| nCode API | http://localhost:18003 | Main API |
| Health Check | http://localhost:18003/health | Status |
| Metrics | http://localhost:18003/metrics | Prometheus |
| Qdrant | http://localhost:6333 | Vector DB |
| Memgraph | bolt://localhost:7687 | Graph DB |
| Marquez UI | http://localhost:3000 | Lineage UI |
| Marquez API | http://localhost:5000 | Lineage API |

---

## System Overview

### Architecture

```
┌────────────────────────────────────────────┐
│            Load Balancer (Optional)         │
└──────────────────┬─────────────────────────┘
                   │
      ┌────────────┴────────────┐
      │    nCode HTTP API       │
      │    (Port 18003)         │
      └────┬──────────┬─────────┘
           │          │          
     ┌─────┴───┐ ┌───┴─────┐ ┌───────┐
     │ Qdrant  │ │Memgraph │ │Marquez│
     │  :6333  │ │  :7687  │ │ :5000 │
     └─────────┘ └─────────┘ └───┬───┘
                                 │
                            ┌────┴────┐
                            │PostgreSQL│
                            │  :5432  │
                            └─────────┘
```

### Components

| Component | Technology | Purpose | Data Location |
|-----------|------------|---------|---------------|
| nCode API | Zig | Main service | In-memory index |
| Qdrant | Rust | Vector search | ./data/qdrant/ |
| Memgraph | C++ | Graph DB | ./data/memgraph/ |
| Marquez | Java | Lineage | ./data/marquez_db/ |
| PostgreSQL | SQL | Marquez backend | ./data/marquez_db/ |

### Resource Requirements

**Minimum (Development):**
- CPU: 2 cores
- RAM: 4GB
- Disk: 20GB

**Recommended (Production):**
- CPU: 8 cores
- RAM: 16GB
- Disk: 100GB (SSD preferred)

**For 100K symbols indexed:**
- RAM: +2GB
- Disk: +10GB

---

## Deployment

### Initial Deployment

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ncode.git
cd ncode/src/serviceCore/nCode

# 2. Review configuration
cp .env.example .env
# Edit .env as needed

# 3. Start services
docker-compose up -d

# 4. Wait for healthy status (30-60 seconds)
./scripts/wait-for-services.sh

# 5. Verify deployment
curl http://localhost:18003/health
```

### Update Deployment

```bash
# 1. Backup current data
./scripts/docker-backup.sh

# 2. Pull latest changes
git pull origin main

# 3. Stop services
docker-compose down

# 4. Update images
docker-compose pull

# 5. Start services
docker-compose up -d

# 6. Verify
curl http://localhost:18003/health
docker-compose ps
```

### Rollback Procedure

```bash
# 1. Stop current version
docker-compose down

# 2. Checkout previous version
git checkout <previous-tag>

# 3. Restore backup (if needed)
./scripts/docker-backup.sh restore backup_YYYY-MM-DD_HH-MM-SS.tar.gz

# 4. Start services
docker-compose up -d

# 5. Verify
curl http://localhost:18003/health
```

---

## Monitoring

### Health Checks

**Automated Health Check Script:**

```bash
#!/bin/bash
# save as: check_health.sh

# nCode API
if curl -f -s http://localhost:18003/health > /dev/null; then
    echo "✓ nCode API: healthy"
else
    echo "✗ nCode API: unhealthy"
    exit 1
fi

# Qdrant
if curl -f -s http://localhost:6333/collections > /dev/null; then
    echo "✓ Qdrant: healthy"
else
    echo "✗ Qdrant: unhealthy"
    exit 1
fi

# Marquez
if curl -f -s http://localhost:5000/api/v1/namespaces > /dev/null; then
    echo "✓ Marquez: healthy"
else
    echo "✗ Marquez: unhealthy"
    exit 1
fi

echo "All services healthy"
```

### Metrics Collection

**Prometheus Configuration:**

```yaml
# Add to prometheus.yml
scrape_configs:
  - job_name: 'ncode'
    static_configs:
      - targets: ['localhost:18003']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Key Metrics to Monitor:**

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| `ncode_requests_total` | Rate change >50% | Investigate |
| `ncode_request_duration_ms` | p95 > 200ms | Check performance |
| `ncode_index_symbols_total` | Unexpected drop | Verify index |
| `ncode_cache_hit_ratio` | < 0.7 | Tune cache |
| `ncode_database_errors` | > 0 | Check DB health |

### Log Monitoring

**Important Log Patterns:**

```bash
# Errors in last hour
docker-compose logs --since=1h | grep ERROR

# Database connection issues
docker-compose logs | grep "connection refused\|timeout"

# Performance warnings
docker-compose logs | grep "slow query\|high latency"

# Memory warnings
docker-compose logs | grep "out of memory\|OOM"
```

---

## Common Operations

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d ncode

# Start with logs visible
docker-compose up
```

### Stopping Services

```bash
# Stop all services gracefully
docker-compose down

# Stop without removing containers
docker-compose stop

# Force stop (if hanging)
docker-compose kill
```

### Viewing Logs

```bash
# All services, realtime
docker-compose logs -f

# Specific service
docker-compose logs -f ncode

# Last 100 lines
docker-compose logs --tail=100

# Since timestamp
docker-compose logs --since="2026-01-18T07:00:00"

# Save logs to file
docker-compose logs > ncode_logs_$(date +%Y%m%d_%H%M%S).log
```

### Scaling Services

```bash
# Scale nCode API (if load balancer configured)
docker-compose up -d --scale ncode=3

# Verify
docker-compose ps
```

### Reindexing

```bash
# 1. Index new project
cd /path/to/project
npx @sourcegraph/scip-typescript index

# 2. Load to databases
python /path/to/ncode/scripts/load_to_databases.py \
    index.scip --all

# 3. Verify in nCode
curl -X POST http://localhost:18003/v1/index/load \
    -H "Content-Type: application/json" \
    -d '{"index_path": "/path/to/index.scip"}'
```

---

## Troubleshooting

### Service Won't Start

**Problem:** Service fails to start or immediately exits

**Diagnosis:**
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs <service_name>

# Check ports
netstat -tuln | grep -E '18003|6333|7687|5000'

# Check disk space
df -h
```

**Solutions:**

1. **Port already in use:**
   ```bash
   # Find process using port
   lsof -i :18003
   # Kill or stop conflicting process
   ```

2. **Insufficient disk space:**
   ```bash
   # Clean up old Docker data
   docker system prune -a
   ```

3. **Configuration error:**
   ```bash
   # Verify .env file
   cat .env
   # Check docker-compose.yml syntax
   docker-compose config
   ```

### High Memory Usage

**Problem:** Service consuming too much memory

**Diagnosis:**
```bash
# Check memory usage
docker stats --no-stream

# Check system memory
free -h

# Check for memory leaks
docker-compose logs | grep "memory\|heap"
```

**Solutions:**

1. **Restart service:**
   ```bash
   docker-compose restart <service_name>
   ```

2. **Increase memory limits:**
   ```yaml
   # In docker-compose.yml
   services:
     ncode:
       deploy:
         resources:
           limits:
             memory: 2G
   ```

3. **Reduce index size:**
   - Index fewer files
   - Clear old indexes

### Slow Queries

**Problem:** API responses are slow

**Diagnosis:**
```bash
# Check metrics
curl http://localhost:18003/metrics | grep duration

# Check database performance
docker-compose exec memgraph bash
# Run Cypher EXPLAIN queries

# Check Qdrant performance
curl http://localhost:6333/collections/code_symbols
```

**Solutions:**

1. **Optimize indexes:**
   - Reduce collection size
   - Add filters to queries
   - Use pagination

2. **Scale resources:**
   - Increase CPU allocation
   - Add more memory
   - Use SSD storage

3. **Enable caching:**
   - Check cache hit ratio
   - Increase cache size

### Database Connection Errors

**Problem:** Cannot connect to database

**Diagnosis:**
```bash
# Test Qdrant
curl http://localhost:6333/collections

# Test Memgraph (requires neo4j client)
python3 << EOF
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687")
driver.verify_connectivity()
EOF

# Test Marquez
curl http://localhost:5000/api/v1/namespaces
```

**Solutions:**

1. **Restart database:**
   ```bash
   docker-compose restart qdrant memgraph marquez
   ```

2. **Check network:**
   ```bash
   docker network ls
   docker network inspect ncode_default
   ```

3. **Verify credentials:**
   - Check .env file
   - Verify environment variables

---

## Incident Response

### Severity Levels

| Level | Definition | Response Time | Example |
|-------|------------|---------------|---------|
| P0 | Complete outage | Immediate | All services down |
| P1 | Major degradation | <15 min | API not responding |
| P2 | Partial degradation | <1 hour | Slow queries |
| P3 | Minor issue | <4 hours | Non-critical errors |

### P0: Complete Outage

**Immediate Actions:**

1. **Verify outage:**
   ```bash
   curl http://localhost:18003/health
   docker-compose ps
   ```

2. **Attempt quick recovery:**
   ```bash
   docker-compose restart
   ```

3. **If restart fails:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

4. **Check system resources:**
   ```bash
   df -h  # Disk space
   free -h  # Memory
   top  # CPU
   ```

5. **Notify stakeholders**

6. **Collect logs:**
   ```bash
   docker-compose logs > incident_$(date +%Y%m%d_%H%M%S).log
   ```

### P1: Major Degradation

**Response Steps:**

1. **Identify affected component:**
   ```bash
   ./scripts/check_health.sh
   ```

2. **Check metrics:**
   ```bash
   curl http://localhost:18003/metrics
   ```

3. **Review recent changes:**
   - Check deployment history
   - Review configuration changes

4. **Restart affected services:**
   ```bash
   docker-compose restart <service>
   ```

5. **Monitor recovery:**
   - Watch logs
   - Check metrics
   - Verify functionality

### Post-Incident Actions

1. **Create incident report**
2. **Update runbook** with lessons learned
3. **Implement preventive measures**
4. **Schedule postmortem meeting**

---

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Check service health
- Review error logs
- Monitor disk usage

**Weekly:**
- Review metrics trends
- Clean up old logs
- Update dependencies

**Monthly:**
- Full backup
- Security updates
- Performance review
- Capacity planning

### Maintenance Windows

**Planned Maintenance Procedure:**

1. **Schedule announcement** (72 hours notice)
2. **Create backup:**
   ```bash
   ./scripts/docker-backup.sh
   ```
3. **Put system in maintenance mode** (if applicable)
4. **Perform updates:**
   ```bash
   docker-compose down
   git pull
   docker-compose pull
   docker-compose up -d
   ```
5. **Run health checks**
6. **Announce completion**

### Log Rotation

**Configure log rotation:**

```bash
# /etc/logrotate.d/ncode
/var/log/ncode/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 ncode ncode
    sharedscripts
    postrotate
        docker-compose restart ncode > /dev/null
    endscript
}
```

---

## Backup & Recovery

### Backup Procedure

**Automated Backup:**

```bash
# Full backup
./scripts/docker-backup.sh

# Backup location
# Default: ./backups/backup_YYYY-MM-DD_HH-MM-SS.tar.gz
```

**What's Backed Up:**
- Qdrant collections and data
- Memgraph graph database
- Marquez metadata
- PostgreSQL database
- Configuration files

### Restore Procedure

```bash
# 1. Stop services
docker-compose down

# 2. Restore backup
./scripts/docker-backup.sh restore backups/backup_YYYY-MM-DD_HH-MM-SS.tar.gz

# 3. Start services
docker-compose up -d

# 4. Verify
curl http://localhost:18003/health
```

### Disaster Recovery

**RTO (Recovery Time Objective):** 1 hour  
**RPO (Recovery Point Objective):** 24 hours

**DR Steps:**

1. **Provision new infrastructure**
2. **Restore from latest backup**
3. **Verify data integrity**
4. **Update DNS/load balancer**
5. **Monitor for issues**
6. **Notify stakeholders**

---

## Performance Tuning

### Database Optimization

**Qdrant:**
```bash
# Optimize collection
curl -X POST 'http://localhost:6333/collections/code_symbols/indexes'
```

**Memgraph:**
```cypher
-- Create indexes
CREATE INDEX ON :Symbol(name);
CREATE INDEX ON :Document(path);
```

### Resource Allocation

**Increase memory limits:**

```yaml
# docker-compose.yml
services:
  ncode:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          memory: 2G
```

### Cache Tuning

**Monitor cache performance:**
```bash
curl http://localhost:18003/metrics | grep cache
```

**Recommendations:**
- Cache hit ratio should be >70%
- If low, increase cache size or TTL
- Monitor cache memory usage

---

## Appendix

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| docker-compose.yml | Service orchestration | ./ |
| .env | Environment variables | ./ |
| Dockerfile | nCode image | ./ |

### Useful Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| docker-backup.sh | Backup/restore | ./scripts/ |
| docker-cleanup.sh | Clean resources | ./scripts/ |
| check_health.sh | Health monitoring | Custom |

### External Documentation

- [SCIP Protocol](https://github.com/sourcegraph/scip)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Memgraph Docs](https://memgraph.com/docs)
- [Marquez Docs](https://marquezproject.ai/)

---

**Document Version:** 1.0.0  
**Last Review:** 2026-01-18  
**Next Review:** 2026-02-18  
**Maintained By:** DevOps Team
