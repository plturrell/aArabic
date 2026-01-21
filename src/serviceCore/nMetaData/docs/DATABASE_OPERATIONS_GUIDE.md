# Database Production Operations Guide

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Audience:** Operations teams and DBAs

---

## Table of Contents

1. [Deployment](#deployment)
2. [Backup & Recovery](#backup--recovery)
3. [Monitoring Setup](#monitoring-setup)
4. [Scaling Strategies](#scaling-strategies)
5. [Security](#security)
6. [Maintenance Windows](#maintenance-windows)

---

## Deployment

### Initial Setup

```bash
#!/bin/bash
# deploy_nmeta_database.sh

# 1. Install database
case "$DB_TYPE" in
  postgresql)
    apt-get install postgresql-14
    ;;
  hana)
    # Follow SAP HANA installation guide
    ;;
  sqlite)
    apt-get install sqlite3
    ;;
esac

# 2. Create database and user
psql -U postgres <<SQL
CREATE DATABASE nmeta;
CREATE USER nmeta WITH PASSWORD '${DB_PASSWORD}';
GRANT ALL PRIVILEGES ON DATABASE nmeta TO nmeta;
SQL

# 3. Run migrations
cd /opt/nmeta
./scripts/run_migrations.sh

# 4. Verify
psql -U nmeta -d nmeta -c "SELECT count(*) FROM schema_migrations;"
```

### Configuration Management

```yaml
# config/production.yaml
database:
  dialect: postgresql
  host: db.example.com
  port: 5432
  database: nmeta
  username: nmeta
  password: ${DB_PASSWORD}  # From environment
  pool:
    min_size: 5
    max_size: 20
    idle_timeout_sec: 300
    max_lifetime_sec: 3600
  ssl:
    enabled: true
    cert: /etc/certs/client.crt
    key: /etc/certs/client.key
```

### Blue-Green Deployment

```bash
# 1. Deploy to green environment
deploy_to green

# 2. Run smoke tests
test_environment green

# 3. Switch traffic
switch_traffic blue to green

# 4. Monitor
monitor_environment green

# 5. Rollback if issues
if [ $? -ne 0 ]; then
  switch_traffic green to blue
fi
```

---

## Backup & Recovery

### Backup Strategy

| Type | Frequency | Retention | Storage |
|------|-----------|-----------|---------|
| **Full** | Daily | 30 days | S3 |
| **Incremental** | Hourly | 7 days | Local + S3 |
| **WAL Archive** | Continuous | 7 days | S3 |
| **Snapshot** | Weekly | 90 days | S3 Glacier |

### PostgreSQL Backup

```bash
#!/bin/bash
# backup_postgresql.sh

BACKUP_DIR="/var/backups/nmeta"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/nmeta_$TIMESTAMP.dump"

# Create backup
pg_dump -U nmeta -F c -b -v -f "$BACKUP_FILE" nmeta

# Compress
gzip "$BACKUP_FILE"

# Upload to S3
aws s3 cp "$BACKUP_FILE.gz" s3://backups/nmeta/

# Cleanup old backups
find "$BACKUP_DIR" -name "*.dump.gz" -mtime +30 -delete
```

### Point-in-Time Recovery (PITR)

```bash
# 1. Stop database
systemctl stop postgresql

# 2. Restore base backup
cd /var/lib/postgresql/14/main
rm -rf *
tar -xzf /backups/base_backup.tar.gz

# 3. Create recovery.conf
cat > recovery.conf <<EOF
restore_command = 'cp /var/backups/wal/%f %p'
recovery_target_time = '2026-01-20 09:00:00'
EOF

# 4. Start recovery
systemctl start postgresql

# 5. Verify
psql -U nmeta -c "SELECT now();"
```

### SAP HANA Backup

```sql
-- Full backup
BACKUP DATA USING FILE ('full_backup');

-- Incremental backup
BACKUP DATA INCREMENTAL USING FILE ('inc_backup');

-- Log backup
BACKUP DATA USING BACKINT ('log_backup');

-- Check backup status
SELECT * FROM M_BACKUP_CATALOG ORDER BY SYS_START_TIME DESC;
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nmeta'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
    scrape_interval: 30s
```

### Key Alerts

```yaml
# alerts.yml
groups:
  - name: database
    rules:
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          
      - alert: HighQueryLatency
        expr: nmeta_query_duration_p95_ms > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High query latency detected"
          
      - alert: LowCacheHitRate
        expr: nmeta_cache_hit_rate < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 80%"
          
      - alert: ConnectionPoolExhausted
        expr: nmeta_connections_active / nmeta_connections_total > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Connection pool nearly exhausted"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "nMetaData Database",
    "rows": [
      {
        "title": "Overview",
        "panels": [
          {"title": "QPS", "metric": "rate(nmeta_queries_total[1m])"},
          {"title": "Latency", "metric": "nmeta_query_duration_avg_ms"},
          {"title": "Error Rate", "metric": "rate(nmeta_queries_failed[1m])"},
          {"title": "Cache Hit Rate", "metric": "nmeta_cache_hit_rate"}
        ]
      },
      {
        "title": "Connections",
        "panels": [
          {"title": "Active", "metric": "nmeta_connections_active"},
          {"title": "Idle", "metric": "nmeta_connections_idle"},
          {"title": "Wait Time", "metric": "nmeta_connection_wait_ms"}
        ]
      }
    ]
  }
}
```

---

## Scaling Strategies

### Vertical Scaling

```bash
# PostgreSQL configuration for larger instance
cat >> /etc/postgresql/14/main/postgresql.conf <<EOF
shared_buffers = 8GB           # 25% of 32GB RAM
effective_cache_size = 24GB    # 75% of 32GB RAM
work_mem = 128MB
maintenance_work_mem = 2GB
max_parallel_workers = 16
EOF

systemctl restart postgresql
```

### Horizontal Scaling

**Read Replicas:**
```bash
# On primary
psql -c "CREATE USER replicator REPLICATION LOGIN PASSWORD 'replpass';"

# On replica
cat > recovery.conf <<EOF
standby_mode = 'on'
primary_conninfo = 'host=primary port=5432 user=replicator password=replpass'
primary_slot_name = 'replica_1'
EOF

# Configure application
# - Writes go to primary
# - Reads go to replicas (round-robin)
```

**Partitioning:**
```sql
-- Partition by namespace
CREATE TABLE datasets_partitioned (
    id BIGINT,
    namespace_id BIGINT,
    name VARCHAR(255),
    created_at TIMESTAMP
) PARTITION BY HASH (namespace_id);

-- Create partitions
CREATE TABLE datasets_p0 PARTITION OF datasets_partitioned
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE datasets_p1 PARTITION OF datasets_partitioned
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
-- ... more partitions
```

---

## Security

### Access Control

```sql
-- PostgreSQL roles
CREATE ROLE nmeta_readonly;
GRANT CONNECT ON DATABASE nmeta TO nmeta_readonly;
GRANT USAGE ON SCHEMA public TO nmeta_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nmeta_readonly;

CREATE ROLE nmeta_readwrite;
GRANT nmeta_readonly TO nmeta_readwrite;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nmeta_readwrite;

-- Create users
CREATE USER app_user WITH PASSWORD 'secure_pass';
GRANT nmeta_readwrite TO app_user;
```

### Encryption

```bash
# SSL/TLS for connections
cat >> /etc/postgresql/14/main/postgresql.conf <<EOF
ssl = on
ssl_cert_file = '/etc/postgresql/ssl/server.crt'
ssl_key_file = '/etc/postgresql/ssl/server.key'
ssl_ca_file = '/etc/postgresql/ssl/ca.crt'
EOF

# Encryption at rest (LUKS)
cryptsetup luksFormat /dev/sdb
cryptsetup open /dev/sdb pgdata
mkfs.ext4 /dev/mapper/pgdata
mount /dev/mapper/pgdata /var/lib/postgresql
```

### Audit Logging

```sql
-- Enable pgaudit
CREATE EXTENSION pgaudit;

-- Configure logging
ALTER SYSTEM SET pgaudit.log = 'all';
ALTER SYSTEM SET pgaudit.log_relation = on;
ALTER SYSTEM SET pgaudit.log_parameter = on;

-- Review logs
SELECT * FROM pgaudit.log ORDER BY timestamp DESC LIMIT 10;
```

---

## Maintenance Windows

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== Weekly Maintenance ==="

# 1. VACUUM
psql -U nmeta -c "VACUUM ANALYZE;"

# 2. Reindex if needed
psql -U nmeta -c "REINDEX DATABASE nmeta;"

# 3. Update statistics
psql -U nmeta -c "ANALYZE;"

# 4. Backup verification
verify_latest_backup

# 5. Cleanup old logs
find /var/log/postgresql -name "*.log" -mtime +7 -delete

# 6. Check disk space
df -h /var/lib/postgresql

echo "=== Maintenance Complete ==="
```

### Monthly Tasks

- [ ] Review and optimize slow queries
- [ ] Update statistics targets
- [ ] Archive old data
- [ ] Test disaster recovery
- [ ] Review security audit logs
- [ ] Update documentation
- [ ] Capacity planning review

### Quarterly Tasks

- [ ] Major version upgrades (if available)
- [ ] Full security audit
- [ ] Disaster recovery drill
- [ ] Performance baseline update
- [ ] Cost optimization review

---

## Operations Checklist

### Daily
- [ ] Check service health
- [ ] Review error logs
- [ ] Verify backups completed
- [ ] Monitor key metrics
- [ ] Check disk space

### Weekly
- [ ] Run maintenance scripts
- [ ] Review slow queries
- [ ] Check for security updates
- [ ] Verify replication lag
- [ ] Test alerting

### Monthly
- [ ] Performance review
- [ ] Capacity planning
- [ ] Security audit
- [ ] Documentation update
- [ ] DR test

---

## Quick Reference Commands

```bash
# Service management
systemctl status nmeta
systemctl restart nmeta

# Database access
psql -U nmeta -d nmeta

# Backup
pg_dump -U nmeta nmeta > backup.sql

# Restore
psql -U nmeta nmeta < backup.sql

# Monitor
watch -n 1 'psql -U nmeta -c "SELECT count(*) FROM pg_stat_activity;"'

# Logs
journalctl -u nmeta -f
tail -f /var/log/postgresql/postgresql-*.log
```

---

**Version History:**
- v1.0 (2026-01-20): Initial production operations guide

**Last Updated:** January 20, 2026
