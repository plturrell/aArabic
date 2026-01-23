# Database Troubleshooting Guide

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Audience:** Operations teams, DBAs, and support engineers

---

## Table of Contents

1. [Overview](#overview)
2. [Diagnostic Tools](#diagnostic-tools)
3. [Common Issues](#common-issues)
4. [PostgreSQL Troubleshooting](#postgresql-troubleshooting)
5. [SAP HANA Troubleshooting](#sap-hana-troubleshooting)
6. [SQLite Troubleshooting](#sqlite-troubleshooting)
7. [Connection Issues](#connection-issues)
8. [Performance Problems](#performance-problems)
9. [Data Integrity Issues](#data-integrity-issues)
10. [Emergency Procedures](#emergency-procedures)

---

## Overview

This guide provides systematic troubleshooting procedures for diagnosing and resolving database issues in nMetaData deployments.

### Troubleshooting Methodology

1. **Identify Symptoms** - What is failing?
2. **Gather Information** - Logs, metrics, system state
3. **Form Hypothesis** - What could cause this?
4. **Test Hypothesis** - Verify with diagnostic queries
5. **Apply Fix** - Implement solution
6. **Verify Resolution** - Confirm issue resolved
7. **Document** - Record issue and solution

### Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| **P0 - Critical** | Service down | Immediate | Database unreachable |
| **P1 - High** | Degraded service | <15 min | High error rate |
| **P2 - Medium** | Minor impact | <2 hours | Slow queries |
| **P3 - Low** | Cosmetic | Best effort | Warning logs |

---

## Diagnostic Tools

### Health Check Script

```bash
#!/bin/bash
# health_check.sh - Quick database health check

echo "=== Database Health Check ==="
echo "Timestamp: $(date)"
echo

# Check nMetaData service
echo "1. Service Status:"
curl -s http://localhost:8080/health || echo "Service NOT responding"
echo

# Check database connections
echo "2. Database Connection:"
psql -h localhost -U nmeta -c "SELECT version();" || echo "PostgreSQL NOT reachable"
echo

# Check disk space
echo "3. Disk Space:"
df -h /var/lib/postgresql
echo

# Check active connections
echo "4. Active Connections:"
psql -h localhost -U nmeta -c "SELECT count(*) FROM pg_stat_activity;"
echo

# Check recent errors
echo "5. Recent Errors (last 10):"
tail -n 10 /var/log/nmeta/errors.log
echo

echo "=== Health Check Complete ==="
```

### Diagnostic Queries

```zig
pub const DiagnosticQueries = struct {
    pub fn checkConnectivity(db_client: *DbClient) !bool {
        const result = db_client.execute("SELECT 1", &.{}) catch return false;
        defer result.deinit();
        return result.rows.items.len == 1;
    }
    
    pub fn getServerInfo(db_client: *DbClient) !ServerInfo {
        const dialect = db_client.vtable.get_dialect(db_client.context);
        
        const version_query = switch (dialect) {
            .PostgreSQL => "SELECT version()",
            .HANA => "SELECT VERSION FROM M_DATABASE",
            .SQLite => "SELECT sqlite_version()",
        };
        
        var result = try db_client.execute(version_query, &.{});
        defer result.deinit();
        
        return ServerInfo{
            .dialect = dialect,
            .version = result.rows.items[0].get(0).?.string,
            .timestamp = std.time.timestamp(),
        };
    }
    
    pub fn getDatabaseSize(db_client: *DbClient, db_name: []const u8) !i64 {
        const dialect = db_client.vtable.get_dialect(db_client.context);
        
        const size_query = switch (dialect) {
            .PostgreSQL => 
                \\SELECT pg_database_size($1)
            ,
            .HANA => 
                \\SELECT SUM(USED_SIZE) FROM M_TABLE_PERSISTENCE_STATISTICS
            ,
            .SQLite => 
                \\SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()
            ,
        };
        
        var result = try db_client.execute(size_query, &[_]Value{
            .{ .string = db_name }
        });
        defer result.deinit();
        
        return result.rows.items[0].get(0).?.int64;
    }
};
```

### Log Analysis

```bash
# Search for errors in last hour
journalctl -u nmeta --since "1 hour ago" | grep -i error

# Count errors by type
journalctl -u nmeta --since "1 hour ago" | grep -i error | sort | uniq -c | sort -rn

# Find slow queries
grep "duration:" /var/log/postgresql/postgresql-*.log | awk '{if($10 > 1000) print}'

# Check connection failures
grep "connection" /var/log/nmeta/app.log | grep -i "failed\|timeout\|refused"
```

---

## Common Issues

### Issue Matrix

| Symptom | Possible Causes | First Steps |
|---------|----------------|-------------|
| Cannot connect | Network, auth, service down | Check service status, network |
| Slow queries | Missing indexes, locks, resources | EXPLAIN ANALYZE, check locks |
| High memory | Queries, leaks, cache | Check query mem, connections |
| Disk full | Logs, data growth, WAL | Check disk usage, cleanup |
| Connection pool exhausted | Leaks, high load, small pool | Check active connections |
| Transaction deadlock | Lock contention, ordering | Review transaction logs |
| Data corruption | Hardware, bugs, crashes | Check integrity, restore backup |

### Quick Diagnosis Commands

```bash
# Check if database is running
pg_isready -h localhost -p 5432

# Check service status
systemctl status nmeta
systemctl status postgresql

# Check recent errors
journalctl -u nmeta -n 50 --no-pager

# Check resource usage
top -b -n 1 | head -n 20
free -h
df -h

# Check network connectivity
netstat -an | grep 5432
ss -tulpn | grep postgres
```

---

## PostgreSQL Troubleshooting

### Connection Refused

**Symptoms:**
```
Error: could not connect to server: Connection refused
Is the server running on host "localhost" and accepting TCP/IP connections?
```

**Diagnosis:**
```bash
# Check if PostgreSQL is running
systemctl status postgresql

# Check if listening on correct port
ss -tulpn | grep 5432

# Check pg_hba.conf for access rules
cat /var/lib/postgresql/data/pg_hba.conf

# Check logs
tail -f /var/log/postgresql/postgresql-*.log
```

**Solutions:**
```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Enable on boot
sudo systemctl enable postgresql

# Update pg_hba.conf if needed
# Add: host all all 0.0.0.0/0 md5
sudo systemctl reload postgresql
```

### Too Many Connections

**Symptoms:**
```
Error: FATAL: sorry, too many clients already
```

**Diagnosis:**
```sql
-- Check current connections
SELECT count(*) as connections,
       count(*) FILTER (WHERE state = 'active') as active,
       count(*) FILTER (WHERE state = 'idle') as idle
FROM pg_stat_activity;

-- Check max connections setting
SHOW max_connections;

-- Find long-running connections
SELECT pid, usename, application_name, state, 
       now() - query_start as duration
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;
```

**Solutions:**
```sql
-- Increase max_connections (requires restart)
ALTER SYSTEM SET max_connections = 200;
-- Then: sudo systemctl restart postgresql

-- Kill idle connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND state_change < now() - interval '1 hour';

-- Use connection pooling (PgBouncer)
-- Better solution than increasing max_connections
```

### Lock Conflicts

**Symptoms:**
- Queries hanging
- Transaction timeouts
- "deadlock detected" errors

**Diagnosis:**
```sql
-- Find blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_query,
    blocking_activity.query AS blocking_query
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_locks.pid = blocked_activity.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_locks.pid = blocking_activity.pid
WHERE NOT blocked_locks.granted;

-- Check for deadlocks in logs
-- grep "deadlock detected" /var/log/postgresql/postgresql-*.log
```

**Solutions:**
```sql
-- Kill blocking query
SELECT pg_terminate_backend(blocking_pid);

-- Cancel instead of terminate (gentler)
SELECT pg_cancel_backend(blocking_pid);

-- Prevention: Use consistent lock ordering
-- Always acquire locks in same order across transactions
```

### High CPU Usage

**Diagnosis:**
```sql
-- Find CPU-intensive queries
SELECT 
    pid,
    usename,
    query,
    state,
    now() - query_start as duration
FROM pg_stat_activity
WHERE state = 'active'
  AND query NOT LIKE '%pg_stat_activity%'
ORDER BY duration DESC;

-- Check for missing indexes
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    seq_tup_read / seq_scan as avg_seq_read
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 10;
```

**Solutions:**
```sql
-- Add missing indexes
CREATE INDEX idx_name ON table_name(column_name);

-- Reduce parallel workers if CPU-bound
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;

-- Kill runaway queries
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE...;
```

### Bloat Issues

**Diagnosis:**
```sql
-- Check table bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as indexes_size,
    n_dead_tup,
    n_live_tup,
    round(100 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

**Solutions:**
```bash
# Manual VACUUM
psql -c "VACUUM ANALYZE datasets;"

# VACUUM FULL (locks table)
psql -c "VACUUM FULL datasets;"

# Tune autovacuum
psql -c "ALTER TABLE datasets SET (autovacuum_vacuum_scale_factor = 0.05);"
```

---

## SAP HANA Troubleshooting

### Out of Memory

**Symptoms:**
```
Error: Cannot allocate memory
```

**Diagnosis:**
```sql
-- Check memory usage
SELECT 
    HOST,
    ROUND(TOTAL_MEMORY_USED_SIZE / 1024 / 1024 / 1024, 2) as USED_GB,
    ROUND(ALLOCATION_LIMIT / 1024 / 1024 / 1024, 2) as LIMIT_GB,
    ROUND(100 * TOTAL_MEMORY_USED_SIZE / ALLOCATION_LIMIT, 2) as USED_PCT
FROM M_HOST_RESOURCE_UTILIZATION;

-- Find memory-intensive queries
SELECT TOP 10
    STATEMENT_STRING,
    ALLOCATED_MEMORY_SIZE / 1024 / 1024 as MEMORY_MB,
    EXECUTION_COUNT,
    AVG_EXECUTION_TIME
FROM M_SQL_PLAN_CACHE
ORDER BY ALLOCATED_MEMORY_SIZE DESC;

-- Check table sizes
SELECT TOP 20
    SCHEMA_NAME,
    TABLE_NAME,
    MEMORY_SIZE_IN_TOTAL / 1024 / 1024 / 1024 as SIZE_GB
FROM M_CS_TABLES
ORDER BY MEMORY_SIZE_IN_TOTAL DESC;
```

**Solutions:**
```sql
-- Unload tables to disk
ALTER TABLE large_table UNLOAD;

-- Merge delta stores
MERGE DELTA OF large_table;

-- Clear plan cache
ALTER SYSTEM CLEAR SQL PLAN CACHE;

-- Increase memory limit
ALTER SYSTEM ALTER CONFIGURATION ('global.ini', 'SYSTEM')
  SET ('memorymanager', 'global_allocation_limit') = '110000'
  WITH RECONFIGURE;
```

### Slow Graph Queries

**Diagnosis:**
```sql
-- Check graph workspace status
SELECT * FROM GRAPH_WORKSPACES;

-- Verify graph statistics
SELECT * FROM GRAPH_WORKSPACE_STATISTICS
WHERE WORKSPACE_NAME = 'lineage_graph';

-- Check expensive graph queries
SELECT *
FROM M_EXPENSIVE_STATEMENTS
WHERE STATEMENT_STRING LIKE '%GRAPH_TABLE%'
ORDER BY DURATION DESC;
```

**Solutions:**
```sql
-- Rebuild graph workspace
DROP GRAPH WORKSPACE lineage_graph;
CREATE GRAPH WORKSPACE lineage_graph ...;

-- Update statistics
UPDATE STATISTICS FOR lineage_graph;

-- Increase workspace size
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
  SET ('graph', 'max_graph_workspace_size') = '20000000000'
  WITH RECONFIGURE;
```

### Column Store Issues

**Diagnosis:**
```sql
-- Check delta store size
SELECT 
    SCHEMA_NAME,
    TABLE_NAME,
    MEMORY_SIZE_IN_DELTA / 1024 / 1024 as DELTA_MB,
    MEMORY_SIZE_IN_MAIN / 1024 / 1024 as MAIN_MB,
    ROUND(100 * MEMORY_SIZE_IN_DELTA / MEMORY_SIZE_IN_TOTAL, 2) as DELTA_PCT
FROM M_CS_TABLES
WHERE MEMORY_SIZE_IN_DELTA > 0
ORDER BY DELTA_PCT DESC;
```

**Solutions:**
```sql
-- Trigger delta merge
MERGE DELTA OF table_name;

-- Enable auto-merge
ALTER TABLE table_name AUTO MERGE ON;

-- Force merge
MERGE DELTA OF table_name FORCE REBUILD MAIN;
```

---

## SQLite Troubleshooting

### Database Locked

**Symptoms:**
```
Error: database is locked
```

**Diagnosis:**
```bash
# Check for other processes
lsof | grep database.db

# Check journal files
ls -la database.db*
```

**Solutions:**
```bash
# Enable WAL mode (better concurrency)
sqlite3 database.db "PRAGMA journal_mode=WAL;"

# Remove stale locks
rm -f database.db-shm database.db-wal

# Increase busy timeout
sqlite3 database.db "PRAGMA busy_timeout=30000;"
```

### Database Corruption

**Symptoms:**
```
Error: database disk image is malformed
```

**Diagnosis:**
```bash
# Check integrity
sqlite3 database.db "PRAGMA integrity_check;"

# Check foreign keys
sqlite3 database.db "PRAGMA foreign_key_check;"
```

**Solutions:**
```bash
# Dump and restore
sqlite3 database.db ".dump" | sqlite3 new_database.db

# Use backup API
sqlite3 database.db ".backup backup.db"

# Reindex
sqlite3 database.db "REINDEX;"
```

### Slow Performance

**Diagnosis:**
```sql
-- Check if indexes exist
SELECT name, sql FROM sqlite_master 
WHERE type='index';

-- Analyze query plan
EXPLAIN QUERY PLAN
SELECT * FROM datasets WHERE namespace_id = ?;

-- Check statistics
SELECT * FROM sqlite_stat1;
```

**Solutions:**
```sql
-- Create missing indexes
CREATE INDEX IF NOT EXISTS idx_namespace 
ON datasets(namespace_id);

-- Update statistics
ANALYZE;

-- Optimize database
PRAGMA optimize;

-- Vacuum
VACUUM;
```

---

## Connection Issues

### Connection Timeout

**Symptoms:**
- "Connection timeout" errors
- Requests hanging
- No response from database

**Diagnosis:**
```bash
# Check network connectivity
ping database_host

# Check port accessibility
telnet database_host 5432

# Check firewall
sudo iptables -L -n | grep 5432

# Check DNS resolution
nslookup database_host
```

**Solutions:**
```bash
# Update firewall rules
sudo firewall-cmd --add-port=5432/tcp --permanent
sudo firewall-cmd --reload

# Check connection string
# Ensure correct host, port, credentials

# Increase timeout
# In connection config: connection_timeout_ms = 10000
```

### Authentication Failed

**Symptoms:**
```
Error: authentication failed for user "nmeta"
```

**Diagnosis:**
```bash
# Check password
psql -h localhost -U nmeta -W

# Check pg_hba.conf
cat /var/lib/postgresql/data/pg_hba.conf

# Check user exists
psql -U postgres -c "\du nmeta"
```

**Solutions:**
```sql
-- Reset password
ALTER USER nmeta PASSWORD 'new_password';

-- Update pg_hba.conf
-- Change 'peer' to 'md5' for TCP connections
-- Reload: sudo systemctl reload postgresql

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE nmeta TO nmeta;
```

---

## Performance Problems

### Sudden Performance Degradation

**Diagnosis Checklist:**
1. Check recent changes (deployments, config)
2. Review slow query log
3. Check system resources (CPU, memory, disk)
4. Verify index usage
5. Check for locks/blocking
6. Review database statistics

**Investigation Script:**
```bash
#!/bin/bash
# investigate_performance.sh

echo "=== Performance Investigation ==="
echo "Time: $(date)"
echo

# System resources
echo "1. System Resources:"
echo "CPU:"
mpstat 1 5
echo "Memory:"
free -h
echo "Disk I/O:"
iostat -x 1 5
echo

# Database metrics
echo "2. Database Metrics:"
psql -U nmeta -c "
SELECT 
    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_queries,
    (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL) as waiting_queries,
    (SELECT ROUND(AVG(extract(epoch from now() - query_start))) FROM pg_stat_activity WHERE state = 'active') as avg_query_time_sec;
"

# Slow queries
echo "3. Slowest Queries (last hour):"
psql -U nmeta -c "
SELECT query, calls, total_exec_time/1000 as total_sec, mean_exec_time as avg_ms
FROM pg_stat_statements
WHERE total_exec_time > 0
ORDER BY total_exec_time DESC
LIMIT 5;
"

# Locks
echo "4. Lock Status:"
psql -U nmeta -c "
SELECT count(*), locktype, mode
FROM pg_locks
GROUP BY locktype, mode
ORDER BY count DESC;
"

echo "=== Investigation Complete ==="
```

---

## Data Integrity Issues

### Missing Data

**Diagnosis:**
```sql
-- Check if data was deleted
SELECT * FROM datasets WHERE id = ? AND deleted_at IS NOT NULL;

-- Check for failed transactions
SELECT * FROM pg_stat_database WHERE datname = 'nmeta';

-- Review audit logs
SELECT * FROM audit_log WHERE table_name = 'datasets' AND record_id = ?;
```

**Solutions:**
```sql
-- Restore from backup
pg_restore -d nmeta backup.dump

-- Point-in-time recovery
# If using WAL archiving

-- Soft delete recovery
UPDATE datasets SET deleted_at = NULL WHERE id = ?;
```

### Duplicate Data

**Diagnosis:**
```sql
-- Find duplicates
SELECT namespace_id, name, count(*)
FROM datasets
GROUP BY namespace_id, name
HAVING count(*) > 1;

-- Check constraints
SELECT conname, contype, conkey
FROM pg_constraint
WHERE conrelid = 'datasets'::regclass;
```

**Solutions:**
```sql
-- Remove duplicates (keep latest)
DELETE FROM datasets
WHERE id NOT IN (
    SELECT MAX(id)
    FROM datasets
    GROUP BY namespace_id, name
);

-- Add unique constraint
ALTER TABLE datasets
ADD CONSTRAINT unique_namespace_name
UNIQUE (namespace_id, name);
```

---

## Emergency Procedures

### Database Crash Recovery

```bash
#!/bin/bash
# emergency_recovery.sh

echo "=== EMERGENCY DATABASE RECOVERY ==="

# 1. Stop application
echo "1. Stopping application..."
systemctl stop nmeta

# 2. Assess database status
echo "2. Checking database..."
if pg_isready; then
    echo "Database is responding"
else
    echo "Database is down - attempting restart"
    systemctl restart postgresql
    sleep 5
fi

# 3. Check for corruption
echo "3. Checking integrity..."
psql -U postgres -c "SELECT pg_database_size('nmeta');"

# 4. Verify backups available
echo "4. Checking backups..."
ls -lh /var/backups/nmeta/

# 5. Test database connection
echo "5. Testing connection..."
psql -U nmeta -d nmeta -c "SELECT count(*) FROM datasets;"

# 6. If OK, restart application
if [ $? -eq 0 ]; then
    echo "6. Database OK - restarting application..."
    systemctl start nmeta
else
    echo "6. Database has issues - DO NOT restart application"
    echo "   Contact DBA for manual recovery"
fi

echo "=== Recovery Script Complete ==="
```

### Rollback Procedure

```bash
#!/bin/bash
# rollback_deployment.sh

echo "=== ROLLBACK PROCEDURE ==="

# 1. Stop new version
systemctl stop nmeta

# 2. Restore previous version
mv /opt/nmeta/current /opt/nmeta/failed
ln -s /opt/nmeta/previous /opt/nmeta/current

# 3. Rollback database migrations
cd /opt/nmeta/current
./scripts/rollback_migrations.sh

# 4. Verify database state
psql -U nmeta -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# 5. Start previous version
systemctl start nmeta

# 6. Verify health
sleep 10
curl http://localhost:8080/health

echo "=== ROLLBACK COMPLETE ==="
```

---

## Troubleshooting Checklist

### Pre-Investigation
- [ ] Document symptoms
- [ ] Record error messages
- [ ] Note when issue started
- [ ] Identify affected users/systems
- [ ] Check recent changes

### Investigation
- [ ] Check logs
- [ ] Review metrics
- [ ] Test connectivity
- [ ] Check system resources
- [ ] Review database queries
- [ ] Check for locks

### Resolution
- [ ] Form hypothesis
- [ ] Test in staging
- [ ] Apply fix
- [ ] Verify resolution
- [ ] Monitor for recurrence

### Post-Incident
- [ ] Document root cause
- [ ] Update runbooks
- [ ] Implement monitoring
- [ ] Schedule post-mortem
- [ ] Create prevention plan

---

## Summary

### Quick Reference

| Issue | Command | Purpose |
|-------|---------|---------|
| Check service | `systemctl status nmeta` | Service running? |
| Check DB | `pg_isready` | Database accessible? |
| Check connections | `SELECT count(*) FROM pg_stat_activity` | How many connections? |
| Find slow queries | `SELECT * FROM pg_stat_statements ORDER BY total_exec_time DESC` | What's slow? |
| Check locks | `SELECT * FROM pg_locks` | Any blocking? |
| Check disk | `df -h` | Space available? |
| Check logs | `journalctl -u nmeta -n 100` | Recent errors? |

### Key Principles

1. **Document Everything** - Record symptoms, steps, outcomes
2. **Test Before Applying** - Verify fixes in staging
3. **Monitor Continuously** - Watch for recurrence
4. **Learn and Improve** - Update procedures
5. **Escalate When Needed** - Don't hesitate to ask for help

---

**Version History:**
- v1.0 (2026-01-20): Initial troubleshooting guide

**Last Updated:** January 20, 2026
