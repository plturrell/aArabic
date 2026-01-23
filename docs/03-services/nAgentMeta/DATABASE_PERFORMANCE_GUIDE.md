# Database Performance Tuning Guide

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Audience:** Database administrators, performance engineers, and developers

---

## Table of Contents

1. [Overview](#overview)
2. [Performance Monitoring](#performance-monitoring)
3. [PostgreSQL Optimization](#postgresql-optimization)
4. [SAP HANA Optimization](#sap-hana-optimization)
5. [SQLite Optimization](#sqlite-optimization)
6. [Query Optimization](#query-optimization)
7. [Index Strategies](#index-strategies)
8. [Connection Pool Tuning](#connection-pool-tuning)
9. [Caching Strategies](#caching-strategies)
10. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

---

## Overview

This guide provides comprehensive performance tuning strategies for the nMetaData database layer across PostgreSQL, SAP HANA, and SQLite.

### Performance Goals

| Database | Target QPS | Target Latency | Use Case |
|----------|-----------|----------------|----------|
| PostgreSQL | 1,000+ | <10ms | Production OLTP |
| SAP HANA | 2,000+ | <5ms | Enterprise scale |
| SQLite | 15,000+ | <1ms | Development/testing |

### Key Performance Factors

1. **Database Configuration** - Proper settings for workload
2. **Query Optimization** - Efficient SQL and indexes
3. **Connection Management** - Pooling and reuse
4. **Caching** - Reduce database load
5. **Monitoring** - Track and diagnose issues

---

## Performance Monitoring

### Essential Metrics

#### Application-Level Metrics

```zig
pub const PerformanceMetrics = struct {
    // Query metrics
    total_queries: u64,
    successful_queries: u64,
    failed_queries: u64,
    avg_query_time_ms: f64,
    p95_query_time_ms: f64,
    p99_query_time_ms: f64,
    
    // Connection metrics
    active_connections: u32,
    idle_connections: u32,
    connection_wait_time_ms: f64,
    connection_failures: u64,
    
    // Cache metrics
    cache_hits: u64,
    cache_misses: u64,
    cache_hit_rate: f64,
    
    pub fn recordQuery(self: *PerformanceMetrics, duration_ms: f64, success: bool) void {
        self.total_queries += 1;
        if (success) {
            self.successful_queries += 1;
        } else {
            self.failed_queries += 1;
        }
        
        // Update running average
        const n = @as(f64, @floatFromInt(self.total_queries));
        self.avg_query_time_ms = 
            (self.avg_query_time_ms * (n - 1.0) + duration_ms) / n;
    }
};
```

#### Database-Level Metrics

**PostgreSQL:**
```sql
-- Active queries
SELECT count(*) as active_queries
FROM pg_stat_activity
WHERE state = 'active';

-- Query performance
SELECT 
    query,
    calls,
    total_exec_time / 1000 as total_time_sec,
    mean_exec_time as avg_time_ms,
    max_exec_time as max_time_ms
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- Cache hit ratio
SELECT 
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
FROM pg_statio_user_tables;

-- Index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

**SAP HANA:**
```sql
-- Expensive statements
SELECT *
FROM M_EXPENSIVE_STATEMENTS
WHERE DURATION > 1000
ORDER BY DURATION DESC
LIMIT 10;

-- Memory usage
SELECT 
    HOST,
    USED_PHYSICAL_MEMORY / 1024 / 1024 as USED_MEMORY_MB,
    FREE_PHYSICAL_MEMORY / 1024 / 1024 as FREE_MEMORY_MB
FROM M_HOST_RESOURCE_UTILIZATION;

-- Table statistics
SELECT 
    SCHEMA_NAME,
    TABLE_NAME,
    RECORD_COUNT,
    MEMORY_SIZE_IN_TOTAL / 1024 / 1024 as SIZE_MB
FROM M_CS_TABLES
ORDER BY MEMORY_SIZE_IN_TOTAL DESC;
```

**SQLite:**
```sql
-- Query plan analysis
EXPLAIN QUERY PLAN
SELECT * FROM datasets WHERE namespace_id = ?;

-- Database statistics
SELECT * FROM dbstat
ORDER BY payload DESC
LIMIT 10;

-- Index list
SELECT name, tbl_name, sql
FROM sqlite_master
WHERE type = 'index';
```

### Monitoring Tools

#### Prometheus Metrics Export

```zig
pub fn exportPrometheusMetrics(metrics: *PerformanceMetrics, writer: anytype) !void {
    // Query metrics
    try writer.print("nmeta_queries_total {d}\n", .{metrics.total_queries});
    try writer.print("nmeta_queries_successful {d}\n", .{metrics.successful_queries});
    try writer.print("nmeta_queries_failed {d}\n", .{metrics.failed_queries});
    try writer.print("nmeta_query_duration_avg_ms {d:.2}\n", .{metrics.avg_query_time_ms});
    try writer.print("nmeta_query_duration_p95_ms {d:.2}\n", .{metrics.p95_query_time_ms});
    
    // Connection metrics
    try writer.print("nmeta_connections_active {d}\n", .{metrics.active_connections});
    try writer.print("nmeta_connections_idle {d}\n", .{metrics.idle_connections});
    try writer.print("nmeta_connection_wait_ms {d:.2}\n", .{metrics.connection_wait_time_ms});
    
    // Cache metrics
    try writer.print("nmeta_cache_hits {d}\n", .{metrics.cache_hits});
    try writer.print("nmeta_cache_misses {d}\n", .{metrics.cache_misses});
    try writer.print("nmeta_cache_hit_rate {d:.4}\n", .{metrics.cache_hit_rate});
}
```

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "nMetaData Performance",
    "panels": [
      {
        "title": "Query Rate",
        "targets": [
          {
            "expr": "rate(nmeta_queries_total[5m])",
            "legendFormat": "QPS"
          }
        ]
      },
      {
        "title": "Query Latency",
        "targets": [
          {
            "expr": "nmeta_query_duration_avg_ms",
            "legendFormat": "Average"
          },
          {
            "expr": "nmeta_query_duration_p95_ms",
            "legendFormat": "P95"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "nmeta_cache_hit_rate",
            "legendFormat": "Hit Rate"
          }
        ]
      }
    ]
  }
}
```

---

## PostgreSQL Optimization

### Configuration Tuning

#### Memory Settings

```ini
# postgresql.conf

# Shared memory (25% of RAM)
shared_buffers = 4GB

# Query working memory
work_mem = 64MB              # Per operation
maintenance_work_mem = 1GB   # For VACUUM, CREATE INDEX

# Cache sizing
effective_cache_size = 12GB  # 75% of RAM

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB
```

#### Connection Settings

```ini
# Connection limits
max_connections = 100        # With connection pooling

# Connection costs
random_page_cost = 1.1       # For SSD
effective_io_concurrency = 200
```

#### Parallel Query Settings

```ini
# Parallel execution
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# Parallel query thresholds
min_parallel_table_scan_size = 8MB
min_parallel_index_scan_size = 512kB
```

### Query Optimization

#### Use EXPLAIN ANALYZE

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT d.*, n.name as namespace_name
FROM datasets d
INNER JOIN namespaces n ON d.namespace_id = n.id
WHERE d.active = true
ORDER BY d.created_at DESC
LIMIT 10;

-- Look for:
-- 1. Sequential scans on large tables
-- 2. High buffer reads
-- 3. Long execution time
-- 4. Missing indexes
```

#### Index Usage

```sql
-- Create composite index for common query pattern
CREATE INDEX idx_datasets_active_created 
ON datasets(active, created_at DESC)
WHERE active = true;

-- Partial index for active datasets
CREATE INDEX idx_datasets_namespace_active 
ON datasets(namespace_id)
WHERE active = true;

-- Index for JSON queries
CREATE INDEX idx_datasets_metadata 
ON datasets USING GIN (metadata jsonb_path_ops);

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
```

### Maintenance

#### Regular VACUUM

```sql
-- Auto-vacuum settings
ALTER TABLE datasets SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

-- Manual VACUUM when needed
VACUUM ANALYZE datasets;

-- VACUUM FULL for major cleanup (requires lock)
VACUUM FULL datasets;
```

#### Statistics Update

```sql
-- Update statistics after bulk operations
ANALYZE datasets;

-- Set statistics target for better planning
ALTER TABLE datasets 
ALTER COLUMN namespace_id 
SET STATISTICS 1000;
```

### Performance Tips

1. **Use connection pooling** (PgBouncer or built-in pool)
2. **Batch INSERT operations** in transactions
3. **Use COPY for bulk inserts** instead of INSERT
4. **Partition large tables** by date or namespace
5. **Archive old data** to keep tables small
6. **Monitor bloat** and vacuum regularly

---

## SAP HANA Optimization

### System Configuration

#### Memory Management

```sql
-- Set global memory limit (90% of available)
ALTER SYSTEM ALTER CONFIGURATION ('global.ini', 'SYSTEM')
  SET ('memorymanager', 'global_allocation_limit') = '90000'
  WITH RECONFIGURE;

-- Column store unload priority
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
  SET ('cs_unload', 'priority') = 'LOW'
  WITH RECONFIGURE;
```

#### Graph Engine Configuration

```sql
-- Increase graph workspace size for lineage
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
  SET ('graph', 'max_graph_workspace_size') = '10000000000'
  WITH RECONFIGURE;

-- Enable graph engine features
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
  SET ('graph', 'enable_experimental_features') = 'true'
  WITH RECONFIGURE;
```

### Column Store Optimization

#### Table Design

```sql
-- Create column store table
CREATE COLUMN TABLE datasets (
    id BIGINT PRIMARY KEY,
    namespace_id BIGINT NOT NULL,
    name NVARCHAR(255) NOT NULL,
    description NCLOB,
    metadata NCLOB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
) UNLOAD PRIORITY 5;

-- Partition by namespace for parallel processing
CREATE COLUMN TABLE datasets_partitioned (
    id BIGINT PRIMARY KEY,
    namespace_id BIGINT NOT NULL,
    name NVARCHAR(255) NOT NULL,
    created_at TIMESTAMP
)
PARTITION BY HASH (namespace_id) PARTITIONS 8;
```

#### Compression

```sql
-- Enable auto-merge for delta store
ALTER TABLE datasets 
  AUTO MERGE ON;

-- Manual merge delta to main
MERGE DELTA OF datasets;

-- Check compression ratio
SELECT 
    SCHEMA_NAME,
    TABLE_NAME,
    MEMORY_SIZE_IN_TOTAL / 1024 / 1024 as MEMORY_MB,
    DISK_SIZE / 1024 / 1024 as DISK_MB,
    MEMORY_SIZE_IN_TOTAL / DISK_SIZE as COMPRESSION_RATIO
FROM M_CS_TABLES
WHERE SCHEMA_NAME = 'NMETA';
```

### Graph Workspace Optimization

#### Create Optimized Graph

```sql
-- Create graph workspace with proper sizing
CREATE GRAPH WORKSPACE lineage_graph
  EDGE TABLE lineage_edges
    SOURCE COLUMN source_id
    TARGET COLUMN target_id
    KEY COLUMN edge_id
  VERTEX TABLE datasets
    KEY COLUMN id;

-- Add indexes for graph operations
CREATE INDEX idx_lineage_source ON lineage_edges(source_id);
CREATE INDEX idx_lineage_target ON lineage_edges(target_id);

-- Update graph statistics
UPDATE STATISTICS FOR lineage_graph;
```

#### Graph Query Optimization

```sql
-- Use shortest path algorithm
SELECT *
FROM GRAPH_TABLE (
    lineage_graph
    MATCH SHORTEST PATH (source:Dataset)-[:DEPENDS_ON*]->(target:Dataset)
    WHERE source.id = ?
    COLUMNS (target.id, target.name, LENGTH(EDGE) as depth)
);

-- Use hints for large graphs
SELECT /*+ USE_GRAPH_WORKSPACE(lineage_graph) */ *
FROM GRAPH_TABLE (
    lineage_graph
    MATCH (a:Dataset)-[:DEPENDS_ON]->(b:Dataset)
    WHERE a.namespace_id = ?
    COLUMNS (a.id, b.id)
);
```

### Monitoring and Profiling

```sql
-- Check expensive statements
SELECT 
    STATEMENT_STRING,
    EXECUTION_COUNT,
    AVG_EXECUTION_TIME / 1000 as AVG_TIME_MS,
    MAX_EXECUTION_TIME / 1000 as MAX_TIME_MS,
    SUM_EXECUTION_TIME / 1000 as TOTAL_TIME_MS
FROM M_SQL_PLAN_CACHE
WHERE AVG_EXECUTION_TIME > 10000
ORDER BY SUM_EXECUTION_TIME DESC
LIMIT 10;

-- Memory consumption per table
SELECT 
    SCHEMA_NAME,
    TABLE_NAME,
    RECORD_COUNT,
    MEMORY_SIZE_IN_TOTAL / 1024 / 1024 as SIZE_MB,
    MEMORY_SIZE_IN_MAIN / MEMORY_SIZE_IN_TOTAL * 100 as MAIN_PERCENT,
    MEMORY_SIZE_IN_DELTA / MEMORY_SIZE_IN_TOTAL * 100 as DELTA_PERCENT
FROM M_CS_TABLES
WHERE SCHEMA_NAME = 'NMETA'
ORDER BY MEMORY_SIZE_IN_TOTAL DESC;
```

### Performance Tips

1. **Use column store** for analytical queries
2. **Leverage graph engine** for lineage (4x faster)
3. **Regular delta merges** to maintain performance
4. **Partition large tables** for parallelism
5. **Monitor memory usage** and adjust limits
6. **Use hints** for complex queries

---

## SQLite Optimization

### PRAGMA Settings

```sql
-- Journal mode for better concurrency
PRAGMA journal_mode = WAL;

-- Synchronous mode (balance safety/performance)
PRAGMA synchronous = NORMAL;  -- or OFF for max performance

-- Cache size (in KB, negative means pages)
PRAGMA cache_size = -64000;  -- 64MB cache

-- Temp store in memory
PRAGMA temp_store = MEMORY;

-- Memory-mapped I/O
PRAGMA mmap_size = 268435456;  -- 256MB

-- Page size (must be set before creating database)
PRAGMA page_size = 4096;

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Optimize on close
PRAGMA optimize;
```

### Index Optimization

```sql
-- Create covering indexes
CREATE INDEX idx_datasets_covering 
ON datasets(namespace_id, active, created_at);

-- Analyze after index creation
ANALYZE datasets;

-- Check index effectiveness
SELECT * FROM sqlite_stat1
WHERE tbl = 'datasets';

-- Verify index usage
EXPLAIN QUERY PLAN
SELECT * FROM datasets 
WHERE namespace_id = ? AND active = true
ORDER BY created_at DESC;
```

### Query Optimization

```sql
-- Use compiled queries (prepared statements)
-- Instead of:
SELECT * FROM datasets WHERE id = 123;

-- Use:
SELECT * FROM datasets WHERE id = ?;
-- And bind parameter 123

-- Batch inserts in transaction
BEGIN TRANSACTION;
INSERT INTO datasets VALUES (?, ?, ?);
-- ... repeat many times
COMMIT;

-- Use WITHOUT ROWID for lookup tables
CREATE TABLE dataset_tags (
    dataset_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (dataset_id, tag)
) WITHOUT ROWID;
```

### Memory Optimization

```sql
-- Limit memory usage
PRAGMA soft_heap_limit = 67108864;  -- 64MB

-- Incremental VACUUM
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA incremental_vacuum;

-- Clear memory caches
PRAGMA shrink_memory;
```

### Performance Tips

1. **Use WAL mode** for better concurrency
2. **Batch operations** in transactions (100x faster)
3. **Use in-memory database** for tests
4. **Create appropriate indexes**
5. **Run ANALYZE** after bulk changes
6. **Keep database size reasonable** (<10GB)

---

## Query Optimization

### General Principles

#### 1. Select Only Needed Columns

```zig
// BAD: Select everything
const sql = "SELECT * FROM datasets";

// GOOD: Select specific columns
const sql = "SELECT id, name, namespace_id FROM datasets";
```

#### 2. Use Indexes Effectively

```zig
// Query that uses index on (namespace_id, created_at)
var qb = try QueryBuilder.init(allocator, dialect);
_ = try qb.select(&[_][]const u8{"id", "name"})
    .from("datasets")
    .where(.{ .expression = "namespace_id = $1" })  // Uses index
    .orderBy("created_at", .DESC);  // Uses same index
```

#### 3. Avoid N+1 Queries

```zig
// BAD: Query in loop
for (dataset_ids) |id| {
    const result = try db.execute(
        "SELECT * FROM datasets WHERE id = $1",
        &[_]Value{.{ .int64 = id }}
    );
}

// GOOD: Single query with IN clause
const result = try db.execute(
    "SELECT * FROM datasets WHERE id = ANY($1)",
    &[_]Value{.{ .array = dataset_ids }}
);
```

#### 4. Use Appropriate JOIN Types

```zig
// Use INNER JOIN when relationship always exists
_ = try qb.select(&[_][]const u8{"d.*", "n.name"})
    .from("datasets d")
    .join(.{
        .type = .Inner,  // Faster than LEFT JOIN
        .table = "namespaces n",
        .condition = "d.namespace_id = n.id",
    });
```

### Query Profiling

```zig
pub fn profileQuery(
    allocator: Allocator,
    db_client: *DbClient,
    sql: []const u8,
    params: []const Value,
) !struct { ResultSet, f64 } {
    const start = std.time.nanoTimestamp();
    
    const result = try db_client.execute(sql, params);
    
    const end = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
    
    std.debug.print("Query took {d:.2}ms\n", .{duration_ms});
    
    return .{ result, duration_ms };
}
```

---

## Index Strategies

### Index Types by Database

| Index Type | PostgreSQL | SAP HANA | SQLite |
|------------|-----------|----------|--------|
| **B-Tree** | ✅ Default | ✅ Yes | ✅ Default |
| **Hash** | ✅ Yes | ✅ Yes | ❌ No |
| **GIN** | ✅ JSONB | ❌ No | ❌ No |
| **GiST** | ✅ Full-text | ❌ No | ❌ No |
| **Covering** | ✅ INCLUDE | ✅ Yes | ✅ Yes |
| **Partial** | ✅ WHERE | ✅ WHERE | ✅ WHERE |

### Index Selection Strategy

```sql
-- For equality lookups
CREATE INDEX idx_datasets_namespace ON datasets(namespace_id);

-- For range queries
CREATE INDEX idx_datasets_created ON datasets(created_at);

-- For composite queries
CREATE INDEX idx_datasets_composite 
ON datasets(namespace_id, active, created_at);

-- For JSON queries (PostgreSQL)
CREATE INDEX idx_datasets_metadata 
ON datasets USING GIN (metadata);

-- Partial index for common filter
CREATE INDEX idx_datasets_active 
ON datasets(namespace_id, created_at)
WHERE active = true;

-- Covering index (PostgreSQL)
CREATE INDEX idx_datasets_covering 
ON datasets(namespace_id) 
INCLUDE (name, description);
```

### Index Maintenance

```sql
-- PostgreSQL: Rebuild bloated index
REINDEX INDEX idx_datasets_namespace;

-- Check index size
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_indexes
JOIN pg_class ON indexname = relname
WHERE tablename = 'datasets';

-- Remove unused indexes
DROP INDEX idx_rarely_used;
```

---

## Connection Pool Tuning

### Pool Configuration

```zig
pub const PoolConfig = struct {
    min_size: usize = 5,         // Minimum connections
    max_size: usize = 20,        // Maximum connections
    idle_timeout_sec: u64 = 300,  // Close idle after 5 min
    max_lifetime_sec: u64 = 3600, // Recycle after 1 hour
    connection_timeout_ms: u64 = 5000,  // Wait up to 5 sec
    validate_on_checkout: bool = true,  // Ping before use
};
```

### Sizing Guidelines

**Formula:** `pool_size = ((core_count * 2) + effective_spindle_count)`

For typical setup:
- **Small (4 cores, SSD):** 10 connections
- **Medium (8 cores, SSD):** 20 connections
- **Large (16 cores, SSD):** 40 connections

### Pool Monitoring

```zig
pub fn monitorPool(pool: *ConnectionPool) PoolStats {
    return PoolStats{
        .size = pool.connections.items.len,
        .active = pool.connections.items.len - pool.available.items.len,
        .idle = pool.available.items.len,
        .wait_count = pool.wait_count,
        .wait_time_ms = pool.total_wait_time_ms / @max(pool.wait_count, 1),
    };
}
```

---

## Caching Strategies

### Query Result Caching

```zig
pub const QueryCache = struct {
    cache: std.StringHashMap(CachedResult),
    max_size: usize = 1000,
    ttl_sec: u64 = 300,  // 5 minutes
    
    const CachedResult = struct {
        result: ResultSet,
        timestamp: i64,
    };
    
    pub fn get(self: *QueryCache, sql: []const u8) ?ResultSet {
        const entry = self.cache.get(sql) orelse return null;
        
        const now = std.time.timestamp();
        if (now - entry.timestamp > self.ttl_sec) {
            _ = self.cache.remove(sql);
            return null;
        }
        
        return entry.result;
    }
    
    pub fn put(self: *QueryCache, sql: []const u8, result: ResultSet) !void {
        if (self.cache.count() >= self.max_size) {
            // Evict oldest entry
            self.evictOldest();
        }
        
        try self.cache.put(sql, .{
            .result = result,
            .timestamp = std.time.timestamp(),
        });
    }
};
```

### Prepared Statement Caching

Already covered in the Driver Guide, but key points:
- Cache up to 100 most-used statements
- LRU eviction policy
- Thread-safe access
- Automatic cleanup

---

## Troubleshooting Performance Issues

### Common Issues and Solutions

#### Issue: Slow Queries

**Symptoms:**
- High average query time
- P95/P99 latency spikes
- Timeout errors

**Diagnosis:**
```sql
-- PostgreSQL: Find slow queries
SELECT 
    query,
    calls,
    mean_exec_time as avg_ms,
    max_exec_time as max_ms
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC;
```

**Solutions:**
1. Add missing indexes
2. Rewrite inefficient queries
3. Update statistics (ANALYZE)
4. Increase work_mem for sorts
5. Consider partitioning

#### Issue: Connection Exhaustion

**Symptoms:**
- "Too many connections" errors
- High connection wait times
- Timeouts acquiring connections

**Diagnosis:**
```sql
-- PostgreSQL: Check active connections
SELECT 
    datname,
    count(*) as connections,
    count(*) FILTER (WHERE state = 'active') as active
FROM pg_stat_activity
GROUP BY datname;
```

**Solutions:**
1. Increase pool size
2. Reduce connection lifetime
3. Fix connection leaks
4. Use PgBouncer for pooling
5. Optimize query performance

#### Issue: High Memory Usage

**Symptoms:**
- Out of memory errors
- Slow performance
- Swapping to disk

**Diagnosis:**
```sql
-- SAP HANA: Memory usage
SELECT 
    HOST,
    USED_PHYSICAL_MEMORY / 1024 / 1024 / 1024 as USED_GB,
    FREE_PHYSICAL_MEMORY / 1024 / 1024 / 1024 as FREE_GB
FROM M_HOST_RESOURCE_UTILIZATION;
```

**Solutions:**
1. Reduce shared_buffers
2. Lower work_mem
3. Optimize queries to use less memory
4. Add more RAM
5. Archive old data

#### Issue: Disk I/O Bottleneck

**Symptoms:**
- High disk wait times
- Slow writes
- WAL lag

**Diagnosis:**
```bash
# Linux: Check I/O wait
iostat -x 1

# Look for high %util and await
```

**Solutions:**
1. Use SSD instead of HDD
2. Increase checkpoint intervals
3. Tune WAL settings
4. Separate WAL and data on different disks
5. Enable compression

---

## Performance Checklist

### Daily Monitoring
- [ ] Check query latency (avg, P95, P99)
- [ ] Monitor QPS and throughput
- [ ] Review error rates
- [ ] Check connection pool usage
- [ ] Verify cache hit rates

### Weekly Maintenance
- [ ] Review slow query log
- [ ] Check index usage statistics
- [ ] Monitor table/index bloat
- [ ] Verify backup completion
- [ ] Review disk space usage

### Monthly Optimization
- [ ] Analyze query patterns
- [ ] Review and optimize indexes
- [ ] Update database statistics
- [ ] Vacuum full if needed
- [ ] Performance regression tests

---

## Summary

### Key Takeaways

1. **Monitor First** - Can't optimize what you don't measure
2. **Index Wisely** - Right indexes dramatically improve performance
3. **Pool Connections** - Reuse connections, don't create new ones
4. **Cache Results** - Avoid repeated expensive queries
5. **Profile Queries** - Find and fix slow queries
6. **Regular Maintenance** - VACUUM, ANALYZE, merge deltas
7. **Database-Specific** - Leverage unique features (HANA graph, PG JSONB)

### Performance Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Average Latency | <10ms | >50ms |
| P95 Latency | <25ms | >100ms |
| QPS | >500 | <100 |
| Error Rate | <0.1% | >1% |
| Cache Hit Rate | >80% | <50% |
| Connection Wait | <1ms | >10ms |

---

**Version History:**
- v1.0 (2026-01-20): Initial performance tuning guide

**Last Updated:** January 20, 2026
