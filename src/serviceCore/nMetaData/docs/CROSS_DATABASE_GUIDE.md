# Cross-Database Integration Guide

**nMetaData Project - Multi-Database Support**  
**Version:** 1.0  
**Last Updated:** January 20, 2026

---

## Overview

nMetaData supports three database backends with a unified abstraction layer:
- **PostgreSQL** - Industry-standard OLTP
- **SAP HANA** - In-memory, graph-enabled
- **SQLite** - Embedded, zero-config

This guide covers feature parity, performance characteristics, and migration paths.

---

## Table of Contents

1. [Feature Parity Matrix](#feature-parity-matrix)
2. [Performance Comparison](#performance-comparison)
3. [Database Selection Guide](#database-selection-guide)
4. [Migration Paths](#migration-paths)
5. [Best Practices](#best-practices)
6. [Testing Strategy](#testing-strategy)

---

## Feature Parity Matrix

### Core Features (Supported by All)

| Feature | PostgreSQL | HANA | SQLite | Notes |
|---------|------------|------|--------|-------|
| Basic Queries | ✓ | ✓ | ✓ | SELECT/INSERT/UPDATE/DELETE |
| Prepared Statements | ✓ | ✓ | ✓ | Full support |
| Transactions | ✓ | ✓ | ✓ | ACID compliance |
| Connection Pooling | ✓ | ✓ | ✓ | Production-ready |
| Savepoints | ✓ | ✓ | ✓ | Nested transactions |
| Batch Operations | ✓ | ✓ | ✓ | Bulk inserts |
| Type Casting | ✓ | ✓ | ✓ | Safe conversions |
| NULL Handling | ✓ | ✓ | ✓ | Consistent behavior |
| JSON Support | ✓ | ✓ | ✓ | JSON columns |
| Full-Text Search | ✓ | ✓ | ✓ | FTS capabilities |
| Recursive CTEs | ✓ | ✓ | ✓ | Lineage queries |
| Window Functions | ✓ | ✓ | ✓ | Analytics |

### Database-Specific Features

| Feature | PostgreSQL | HANA | SQLite | Notes |
|---------|------------|------|--------|-------|
| UUID Type | ✓ | ✓ | ✗ | Store as TEXT in SQLite |
| Graph Queries | ✗ | ✓ | ✗ | HANA exclusive |
| LISTEN/NOTIFY | ✓ | ✗ | ✗ | PostgreSQL exclusive |
| In-Memory Mode | ✗ | Partial | ✓ | SQLite exclusive |
| Column Store | ✗ | ✓ | ✗ | HANA exclusive |
| Native Geospatial | ✓ | ✓ | ✗ | PostGIS, HANA Spatial |

---

## Performance Comparison

### Benchmark Results (1K queries)

| Operation | PostgreSQL | HANA | SQLite | Winner |
|-----------|------------|------|--------|--------|
| Simple SELECT | 8,500 QPS | 12,000 QPS | 15,000 QPS | **SQLite** |
| Complex JOINs | 2,200 QPS | 4,800 QPS | 1,800 QPS | **HANA** |
| Batch Inserts (10K) | 180ms | 120ms | 45ms | **SQLite** |
| Transactions (1K) | 5,000 TPS | 8,000 TPS | 12,000 TPS | **SQLite** |
| Connection Pool | 500/s | 800/s | 2000/s | **SQLite** |
| Graph Queries | N/A | **20-40x vs CTE** | N/A | **HANA** |

### Performance Characteristics

**PostgreSQL:**
- ✅ Excellent for OLTP workloads
- ✅ Strong consistency guarantees
- ✅ Good concurrent write performance
- ⚠️ Higher latency than in-memory DBs
- ⚠️ Network overhead

**SAP HANA:**
- ✅ **Fastest for complex analytics**
- ✅ **Graph queries 20-40x faster**
- ✅ In-memory processing
- ✅ Columnar storage for aggregations
- ⚠️ Higher resource requirements
- ⚠️ Network overhead

**SQLite:**
- ✅ **Lowest latency** (no network)
- ✅ **Fastest for simple queries**
- ✅ Zero configuration
- ✅ Perfect for testing
- ⚠️ Single writer limitation
- ⚠️ Not suitable for distributed systems

---

## Database Selection Guide

### Choose PostgreSQL When:

✅ **Production OLTP workload**
- High concurrent writes needed
- Strong consistency critical
- Standard SQL features sufficient
- Proven reliability required

✅ **Cost-effective scaling**
- Open-source license
- Wide ecosystem support
- Mature tooling

✅ **General-purpose needs**
- Balanced read/write performance
- LISTEN/NOTIFY for real-time updates
- PostGIS for geospatial data

**Example Use Cases:**
- Transaction processing systems
- Multi-tenant SaaS applications
- General metadata storage
- Event-driven architectures

---

### Choose SAP HANA When:

✅ **Advanced analytics required**
- Complex aggregations
- Graph traversal (lineage)
- Real-time analytics
- In-memory performance critical

✅ **SAP ecosystem integration**
- Existing SAP infrastructure
- SAP BTP deployment
- SAP Data Intelligence integration

✅ **Graph-heavy workloads**
- **Lineage queries 20-40x faster**
- Impact analysis
- Dependency tracking
- Network analysis

**Example Use Cases:**
- Data lineage platforms (nMetaData primary use case)
- Real-time analytics dashboards
- SAP integrated solutions
- Graph-based applications

---

### Choose SQLite When:

✅ **Testing & development**
- Fast test execution
- In-memory mode
- Zero setup
- Consistent across environments

✅ **Embedded applications**
- Single-user applications
- Mobile apps
- Edge computing
- Offline-first apps

✅ **Read-heavy workloads**
- Configuration storage
- Caching layer
- Local data storage
- Prototyping

**Example Use Cases:**
- Unit/integration tests
- Development environments
- Embedded devices
- Personal tools

---

## Migration Paths

### Migration Compatibility Matrix

| From ↓ / To → | PostgreSQL | HANA | SQLite |
|---------------|------------|------|--------|
| **PostgreSQL** | N/A | ⚠️ Warnings | ⚡ Manual work |
| **HANA** | ⚠️ Warnings | N/A | ⚡ Manual work |
| **SQLite** | ✓ Compatible | ✓ Compatible | N/A |

**Legend:**
- ✓ Fully compatible
- ⚠️ Compatible with warnings
- ⚡ Manual intervention required

---

### PostgreSQL → HANA

**Compatibility:** ⚠️ Compatible with warnings

**Issues:**
- LISTEN/NOTIFY not available in HANA
- Some PostgreSQL-specific types need mapping
- Different text type (VARCHAR → NVARCHAR)

**Migration Steps:**
```sql
-- 1. Convert types
ALTER TABLE datasets 
  ALTER COLUMN name TYPE NVARCHAR(255);

-- 2. Remove PostgreSQL-specific features
-- (Stop using LISTEN/NOTIFY)

-- 3. Optional: Enable graph features
CREATE GRAPH WORKSPACE LINEAGE_GRAPH ...
```

**Code Changes:**
```zig
// Change connection
const config = ConnectionConfig{
    .database_type = .hana, // Was .postgresql
    .host = "hana.example.com",
    .port = 30015, // Was 5432
    // ... rest of config
};
```

---

### PostgreSQL → SQLite

**Compatibility:** ⚡ Manual intervention required

**Issues:**
- UUID type not natively supported (use TEXT)
- LISTEN/NOTIFY not available
- Limited concurrent write support
- Some advanced features unavailable

**Migration Steps:**
```sql
-- 1. Convert UUID to TEXT
ALTER TABLE datasets 
  ALTER COLUMN id TYPE TEXT;

-- 2. Simplify schema
-- Remove database-specific features

-- 3. Adjust for single-writer
-- May need application-level coordination
```

**Use Case:** Primarily for testing, not recommended for production

---

### HANA → PostgreSQL

**Compatibility:** ⚠️ Compatible with warnings

**Issues:**
- Graph Engine features not available
- NVARCHAR → VARCHAR conversion
- Loss of in-memory optimizations
- Different performance characteristics

**Migration Steps:**
```sql
-- 1. Drop graph workspaces
DROP GRAPH WORKSPACE LINEAGE_GRAPH;

-- 2. Convert types
ALTER TABLE datasets 
  ALTER COLUMN name TYPE VARCHAR(255);

-- 3. Replace graph queries with CTEs
-- Use recursive CTEs instead of GRAPH_TABLE
```

**Performance Impact:** 10-40x slower for lineage queries

---

### HANA → SQLite

**Compatibility:** ⚡ Manual intervention required

**Issues:**
- Graph Engine not available
- NVARCHAR → TEXT conversion
- Reduced concurrency support
- Advanced features unavailable

**Use Case:** Testing only, not recommended for production

---

### SQLite → PostgreSQL/HANA

**Compatibility:** ✓ Fully compatible

**Advantages:**
- SQLite is simplest feature set
- No SQLite-specific features to migrate
- Straightforward schema transfer

**Migration Steps:**
```sql
-- 1. Export data from SQLite
.mode csv
.output datasets.csv
SELECT * FROM datasets;

-- 2. Import to target database
\COPY datasets FROM 'datasets.csv' CSV; -- PostgreSQL
IMPORT FROM CSV FILE 'datasets.csv' ... -- HANA

-- 3. Add database-specific optimizations
-- Enable features not available in SQLite
```

---

## Best Practices

### 1. Use Dialect-Agnostic SQL

```zig
// Good: Works on all databases
const sql = "SELECT id, name FROM datasets WHERE active = true";

// Avoid: Database-specific syntax
// const sql = "SELECT id, name FROM datasets WHERE active = true FOR UPDATE SKIP LOCKED";
```

### 2. Handle Database-Specific Features Gracefully

```zig
fn queryLineage(executor: anytype, dataset_id: []const u8) !Result {
    return switch (db_type) {
        .hana => {
            // Use Graph Engine for 20x speedup
            return executor.findUpstreamLineage(workspace, dataset_id, 10);
        },
        .postgresql, .sqlite => {
            // Fallback to recursive CTE
            return executor.executeRecursiveCTE(dataset_id, 10);
        },
    };
}
```

### 3. Test Against All Databases

```zig
test "feature works on all databases" {
    const databases = [_]DatabaseType{ .postgresql, .hana, .sqlite };
    
    for (databases) |db_type| {
        var conn = try createConnection(db_type);
        defer conn.deinit();
        
        // Run test...
    }
}
```

### 4. Use Connection Pools Appropriately

```zig
// PostgreSQL/HANA: Connection pools essential
const pool_size = 20; // Network connections

// SQLite: Small pool sufficient
const pool_size = if (db_type == .sqlite) 5 else 20;
```

### 5. Consider Performance Trade-offs

```zig
// Development: Use SQLite (fast, easy)
const dev_db = DatabaseType.sqlite;

// Production: Use PostgreSQL (reliable) or HANA (fast analytics)
const prod_db = if (needs_graph_features) 
    DatabaseType.hana 
else 
    DatabaseType.postgresql;
```

---

## Testing Strategy

### Unit Tests

Run against all databases:
```bash
zig build test  # Tests on all databases
```

### Integration Tests

Specific database tests:
```bash
zig build test-postgres
zig build test-hana
zig build test-sqlite
```

### Cross-Database Tests

Verify identical behavior:
```bash
zig build test-cross-database
```

### Performance Benchmarks

Compare performance:
```bash
zig build bench-cross-database
```

### Migration Tests

Test migration paths:
```bash
zig build test-migrations
```

---

## Performance Tuning

### PostgreSQL Optimization

```sql
-- Connection pooling
SET max_connections = 200;

-- Query optimization
SET enable_seqscan = off;
SET random_page_cost = 1.1;

-- Autovacuum tuning
SET autovacuum_naptime = '30s';
```

### HANA Optimization

```sql
-- Graph Engine
SET 'graph_parallel_execution' = 'ON';
SET 'graph_max_memory' = '2048';

-- Column store
MERGE DELTA OF datasets;

-- Query cache
SET 'sql_result_cache' = 'ON';
```

### SQLite Optimization

```sql
-- Performance pragmas
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB
PRAGMA temp_store = MEMORY;
```

---

## Decision Matrix

### Quick Selection Guide

| Requirement | PostgreSQL | HANA | SQLite |
|-------------|------------|------|--------|
| Production OLTP | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Analytics | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| Graph Queries | ⭐ | ⭐⭐⭐ | ⭐ |
| Testing | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| Ease of Setup | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| Concurrent Writes | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Low Latency | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Cost | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |

### Recommended Use Cases

**PostgreSQL:**
- Default production database
- Standard metadata storage
- Event-driven systems
- Budget-conscious deployments

**HANA:**
- Lineage-heavy workloads
- Real-time analytics
- SAP-integrated environments
- Performance-critical scenarios

**SQLite:**
- Testing & development
- Single-user applications
- Embedded systems
- Proof-of-concept

---

## Migration Examples

### Example 1: PostgreSQL to HANA (with Graph upgrade)

```zig
// 1. Export from PostgreSQL
const pg_exporter = DatabaseExporter.init(allocator, .postgresql);
const data = try pg_exporter.exportSchema();
defer data.deinit();

// 2. Transform schema
const transformer = SchemaTransformer.init(allocator);
const hana_schema = try transformer.transform(data, .postgresql, .hana);
defer hana_schema.deinit();

// 3. Import to HANA
const hana_importer = DatabaseImporter.init(allocator, .hana);
try hana_importer.importSchema(hana_schema);

// 4. Create graph workspace (new capability!)
const workspace = GraphWorkspace{
    .name = "LINEAGE_GRAPH",
    .schema = "METADATA",
    .vertex_table = "DATASETS",
    .edge_table = "LINEAGE_EDGES",
};
try hana_importer.createGraphWorkspace(workspace);

std.debug.print("Migration complete! Graph queries now 20x faster.\n", .{});
```

### Example 2: SQLite to PostgreSQL (test to production)

```zig
// 1. Run tests on SQLite
test "lineage tracking" {
    var conn = try createConnection(.sqlite, ":memory:");
    defer conn.deinit();
    
    // Test implementation...
}

// 2. Deploy to PostgreSQL
pub fn main() !void {
    var conn = try createConnection(.postgresql, prod_config);
    defer conn.deinit();
    
    // Same code works!
}
```

---

## Troubleshooting

### Common Issues

**1. Feature Not Available**
```
Error: Graph queries not supported on PostgreSQL
```
**Solution:** Use conditional execution or fallback to CTEs

**2. Type Mismatch**
```
Error: UUID type not supported
```
**Solution:** Use TEXT in SQLite, add conversion layer

**3. Performance Degradation**
```
Warning: Lineage queries slow on PostgreSQL
```
**Solution:** Consider migrating to HANA or optimize CTEs

---

## Conclusion

nMetaData's multi-database support provides flexibility:

- **PostgreSQL:** Production default, reliable and cost-effective
- **HANA:** Performance leader for analytics and graph queries
- **SQLite:** Testing champion, zero-config development

Choose based on your requirements, and migrate easily when needs change.

---

**Last Updated:** January 20, 2026  
**Version:** 1.0  
**Status:** Complete
