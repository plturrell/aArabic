# Database Selection Guide

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Audience:** Architects, developers, and operations teams

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Database Comparison Matrix](#database-comparison-matrix)
3. [PostgreSQL](#postgresql)
4. [SAP HANA](#sap-hana)
5. [SQLite](#sqlite)
6. [Decision Tree](#decision-tree)
7. [Workload Analysis](#workload-analysis)
8. [Migration Considerations](#migration-considerations)
9. [Cost Analysis](#cost-analysis)
10. [Real-World Use Cases](#real-world-use-cases)

---

## Executive Summary

The nMetaData service supports three database backends, each optimized for different use cases. This guide helps you select the right database for your specific requirements.

### Quick Recommendation

| If you need... | Choose | Reason |
|----------------|--------|--------|
| **General purpose** | PostgreSQL | Best balance of features, performance, and maturity |
| **Graph/lineage queries** | SAP HANA | 4x faster on graph operations with native graph engine |
| **Development/testing** | SQLite | Fastest setup, in-memory mode, zero configuration |
| **High write throughput** | PostgreSQL | Optimized OLTP performance |
| **Real-time analytics** | SAP HANA | Column store, in-memory architecture |
| **Minimal infrastructure** | SQLite | Embedded, no server required |

---

## Database Comparison Matrix

### Performance Characteristics

| Metric | PostgreSQL | SAP HANA | SQLite |
|--------|-----------|----------|--------|
| **Query Throughput** | 1,000+ QPS | 2,000+ QPS | 15,000+ QPS* |
| **Dataset Operations** | 10ms | 10ms | 10ms |
| **Simple Lineage (CTE)** | 20ms | 20ms | 20ms |
| **Graph Lineage** | 20ms (CTE) | **5ms** (native) | 20ms (CTE) |
| **Complex Joins** | Excellent | Excellent | Good |
| **Concurrent Writes** | Excellent | Excellent | Limited |
| **Write Latency** | 5-10ms | 5-10ms | 1-5ms |
| **Read Latency** | 1-5ms | 0.5-3ms | 0.1-1ms |

\* SQLite performance is for in-memory mode; disk-based is ~1,000 QPS

### Feature Support

| Feature | PostgreSQL | SAP HANA | SQLite |
|---------|-----------|----------|--------|
| **ACID Transactions** | ✅ Full | ✅ Full | ✅ Full |
| **Connection Pooling** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Prepared Statements** | ✅ Yes | ✅ Yes | ✅ Yes |
| **CTEs (WITH)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Window Functions** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Full-Text Search** | ✅ Yes | ✅ Yes | ✅ FTS5 |
| **JSON Support** | ✅ JSONB | ✅ Native | ✅ JSON1 |
| **Graph Engine** | ❌ No | ✅ Native | ❌ No |
| **Column Store** | ❌ Row-based | ✅ Native | ❌ Row-based |
| **Spatial Data** | ✅ PostGIS | ✅ Native | ✅ Limited |
| **Replication** | ✅ Streaming | ✅ System | ❌ No |
| **Partitioning** | ✅ Yes | ✅ Yes | ❌ No |
| **Parallel Queries** | ✅ Yes | ✅ Yes | ❌ No |

### Operational Characteristics

| Aspect | PostgreSQL | SAP HANA | SQLite |
|--------|-----------|----------|--------|
| **Setup Complexity** | Medium | High | Minimal |
| **Operational Cost** | Medium | High | Low |
| **Backup/Recovery** | Excellent | Excellent | Simple |
| **Monitoring** | Mature tools | SAP tools | Basic |
| **Scaling** | Vertical + sharding | Vertical + scale-out | Vertical only |
| **Cloud Support** | Excellent | SAP Cloud | N/A |
| **License** | Open Source | Commercial | Public Domain |
| **Community** | Large | Specialized | Large |

---

## PostgreSQL

### Overview

PostgreSQL is a powerful, open-source relational database with a strong focus on standards compliance and extensibility.

### Strengths

1. **Mature Ecosystem**
   - 30+ years of development
   - Large community and tooling
   - Extensive documentation
   - Wide cloud provider support

2. **Feature Rich**
   - Advanced SQL support
   - JSONB for semi-structured data
   - Full-text search
   - Geographic data (PostGIS)
   - Custom extensions

3. **ACID Compliance**
   - MVCC for high concurrency
   - Multiple isolation levels
   - Robust transaction support
   - Point-in-time recovery

4. **Performance**
   - Excellent for mixed OLTP/OLAP workloads
   - Query planner optimization
   - Index types: B-tree, Hash, GiST, GIN, BRIN
   - Parallel query execution

5. **Open Source**
   - PostgreSQL license (permissive)
   - No vendor lock-in
   - Free for all use cases

### Weaknesses

1. **No Native Graph Engine**
   - Graph queries require CTEs
   - 4x slower than HANA for lineage
   - Complex graph algorithms can be slow

2. **Row-based Storage**
   - Not optimized for analytical queries
   - Higher storage overhead
   - Slower column scans

3. **Write Amplification**
   - MVCC creates multiple row versions
   - Requires VACUUM maintenance
   - Can lead to bloat

### Best For

- **General purpose metadata storage**
- **Standard OLTP workloads**
- **Applications requiring PostgreSQL ecosystem**
- **Cost-conscious deployments**
- **When graph performance is not critical**

### Configuration Recommendations

```yaml
# Recommended PostgreSQL settings for nMetaData
shared_buffers: 4GB              # 25% of RAM
effective_cache_size: 12GB       # 75% of RAM
work_mem: 64MB                   # Per operation
maintenance_work_mem: 1GB        # For VACUUM, etc.
max_connections: 100             # With connection pooling
random_page_cost: 1.1            # For SSD
effective_io_concurrency: 200    # For SSD
max_parallel_workers: 8          # CPU cores
max_parallel_workers_per_gather: 4
```

### Performance Tuning

```sql
-- Create indexes for common queries
CREATE INDEX idx_datasets_namespace ON datasets(namespace_id);
CREATE INDEX idx_jobs_name ON jobs(namespace_id, name);
CREATE INDEX idx_runs_job_timestamp ON runs(job_id, started_at DESC);

-- Analyze tables regularly
ANALYZE datasets;
ANALYZE jobs;
ANALYZE runs;

-- Monitor slow queries
CREATE EXTENSION pg_stat_statements;
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

---

## SAP HANA

### Overview

SAP HANA is an in-memory, column-oriented database platform designed for high-performance analytics and transactions.

### Strengths

1. **Native Graph Engine**
   - 4x faster lineage queries
   - Optimized graph algorithms
   - Graph workspaces
   - Efficient traversal operations

2. **In-Memory Architecture**
   - Sub-millisecond query latency
   - Real-time analytics
   - No separate OLAP system needed
   - Column store for analytics

3. **Advanced Analytics**
   - Built-in machine learning
   - Predictive analytics
   - Text analytics
   - Spatial processing

4. **High Performance**
   - 2,000+ QPS
   - Parallel processing
   - Optimized for large datasets
   - Compression algorithms

5. **Enterprise Features**
   - Advanced security
   - Multi-tenancy
   - Disaster recovery
   - High availability

### Weaknesses

1. **Cost**
   - Commercial license required
   - Higher infrastructure costs
   - Pricing based on memory
   - Enterprise-focused pricing

2. **Complexity**
   - Steeper learning curve
   - Specialized administration
   - SAP-specific tools
   - Less community support

3. **Vendor Lock-in**
   - Proprietary technology
   - Limited cloud options (SAP Cloud)
   - Migration challenges
   - SAP ecosystem dependency

4. **Memory Requirements**
   - High RAM needs
   - All data in memory
   - Expensive for large datasets
   - Careful capacity planning needed

### Best For

- **Graph-heavy workloads** (lineage, dependencies)
- **Real-time analytics** requirements
- **SAP ecosystem** integration
- **Enterprise deployments** with budget
- **High-performance** requirements
- **Complex analytical** queries

### Configuration Recommendations

```sql
-- Enable graph workspace
CREATE GRAPH WORKSPACE lineage_workspace;

-- Optimize for lineage queries
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
  SET ('graph', 'max_graph_workspace_size') = '10000000000'
  WITH RECONFIGURE;

-- Configure column store
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
  SET ('cs', 'compression_type') = 'DEFAULT'
  WITH RECONFIGURE;

-- Set memory limits
ALTER SYSTEM ALTER CONFIGURATION ('global.ini', 'SYSTEM')
  SET ('memorymanager', 'global_allocation_limit') = '90000'
  WITH RECONFIGURE;
```

### Performance Tuning

```sql
-- Create graph workspace for lineage
CREATE GRAPH WORKSPACE lineage_graph
  EDGE TABLE lineage_edges
    SOURCE COLUMN source_id
    TARGET COLUMN target_id
    KEY COLUMN edge_id
  VERTEX TABLE datasets
    KEY COLUMN id;

-- Optimize column store tables
ALTER TABLE datasets UNLOAD;
MERGE DELTA OF datasets;

-- Monitor performance
SELECT * FROM M_EXPENSIVE_STATEMENTS
WHERE DURATION > 1000
ORDER BY DURATION DESC;
```

---

## SQLite

### Overview

SQLite is a self-contained, serverless, zero-configuration database engine designed for embedded use.

### Strengths

1. **Zero Configuration**
   - No server to install
   - No configuration needed
   - Single file database
   - Embedded in application

2. **Development Speed**
   - Instant setup
   - In-memory mode
   - Fast tests
   - Local development

3. **Performance**
   - 15,000+ QPS (in-memory)
   - Low latency
   - Minimal overhead
   - Efficient for read-heavy workloads

4. **Portability**
   - Cross-platform
   - No dependencies
   - Easy to distribute
   - Simple backups (copy file)

5. **Public Domain**
   - No license restrictions
   - Free for any use
   - Well-documented
   - Stable API

### Weaknesses

1. **Concurrency Limitations**
   - Single writer at a time
   - Database-level locking
   - Not suitable for high-write scenarios
   - Concurrent reads OK

2. **No Network Access**
   - Same-machine only
   - No distributed queries
   - No replication
   - No clustering

3. **Limited Scalability**
   - Vertical scaling only
   - Performance degrades with size
   - No parallel queries
   - Table size limits

4. **Feature Gaps**
   - No user management
   - Limited ALTER TABLE
   - No stored procedures
   - No graph engine

### Best For

- **Development and testing**
- **Local applications**
- **Embedded systems**
- **Small to medium datasets** (<100GB)
- **Read-heavy workloads**
- **Single-user scenarios**
- **CI/CD pipelines** (fast tests)

### Configuration Recommendations

```sql
-- Optimize SQLite for nMetaData
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging
PRAGMA synchronous = NORMAL;         -- Balance safety/performance
PRAGMA cache_size = -64000;          -- 64MB cache
PRAGMA temp_store = MEMORY;          -- Temp tables in memory
PRAGMA mmap_size = 268435456;        -- 256MB memory-mapped I/O
PRAGMA page_size = 4096;             -- Match OS page size

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Load useful extensions
.load json1
.load fts5
```

### Performance Tuning

```sql
-- Create indexes
CREATE INDEX idx_datasets_namespace ON datasets(namespace);
CREATE INDEX idx_jobs_name ON jobs(namespace, name);

-- Use ANALYZE
ANALYZE;

-- Batch inserts in transactions
BEGIN TRANSACTION;
INSERT INTO datasets VALUES (...);
INSERT INTO datasets VALUES (...);
-- ... more inserts
COMMIT;

-- In-memory mode for tests
:memory:  -- Special database name
```

---

## Decision Tree

Use this flowchart to select the appropriate database:

```
START
  |
  v
Is this for production? ──NO──> Use SQLite (Development/Testing)
  |
  YES
  |
  v
Is graph/lineage performance critical? ──YES──> Budget for enterprise?
  |                                               |
  NO                                              YES ──> Use SAP HANA
  |                                               |
  v                                               NO
Do you need > 10,000 QPS?                        |
  |                                               v
  YES ──> Use PostgreSQL                    Use PostgreSQL
  |       (with read replicas)              (consider migration plan)
  NO
  |
  v
Use PostgreSQL
(Standard choice)
```

### Detailed Decision Criteria

#### Choose PostgreSQL if:

- ✅ General purpose metadata storage
- ✅ Standard OLTP workloads
- ✅ Budget constraints
- ✅ Open-source requirement
- ✅ Mature ecosystem needed
- ✅ Cloud-agnostic deployment
- ✅ Graph performance acceptable (20ms lineage)

#### Choose SAP HANA if:

- ✅ Graph queries are critical (need <10ms lineage)
- ✅ Real-time analytics required
- ✅ Enterprise budget available
- ✅ SAP ecosystem integration
- ✅ In-memory performance needed
- ✅ Complex analytical queries
- ✅ Large-scale deployment (1000s of datasets)

#### Choose SQLite if:

- ✅ Development environment
- ✅ Testing/CI pipeline
- ✅ Embedded application
- ✅ Single-user scenario
- ✅ Small dataset (<10GB)
- ✅ Read-heavy workload
- ✅ Zero administration desired

---

## Workload Analysis

### OLTP (Online Transaction Processing)

**Characteristics:**
- High volume of short transactions
- Insert/update/delete heavy
- Concurrent users
- Low latency requirements

**Recommendation:** **PostgreSQL** (or SAP HANA)
- Excellent MVCC for concurrency
- Optimized for short transactions
- Mature transaction management

### OLAP (Online Analytical Processing)

**Characteristics:**
- Complex queries
- Large data scans
- Aggregations and analytics
- Reporting and BI

**Recommendation:** **SAP HANA**
- Column store optimization
- In-memory processing
- Parallel execution
- Advanced analytics

### Mixed Workload (HTAP)

**Characteristics:**
- Both OLTP and OLAP
- Real-time analytics
- Operational reporting

**Recommendation:** **SAP HANA** > **PostgreSQL**
- HANA: Native HTAP support
- PostgreSQL: Good enough for most cases

### Graph Workload

**Characteristics:**
- Lineage queries
- Dependency analysis
- Impact assessment
- Relationship traversal

**Recommendation:** **SAP HANA** (4x faster)
- Native graph engine
- Optimized algorithms
- Graph workspaces

---

## Migration Considerations

### Migration Paths

All database combinations are supported:

| From | To | Complexity | Downtime |
|------|----|-----------| |
| SQLite | PostgreSQL | Low | Minutes |
| SQLite | SAP HANA | Low | Minutes |
| PostgreSQL | SAP HANA | Medium | Hours |
| SAP HANA | PostgreSQL | Medium | Hours |
| PostgreSQL | SQLite | Low | Minutes |
| SAP HANA | SQLite | Low | Minutes |

### Migration Process

1. **Export Data** from source database
2. **Transform Schema** (if needed)
3. **Import Data** to target database
4. **Validate** data integrity
5. **Update Connection** strings
6. **Test Application** functionality
7. **Monitor Performance**

### Example Migration

```bash
# PostgreSQL to HANA
./scripts/migrate_database.sh \
  --from postgresql://localhost/nmeta \
  --to hana://hanacloud.sap/nmeta \
  --validate \
  --parallel 4

# Expected output:
# Exporting from PostgreSQL... 100%
# Transforming schema... Done
# Importing to HANA... 100%
# Validating data... 10000/10000 rows OK
# Migration complete: 15 minutes
```

### Zero-Downtime Migration

For production systems:

1. Set up target database
2. Initial data sync
3. Enable change data capture (CDC)
4. Continuous replication
5. Switch application (blue-green deployment)
6. Verify and rollback capability

---

## Cost Analysis

### Total Cost of Ownership (TCO)

#### PostgreSQL

**Infrastructure:**
- Compute: $200-500/month (cloud VM)
- Storage: $100-200/month (500GB SSD)
- Backup: $50-100/month
- **Total: $350-800/month**

**Operational:**
- DBA time: Low (automated tools)
- Maintenance: Moderate
- Monitoring: Free (Prometheus/Grafana)
- **License: $0 (open source)**

**Annual TCO: $4,200-9,600**

#### SAP HANA

**Infrastructure:**
- Compute: $1,000-3,000/month (SAP Cloud)
- Storage: $300-600/month
- Backup: $200-400/month
- **Total: $1,500-4,000/month**

**Operational:**
- DBA time: High (specialized skills)
- Maintenance: High
- Monitoring: SAP tools (included)
- **License: Included in cloud pricing**

**Annual TCO: $18,000-48,000+**

#### SQLite

**Infrastructure:**
- Compute: $0 (embedded)
- Storage: $0 (local disk)
- Backup: $0 (file copy)
- **Total: $0**

**Operational:**
- DBA time: Minimal
- Maintenance: Minimal
- Monitoring: Basic
- **License: $0 (public domain)**

**Annual TCO: $0-1,000** (developer time)

### ROI Analysis

**When HANA makes sense:**
- Graph queries >10,000/day
- Lineage latency critical (<10ms required)
- Enterprise scale (1,000+ datasets)
- Real-time analytics required

**Break-even calculation:**
- HANA premium: ~$30,000/year
- Performance gain: 4x on graph queries
- If graph queries save 1 minute per day per data engineer
- Team of 10 engineers: 10 * 365 * 1 min = 60 hours/year
- At $150/hour: $9,000 savings
- Need ~30 engineers for break-even on performance alone

**Conclusion:** HANA justified for:
- Large teams (>30 data engineers)
- High-frequency lineage queries
- Enterprise requirements
- SAP ecosystem

---

## Real-World Use Cases

### Use Case 1: Startup (50 datasets)

**Requirements:**
- 50 datasets
- 5 data engineers
- Budget: $500/month
- Growth expected

**Recommendation: PostgreSQL**
- Cost-effective
- Room to grow
- Mature tooling
- Easy to find expertise

**Configuration:**
```yaml
database: postgresql
instance: Small (2 vCPU, 8GB RAM)
storage: 100GB SSD
backup: Daily
monitoring: Prometheus + Grafana
```

### Use Case 2: Enterprise (5,000 datasets)

**Requirements:**
- 5,000 datasets
- 100 data engineers
- Lineage queries: 50,000/day
- Budget: $5,000/month

**Recommendation: SAP HANA**
- Graph performance critical
- High query volume
- Enterprise scale
- ROI positive

**Configuration:**
```yaml
database: sap_hana
instance: Large (32 vCPU, 256GB RAM)
storage: 1TB column store
backup: Continuous
monitoring: SAP Cloud Platform
graph_workspace: Enabled
```

### Use Case 3: CI/CD Pipeline

**Requirements:**
- Fast test execution
- Isolated environments
- Disposable databases
- Cost: Free

**Recommendation: SQLite**
- In-memory mode
- No setup time
- Fast tests
- No infrastructure

**Configuration:**
```yaml
database: sqlite
mode: :memory:
location: In-memory
backup: Not needed (ephemeral)
parallelism: Per test process
```

### Use Case 4: SaaS Platform (Multi-tenant)

**Requirements:**
- 100+ tenants
- Isolation required
- Variable load
- Cost efficiency

**Recommendation: PostgreSQL**
- Row-level security
- Schema per tenant
- Connection pooling
- Cost-effective scaling

**Configuration:**
```yaml
database: postgresql
architecture: Multi-tenant
isolation: Schema per tenant
pooling: PgBouncer (1000 connections)
scaling: Horizontal (read replicas)
```

---

## Summary Recommendations

### Default Choice: PostgreSQL

For most users, **PostgreSQL is the recommended starting point**:
- Excellent all-around performance
- Mature ecosystem
- Cost-effective
- Easy migration path to HANA if needed

### When to Use HANA

Upgrade to **SAP HANA** when:
- Graph query performance becomes bottleneck
- Real-time analytics required
- Enterprise budget available
- Scale demands it (>1,000 datasets, >10 engineers)

### When to Use SQLite

Use **SQLite** for:
- Development environments
- Testing and CI/CD
- Embedded applications
- Learning and prototyping

### Migration Strategy

1. **Start with SQLite** for development
2. **Deploy PostgreSQL** for production
3. **Monitor graph query performance**
4. **Migrate to HANA** if justified by ROI

---

## Quick Reference

| Scenario | Database | Reason |
|----------|----------|--------|
| **Getting Started** | SQLite | Zero config, fast development |
| **Production (default)** | PostgreSQL | Best balance of features/cost |
| **High-scale enterprise** | SAP HANA | Performance at scale |
| **Graph-heavy** | SAP HANA | 4x faster lineage queries |
| **Budget-constrained** | PostgreSQL | Open source, low TCO |
| **SAP ecosystem** | SAP HANA | Native integration |
| **Testing/CI** | SQLite | Fast, isolated, disposable |
| **Multi-tenant SaaS** | PostgreSQL | RLS, cost-effective scaling |
| **Real-time analytics** | SAP HANA | In-memory, column store |
| **Small team (<10)** | PostgreSQL | Easy management |
| **Large team (>50)** | SAP HANA | ROI positive |

---

**Version History:**
- v1.0 (2026-01-20): Initial comprehensive database selection guide

**Last Updated:** January 20, 2026
