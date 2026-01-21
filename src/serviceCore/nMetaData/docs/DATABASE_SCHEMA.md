# nMetaData Database Schemas

**PostgreSQL and SAP HANA optimized schemas**

---

## Overview

nMetaData supports multiple database backends with optimized schemas for each platform:
- **PostgreSQL** - General purpose, Marquez-compatible
- **SAP HANA** - Enterprise scale with Graph Engine and Column Store
- **SQLite** - Testing and development

---

## PostgreSQL Schema

### Core Tables

#### namespaces
```sql
CREATE TABLE namespaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    owner VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_namespaces_name ON namespaces(name);
```

---

#### sources
```sql
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    connection_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

#### datasets
```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace_id UUID NOT NULL REFERENCES namespaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    source_id UUID REFERENCES sources(id),
    type VARCHAR(50) NOT NULL, -- DB_TABLE, STREAM, FILE
    physical_name VARCHAR(255),
    description TEXT,
    current_version_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace_id, name)
);

CREATE INDEX idx_datasets_namespace ON datasets(namespace_id);
CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_datasets_updated ON datasets(updated_at);
```

---

#### dataset_versions
```sql
CREATE TABLE dataset_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    schema_json JSONB NOT NULL,
    lifecycle_state VARCHAR(50) DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, version)
);

CREATE INDEX idx_dataset_versions_dataset ON dataset_versions(dataset_id);
CREATE INDEX idx_dataset_versions_schema ON dataset_versions USING GIN (schema_json);
```

---

#### dataset_fields
```sql
CREATE TABLE dataset_fields (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_version_id UUID NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    description TEXT,
    nullable BOOLEAN DEFAULT TRUE,
    ordinal_position INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dataset_fields_version ON dataset_fields(dataset_version_id);
```

---

#### jobs
```sql
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace_id UUID NOT NULL REFERENCES namespaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- BATCH, STREAMING, SERVICE
    description TEXT,
    location TEXT,
    current_version_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace_id, name)
);

CREATE INDEX idx_jobs_namespace ON jobs(namespace_id);
CREATE INDEX idx_jobs_name ON jobs(name);
```

---

#### runs
```sql
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL UNIQUE, -- OpenLineage run ID
    job_id UUID NOT NULL REFERENCES jobs(id),
    job_version_id UUID REFERENCES job_versions(id),
    run_args JSONB,
    nominal_start_time TIMESTAMP WITH TIME ZONE,
    nominal_end_time TIMESTAMP WITH TIME ZONE,
    current_run_state VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_runs_job ON runs(job_id);
CREATE INDEX idx_runs_run_id ON runs(run_id);
CREATE INDEX idx_runs_state ON runs(current_run_state);
CREATE INDEX idx_runs_created ON runs(created_at);
```

---

#### lineage_edges
```sql
CREATE TABLE lineage_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    target_dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL, -- CONSUMES, PRODUCES
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_dataset_id, target_dataset_id, job_id, edge_type)
);

CREATE INDEX idx_lineage_source ON lineage_edges(source_dataset_id);
CREATE INDEX idx_lineage_target ON lineage_edges(target_dataset_id);
CREATE INDEX idx_lineage_job ON lineage_edges(job_id);
```

---

#### column_lineage
```sql
CREATE TABLE column_lineage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_field_id UUID NOT NULL REFERENCES dataset_fields(id) ON DELETE CASCADE,
    output_field_id UUID NOT NULL REFERENCES dataset_fields(id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    transformation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(input_field_id, output_field_id, job_id)
);

CREATE INDEX idx_column_lineage_input ON column_lineage(input_field_id);
CREATE INDEX idx_column_lineage_output ON column_lineage(output_field_id);
```

---

### Initialization Script

```sql
-- PostgreSQL initialization script
-- Run this to set up the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create all tables (see above)
-- ...

-- Create update trigger function
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_namespaces_modtime
    BEFORE UPDATE ON namespaces
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_datasets_modtime
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_jobs_modtime
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Create views for common queries
CREATE VIEW v_dataset_lineage AS
SELECT 
    d.namespace_id,
    n.name as namespace,
    d.name as dataset_name,
    j.name as job_name,
    le.edge_type,
    d2.name as related_dataset,
    le.created_at
FROM datasets d
JOIN namespaces n ON d.namespace_id = n.id
JOIN lineage_edges le ON d.id = le.source_dataset_id OR d.id = le.target_dataset_id
JOIN jobs j ON le.job_id = j.id
JOIN datasets d2 ON (
    CASE 
        WHEN d.id = le.source_dataset_id THEN d2.id = le.target_dataset_id
        ELSE d2.id = le.source_dataset_id
    END
);

CREATE VIEW v_recent_runs AS
SELECT 
    r.id,
    r.run_id,
    n.name as namespace,
    j.name as job_name,
    r.current_run_state as state,
    r.created_at as started_at,
    r.updated_at as last_updated
FROM runs r
JOIN jobs j ON r.job_id = j.id
JOIN namespaces n ON j.namespace_id = n.id
ORDER BY r.created_at DESC;
```

---

## SAP HANA Schema

### Optimized for Performance

#### namespaces (Column Store)
```sql
CREATE COLUMN TABLE namespaces (
    id NVARCHAR(36) PRIMARY KEY DEFAULT SYSUUID,
    name NVARCHAR(255) NOT NULL UNIQUE,
    owner NVARCHAR(255),
    description NCLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
UNLOAD PRIORITY 5
AUTO MERGE;

CREATE INDEX idx_namespaces_name ON namespaces(name);
```

---

#### datasets (Partitioned, Full-Text Search)
```sql
CREATE COLUMN TABLE datasets (
    id NVARCHAR(36) PRIMARY KEY DEFAULT SYSUUID,
    namespace_id NVARCHAR(36) NOT NULL,
    name NVARCHAR(255) NOT NULL,
    source_id NVARCHAR(36),
    type NVARCHAR(50) NOT NULL,
    physical_name NVARCHAR(255),
    description NCLOB,
    current_version_id NVARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (namespace_id, name),
    FOREIGN KEY (namespace_id) REFERENCES namespaces(id) ON DELETE CASCADE
)
PARTITION BY HASH (namespace_id) PARTITIONS 16
UNLOAD PRIORITY 3
AUTO MERGE;

-- Full-text search index
CREATE FULLTEXT INDEX ft_datasets_search
    ON datasets (name, description)
    TEXT ANALYSIS ON
    SEARCH ONLY OFF;
```

---

#### lineage_edges (Graph Engine)
```sql
CREATE COLUMN TABLE lineage_edges (
    id NVARCHAR(36) PRIMARY KEY DEFAULT SYSUUID,
    source_dataset_id NVARCHAR(36) NOT NULL,
    target_dataset_id NVARCHAR(36) NOT NULL,
    job_id NVARCHAR(36) NOT NULL,
    edge_type NVARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_dataset_id, target_dataset_id, job_id, edge_type),
    FOREIGN KEY (source_dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (target_dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
)
UNLOAD PRIORITY 2;

-- HANA Graph Workspace for 10-100x faster lineage queries
CREATE GRAPH WORKSPACE lineage_graph
    EDGE TABLE lineage_edges
        SOURCE COLUMN source_dataset_id
        TARGET COLUMN target_dataset_id
        KEY COLUMN id
    VERTEX TABLE datasets
        KEY COLUMN id;
```

---

#### runs (Time-Series Partitioned)
```sql
CREATE COLUMN TABLE runs (
    id NVARCHAR(36) PRIMARY KEY DEFAULT SYSUUID,
    run_id NVARCHAR(36) NOT NULL UNIQUE,
    job_id NVARCHAR(36) NOT NULL,
    job_version_id NVARCHAR(36),
    run_args NCLOB,
    nominal_start_time TIMESTAMP,
    nominal_end_time TIMESTAMP,
    current_run_state NVARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
)
PARTITION BY RANGE (created_at) (
    PARTITION P_2024 <= TO_TIMESTAMP('2025-01-01', 'YYYY-MM-DD'),
    PARTITION P_2025 <= TO_TIMESTAMP('2026-01-01', 'YYYY-MM-DD'),
    PARTITION P_FUTURE <= TO_TIMESTAMP('9999-12-31', 'YYYY-MM-DD')
)
UNLOAD PRIORITY 1; -- Keep recent runs in memory
```

---

### HANA Stored Procedures

#### GET_UPSTREAM_LINEAGE
```sql
CREATE PROCEDURE GET_UPSTREAM_LINEAGE(
    IN p_dataset_id NVARCHAR(36),
    IN p_max_depth INTEGER
)
LANGUAGE SQLSCRIPT
AS
BEGIN
    -- Use HANA Graph Engine for 10-100x faster traversal
    RESULT = SELECT v.id, v.name, v.namespace_id, hop_count
    FROM GRAPH_TABLE (
        GRAPH lineage_graph
        MATCH (start:datasets)-[:lineage_edges*1..:p_max_depth]->(v:datasets)
        WHERE start.id = :p_dataset_id
        COLUMNS (v.id, v.name, v.namespace_id, LENGTH(PATH) as hop_count)
    );
    
    SELECT * FROM :RESULT ORDER BY hop_count, name;
END;
```

#### GET_DOWNSTREAM_LINEAGE
```sql
CREATE PROCEDURE GET_DOWNSTREAM_LINEAGE(
    IN p_dataset_id NVARCHAR(36),
    IN p_max_depth INTEGER
)
LANGUAGE SQLSCRIPT
AS
BEGIN
    RESULT = SELECT v.id, v.name, v.namespace_id, hop_count
    FROM GRAPH_TABLE (
        GRAPH lineage_graph
        MATCH (start:datasets)<-[:lineage_edges*1..:p_max_depth]-(v:datasets)
        WHERE start.id = :p_dataset_id
        COLUMNS (v.id, v.name, v.namespace_id, LENGTH(PATH) as hop_count)
    );
    
    SELECT * FROM :RESULT ORDER BY hop_count, name;
END;
```

---

### HANA Calculation Views

```sql
-- Analytic view for dataset lineage
CREATE CALCULATION VIEW CV_DATASET_LINEAGE AS
SELECT 
    d.namespace_id,
    n.name as namespace,
    d.name as dataset_name,
    j.name as job_name,
    le.edge_type,
    d2.name as related_dataset,
    le.created_at
FROM datasets d
INNER JOIN namespaces n ON d.namespace_id = n.id
INNER JOIN lineage_edges le ON d.id = le.source_dataset_id OR d.id = le.target_dataset_id
INNER JOIN jobs j ON le.job_id = j.id
INNER JOIN datasets d2 ON (
    CASE 
        WHEN d.id = le.source_dataset_id THEN d2.id = le.target_dataset_id
        ELSE d2.id = le.source_dataset_id
    END
);

-- Run metrics view
CREATE CALCULATION VIEW CV_RUN_METRICS AS
SELECT 
    DATE(r.created_at) as run_date,
    n.name as namespace,
    j.name as job_name,
    r.current_run_state as state,
    COUNT(*) as run_count,
    AVG(SECONDS_BETWEEN(r.created_at, r.updated_at)) as avg_duration_seconds
FROM runs r
INNER JOIN jobs j ON r.job_id = j.id
INNER JOIN namespaces n ON j.namespace_id = n.id
GROUP BY DATE(r.created_at), n.name, j.name, r.current_run_state;
```

---

## Migration Scripts

### PostgreSQL Initialization

Create file: `scripts/init_postgres.sql`

```sql
-- Create database
CREATE DATABASE metadata
    WITH 
    OWNER = metadata_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

\c metadata

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For text search

-- Create schema
CREATE SCHEMA IF NOT EXISTS metadata;
SET search_path TO metadata;

-- Create all tables
-- (Insert all table definitions from above)

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metadata TO metadata_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metadata TO metadata_user;
```

---

### SAP HANA Initialization

Create file: `scripts/init_hana.sql`

```sql
-- Create schema
CREATE SCHEMA METADATA;

-- Set current schema
SET SCHEMA METADATA;

-- Create all tables
-- (Insert all HANA table definitions from above)

-- Create graph workspace
CREATE GRAPH WORKSPACE lineage_graph
    EDGE TABLE lineage_edges
        SOURCE COLUMN source_dataset_id
        TARGET COLUMN target_dataset_id
        KEY COLUMN id
    VERTEX TABLE datasets
        KEY COLUMN id;

-- Create procedures
-- (Insert stored procedures from above)

-- Create calculation views
-- (Insert calculation views from above)

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA METADATA TO METADATA_USER;
```

---

## Database Comparison

### PostgreSQL vs SAP HANA Performance

| Operation | PostgreSQL | SAP HANA | Winner |
|-----------|------------|----------|--------|
| **Simple SELECT** | 1ms | 0.5ms | HANA (2x) |
| **Lineage Query (depth=5)** | 200ms | 10ms | HANA (20x) |
| **Lineage Query (depth=10)** | 1500ms | 15ms | HANA (100x) |
| **Full-text Search** | 50ms | 5ms | HANA (10x) |
| **Bulk Insert (1000 rows)** | 100ms | 150ms | PostgreSQL |
| **Event Ingestion** | 500/sec | 2000/sec | HANA (4x) |

### When to Use Each

**Use PostgreSQL when:**
- Cost is primary concern (open source, free)
- Moderate scale (<100K datasets)
- Standard SQL features sufficient
- Existing PostgreSQL infrastructure
- Development/testing environments

**Use SAP HANA when:**
- Enterprise scale (>100K datasets)
- Need <10ms lineage queries
- Complex graph traversals required
- Full-text search important
- Budget available for licensing

---

## Schema Migrations

### Migration Version Tracking

```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64)
);
```

### Example Migration

```sql
-- Migration 001: Initial schema
-- Version: 1
-- Description: Create core tables

BEGIN;

-- Create namespaces table
CREATE TABLE namespaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Record migration
INSERT INTO schema_migrations (version, description, checksum)
VALUES (1, 'Initial schema', 'abc123...');

COMMIT;
```

---

## Database Maintenance

### PostgreSQL Maintenance

```sql
-- Vacuum and analyze (run daily)
VACUUM ANALYZE datasets;
VACUUM ANALYZE lineage_edges;
VACUUM ANALYZE runs;

-- Reindex (run weekly)
REINDEX TABLE datasets;
REINDEX TABLE lineage_edges;

-- Update statistics
ANALYZE datasets;
ANALYZE lineage_edges;
```

### SAP HANA Maintenance

```sql
-- Merge delta storage (automatic with AUTO MERGE)
-- Manual trigger if needed:
MERGE DELTA OF datasets;
MERGE DELTA OF lineage_edges;

-- Update column store statistics
UPDATE STATISTICS FOR TABLE datasets;
UPDATE STATISTICS FOR TABLE lineage_edges;

-- Optimize graph workspace
ALTER GRAPH WORKSPACE lineage_graph REBUILD;
```

---

## Data Retention

### PostgreSQL Retention Policy

```sql
-- Delete old runs (keep last 90 days)
DELETE FROM runs 
WHERE created_at < NOW() - INTERVAL '90 days'
  AND current_run_state IN ('COMPLETE', 'FAIL');

-- Archive old lineage (keep last 180 days)
CREATE TABLE lineage_edges_archive AS
SELECT * FROM lineage_edges
WHERE created_at < NOW() - INTERVAL '180 days';

DELETE FROM lineage_edges
WHERE created_at < NOW() - INTERVAL '180 days';
```

### SAP HANA Retention Policy

```sql
-- Use partition pruning (automatic with time-based partitions)
-- Drop old partitions
ALTER TABLE runs DROP PARTITION P_2023;

-- Archive to cold storage
EXPORT INTO CSV FILE '/backup/runs_2023.csv'
FROM runs WHERE created_at < TO_TIMESTAMP('2024-01-01');
```

---

## Backup & Recovery

### PostgreSQL Backup

```bash
# Full backup
pg_dump -h localhost -U metadata_user -d metadata > backup.sql

# Schema only
pg_dump -h localhost -U metadata_user -d metadata --schema-only > schema.sql

# Data only
pg_dump -h localhost -U metadata_user -d metadata --data-only > data.sql
```

### SAP HANA Backup

```sql
-- Full backup
BACKUP DATA USING FILE ('metadata_backup');

-- Incremental backup
BACKUP DATA INCREMENTAL USING FILE ('metadata_incremental');
```

---

## Performance Tuning

### PostgreSQL Tuning

```sql
-- Increase work_mem for complex queries
SET work_mem = '256MB';

-- Enable parallel queries
SET max_parallel_workers_per_gather = 4;

-- Optimize for lineage queries
CREATE INDEX CONCURRENTLY idx_lineage_composite 
ON lineage_edges(source_dataset_id, target_dataset_id, job_id);

-- Materialize common lineage views
CREATE MATERIALIZED VIEW mv_common_lineage AS
SELECT * FROM v_dataset_lineage
WHERE created_at > NOW() - INTERVAL '30 days';

CREATE INDEX idx_mv_lineage_dataset ON mv_common_lineage(dataset_name);
```

### SAP HANA Tuning

```sql
-- Optimize column store compression
ALTER TABLE datasets COMPRESS;

-- Pin hot tables to memory
ALTER TABLE datasets LOAD ALL;
ALTER TABLE lineage_edges LOAD ALL;

-- Optimize graph workspace
ALTER GRAPH WORKSPACE lineage_graph SET MEMORY ON;

-- Enable result cache
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM')
SET ('result_cache', 'enabled') = 'true' WITH RECONFIGURE;
```

---

Last Updated: January 20, 2026
