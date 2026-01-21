-- KV Cache Database Schema for PostgreSQL
-- This schema stores metadata and versioning for the database-backed KV cache tier

-- ============================================================================
-- Extensions
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For text search optimization

-- ============================================================================
-- Main Metadata Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS kv_cache_metadata (
    id BIGSERIAL PRIMARY KEY,
    
    -- Cache entry identification
    model_id VARCHAR(255) NOT NULL,
    layer INTEGER NOT NULL,
    token_start INTEGER NOT NULL,
    token_end INTEGER NOT NULL,
    
    -- Compression information
    compression_algorithm VARCHAR(50) NOT NULL DEFAULT 'fp16',
    compressed_size BIGINT NOT NULL,
    original_size BIGINT NOT NULL,
    compression_ratio FLOAT GENERATED ALWAYS AS (
        CASE 
            WHEN compressed_size > 0 
            THEN original_size::FLOAT / compressed_size::FLOAT 
            ELSE 0 
        END
    ) STORED,
    
    -- Storage location
    storage_backend VARCHAR(50) NOT NULL DEFAULT 'dragonfly', -- dragonfly, postgres, qdrant, ssd
    storage_key VARCHAR(500), -- Key in DragonflyDB or path in storage
    vector_id UUID, -- ID in Qdrant
    
    -- Timestamps and access tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    
    -- Versioning
    version INTEGER DEFAULT 1,
    parent_version_id BIGINT REFERENCES kv_cache_metadata(id),
    
    -- Metadata (JSON for flexibility)
    metadata JSONB DEFAULT '{}'::JSONB,
    
    -- Constraints
    CONSTRAINT unique_cache_entry UNIQUE (model_id, layer, token_start, token_end, version),
    CONSTRAINT valid_token_range CHECK (token_end > token_start),
    CONSTRAINT valid_sizes CHECK (compressed_size > 0 AND original_size > 0)
);

-- ============================================================================
-- Version History Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS kv_cache_versions (
    id BIGSERIAL PRIMARY KEY,
    cache_id BIGINT NOT NULL REFERENCES kv_cache_metadata(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    
    -- Change tracking
    changed_by VARCHAR(255),
    change_reason TEXT,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Before/after sizes for comparison
    size_before BIGINT,
    size_after BIGINT,
    
    -- Metadata snapshot
    metadata_snapshot JSONB,
    
    CONSTRAINT unique_version UNIQUE (cache_id, version)
);

-- ============================================================================
-- Access Log Table (Optional - for analytics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kv_cache_access_log (
    id BIGSERIAL PRIMARY KEY,
    cache_id BIGINT NOT NULL REFERENCES kv_cache_metadata(id) ON DELETE CASCADE,
    
    -- Access details
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_type VARCHAR(50) NOT NULL, -- 'read', 'write', 'evict'
    latency_us BIGINT, -- Microseconds
    
    -- Request context
    request_id UUID,
    model_id VARCHAR(255),
    
    -- Performance metrics
    cache_hit BOOLEAN DEFAULT TRUE,
    tier_accessed VARCHAR(50), -- 'dragonfly', 'postgres', 'qdrant', 'ssd'
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::JSONB
);

-- ============================================================================
-- Statistics Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS kv_cache_stats (
    id BIGSERIAL PRIMARY KEY,
    
    -- Time bucket (for aggregation)
    bucket_time TIMESTAMP WITH TIME ZONE NOT NULL,
    bucket_interval VARCHAR(20) NOT NULL, -- 'minute', 'hour', 'day'
    
    -- Model/layer dimensions
    model_id VARCHAR(255),
    layer INTEGER,
    
    -- Aggregate statistics
    total_reads BIGINT DEFAULT 0,
    total_writes BIGINT DEFAULT 0,
    total_evictions BIGINT DEFAULT 0,
    
    cache_hits BIGINT DEFAULT 0,
    cache_misses BIGINT DEFAULT 0,
    hit_rate FLOAT GENERATED ALWAYS AS (
        CASE 
            WHEN (cache_hits + cache_misses) > 0 
            THEN cache_hits::FLOAT / (cache_hits + cache_misses)::FLOAT 
            ELSE 0 
        END
    ) STORED,
    
    -- Latency statistics (microseconds)
    avg_latency_us BIGINT,
    p50_latency_us BIGINT,
    p95_latency_us BIGINT,
    p99_latency_us BIGINT,
    
    -- Size statistics
    total_compressed_bytes BIGINT DEFAULT 0,
    total_original_bytes BIGINT DEFAULT 0,
    avg_compression_ratio FLOAT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::JSONB,
    
    CONSTRAINT unique_stats_bucket UNIQUE (bucket_time, bucket_interval, model_id, layer)
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX idx_cache_model_layer ON kv_cache_metadata(model_id, layer, token_start);
CREATE INDEX idx_cache_accessed ON kv_cache_metadata(accessed_at DESC);
CREATE INDEX idx_cache_created ON kv_cache_metadata(created_at DESC);
CREATE INDEX idx_cache_version ON kv_cache_metadata(model_id, layer, version);

-- Access pattern indexes
CREATE INDEX idx_cache_access_count ON kv_cache_metadata(access_count DESC);
CREATE INDEX idx_cache_storage_backend ON kv_cache_metadata(storage_backend);

-- Vector ID lookup
CREATE INDEX idx_cache_vector_id ON kv_cache_metadata(vector_id) WHERE vector_id IS NOT NULL;

-- JSONB indexes for metadata search
CREATE INDEX idx_cache_metadata_gin ON kv_cache_metadata USING GIN (metadata);

-- Version history indexes
CREATE INDEX idx_versions_cache_id ON kv_cache_versions(cache_id, version DESC);
CREATE INDEX idx_versions_changed_at ON kv_cache_versions(changed_at DESC);

-- Access log indexes
CREATE INDEX idx_access_log_cache_id ON kv_cache_access_log(cache_id);
CREATE INDEX idx_access_log_timestamp ON kv_cache_access_log(accessed_at DESC);
CREATE INDEX idx_access_log_request ON kv_cache_access_log(request_id);
CREATE INDEX idx_access_log_tier ON kv_cache_access_log(tier_accessed);

-- Statistics indexes
CREATE INDEX idx_stats_bucket ON kv_cache_stats(bucket_time DESC, bucket_interval);
CREATE INDEX idx_stats_model ON kv_cache_stats(model_id, bucket_time DESC);

-- ============================================================================
-- Partitioning (for large-scale deployments)
-- ============================================================================

-- Partition by model_id for horizontal scaling
-- Note: This requires PostgreSQL 10+ with declarative partitioning

-- Example: Create partitioned table (uncomment if needed)
/*
CREATE TABLE kv_cache_metadata_partitioned (
    LIKE kv_cache_metadata INCLUDING ALL
) PARTITION BY LIST (model_id);

-- Create partitions per model
CREATE TABLE kv_cache_metadata_model_llama PARTITION OF kv_cache_metadata_partitioned
    FOR VALUES IN ('Llama-3.3-70B-Instruct');

CREATE TABLE kv_cache_metadata_model_default PARTITION OF kv_cache_metadata_partitioned
    DEFAULT;
*/

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Function to update accessed_at and access_count
CREATE OR REPLACE FUNCTION update_cache_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.accessed_at = NOW();
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger on SELECT (via application-level UPDATE)
CREATE TRIGGER trigger_update_access
    BEFORE UPDATE ON kv_cache_metadata
    FOR EACH ROW
    WHEN (OLD.accessed_at IS DISTINCT FROM NEW.accessed_at)
    EXECUTE FUNCTION update_cache_access();

-- Function to create version history
CREATE OR REPLACE FUNCTION create_version_history()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.version <> NEW.version THEN
        INSERT INTO kv_cache_versions (
            cache_id, version, size_before, size_after, metadata_snapshot
        ) VALUES (
            NEW.id, NEW.version, OLD.compressed_size, NEW.compressed_size, OLD.metadata
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_version_history
    AFTER UPDATE ON kv_cache_metadata
    FOR EACH ROW
    EXECUTE FUNCTION create_version_history();

-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function to clean old access logs (keep last 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_access_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM kv_cache_access_log
    WHERE accessed_at < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate statistics
CREATE OR REPLACE FUNCTION aggregate_cache_stats(
    p_interval VARCHAR DEFAULT 'hour'
)
RETURNS VOID AS $$
DECLARE
    bucket_start TIMESTAMP WITH TIME ZONE;
    bucket_interval INTERVAL;
BEGIN
    -- Determine bucket interval
    CASE p_interval
        WHEN 'minute' THEN bucket_interval := INTERVAL '1 minute';
        WHEN 'hour' THEN bucket_interval := INTERVAL '1 hour';
        WHEN 'day' THEN bucket_interval := INTERVAL '1 day';
        ELSE bucket_interval := INTERVAL '1 hour';
    END CASE;
    
    bucket_start := date_trunc(p_interval, NOW() - bucket_interval);
    
    -- Aggregate from access log
    INSERT INTO kv_cache_stats (
        bucket_time, bucket_interval, model_id, layer,
        total_reads, total_writes, cache_hits, cache_misses,
        avg_latency_us
    )
    SELECT
        bucket_start,
        p_interval,
        model_id,
        NULL, -- Aggregate across all layers
        COUNT(*) FILTER (WHERE access_type = 'read'),
        COUNT(*) FILTER (WHERE access_type = 'write'),
        COUNT(*) FILTER (WHERE cache_hit = TRUE),
        COUNT(*) FILTER (WHERE cache_hit = FALSE),
        AVG(latency_us)::BIGINT
    FROM kv_cache_access_log
    WHERE accessed_at >= bucket_start
      AND accessed_at < bucket_start + bucket_interval
    GROUP BY model_id
    ON CONFLICT (bucket_time, bucket_interval, model_id, layer)
    DO UPDATE SET
        total_reads = EXCLUDED.total_reads,
        total_writes = EXCLUDED.total_writes,
        cache_hits = EXCLUDED.cache_hits,
        cache_misses = EXCLUDED.cache_misses,
        avg_latency_us = EXCLUDED.avg_latency_us;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Scheduled Jobs (using pg_cron extension if available)
-- ============================================================================

-- Note: Requires pg_cron extension
-- CREATE EXTENSION IF NOT EXISTS pg_cron;

/*
-- Schedule cleanup job (daily at 2 AM)
SELECT cron.schedule('cleanup-access-logs', '0 2 * * *', 
    'SELECT cleanup_old_access_logs()');

-- Schedule stats aggregation (every hour)
SELECT cron.schedule('aggregate-stats-hourly', '0 * * * *', 
    'SELECT aggregate_cache_stats(''hour'')');

-- Schedule stats aggregation (daily at 1 AM)
SELECT cron.schedule('aggregate-stats-daily', '0 1 * * *', 
    'SELECT aggregate_cache_stats(''day'')');
*/

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View: Recent cache entries
CREATE OR REPLACE VIEW recent_cache_entries AS
SELECT 
    id, model_id, layer, token_start, token_end,
    compression_algorithm, compression_ratio,
    accessed_at, access_count, version
FROM kv_cache_metadata
WHERE accessed_at > NOW() - INTERVAL '1 hour'
ORDER BY accessed_at DESC;

-- View: Hot cache entries (frequently accessed)
CREATE OR REPLACE VIEW hot_cache_entries AS
SELECT 
    id, model_id, layer, token_start, token_end,
    access_count, accessed_at,
    compression_ratio, storage_backend
FROM kv_cache_metadata
WHERE access_count > 10
  AND accessed_at > NOW() - INTERVAL '24 hours'
ORDER BY access_count DESC, accessed_at DESC;

-- View: Cache statistics summary
CREATE OR REPLACE VIEW cache_stats_summary AS
SELECT 
    model_id,
    COUNT(*) as total_entries,
    SUM(compressed_size) as total_compressed_bytes,
    SUM(original_size) as total_original_bytes,
    AVG(compression_ratio)::NUMERIC(10,2) as avg_compression_ratio,
    AVG(access_count)::NUMERIC(10,2) as avg_access_count,
    MAX(accessed_at) as last_accessed,
    COUNT(DISTINCT layer) as layers_cached
FROM kv_cache_metadata
GROUP BY model_id
ORDER BY total_compressed_bytes DESC;

-- ============================================================================
-- Grants (adjust as needed)
-- ============================================================================

-- Grant permissions to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO kv_cache_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO kv_cache_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO kv_cache_app;

-- ============================================================================
-- Initial Data / Seed (if needed)
-- ============================================================================

-- Example: Insert a test entry
/*
INSERT INTO kv_cache_metadata (
    model_id, layer, token_start, token_end,
    compression_algorithm, compressed_size, original_size,
    storage_backend, storage_key
) VALUES (
    'test-model', 0, 0, 128,
    'fp16', 1024, 2048,
    'dragonfly', 'kv:test-model:0:0'
);
*/

-- ============================================================================
-- End of Schema
-- ============================================================================

COMMENT ON TABLE kv_cache_metadata IS 'Main metadata table for KV cache entries';
COMMENT ON TABLE kv_cache_versions IS 'Version history for cache entries';
COMMENT ON TABLE kv_cache_access_log IS 'Access log for analytics and debugging';
COMMENT ON TABLE kv_cache_stats IS 'Aggregated statistics for monitoring';
