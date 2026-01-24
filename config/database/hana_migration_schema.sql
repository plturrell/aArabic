-- ============================================================================
-- SAP HANA Unified Schema for aArabic System
-- Created: January 24, 2026
-- Purpose: Complete database schema for all migrated services
-- ============================================================================

-- Database: NOPENAI_DB
-- User: SHIMMY_USER

-- ============================================================================
-- nAgentFlow Tables
-- ============================================================================

-- Workflow state storage
CREATE COLUMN TABLE WORKFLOW_STATE (
    workflow_id VARCHAR(128) PRIMARY KEY,
    state_data BLOB,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    metadata NCLOB
);

CREATE INDEX idx_workflow_status ON WORKFLOW_STATE(status);
CREATE INDEX idx_workflow_updated ON WORKFLOW_STATE(updated_at);

COMMENT ON TABLE WORKFLOW_STATE IS 'Stores workflow execution state for nAgentFlow';

-- Workflow runtime cache
CREATE COLUMN TABLE WORKFLOW_CACHE (
    cache_key VARCHAR(512) PRIMARY KEY,
    cache_value BLOB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_workflow_cache_expires ON WORKFLOW_CACHE(expires_at);

COMMENT ON TABLE WORKFLOW_CACHE IS 'Runtime cache for workflow execution data';

-- Workflow execution history/logs
CREATE COLUMN TABLE EXECUTION_LOG (
    log_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workflow_id VARCHAR(128) NOT NULL,
    step_name VARCHAR(256),
    status VARCHAR(50),
    message NCLOB,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata NCLOB
);

CREATE INDEX idx_execution_log_workflow ON EXECUTION_LOG(workflow_id);
CREATE INDEX idx_execution_log_created ON EXECUTION_LOG(created_at);

COMMENT ON TABLE EXECUTION_LOG IS 'Audit log for workflow executions';

-- ============================================================================
-- nAgentMeta Tables (Standard relational tables)
-- ============================================================================

-- Metadata storage example - adapt to your schema
CREATE COLUMN TABLE AGENT_METADATA (
    agent_id VARCHAR(128) PRIMARY KEY,
    agent_name VARCHAR(256) NOT NULL,
    agent_type VARCHAR(100),
    configuration NCLOB,
    capabilities NCLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50),
    metadata NCLOB
);

CREATE INDEX idx_agent_type ON AGENT_METADATA(agent_type);
CREATE INDEX idx_agent_status ON AGENT_METADATA(status);

COMMENT ON TABLE AGENT_METADATA IS 'Agent metadata and configuration';

-- Graph nodes for HANA graph engine
CREATE COLUMN TABLE GRAPH_NODES (
    node_id VARCHAR(128) PRIMARY KEY,
    node_type VARCHAR(100) NOT NULL,
    properties NCLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Graph edges for HANA graph engine
CREATE COLUMN TABLE GRAPH_EDGES (
    edge_id VARCHAR(128) PRIMARY KEY,
    source_node_id VARCHAR(128) NOT NULL,
    target_node_id VARCHAR(128) NOT NULL,
    edge_type VARCHAR(100) NOT NULL,
    properties NCLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES GRAPH_NODES(node_id),
    FOREIGN KEY (target_node_id) REFERENCES GRAPH_NODES(node_id)
);

CREATE INDEX idx_graph_edges_source ON GRAPH_EDGES(source_node_id);
CREATE INDEX idx_graph_edges_target ON GRAPH_EDGES(target_node_id);
CREATE INDEX idx_graph_edges_type ON GRAPH_EDGES(edge_type);

COMMENT ON TABLE GRAPH_NODES IS 'Graph nodes for knowledge graph';
COMMENT ON TABLE GRAPH_EDGES IS 'Graph relationships';

-- ============================================================================
-- nLocalModels Tables (LLM Caching)
-- ============================================================================

-- KV cache for attention states
CREATE COLUMN TABLE KV_CACHE (
    key VARCHAR(512) PRIMARY KEY,
    value BLOB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    size_bytes INTEGER,
    access_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kv_cache_expires ON KV_CACHE(expires_at);
CREATE INDEX idx_kv_cache_accessed ON KV_CACHE(last_accessed);

COMMENT ON TABLE KV_CACHE IS 'Key-value cache for LLM attention states';

-- Prompt caching for LLM
CREATE COLUMN TABLE PROMPT_CACHE (
    hash VARCHAR(64) PRIMARY KEY,
    state BLOB NOT NULL,
    prompt_text NCLOB,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tokens INTEGER,
    model_version VARCHAR(100),
    hit_count INTEGER DEFAULT 0
);

CREATE INDEX idx_prompt_cache_expires ON PROMPT_CACHE(expires_at);
CREATE INDEX idx_prompt_cache_hits ON PROMPT_CACHE(hit_count);

COMMENT ON TABLE PROMPT_CACHE IS 'Cached prompt states for LLM inference';

-- Session state persistence
CREATE COLUMN TABLE SESSION_STATE (
    session_id VARCHAR(128) PRIMARY KEY,
    data BLOB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(128),
    conversation_turns INTEGER DEFAULT 0
);

CREATE INDEX idx_session_expires ON SESSION_STATE(expires_at);
CREATE INDEX idx_session_user ON SESSION_STATE(user_id);

COMMENT ON TABLE SESSION_STATE IS 'User session state for LLM conversations';

-- Tensor storage for model weights
CREATE COLUMN TABLE TENSOR_STORAGE (
    tensor_id VARCHAR(256) PRIMARY KEY,
    tensor_data BLOB NOT NULL,
    metadata VARCHAR(1024),
    model_name VARCHAR(256),
    layer_name VARCHAR(256),
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    size_bytes BIGINT,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX idx_tensor_expires ON TENSOR_STORAGE(expires_at);
CREATE INDEX idx_tensor_model ON TENSOR_STORAGE(model_name);

COMMENT ON TABLE TENSOR_STORAGE IS 'Tensor storage for model weights and activations';

-- ============================================================================
-- Common Tables (Shared Utilities)
-- ============================================================================

-- System configuration
CREATE COLUMN TABLE SYSTEM_CONFIG (
    config_key VARCHAR(256) PRIMARY KEY,
    config_value NCLOB NOT NULL,
    config_type VARCHAR(50),
    description VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(128)
);

COMMENT ON TABLE SYSTEM_CONFIG IS 'System-wide configuration settings';

-- Health check status
CREATE COLUMN TABLE HEALTH_STATUS (
    service_name VARCHAR(100) PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    error_message VARCHAR(1024),
    metadata NCLOB
);

CREATE INDEX idx_health_status_check ON HEALTH_STATUS(last_check);

COMMENT ON TABLE HEALTH_STATUS IS 'Service health monitoring';

-- ============================================================================
-- TTL Cleanup Procedures
-- ============================================================================

-- Procedure to clean up expired cache entries
CREATE PROCEDURE CLEANUP_EXPIRED_CACHE()
LANGUAGE SQLSCRIPT AS
BEGIN
    -- Clean up workflow cache
    DELETE FROM WORKFLOW_CACHE WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up KV cache
    DELETE FROM KV_CACHE WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up prompt cache
    DELETE FROM PROMPT_CACHE WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up session state
    DELETE FROM SESSION_STATE WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Clean up tensor storage
    DELETE FROM TENSOR_STORAGE WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Log cleanup
    INSERT INTO EXECUTION_LOG (workflow_id, step_name, status, message)
    VALUES ('SYSTEM', 'TTL_CLEANUP', 'SUCCESS', 
            'Cleaned up expired cache entries at ' || CURRENT_TIMESTAMP);
END;

-- Schedule TTL cleanup (every hour)
-- Note: Use SAP HANA Job Scheduler or external scheduler
-- Example cron: 0 * * * * (every hour)

-- ============================================================================
-- Statistics and Monitoring Views
-- ============================================================================

-- Cache statistics view
CREATE VIEW V_CACHE_STATISTICS AS
SELECT 
    'WORKFLOW_CACHE' as cache_type,
    COUNT(*) as entry_count,
    SUM(LENGTH(cache_value))/1024/1024 as size_mb,
    COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_count,
    AVG(access_count) as avg_access_count
FROM WORKFLOW_CACHE
UNION ALL
SELECT 
    'KV_CACHE' as cache_type,
    COUNT(*) as entry_count,
    SUM(size_bytes)/1024/1024 as size_mb,
    COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_count,
    AVG(access_count) as avg_access_count
FROM KV_CACHE
UNION ALL
SELECT 
    'PROMPT_CACHE' as cache_type,
    COUNT(*) as entry_count,
    SUM(LENGTH(state))/1024/1024 as size_mb,
    COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_count,
    AVG(hit_count) as avg_access_count
FROM PROMPT_CACHE
UNION ALL
SELECT 
    'SESSION_STATE' as cache_type,
    COUNT(*) as entry_count,
    SUM(LENGTH(data))/1024/1024 as size_mb,
    COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_count,
    AVG(conversation_turns) as avg_access_count
FROM SESSION_STATE
UNION ALL
SELECT 
    'TENSOR_STORAGE' as cache_type,
    COUNT(*) as entry_count,
    SUM(size_bytes)/1024/1024 as size_mb,
    COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_count,
    AVG(access_count) as avg_access_count
FROM TENSOR_STORAGE;

-- Workflow execution statistics
CREATE VIEW V_WORKFLOW_STATISTICS AS
SELECT 
    DATE(created_at) as execution_date,
    status,
    COUNT(*) as count,
    AVG(execution_time_ms) as avg_time_ms,
    MIN(execution_time_ms) as min_time_ms,
    MAX(execution_time_ms) as max_time_ms
FROM EXECUTION_LOG
WHERE created_at >= ADD_DAYS(CURRENT_DATE, -7)
GROUP BY DATE(created_at), status;

-- ============================================================================
-- Performance Optimization
-- ============================================================================

-- Enable auto merge for column store optimization
ALTER TABLE WORKFLOW_STATE AUTO MERGE ON;
ALTER TABLE WORKFLOW_CACHE AUTO MERGE ON;
ALTER TABLE KV_CACHE AUTO MERGE ON;
ALTER TABLE PROMPT_CACHE AUTO MERGE ON;
ALTER TABLE SESSION_STATE AUTO MERGE ON;
ALTER TABLE TENSOR_STORAGE AUTO MERGE ON;

-- Enable delta merge optimization
ALTER TABLE WORKFLOW_STATE RECLAIM DATAVOLUME DEFRAGMENT;
ALTER TABLE KV_CACHE RECLAIM DATAVOLUME DEFRAGMENT;
ALTER TABLE PROMPT_CACHE RECLAIM DATAVOLUME DEFRAGMENT;

-- ============================================================================
-- Permissions Setup
-- ============================================================================

-- Grant permissions to SHIMMY_USER
GRANT SELECT, INSERT, UPDATE, DELETE ON WORKFLOW_STATE TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON WORKFLOW_CACHE TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON EXECUTION_LOG TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON AGENT_METADATA TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON GRAPH_NODES TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON GRAPH_EDGES TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON KV_CACHE TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON PROMPT_CACHE TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON SESSION_STATE TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TENSOR_STORAGE TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON SYSTEM_CONFIG TO SHIMMY_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON HEALTH_STATUS TO SHIMMY_USER;
GRANT SELECT ON V_CACHE_STATISTICS TO SHIMMY_USER;
GRANT SELECT ON V_WORKFLOW_STATISTICS TO SHIMMY_USER;
GRANT EXECUTE ON PROCEDURE CLEANUP_EXPIRED_CACHE TO SHIMMY_USER;

-- ============================================================================
-- Sample Data (Optional - for testing)
-- ============================================================================

-- Insert sample system config
INSERT INTO SYSTEM_CONFIG (config_key, config_value, config_type, description)
VALUES 
    ('cache.default_ttl', '3600', 'integer', 'Default cache TTL in seconds'),
    ('cache.cleanup_interval', '60', 'integer', 'Cache cleanup interval in seconds'),
    ('workflow.max_retries', '3', 'integer', 'Maximum workflow retry attempts'),
    ('llm.max_context_length', '8192', 'integer', 'Maximum LLM context length');

-- Insert initial health status
INSERT INTO HEALTH_STATUS (service_name, status, response_time_ms)
VALUES 
    ('nAgentFlow', 'HEALTHY', 5),
    ('nAgentMeta', 'HEALTHY', 3),
    ('nLocalModels', 'HEALTHY', 10);

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- List all tables
SELECT TABLE_NAME, TABLE_TYPE, RECORD_COUNT 
FROM TABLES 
WHERE SCHEMA_NAME = 'NOPENAI_DB'
ORDER BY TABLE_NAME;

-- Check table sizes
SELECT 
    TABLE_NAME,
    RECORD_COUNT,
    DISK_SIZE/1024/1024 as SIZE_MB,
    MEMORY_SIZE_IN_MAIN/1024/1024 as MEMORY_MB
FROM M_TABLES
WHERE SCHEMA_NAME = 'NOPENAI_DB'
ORDER BY MEMORY_SIZE_IN_MAIN DESC;

-- Check cache statistics
SELECT * FROM V_CACHE_STATISTICS;

-- ============================================================================
-- Completion Status
-- ============================================================================

-- Schema creation completed successfully
SELECT 
    COUNT(*) as table_count,
    'Schema migration complete' as status
FROM TABLES 
WHERE SCHEMA_NAME = 'NOPENAI_DB';

-- End of schema migration