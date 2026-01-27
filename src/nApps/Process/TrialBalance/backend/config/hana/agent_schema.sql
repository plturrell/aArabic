-- ============================================================================
-- HANA Agent Schema for Trial Balance Application
-- Following nLocalModels HANA table patterns
-- ============================================================================

-- Drop existing tables (for clean reinstall)
DROP TABLE IF EXISTS AGENT_PERFORMANCE;
DROP TABLE IF EXISTS AGENT_ASSIGNMENTS;
DROP TABLE IF EXISTS AGENT_REGISTRY;

-- ============================================================================
-- AGENT_REGISTRY - Agent profiles and capabilities
-- ============================================================================

CREATE COLUMN TABLE AGENT_REGISTRY (
    agent_id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    role VARCHAR(64) NOT NULL,  -- 'maker', 'checker', 'manager'
    
    -- Capabilities stored as JSON array
    capabilities_json NCLOB,
    
    -- Capacity and load management
    capacity INTEGER NOT NULL DEFAULT 5,
    current_load INTEGER NOT NULL DEFAULT 0,
    
    -- Availability status
    availability VARCHAR(32) NOT NULL DEFAULT 'available',  -- 'available', 'busy', 'offline', 'on_leave'
    
    -- Performance tracking
    performance_score DECIMAL(5,4) NOT NULL DEFAULT 0.8000,  -- 0.0000 to 1.0000
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_capacity CHECK (capacity > 0),
    CONSTRAINT chk_current_load CHECK (current_load >= 0 AND current_load <= capacity),
    CONSTRAINT chk_availability CHECK (availability IN ('available', 'busy', 'offline', 'on_leave')),
    CONSTRAINT chk_performance CHECK (performance_score >= 0.0 AND performance_score <= 1.0)
) UNLOAD PRIORITY 5 AUTO MERGE;

-- Indexes for agent queries
CREATE INDEX idx_agent_availability ON AGENT_REGISTRY(availability, current_load);
CREATE INDEX idx_agent_role ON AGENT_REGISTRY(role);
CREATE INDEX idx_agent_performance ON AGENT_REGISTRY(performance_score DESC);

-- Full-text index for capability search
CREATE FULLTEXT INDEX idx_agent_capabilities ON AGENT_REGISTRY(capabilities_json)
    FUZZY SEARCH INDEX ON
    FAST PREPROCESS ON
    TEXT ANALYSIS ON;

-- ============================================================================
-- AGENT_ASSIGNMENTS - Task assignment tracking
-- ============================================================================

CREATE COLUMN TABLE AGENT_ASSIGNMENTS (
    assignment_id VARCHAR(128) PRIMARY KEY,
    task_id VARCHAR(128) NOT NULL,
    agent_id VARCHAR(128) NOT NULL,
    
    -- Assignment metadata
    score DECIMAL(5,4) NOT NULL,  -- Assignment confidence score
    method VARCHAR(32) NOT NULL,  -- 'hungarian', 'greedy', 'ai_enhanced', 'round_robin'
    
    -- Status tracking
    status VARCHAR(32) NOT NULL DEFAULT 'active',  -- 'active', 'completed', 'failed', 'cancelled'
    
    -- Performance data (filled on completion)
    success BOOLEAN DEFAULT NULL,
    duration_ms BIGINT DEFAULT NULL,
    
    -- Timestamps
    assigned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP DEFAULT NULL,
    
    -- Foreign key
    FOREIGN KEY (agent_id) REFERENCES AGENT_REGISTRY(agent_id),
    
    -- Constraints
    CONSTRAINT chk_assignment_score CHECK (score >= 0.0 AND score <= 1.0),
    CONSTRAINT chk_assignment_method CHECK (method IN ('hungarian', 'greedy', 'ai_enhanced', 'round_robin')),
    CONSTRAINT chk_assignment_status CHECK (status IN ('active', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_duration CHECK (duration_ms IS NULL OR duration_ms >= 0)
) UNLOAD PRIORITY 5 AUTO MERGE;

-- Indexes for assignment queries
CREATE INDEX idx_assignment_task ON AGENT_ASSIGNMENTS(task_id);
CREATE INDEX idx_assignment_agent ON AGENT_ASSIGNMENTS(agent_id, status);
CREATE INDEX idx_assignment_status ON AGENT_ASSIGNMENTS(status, assigned_at);
CREATE INDEX idx_assignment_completed ON AGENT_ASSIGNMENTS(completed_at) WHERE completed_at IS NOT NULL;

-- ============================================================================
-- AGENT_PERFORMANCE - Historical performance tracking
-- ============================================================================

CREATE COLUMN TABLE AGENT_PERFORMANCE (
    record_id VARCHAR(128) PRIMARY KEY,
    agent_id VARCHAR(128) NOT NULL,
    task_id VARCHAR(128) NOT NULL,
    
    -- Performance metrics
    success BOOLEAN NOT NULL,
    duration_ms BIGINT NOT NULL,
    
    -- Task metadata
    task_complexity VARCHAR(32),  -- 'low', 'medium', 'high'
    task_priority INTEGER,
    
    -- Timestamp
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (agent_id) REFERENCES AGENT_REGISTRY(agent_id),
    
    -- Constraints
    CONSTRAINT chk_perf_duration CHECK (duration_ms >= 0),
    CONSTRAINT chk_perf_complexity CHECK (task_complexity IN ('low', 'medium', 'high') OR task_complexity IS NULL),
    CONSTRAINT chk_perf_priority CHECK (task_priority IS NULL OR (task_priority >= 1 AND task_priority <= 10))
) UNLOAD PRIORITY 7 AUTO MERGE;

-- Indexes for performance analytics
CREATE INDEX idx_perf_agent_time ON AGENT_PERFORMANCE(agent_id, recorded_at DESC);
CREATE INDEX idx_perf_success ON AGENT_PERFORMANCE(agent_id, success);
CREATE INDEX idx_perf_complexity ON AGENT_PERFORMANCE(task_complexity, success);

-- ============================================================================
-- VIEWS - Analytics and reporting
-- ============================================================================

-- Agent performance summary
CREATE VIEW V_AGENT_PERFORMANCE_SUMMARY AS
SELECT 
    a.agent_id,
    a.name,
    a.role,
    a.performance_score,
    a.current_load,
    a.capacity,
    COUNT(DISTINCT aa.assignment_id) as total_assignments,
    COUNT(DISTINCT CASE WHEN aa.status = 'completed' AND aa.success = TRUE THEN aa.assignment_id END) as successful_tasks,
    COUNT(DISTINCT CASE WHEN aa.status = 'completed' AND aa.success = FALSE THEN aa.assignment_id END) as failed_tasks,
    AVG(CASE WHEN aa.status = 'completed' THEN aa.duration_ms END) as avg_duration_ms,
    CASE 
        WHEN COUNT(DISTINCT CASE WHEN aa.status = 'completed' THEN aa.assignment_id END) > 0 
        THEN CAST(COUNT(DISTINCT CASE WHEN aa.status = 'completed' AND aa.success = TRUE THEN aa.assignment_id END) AS DECIMAL) / 
             CAST(COUNT(DISTINCT CASE WHEN aa.status = 'completed' THEN aa.assignment_id END) AS DECIMAL)
        ELSE 0.0 
    END as success_rate
FROM AGENT_REGISTRY a
LEFT JOIN AGENT_ASSIGNMENTS aa ON a.agent_id = aa.agent_id
GROUP BY a.agent_id, a.name, a.role, a.performance_score, a.current_load, a.capacity;

-- Current workload distribution
CREATE VIEW V_AGENT_WORKLOAD AS
SELECT 
    a.agent_id,
    a.name,
    a.role,
    a.capacity,
    a.current_load,
    COUNT(aa.assignment_id) as active_assignments,
    CAST(a.current_load AS DECIMAL) / CAST(a.capacity AS DECIMAL) as utilization_rate
FROM AGENT_REGISTRY a
LEFT JOIN AGENT_ASSIGNMENTS aa ON a.agent_id = aa.agent_id AND aa.status = 'active'
GROUP BY a.agent_id, a.name, a.role, a.capacity, a.current_load
ORDER BY utilization_rate DESC;

-- Recent assignment activity
CREATE VIEW V_RECENT_ASSIGNMENTS AS
SELECT 
    aa.assignment_id,
    aa.task_id,
    aa.agent_id,
    a.name as agent_name,
    a.role as agent_role,
    aa.score,
    aa.method,
    aa.status,
    aa.success,
    aa.duration_ms,
    aa.assigned_at,
    aa.completed_at
FROM AGENT_ASSIGNMENTS aa
JOIN AGENT_REGISTRY a ON aa.agent_id = a.agent_id
WHERE aa.assigned_at >= ADD_DAYS(CURRENT_TIMESTAMP, -7)
ORDER BY aa.assigned_at DESC;

-- ============================================================================
-- PROCEDURES - Common operations
-- ============================================================================

-- Procedure: Update agent load
CREATE PROCEDURE P_UPDATE_AGENT_LOAD(
    IN p_agent_id VARCHAR(128),
    IN p_delta INTEGER
)
LANGUAGE SQLSCRIPT AS
BEGIN
    UPDATE AGENT_REGISTRY
    SET current_load = current_load + p_delta,
        updated_at = CURRENT_TIMESTAMP
    WHERE agent_id = p_agent_id;
END;

-- Procedure: Complete assignment
CREATE PROCEDURE P_COMPLETE_ASSIGNMENT(
    IN p_assignment_id VARCHAR(128),
    IN p_success BOOLEAN,
    IN p_duration_ms BIGINT
)
LANGUAGE SQLSCRIPT AS
BEGIN
    DECLARE v_agent_id VARCHAR(128);
    DECLARE v_task_id VARCHAR(128);
    
    -- Get assignment details
    SELECT agent_id, task_id INTO v_agent_id, v_task_id
    FROM AGENT_ASSIGNMENTS
    WHERE assignment_id = p_assignment_id;
    
    -- Update assignment
    UPDATE AGENT_ASSIGNMENTS
    SET status = 'completed',
        success = p_success,
        duration_ms = p_duration_ms,
        completed_at = CURRENT_TIMESTAMP
    WHERE assignment_id = p_assignment_id;
    
    -- Decrease agent load
    CALL P_UPDATE_AGENT_LOAD(v_agent_id, -1);
    
    -- Record performance
    INSERT INTO AGENT_PERFORMANCE (
        record_id, agent_id, task_id, success, duration_ms, recorded_at
    ) VALUES (
        'perf_' || CURRENT_TIMESTAMP,
        v_agent_id,
        v_task_id,
        p_success,
        p_duration_ms,
        CURRENT_TIMESTAMP
    );
    
    -- Update performance score (exponential moving average)
    UPDATE AGENT_REGISTRY
    SET performance_score = CASE
            WHEN p_success = TRUE 
            THEN LEAST(performance_score + 0.05, 1.0)
            ELSE GREATEST(performance_score - 0.05, 0.0)
        END,
        updated_at = CURRENT_TIMESTAMP
    WHERE agent_id = v_agent_id;
END;

-- ============================================================================
-- INITIAL DATA - Seed with sample agents
-- ============================================================================

-- Sample agents for testing
INSERT INTO AGENT_REGISTRY (agent_id, name, role, capabilities_json, capacity, performance_score) VALUES
('agent-001', 'Alice Chen', 'checker', '["accounting", "audit", "compliance"]', 5, 0.92),
('agent-002', 'Bob Smith', 'checker', '["accounting", "tax"]', 5, 0.85),
('agent-003', 'Carol Wang', 'manager', '["accounting", "audit", "compliance", "management"]', 10, 0.95),
('agent-004', 'David Lee', 'maker', '["data_entry", "accounting"]', 8, 0.88);

-- ============================================================================
-- GRANTS - Set permissions
-- ============================================================================

-- Grant access to application user (update with actual username)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON AGENT_REGISTRY TO TRIAL_BALANCE_USER;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON AGENT_ASSIGNMENTS TO TRIAL_BALANCE_USER;
-- GRANT SELECT, INSERT ON AGENT_PERFORMANCE TO TRIAL_BALANCE_USER;
-- GRANT SELECT ON V_AGENT_PERFORMANCE_SUMMARY TO TRIAL_BALANCE_USER;
-- GRANT SELECT ON V_AGENT_WORKLOAD TO TRIAL_BALANCE_USER;
-- GRANT SELECT ON V_RECENT_ASSIGNMENTS TO TRIAL_BALANCE_USER;

-- ============================================================================
-- MAINTENANCE - Automatic cleanup jobs
-- ============================================================================

-- Clean up old performance records (keep last 90 days)
-- This would be scheduled as a HANA job
-- DELETE FROM AGENT_PERFORMANCE 
-- WHERE recorded_at < ADD_DAYS(CURRENT_TIMESTAMP, -90);

-- Archive completed assignments older than 30 days
-- This would be scheduled as a HANA job
-- UPDATE AGENT_ASSIGNMENTS 
-- SET status = 'archived'
-- WHERE status = 'completed' 
--   AND completed_at < ADD_DAYS(CURRENT_TIMESTAMP, -30);

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify tables created
SELECT TABLE_NAME, RECORD_COUNT 
FROM M_TABLES 
WHERE SCHEMA_NAME = CURRENT_SCHEMA
  AND TABLE_NAME LIKE 'AGENT_%'
ORDER BY TABLE_NAME;

-- Show sample data
SELECT * FROM AGENT_REGISTRY;
SELECT * FROM V_AGENT_WORKLOAD;