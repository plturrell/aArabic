-- ============================================================================
-- Trial Balance Audit and Exception Tables
-- ============================================================================

SET SCHEMA TB_SCHEMA;

-- ============================================================================
-- 6. TB_AUDIT_LOG - Complete Audit Trail
-- ============================================================================

CREATE COLUMN TABLE TB_AUDIT_LOG (
    -- Primary Key
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    
    -- Process Identification
    mandt VARCHAR(3),
    process_id VARCHAR(50),                       -- Workflow Process ID
    process_type VARCHAR(50) NOT NULL,            -- EXTRACTION, VALIDATION, CALCULATION, etc.
    workflow_stage VARCHAR(50),                   -- Petri net place/transition
    
    -- Entity Information
    entity_type VARCHAR(50),                      -- TABLE, CALCULATION, APPROVAL, etc.
    entity_id VARCHAR(100),
    entity_name VARCHAR(200),
    
    -- Action Details
    action VARCHAR(50) NOT NULL,                  -- INSERT, UPDATE, DELETE, APPROVE, etc.
    action_status VARCHAR(20) NOT NULL,           -- SUCCESS, FAILED, WARNING
    
    -- User Information
    user_id VARCHAR(50),
    user_name VARCHAR(100),
    user_role VARCHAR(50),
    
    -- Timing
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    execution_time_ms INTEGER,
    
    -- Before/After State (for data changes)
    old_values NCLOB,
    new_values NCLOB,
    
    -- Details and Context
    details NCLOB,
    error_message NCLOB,
    stack_trace NCLOB,
    
    -- Source Information
    source_ip VARCHAR(45),
    source_application VARCHAR(50),
    source_component VARCHAR(50),
    
    -- Compliance
    compliance_flags NCLOB,                       -- JSON with compliance markers
    
    -- Session Information
    session_id VARCHAR(50),
    correlation_id VARCHAR(50)
)
UNLOAD PRIORITY 7 AUTO MERGE;

-- Partitioning by month for performance
ALTER TABLE TB_AUDIT_LOG PARTITION BY RANGE (timestamp) (
    PARTITION p_2025_01 VALUES < '2025-02-01',
    PARTITION p_2025_02 VALUES < '2025-03-01',
    PARTITION p_2025_03 VALUES < '2025-04-01',
    PARTITION p_2025_04 VALUES < '2025-05-01',
    PARTITION p_2025_05 VALUES < '2025-06-01',
    PARTITION p_2025_06 VALUES < '2025-07-01',
    PARTITION p_2025_07 VALUES < '2025-08-01',
    PARTITION p_2025_08 VALUES < '2025-09-01',
    PARTITION p_2025_09 VALUES < '2025-10-01',
    PARTITION p_2025_10 VALUES < '2025-11-01',
    PARTITION p_2025_11 VALUES < '2025-12-01',
    PARTITION p_2025_12 VALUES < '2026-01-01',
    PARTITION p_others VALUES < '9999-12-31'
);

-- Indexes
CREATE INDEX idx_tb_audit_process ON TB_AUDIT_LOG(process_id);
CREATE INDEX idx_tb_audit_process_type ON TB_AUDIT_LOG(process_type);
CREATE INDEX idx_tb_audit_entity ON TB_AUDIT_LOG(entity_type, entity_id);
CREATE INDEX idx_tb_audit_user ON TB_AUDIT_LOG(user_id);
CREATE INDEX idx_tb_audit_timestamp ON TB_AUDIT_LOG(timestamp);
CREATE INDEX idx_tb_audit_status ON TB_AUDIT_LOG(action_status);

COMMENT ON TABLE TB_AUDIT_LOG IS 'Complete audit trail for all Trial Balance operations';

-- ============================================================================
-- 7. TB_EXCEPTIONS - Exception Tracking and Management
-- ============================================================================

CREATE COLUMN TABLE TB_EXCEPTIONS (
    -- Primary Key
    exception_id VARCHAR(36) PRIMARY KEY,
    
    -- Exception Identification
    mandt VARCHAR(3),
    rbukrs VARCHAR(4),
    gjahr VARCHAR(4),
    period VARCHAR(3),
    
    -- Exception Type and Severity
    exception_type VARCHAR(50) NOT NULL,          -- BALANCE_MISMATCH, FX_MISSING, THRESHOLD_EXCEEDED, etc.
    exception_category VARCHAR(50),               -- DATA_QUALITY, PROCESS, BUSINESS_RULE
    severity VARCHAR(20) NOT NULL,                -- CRITICAL, HIGH, MEDIUM, LOW, INFO
    priority INTEGER DEFAULT 3,                   -- 1=Highest, 5=Lowest
    
    -- Source Information
    source_document VARCHAR(50),
    source_table VARCHAR(50),
    source_record_id VARCHAR(100),
    workflow_stage VARCHAR(50),
    
    -- Exception Details
    title VARCHAR(200) NOT NULL,
    description VARCHAR(500),
    error_message NCLOB,
    error_code VARCHAR(20),
    
    -- Impact Assessment
    impacted_accounts NCLOB,                      -- JSON array of affected accounts
    impact_amount DECIMAL(23,2),
    impact_description VARCHAR(500),
    
    -- Resolution Management
    resolution_status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, IN_PROGRESS, RESOLVED, CLOSED, ESCALATED
    resolution_type VARCHAR(50),                  -- CORRECTED, WAIVED, DEFERRED, NO_ACTION
    resolution_description NCLOB,
    resolution_date TIMESTAMP,
    
    -- Assignment
    assigned_to VARCHAR(50),
    assigned_to_role VARCHAR(50),
    assigned_at TIMESTAMP,
    
    -- Escalation
    escalated BOOLEAN DEFAULT FALSE,
    escalated_to VARCHAR(50),
    escalated_at TIMESTAMP,
    escalation_reason VARCHAR(500),
    
    -- SLA Tracking
    sla_due_date TIMESTAMP,
    sla_breached BOOLEAN DEFAULT FALSE,
    
    -- Related Records
    related_exceptions NCLOB,                     -- JSON array of related exception IDs
    related_audit_log_ids NCLOB,                  -- JSON array of audit log IDs
    
    -- Timestamps
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    first_occurrence_at TIMESTAMP,
    last_occurrence_at TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_by VARCHAR(50)
)
UNLOAD PRIORITY 5 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_exc_company_period ON TB_EXCEPTIONS(mandt, rbukrs, gjahr, period);
CREATE INDEX idx_tb_exc_type ON TB_EXCEPTIONS(exception_type);
CREATE INDEX idx_tb_exc_severity ON TB_EXCEPTIONS(severity);
CREATE INDEX idx_tb_exc_status ON TB_EXCEPTIONS(resolution_status);
CREATE INDEX idx_tb_exc_assigned ON TB_EXCEPTIONS(assigned_to);
CREATE INDEX idx_tb_exc_sla ON TB_EXCEPTIONS(sla_due_date);
CREATE INDEX idx_tb_exc_escalated ON TB_EXCEPTIONS(escalated);

COMMENT ON TABLE TB_EXCEPTIONS IS 'Exception tracking and management for Trial Balance process';

-- ============================================================================
-- 8. TB_WORKFLOW_STATE - Workflow Execution State
-- ============================================================================

CREATE COLUMN TABLE TB_WORKFLOW_STATE (
    -- Primary Key
    workflow_id VARCHAR(36) PRIMARY KEY,
    
    -- Workflow Identification
    workflow_name VARCHAR(100) NOT NULL,
    workflow_version VARCHAR(20),
    workflow_type VARCHAR(50),                    -- MONTHLY_CLOSE, QUARTERLY_CLOSE, DAILY_CLOSE
    
    -- Execution Context
    mandt VARCHAR(3),
    rbukrs VARCHAR(4),
    gjahr VARCHAR(4),
    period VARCHAR(3),
    execution_mode VARCHAR(20),                   -- MANUAL, SCHEDULED, TRIGGERED
    
    -- Current State
    current_place VARCHAR(50),                    -- Current Petri net place
    current_transition VARCHAR(50),               -- Current transition
    status VARCHAR(20) NOT NULL,                  -- RUNNING, COMPLETED, FAILED, PAUSED
    
    -- Progress Tracking
    total_stages INTEGER,
    completed_stages INTEGER,
    current_stage_name VARCHAR(100),
    progress_percentage DECIMAL(5,2),
    
    -- Timing Information
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    estimated_completion TIMESTAMP,
    completed_at TIMESTAMP,
    total_duration_ms BIGINT,
    
    -- State Data
    state_data BLOB,                              -- Serialized workflow state
    petri_net_marking NCLOB,                      -- Current marking (JSON)
    variables NCLOB,                              -- Workflow variables (JSON)
    
    -- Approval Status
    requires_maker_approval BOOLEAN DEFAULT FALSE,
    maker_approved BOOLEAN DEFAULT FALSE,
    maker_approved_by VARCHAR(50),
    maker_approved_at TIMESTAMP,
    
    requires_checker_approval BOOLEAN DEFAULT FALSE,
    checker_approved BOOLEAN DEFAULT FALSE,
    checker_approved_by VARCHAR(50),
    checker_approved_at TIMESTAMP,
    
    -- Error Handling
    error_count INTEGER DEFAULT 0,
    last_error NCLOB,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Metadata
    metadata NCLOB,                               -- Additional context (JSON)
    tags NCLOB,                                   -- Tags for categorization (JSON array)
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
UNLOAD PRIORITY 5 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_wf_status ON TB_WORKFLOW_STATE(status);
CREATE INDEX idx_tb_wf_company_period ON TB_WORKFLOW_STATE(mandt, rbukrs, gjahr, period);
CREATE INDEX idx_tb_wf_started ON TB_WORKFLOW_STATE(started_at);
CREATE INDEX idx_tb_wf_type ON TB_WORKFLOW_STATE(workflow_type);

COMMENT ON TABLE TB_WORKFLOW_STATE IS 'Current state of workflow executions';

-- ============================================================================
-- 9. TB_COMMENTARY - Business Commentary for Variances
-- ============================================================================

CREATE COLUMN TABLE TB_COMMENTARY (
    -- Primary Key
    commentary_id VARCHAR(36) PRIMARY KEY,
    
    -- Link to Trial Balance
    tb_id VARCHAR(36),
    
    -- Organizational Context
    mandt VARCHAR(3),
    rbukrs VARCHAR(4),
    gjahr VARCHAR(4),
    period VARCHAR(3),
    
    -- Account Context
    racct VARCHAR(10),
    account_name VARCHAR(50),
    ifrs_schedule VARCHAR(10),
    
    -- Variance Information
    variance_amount DECIMAL(23,2),
    variance_percentage DECIMAL(9,4),
    threshold_type VARCHAR(20),                   -- BS or PL
    exceeds_threshold BOOLEAN,
    
    -- Commentary Content
    commentary_text NCLOB NOT NULL,
    major_drivers NCLOB,                          -- JSON array of major drivers
    supporting_details NCLOB,                     -- Additional details
    
    -- AI-Generated Content
    ai_generated BOOLEAN DEFAULT FALSE,
    ai_model_version VARCHAR(50),
    ai_confidence_score DECIMAL(5,4),
    human_reviewed BOOLEAN DEFAULT FALSE,
    
    -- Source Information
    source_type VARCHAR(50),                      -- BUSINESS_FINANCE, ACCOUNTING, AUTO_GENERATED
    provided_by VARCHAR(50),
    provided_by_role VARCHAR(50),
    
    -- Quality Metrics
    completeness_score DECIMAL(5,4),
    relevance_score DECIMAL(5,4),
    
    -- Approval
    approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(50),
    approved_at TIMESTAMP,
    
    -- Timestamps
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    CONSTRAINT fk_tb_commentary_tb FOREIGN KEY (tb_id) REFERENCES TB_TRIAL_BALANCE(tb_id) ON DELETE CASCADE
)
UNLOAD PRIORITY 7 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_comm_tb ON TB_COMMENTARY(tb_id);
CREATE INDEX idx_tb_comm_account ON TB_COMMENTARY(racct);
CREATE INDEX idx_tb_comm_threshold ON TB_COMMENTARY(exceeds_threshold);
CREATE INDEX idx_tb_comm_approved ON TB_COMMENTARY(approved);

COMMENT ON TABLE TB_COMMENTARY IS 'Business commentary for variance explanations';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_AUDIT_LOG TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_EXCEPTIONS TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_WORKFLOW_STATE TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_COMMENTARY TO TB_APP_USER;