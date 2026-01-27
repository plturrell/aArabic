-- ============================================================================
-- Trial Balance Audit and Exception Tables for SQLite
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ============================================================================
-- 6. TB_AUDIT_LOG - Complete Audit Trail
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_AUDIT_LOG (
    -- Primary Key
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Process Identification
    mandt TEXT,
    process_id TEXT,
    process_type TEXT NOT NULL,
    workflow_stage TEXT,
    
    -- Entity Information
    entity_type TEXT,
    entity_id TEXT,
    entity_name TEXT,
    
    -- Action Details
    action TEXT NOT NULL,
    action_status TEXT NOT NULL,
    
    -- User Information
    user_id TEXT,
    user_name TEXT,
    user_role TEXT,
    
    -- Timing
    timestamp TEXT DEFAULT (datetime('now')) NOT NULL,
    execution_time_ms INTEGER,
    
    -- Before/After State
    old_values TEXT,
    new_values TEXT,
    
    -- Details and Context
    details TEXT,
    error_message TEXT,
    stack_trace TEXT,
    
    -- Source Information
    source_ip TEXT,
    source_application TEXT,
    source_component TEXT,
    
    -- Compliance
    compliance_flags TEXT,
    
    -- Session Information
    session_id TEXT,
    correlation_id TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_audit_process ON TB_AUDIT_LOG(process_id);
CREATE INDEX IF NOT EXISTS idx_tb_audit_process_type ON TB_AUDIT_LOG(process_type);
CREATE INDEX IF NOT EXISTS idx_tb_audit_entity ON TB_AUDIT_LOG(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_tb_audit_timestamp ON TB_AUDIT_LOG(timestamp);

-- ============================================================================
-- 7. TB_EXCEPTIONS - Exception Tracking and Management
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_EXCEPTIONS (
    -- Primary Key
    exception_id TEXT PRIMARY KEY,
    
    -- Exception Identification
    mandt TEXT,
    rbukrs TEXT,
    gjahr TEXT,
    period TEXT,
    
    -- Exception Type and Severity
    exception_type TEXT NOT NULL,
    exception_category TEXT,
    severity TEXT NOT NULL,
    priority INTEGER DEFAULT 3,
    
    -- Source Information
    source_document TEXT,
    source_table TEXT,
    source_record_id TEXT,
    workflow_stage TEXT,
    
    -- Exception Details
    title TEXT NOT NULL,
    description TEXT,
    error_message TEXT,
    error_code TEXT,
    
    -- Impact Assessment
    impacted_accounts TEXT,
    impact_amount REAL,
    impact_description TEXT,
    
    -- Resolution Management
    resolution_status TEXT DEFAULT 'OPEN',
    resolution_type TEXT,
    resolution_description TEXT,
    resolution_date TEXT,
    
    -- Assignment
    assigned_to TEXT,
    assigned_to_role TEXT,
    assigned_at TEXT,
    
    -- Escalation
    escalated INTEGER DEFAULT 0,
    escalated_to TEXT,
    escalated_at TEXT,
    escalation_reason TEXT,
    
    -- SLA Tracking
    sla_due_date TEXT,
    sla_breached INTEGER DEFAULT 0,
    
    -- Related Records
    related_exceptions TEXT,
    related_audit_log_ids TEXT,
    
    -- Timestamps
    detected_at TEXT DEFAULT (datetime('now')),
    first_occurrence_at TEXT,
    last_occurrence_at TEXT,
    occurrence_count INTEGER DEFAULT 1,
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    resolved_by TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_exc_company_period ON TB_EXCEPTIONS(mandt, rbukrs, gjahr, period);
CREATE INDEX IF NOT EXISTS idx_tb_exc_type ON TB_EXCEPTIONS(exception_type);
CREATE INDEX IF NOT EXISTS idx_tb_exc_severity ON TB_EXCEPTIONS(severity);
CREATE INDEX IF NOT EXISTS idx_tb_exc_status ON TB_EXCEPTIONS(resolution_status);

-- ============================================================================
-- 8. TB_WORKFLOW_STATE - Workflow Execution State
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_WORKFLOW_STATE (
    -- Primary Key
    workflow_id TEXT PRIMARY KEY,
    
    -- Workflow Identification
    workflow_name TEXT NOT NULL,
    workflow_version TEXT,
    workflow_type TEXT,
    
    -- Execution Context
    mandt TEXT,
    rbukrs TEXT,
    gjahr TEXT,
    period TEXT,
    execution_mode TEXT,
    
    -- Current State
    current_place TEXT,
    current_transition TEXT,
    status TEXT NOT NULL,
    
    -- Progress Tracking
    total_stages INTEGER,
    completed_stages INTEGER,
    current_stage_name TEXT,
    progress_percentage REAL,
    
    -- Timing Information
    started_at TEXT DEFAULT (datetime('now')),
    estimated_completion TEXT,
    completed_at TEXT,
    total_duration_ms INTEGER,
    
    -- State Data
    state_data BLOB,
    petri_net_marking TEXT,
    variables TEXT,
    
    -- Approval Status
    requires_maker_approval INTEGER DEFAULT 0,
    maker_approved INTEGER DEFAULT 0,
    maker_approved_by TEXT,
    maker_approved_at TEXT,
    
    requires_checker_approval INTEGER DEFAULT 0,
    checker_approved INTEGER DEFAULT 0,
    checker_approved_by TEXT,
    checker_approved_at TEXT,
    
    -- Error Handling
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Metadata
    metadata TEXT,
    tags TEXT,
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_wf_status ON TB_WORKFLOW_STATE(status);
CREATE INDEX IF NOT EXISTS idx_tb_wf_company_period ON TB_WORKFLOW_STATE(mandt, rbukrs, gjahr, period);
CREATE INDEX IF NOT EXISTS idx_tb_wf_started ON TB_WORKFLOW_STATE(started_at);

-- ============================================================================
-- 9. TB_COMMENTARY - Business Commentary for Variances
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_COMMENTARY (
    -- Primary Key
    commentary_id TEXT PRIMARY KEY,
    
    -- Link to Trial Balance
    tb_id TEXT,
    
    -- Organizational Context
    mandt TEXT,
    rbukrs TEXT,
    gjahr TEXT,
    period TEXT,
    
    -- Account Context
    racct TEXT,
    account_name TEXT,
    ifrs_schedule TEXT,
    
    -- Variance Information
    variance_amount REAL,
    variance_percentage REAL,
    threshold_type TEXT,
    exceeds_threshold INTEGER,
    
    -- Commentary Content
    commentary_text TEXT NOT NULL,
    major_drivers TEXT,
    supporting_details TEXT,
    
    -- AI-Generated Content
    ai_generated INTEGER DEFAULT 0,
    ai_model_version TEXT,
    ai_confidence_score REAL,
    human_reviewed INTEGER DEFAULT 0,
    
    -- Source Information
    source_type TEXT,
    provided_by TEXT,
    provided_by_role TEXT,
    
    -- Quality Metrics
    completeness_score REAL,
    relevance_score REAL,
    
    -- Approval
    approved INTEGER DEFAULT 0,
    approved_by TEXT,
    approved_at TEXT,
    
    -- Timestamps
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    
    FOREIGN KEY (tb_id) REFERENCES TB_TRIAL_BALANCE(tb_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_comm_tb ON TB_COMMENTARY(tb_id);
CREATE INDEX IF NOT EXISTS idx_tb_comm_account ON TB_COMMENTARY(racct);
CREATE INDEX IF NOT EXISTS idx_tb_comm_threshold ON TB_COMMENTARY(exceeds_threshold);