-- ============================================================================
-- ODPS Data Quality Columns Migration
-- Adds ODPS-aligned data quality tracking columns to existing tables
-- Reference: models/odps/primary/*.odps.yaml
-- ============================================================================

-- ============================================================================
-- TB_JOURNAL_ENTRIES - Add ODPS Data Quality Tracking
-- Reference: odps/primary/acdoca-journal-entries.odps.yaml
-- ============================================================================

-- GCOA Mapping Status (for TB005 validation)
ALTER TABLE TB_JOURNAL_ENTRIES ADD COLUMN gcoa_mapping_status TEXT DEFAULT 'mapped';

-- Data Quality Score (per-record)
ALTER TABLE TB_JOURNAL_ENTRIES ADD COLUMN data_quality_score REAL DEFAULT 100.0;

-- Validation Rule Status (JSON array of passed/failed rules)
ALTER TABLE TB_JOURNAL_ENTRIES ADD COLUMN validation_rule_results TEXT;

-- ODPS Product Reference
ALTER TABLE TB_JOURNAL_ENTRIES ADD COLUMN odps_product_id TEXT DEFAULT 'urn:uuid:acdoca-journal-entries-v1';

-- ============================================================================
-- TB_GL_ACCOUNTS - Add ODPS Data Quality Tracking
-- Reference: odps/primary/account-master.odps.yaml
-- ============================================================================

-- GCOA Version (for TB006 validation)
ALTER TABLE TB_GL_ACCOUNTS ADD COLUMN gcoa_version TEXT;

-- GCOA Mapping Status (for TB005 validation)
ALTER TABLE TB_GL_ACCOUNTS ADD COLUMN gcoa_mapping_status TEXT DEFAULT 'mapped';

-- Data Quality Score
ALTER TABLE TB_GL_ACCOUNTS ADD COLUMN data_quality_score REAL DEFAULT 100.0;

-- ODPS Product Reference
ALTER TABLE TB_GL_ACCOUNTS ADD COLUMN odps_product_id TEXT DEFAULT 'urn:uuid:account-master-v1';

-- ============================================================================
-- TB_EXCHANGE_RATES - Add ODPS Data Quality Tracking
-- Reference: odps/primary/exchange-rates.odps.yaml
-- ============================================================================

-- Validation Status (FX001-FX007 compliance)
ALTER TABLE TB_EXCHANGE_RATES ADD COLUMN validation_status TEXT DEFAULT 'valid';

-- Rate Source (for FX007 - GROUP_TREASURY, ECB, FED)
ALTER TABLE TB_EXCHANGE_RATES ADD COLUMN rate_source TEXT DEFAULT 'GROUP_TREASURY';

-- Data Quality Score
ALTER TABLE TB_EXCHANGE_RATES ADD COLUMN data_quality_score REAL DEFAULT 100.0;

-- ODPS Product Reference
ALTER TABLE TB_EXCHANGE_RATES ADD COLUMN odps_product_id TEXT DEFAULT 'urn:uuid:exchange-rates-v1';

-- ============================================================================
-- TB_TRIAL_BALANCE - Add ODPS Data Quality Tracking (Enhanced)
-- Reference: odps/primary/trial-balance-aggregated.odps.yaml
-- ============================================================================

-- GCOA Mapping Status (for TB005 validation)
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN gcoa_mapping_status TEXT DEFAULT 'mapped';

-- GCOA Version (for TB006 validation)
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN gcoa_version TEXT;

-- Data Quality Score (aggregate)
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN data_quality_score REAL DEFAULT 92.0;

-- Individual Quality Dimension Scores
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN quality_completeness REAL DEFAULT 95.0;
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN quality_accuracy REAL DEFAULT 98.0;
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN quality_consistency REAL DEFAULT 90.0;
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN quality_timeliness REAL DEFAULT 95.0;

-- Validation Rule Results (TB001-TB006)
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN tb001_passed INTEGER DEFAULT 1;  -- Balance Equation
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN tb002_passed INTEGER DEFAULT 1;  -- Debit Credit Balance
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN tb003_passed INTEGER DEFAULT 1;  -- IFRS Classification
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN tb004_passed INTEGER DEFAULT 1;  -- Period Data Accuracy
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN tb005_passed INTEGER DEFAULT 1;  -- GCOA Mapping Completeness
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN tb006_passed INTEGER DEFAULT 1;  -- Global Mapping Currency

-- DOI Threshold Details
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN threshold_amount REAL DEFAULT 100000000.0;  -- $100M for BS
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN threshold_percentage REAL DEFAULT 0.10;  -- 10%

-- Major Driver (VAR008)
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN major_driver TEXT;

-- Exception Status (VAR007)
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN is_exception INTEGER DEFAULT 0;

-- ODPS Product Reference
ALTER TABLE TB_TRIAL_BALANCE ADD COLUMN odps_product_id TEXT DEFAULT 'urn:uuid:trial-balance-aggregated-v1';

-- ============================================================================
-- TB_BALANCE_HISTORY - Add ODPS Data Quality Tracking
-- ============================================================================

-- Individual Quality Dimension Scores
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN quality_completeness REAL;
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN quality_accuracy REAL;
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN quality_consistency REAL;
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN quality_timeliness REAL;

-- Validation Rules Summary
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN rules_passed_count INTEGER;
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN rules_failed_count INTEGER;
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN rules_passed_list TEXT;  -- JSON array
ALTER TABLE TB_BALANCE_HISTORY ADD COLUMN rules_failed_list TEXT;  -- JSON array

-- ============================================================================
-- New Table: TB_DATA_QUALITY_LOG - ODPS Data Quality Audit Trail
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_DATA_QUALITY_LOG (
    -- Primary Key
    log_id TEXT PRIMARY KEY,
    
    -- Reference
    entity_type TEXT NOT NULL,  -- 'JOURNAL_ENTRY', 'TRIAL_BALANCE', 'VARIANCE'
    entity_id TEXT NOT NULL,
    
    -- Quality Assessment
    rule_id TEXT NOT NULL,  -- 'TB001', 'VAR003', etc.
    rule_name TEXT,
    rule_description TEXT,
    
    -- Result
    passed INTEGER NOT NULL,  -- 0 = failed, 1 = passed
    score REAL,
    severity TEXT DEFAULT 'error',  -- 'error', 'warning', 'info'
    
    -- Details
    expected_value TEXT,
    actual_value TEXT,
    error_message TEXT,
    
    -- Control Reference (from DOI requirements)
    control_id TEXT,  -- 'VAL-001', 'REC-002', etc.
    control_name TEXT,
    
    -- ODPS Reference
    odps_product_id TEXT,
    odps_rule_ref TEXT,
    
    -- Timestamp
    assessed_at TEXT DEFAULT (datetime('now')),
    assessed_by TEXT,
    
    -- Period Context
    fiscal_year TEXT,
    period TEXT,
    company_code TEXT
);

-- Indexes for TB_DATA_QUALITY_LOG
CREATE INDEX IF NOT EXISTS idx_dq_entity ON TB_DATA_QUALITY_LOG(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_dq_rule ON TB_DATA_QUALITY_LOG(rule_id);
CREATE INDEX IF NOT EXISTS idx_dq_passed ON TB_DATA_QUALITY_LOG(passed);
CREATE INDEX IF NOT EXISTS idx_dq_period ON TB_DATA_QUALITY_LOG(fiscal_year, period);

-- ============================================================================
-- New Table: TB_VARIANCE_DETAILS - ODPS Variance Analysis Details
-- Reference: odps/primary/variances.odps.yaml
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_VARIANCE_DETAILS (
    -- Primary Key
    variance_id TEXT PRIMARY KEY,
    
    -- Reference to Trial Balance
    tb_id TEXT NOT NULL,
    
    -- Account Information
    racct TEXT NOT NULL,
    account_name TEXT,
    account_type TEXT,  -- Asset, Liability, Equity, Revenue, Expense
    
    -- Period Information
    current_period TEXT NOT NULL,
    comparative_period TEXT NOT NULL,
    comparison_type TEXT NOT NULL,  -- 'MTD', 'YTD', 'YOY', 'QTD'
    
    -- Balances (ODPS Field: current_period_balance, comparative_period_balance)
    current_balance REAL NOT NULL,
    comparative_balance REAL NOT NULL,
    
    -- Variance Calculation (VAR001, VAR002)
    variance_amount REAL NOT NULL,
    variance_percentage REAL,
    
    -- Materiality (VAR003, VAR004)
    is_material INTEGER DEFAULT 0,
    threshold_amount REAL,
    threshold_percentage REAL,
    
    -- Commentary (VAR005, VAR006)
    has_commentary INTEGER DEFAULT 0,
    commentary TEXT,
    commentary_updated_at TEXT,
    commentary_updated_by TEXT,
    
    -- Major Driver (VAR008)
    major_driver TEXT,
    driver_category TEXT,
    
    -- Exception Flagging (VAR007)
    is_exception INTEGER DEFAULT 0,
    exception_reason TEXT,
    escalated_to TEXT,
    
    -- Data Quality
    var001_passed INTEGER DEFAULT 1,
    var002_passed INTEGER DEFAULT 1,
    data_quality_score REAL DEFAULT 100.0,
    
    -- ODPS Product Reference
    odps_product_id TEXT DEFAULT 'urn:uuid:variances-v1',
    
    -- Audit Fields
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    
    FOREIGN KEY (tb_id) REFERENCES TB_TRIAL_BALANCE(tb_id)
);

-- Indexes for TB_VARIANCE_DETAILS
CREATE INDEX IF NOT EXISTS idx_var_tb ON TB_VARIANCE_DETAILS(tb_id);
CREATE INDEX IF NOT EXISTS idx_var_account ON TB_VARIANCE_DETAILS(racct);
CREATE INDEX IF NOT EXISTS idx_var_material ON TB_VARIANCE_DETAILS(is_material);
CREATE INDEX IF NOT EXISTS idx_var_exception ON TB_VARIANCE_DETAILS(is_exception);
CREATE INDEX IF NOT EXISTS idx_var_commentary ON TB_VARIANCE_DETAILS(has_commentary);

-- ============================================================================
-- New Table: TB_COMMENTARY_COVERAGE - Track VAR006 Compliance (90% rule)
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_COMMENTARY_COVERAGE (
    -- Primary Key
    coverage_id TEXT PRIMARY KEY,
    
    -- Period Context
    fiscal_year TEXT NOT NULL,
    period TEXT NOT NULL,
    company_code TEXT NOT NULL,
    
    -- Coverage Metrics
    total_variances INTEGER NOT NULL,
    material_variances INTEGER NOT NULL,
    variances_with_commentary INTEGER NOT NULL,
    coverage_percentage REAL NOT NULL,
    
    -- DOI Compliance
    meets_90_percent INTEGER NOT NULL,  -- VAR006 check
    exceptions_count INTEGER DEFAULT 0,
    
    -- Breakdown by Account Type
    bs_material_count INTEGER,
    bs_explained_count INTEGER,
    pl_material_count INTEGER,
    pl_explained_count INTEGER,
    
    -- Timestamp
    calculated_at TEXT DEFAULT (datetime('now')),
    
    UNIQUE(fiscal_year, period, company_code)
);

-- Index
CREATE INDEX IF NOT EXISTS idx_cov_period ON TB_COMMENTARY_COVERAGE(fiscal_year, period);