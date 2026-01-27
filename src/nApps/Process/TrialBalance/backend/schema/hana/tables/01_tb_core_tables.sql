-- ============================================================================
-- Trial Balance Core Tables for SAP HANA Cloud
-- Aligned with SAP S/4HANA GL Structure and IFRS Requirements
-- DOI Version: 2.0 (2025-10-23)
-- ============================================================================

-- Set schema context
SET SCHEMA TB_SCHEMA;

-- ============================================================================
-- 1. TB_JOURNAL_ENTRIES - Journal Entry Data from S/4 ACDOCA
-- ============================================================================

CREATE COLUMN TABLE TB_JOURNAL_ENTRIES (
    -- Primary Key Components
    entry_id VARCHAR(50) PRIMARY KEY,
    
    -- SAP S/4 Standard Fields
    mandt VARCHAR(3) NOT NULL,                    -- Client
    rldnr VARCHAR(2) NOT NULL,                    -- Ledger (0L=Leading)
    rbukrs VARCHAR(4) NOT NULL,                   -- Company Code
    gjahr VARCHAR(4) NOT NULL,                    -- Fiscal Year
    belnr VARCHAR(10) NOT NULL,                   -- Document Number
    buzei VARCHAR(3) NOT NULL,                    -- Line Item
    
    -- Posting Date Information
    budat DATE NOT NULL,                          -- Posting Date
    bldat DATE,                                   -- Document Date
    cpudt DATE,                                   -- Entry Date
    blart VARCHAR(2),                             -- Document Type
    
    -- Account Information
    racct VARCHAR(10) NOT NULL,                   -- GL Account Number
    rtcur VARCHAR(5) NOT NULL,                    -- Transaction Currency
    rwcur VARCHAR(5),                             -- Company Code Currency
    
    -- Debit/Credit Information
    drcrk VARCHAR(1) NOT NULL,                    -- Debit/Credit Indicator (S/H)
    poper VARCHAR(3),                             -- Posting Period
    
    -- Amount Fields (DECIMAL(23,2) matches S/4 precision)
    tsl DECIMAL(23,2),                            -- Amount in Transaction Currency
    hsl DECIMAL(23,2),                            -- Amount in Local Currency
    wsl DECIMAL(23,2),                            -- Amount in Document Currency
    ksl DECIMAL(23,2),                            -- Amount in Group Currency
    
    -- Organizational Units
    kostl VARCHAR(10),                            -- Cost Center
    prctr VARCHAR(10),                            -- Profit Center
    segment VARCHAR(10),                          -- Segment for Segmental Reporting
    psegment VARCHAR(10),                         -- Partner Segment
    
    -- Additional Classification
    sgtxt VARCHAR(50),                            -- Line Item Text
    xref1 VARCHAR(12),                            -- Reference Key 1
    xref2 VARCHAR(12),                            -- Reference Key 2
    xref3 VARCHAR(20),                            -- Reference Key 3
    
    -- Customer/Vendor
    kunnr VARCHAR(10),                            -- Customer Number
    lifnr VARCHAR(10),                            -- Vendor Number
    
    -- IFRS Classification (Enhanced)
    ifrs_schedule VARCHAR(10),                    -- IFRS Schedule (01, 1A, 2O, etc.)
    ifrs_category VARCHAR(50),                    -- IFRS Category
    account_type VARCHAR(20),                     -- Asset/Liability/Equity/Revenue/Expense
    
    -- Source System
    source_system VARCHAR(20) DEFAULT 'S4HANA',   -- Source System ID
    source_table VARCHAR(20) DEFAULT 'ACDOCA',    -- Source Table
    
    -- Processing Status
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated BOOLEAN DEFAULT FALSE,
    validation_errors NCLOB,
    processed_at TIMESTAMP,
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_drcrk CHECK (drcrk IN ('S', 'H'))
) 
UNLOAD PRIORITY 5 AUTO MERGE;

-- Indexes for Performance
CREATE INDEX idx_tb_je_company_year ON TB_JOURNAL_ENTRIES(mandt, rbukrs, gjahr);
CREATE INDEX idx_tb_je_account ON TB_JOURNAL_ENTRIES(racct);
CREATE INDEX idx_tb_je_posting_date ON TB_JOURNAL_ENTRIES(budat);
CREATE INDEX idx_tb_je_document ON TB_JOURNAL_ENTRIES(mandt, rbukrs, gjahr, belnr);
CREATE INDEX idx_tb_je_cost_center ON TB_JOURNAL_ENTRIES(kostl);
CREATE INDEX idx_tb_je_profit_center ON TB_JOURNAL_ENTRIES(prctr);
CREATE INDEX idx_tb_je_ifrs_schedule ON TB_JOURNAL_ENTRIES(ifrs_schedule);
CREATE INDEX idx_tb_je_validated ON TB_JOURNAL_ENTRIES(validated);

COMMENT ON TABLE TB_JOURNAL_ENTRIES IS 'Journal entries extracted from SAP S/4HANA ACDOCA table';

-- ============================================================================
-- 2. TB_GL_ACCOUNTS - Chart of Accounts from S/4 SKA1
-- ============================================================================

CREATE COLUMN TABLE TB_GL_ACCOUNTS (
    -- Primary Key
    account_id VARCHAR(50) PRIMARY KEY,
    
    -- SAP S/4 Standard Fields
    mandt VARCHAR(3) NOT NULL,
    saknr VARCHAR(10) NOT NULL,                   -- GL Account Number
    ktopl VARCHAR(4) NOT NULL,                    -- Chart of Accounts
    
    -- Account Description
    txt20 VARCHAR(20),                            -- GL Account Short Text
    txt50 VARCHAR(50),                            -- GL Account Long Text
    
    -- Account Classification
    xbilk VARCHAR(1),                             -- Balance Sheet Account Indicator
    gvtyp VARCHAR(2),                             -- P&L Statement Account Type
    ktoks VARCHAR(4),                             -- Account Group
    
    -- IFRS Classification
    ifrs_schedule VARCHAR(10),                    -- IFRS Schedule Code
    ifrs_category VARCHAR(50),                    -- IFRS Category Name
    ifrs_subcategory VARCHAR(50),                 -- IFRS Subcategory
    account_type VARCHAR(20) NOT NULL,            -- Asset/Liability/Equity/Revenue/Expense
    
    -- Account Hierarchy
    parent_account VARCHAR(10),                   -- Parent Account for Hierarchy
    hierarchy_level INTEGER DEFAULT 1,            -- Level in Hierarchy (1-10)
    sort_key VARCHAR(10),                         -- Sort Key for Reporting
    
    -- Balance Sheet Classification
    bs_section VARCHAR(20),                       -- BS Section (Assets/Liabilities)
    bs_category VARCHAR(50),                      -- BS Category (Current/Non-Current)
    
    -- P&L Classification  
    pl_section VARCHAR(20),                       -- P&L Section (Revenue/Expense)
    pl_category VARCHAR(50),                      -- P&L Category (Operating/Non-Operating)
    
    -- Account Properties
    waers VARCHAR(5),                             -- Currency Key
    xloev VARCHAR(1),                             -- Deletion Flag
    xspeb VARCHAR(1),                             -- Blocked for Posting
    
    -- Reconciliation Account
    xopvw VARCHAR(1),                             -- Reconciliation Account Type
    mustr VARCHAR(16),                            -- Sample Account Number
    
    -- Source System
    source_system VARCHAR(20) DEFAULT 'S4HANA',
    source_table VARCHAR(20) DEFAULT 'SKA1',
    
    -- Processing Status
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique Constraint
    CONSTRAINT uk_tb_gl_accounts UNIQUE (mandt, saknr, ktopl)
)
UNLOAD PRIORITY 7 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_gl_account_num ON TB_GL_ACCOUNTS(saknr);
CREATE INDEX idx_tb_gl_chart ON TB_GL_ACCOUNTS(ktopl);
CREATE INDEX idx_tb_gl_ifrs_schedule ON TB_GL_ACCOUNTS(ifrs_schedule);
CREATE INDEX idx_tb_gl_account_type ON TB_GL_ACCOUNTS(account_type);
CREATE INDEX idx_tb_gl_parent ON TB_GL_ACCOUNTS(parent_account);

COMMENT ON TABLE TB_GL_ACCOUNTS IS 'GL Account master data from SAP S/4HANA SKA1';

-- ============================================================================
-- 3. TB_EXCHANGE_RATES - FX Rates from S/4 TCURR
-- ============================================================================

CREATE COLUMN TABLE TB_EXCHANGE_RATES (
    -- Primary Key
    rate_id VARCHAR(50) PRIMARY KEY,
    
    -- SAP S/4 Standard Fields
    mandt VARCHAR(3) NOT NULL,
    kurst VARCHAR(4) NOT NULL,                    -- Exchange Rate Type (M, B, G)
    fcurr VARCHAR(5) NOT NULL,                    -- From Currency
    tcurr VARCHAR(5) NOT NULL,                    -- To Currency
    gdatu DATE NOT NULL,                          -- Valid From Date
    
    -- Exchange Rate Information
    ukurs DECIMAL(9,5) NOT NULL,                  -- Exchange Rate
    ffact INTEGER DEFAULT 1,                      -- From Factor
    tfact INTEGER DEFAULT 1,                      -- To Factor
    
    -- Additional Fields
    abwct VARCHAR(1),                             -- Exchange Rate Spread
    abwga VARCHAR(4),                             -- Inv. Postings: Exch. Rate Spread
    
    -- Rate Type Descriptions
    rate_type_desc VARCHAR(50),                   -- Description of Rate Type
    
    -- Source Information
    source_table VARCHAR(20) DEFAULT 'TCURR',
    source_system VARCHAR(20) DEFAULT 'S4HANA',
    
    -- Processing Status
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique Constraint
    CONSTRAINT uk_tb_exchange_rates UNIQUE (mandt, kurst, fcurr, tcurr, gdatu)
)
UNLOAD PRIORITY 7 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_fx_currency_pair ON TB_EXCHANGE_RATES(fcurr, tcurr);
CREATE INDEX idx_tb_fx_rate_type ON TB_EXCHANGE_RATES(kurst);
CREATE INDEX idx_tb_fx_valid_date ON TB_EXCHANGE_RATES(gdatu);
CREATE INDEX idx_tb_fx_active ON TB_EXCHANGE_RATES(is_active);

COMMENT ON TABLE TB_EXCHANGE_RATES IS 'Exchange rates from SAP S/4HANA TCURR';

-- ============================================================================
-- 4. TB_TRIAL_BALANCE - Calculated Trial Balance Results
-- ============================================================================

CREATE COLUMN TABLE TB_TRIAL_BALANCE (
    -- Primary Key
    tb_id VARCHAR(36) PRIMARY KEY,
    
    -- Organizational Dimension
    mandt VARCHAR(3) NOT NULL,
    rbukrs VARCHAR(4) NOT NULL,                   -- Company Code
    gjahr VARCHAR(4) NOT NULL,                    -- Fiscal Year
    period VARCHAR(3) NOT NULL,                   -- Period (001-012 or YYYYMMDD)
    
    -- Account Dimension
    racct VARCHAR(10) NOT NULL,                   -- GL Account
    account_name VARCHAR(50),
    
    -- IFRS Classification
    ifrs_schedule VARCHAR(10),
    ifrs_category VARCHAR(50),
    account_type VARCHAR(20),
    
    -- Organizational Units
    kostl VARCHAR(10),                            -- Cost Center
    prctr VARCHAR(10),                            -- Profit Center
    segment VARCHAR(10),                          -- Segment
    
    -- Currency
    currency VARCHAR(5) NOT NULL,
    
    -- Balance Amounts
    opening_balance DECIMAL(23,2) DEFAULT 0,
    debit_amount DECIMAL(23,2) DEFAULT 0,
    credit_amount DECIMAL(23,2) DEFAULT 0,
    closing_balance DECIMAL(23,2) DEFAULT 0,
    
    -- Multi-Currency Balances
    local_currency VARCHAR(5),
    local_currency_balance DECIMAL(23,2),
    group_currency VARCHAR(5) DEFAULT 'USD',
    group_currency_balance DECIMAL(23,2),
    
    -- Calculation Metadata
    calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    calculation_status VARCHAR(20) DEFAULT 'DRAFT',
    calculation_method VARCHAR(50),
    is_balanced BOOLEAN,
    balance_difference DECIMAL(23,2) DEFAULT 0,
    
    -- Variance Information (for reporting)
    prior_period_balance DECIMAL(23,2),
    variance_absolute DECIMAL(23,2),
    variance_percentage DECIMAL(9,4),
    exceeds_threshold BOOLEAN DEFAULT FALSE,
    
    -- Commentary
    has_commentary BOOLEAN DEFAULT FALSE,
    commentary NCLOB,
    commentary_updated_at TIMESTAMP,
    commentary_updated_by VARCHAR(50),
    
    -- Approval Workflow
    maker_id VARCHAR(50),
    maker_approved_at TIMESTAMP,
    checker_id VARCHAR(50),
    checker_approved_at TIMESTAMP,
    approval_status VARCHAR(20) DEFAULT 'PENDING',
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique Constraint for Logical Key
    CONSTRAINT uk_tb_trial_balance UNIQUE (mandt, rbukrs, gjahr, period, racct, kostl, prctr, segment, currency)
)
UNLOAD PRIORITY 5 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_company_period ON TB_TRIAL_BALANCE(mandt, rbukrs, gjahr, period);
CREATE INDEX idx_tb_account ON TB_TRIAL_BALANCE(racct);
CREATE INDEX idx_tb_ifrs_schedule ON TB_TRIAL_BALANCE(ifrs_schedule);
CREATE INDEX idx_tb_threshold ON TB_TRIAL_BALANCE(exceeds_threshold);
CREATE INDEX idx_tb_status ON TB_TRIAL_BALANCE(calculation_status);
CREATE INDEX idx_tb_approval ON TB_TRIAL_BALANCE(approval_status);

COMMENT ON TABLE TB_TRIAL_BALANCE IS 'Calculated trial balance results with variance analysis';

-- ============================================================================
-- 5. TB_BALANCE_HISTORY - Historical Snapshots
-- ============================================================================

CREATE COLUMN TABLE TB_BALANCE_HISTORY (
    -- Primary Key
    snapshot_id VARCHAR(36) PRIMARY KEY,
    
    -- Snapshot Identification
    mandt VARCHAR(3) NOT NULL,
    rbukrs VARCHAR(4) NOT NULL,
    gjahr VARCHAR(4) NOT NULL,
    period VARCHAR(3) NOT NULL,
    snapshot_type VARCHAR(20) NOT NULL,           -- DAILY, MONTHLY, QUARTERLY, YEAR_END
    snapshot_date DATE NOT NULL,
    
    -- Summary Balances
    total_assets DECIMAL(23,2),
    total_liabilities DECIMAL(23,2),
    total_equity DECIMAL(23,2),
    total_debits DECIMAL(23,2),
    total_credits DECIMAL(23,2),
    balance_difference DECIMAL(23,2),
    is_balanced BOOLEAN,
    
    -- P&L Summary
    total_revenue DECIMAL(23,2),
    total_expenses DECIMAL(23,2),
    net_income DECIMAL(23,2),
    
    -- Snapshot Data (Full JSON)
    snapshot_data NCLOB,                          -- Complete TB snapshot in JSON
    snapshot_compressed BLOB,                     -- Compressed snapshot for archive
    
    -- Metadata
    record_count INTEGER,
    calculation_time_ms INTEGER,
    data_quality_score DECIMAL(5,4),
    
    -- Audit Fields
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique Constraint
    CONSTRAINT uk_tb_balance_history UNIQUE (mandt, rbukrs, gjahr, period, snapshot_type, snapshot_date)
)
UNLOAD PRIORITY 9 AUTO MERGE;

-- Indexes
CREATE INDEX idx_tb_hist_company_period ON TB_BALANCE_HISTORY(mandt, rbukrs, gjahr, period);
CREATE INDEX idx_tb_hist_snapshot_type ON TB_BALANCE_HISTORY(snapshot_type);
CREATE INDEX idx_tb_hist_date ON TB_BALANCE_HISTORY(snapshot_date);

COMMENT ON TABLE TB_BALANCE_HISTORY IS 'Historical trial balance snapshots for trend analysis';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_JOURNAL_ENTRIES TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_GL_ACCOUNTS TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_EXCHANGE_RATES TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_TRIAL_BALANCE TO TB_APP_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON TB_BALANCE_HISTORY TO TB_APP_USER;