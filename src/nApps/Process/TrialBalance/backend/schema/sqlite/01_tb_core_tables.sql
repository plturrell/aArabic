-- ============================================================================
-- Trial Balance Core Tables for SQLite (Development)
-- Simplified version of HANA schema for local development
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ============================================================================
-- 1. TB_JOURNAL_ENTRIES - Journal Entry Data
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_JOURNAL_ENTRIES (
    -- Primary Key
    entry_id TEXT PRIMARY KEY,
    
    -- SAP S/4 Standard Fields
    mandt TEXT NOT NULL,
    rldnr TEXT NOT NULL,
    rbukrs TEXT NOT NULL,
    gjahr TEXT NOT NULL,
    belnr TEXT NOT NULL,
    buzei TEXT NOT NULL,
    
    -- Posting Date Information
    budat TEXT NOT NULL,
    bldat TEXT,
    cpudt TEXT,
    blart TEXT,
    
    -- Account Information
    racct TEXT NOT NULL,
    rtcur TEXT NOT NULL,
    rwcur TEXT,
    
    -- Debit/Credit Information
    drcrk TEXT NOT NULL CHECK(drcrk IN ('S', 'H')),
    poper TEXT,
    
    -- Amount Fields
    tsl REAL,
    hsl REAL,
    wsl REAL,
    ksl REAL,
    
    -- Organizational Units
    kostl TEXT,
    prctr TEXT,
    segment TEXT,
    psegment TEXT,
    
    -- Additional Classification
    sgtxt TEXT,
    xref1 TEXT,
    xref2 TEXT,
    xref3 TEXT,
    
    -- Customer/Vendor
    kunnr TEXT,
    lifnr TEXT,
    
    -- IFRS Classification
    ifrs_schedule TEXT,
    ifrs_category TEXT,
    account_type TEXT,
    
    -- Source System
    source_system TEXT DEFAULT 'S4HANA',
    source_table TEXT DEFAULT 'ACDOCA',
    
    -- Processing Status
    extracted_at TEXT DEFAULT (datetime('now')),
    validated INTEGER DEFAULT 0,
    validation_errors TEXT,
    processed_at TEXT,
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_tb_je_company_year ON TB_JOURNAL_ENTRIES(mandt, rbukrs, gjahr);
CREATE INDEX IF NOT EXISTS idx_tb_je_account ON TB_JOURNAL_ENTRIES(racct);
CREATE INDEX IF NOT EXISTS idx_tb_je_posting_date ON TB_JOURNAL_ENTRIES(budat);
CREATE INDEX IF NOT EXISTS idx_tb_je_document ON TB_JOURNAL_ENTRIES(mandt, rbukrs, gjahr, belnr);
CREATE INDEX IF NOT EXISTS idx_tb_je_validated ON TB_JOURNAL_ENTRIES(validated);

-- ============================================================================
-- 2. TB_GL_ACCOUNTS - Chart of Accounts
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_GL_ACCOUNTS (
    -- Primary Key
    account_id TEXT PRIMARY KEY,
    
    -- SAP S/4 Standard Fields
    mandt TEXT NOT NULL,
    saknr TEXT NOT NULL,
    ktopl TEXT NOT NULL,
    
    -- Account Description
    txt20 TEXT,
    txt50 TEXT,
    
    -- Account Classification
    xbilk TEXT,
    gvtyp TEXT,
    ktoks TEXT,
    
    -- IFRS Classification
    ifrs_schedule TEXT,
    ifrs_category TEXT,
    ifrs_subcategory TEXT,
    account_type TEXT NOT NULL,
    
    -- Account Hierarchy
    parent_account TEXT,
    hierarchy_level INTEGER DEFAULT 1,
    sort_key TEXT,
    
    -- Balance Sheet Classification
    bs_section TEXT,
    bs_category TEXT,
    
    -- P&L Classification
    pl_section TEXT,
    pl_category TEXT,
    
    -- Account Properties
    waers TEXT,
    xloev TEXT,
    xspeb TEXT,
    
    -- Source System
    source_system TEXT DEFAULT 'S4HANA',
    source_table TEXT DEFAULT 'SKA1',
    
    -- Processing Status
    extracted_at TEXT DEFAULT (datetime('now')),
    last_updated TEXT DEFAULT (datetime('now')),
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    
    UNIQUE(mandt, saknr, ktopl)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_gl_account_num ON TB_GL_ACCOUNTS(saknr);
CREATE INDEX IF NOT EXISTS idx_tb_gl_ifrs_schedule ON TB_GL_ACCOUNTS(ifrs_schedule);
CREATE INDEX IF NOT EXISTS idx_tb_gl_account_type ON TB_GL_ACCOUNTS(account_type);

-- ============================================================================
-- 3. TB_EXCHANGE_RATES - FX Rates
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_EXCHANGE_RATES (
    -- Primary Key
    rate_id TEXT PRIMARY KEY,
    
    -- SAP S/4 Standard Fields
    mandt TEXT NOT NULL,
    kurst TEXT NOT NULL,
    fcurr TEXT NOT NULL,
    tcurr TEXT NOT NULL,
    gdatu TEXT NOT NULL,
    
    -- Exchange Rate Information
    ukurs REAL NOT NULL,
    ffact INTEGER DEFAULT 1,
    tfact INTEGER DEFAULT 1,
    
    -- Additional Fields
    abwct TEXT,
    abwga TEXT,
    rate_type_desc TEXT,
    
    -- Source Information
    source_table TEXT DEFAULT 'TCURR',
    source_system TEXT DEFAULT 'S4HANA',
    
    -- Processing Status
    extracted_at TEXT DEFAULT (datetime('now')),
    is_active INTEGER DEFAULT 1,
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    
    UNIQUE(mandt, kurst, fcurr, tcurr, gdatu)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_fx_currency_pair ON TB_EXCHANGE_RATES(fcurr, tcurr);
CREATE INDEX IF NOT EXISTS idx_tb_fx_rate_type ON TB_EXCHANGE_RATES(kurst);
CREATE INDEX IF NOT EXISTS idx_tb_fx_valid_date ON TB_EXCHANGE_RATES(gdatu);

-- ============================================================================
-- 4. TB_TRIAL_BALANCE - Calculated Trial Balance Results
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_TRIAL_BALANCE (
    -- Primary Key
    tb_id TEXT PRIMARY KEY,
    
    -- Organizational Dimension
    mandt TEXT NOT NULL,
    rbukrs TEXT NOT NULL,
    gjahr TEXT NOT NULL,
    period TEXT NOT NULL,
    
    -- Account Dimension
    racct TEXT NOT NULL,
    account_name TEXT,
    
    -- IFRS Classification
    ifrs_schedule TEXT,
    ifrs_category TEXT,
    account_type TEXT,
    
    -- Organizational Units
    kostl TEXT,
    prctr TEXT,
    segment TEXT,
    
    -- Currency
    currency TEXT NOT NULL,
    
    -- Balance Amounts
    opening_balance REAL DEFAULT 0,
    debit_amount REAL DEFAULT 0,
    credit_amount REAL DEFAULT 0,
    closing_balance REAL DEFAULT 0,
    
    -- Multi-Currency Balances
    local_currency TEXT,
    local_currency_balance REAL,
    group_currency TEXT DEFAULT 'USD',
    group_currency_balance REAL,
    
    -- Calculation Metadata
    calculation_date TEXT DEFAULT (datetime('now')),
    calculation_status TEXT DEFAULT 'DRAFT',
    calculation_method TEXT,
    is_balanced INTEGER,
    balance_difference REAL DEFAULT 0,
    
    -- Variance Information
    prior_period_balance REAL,
    variance_absolute REAL,
    variance_percentage REAL,
    exceeds_threshold INTEGER DEFAULT 0,
    
    -- Commentary
    has_commentary INTEGER DEFAULT 0,
    commentary TEXT,
    commentary_updated_at TEXT,
    commentary_updated_by TEXT,
    
    -- Approval Workflow
    maker_id TEXT,
    maker_approved_at TEXT,
    checker_id TEXT,
    checker_approved_at TEXT,
    approval_status TEXT DEFAULT 'PENDING',
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_by TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    
    -- Unique constraint without expressions (SQLite limitation)
    -- Use trigger or application logic to enforce uniqueness with NULLs
    UNIQUE(mandt, rbukrs, gjahr, period, racct, currency)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_company_period ON TB_TRIAL_BALANCE(mandt, rbukrs, gjahr, period);
CREATE INDEX IF NOT EXISTS idx_tb_account ON TB_TRIAL_BALANCE(racct);
CREATE INDEX IF NOT EXISTS idx_tb_threshold ON TB_TRIAL_BALANCE(exceeds_threshold);
CREATE INDEX IF NOT EXISTS idx_tb_status ON TB_TRIAL_BALANCE(calculation_status);

-- ============================================================================
-- 5. TB_BALANCE_HISTORY - Historical Snapshots
-- ============================================================================

CREATE TABLE IF NOT EXISTS TB_BALANCE_HISTORY (
    -- Primary Key
    snapshot_id TEXT PRIMARY KEY,
    
    -- Snapshot Identification
    mandt TEXT NOT NULL,
    rbukrs TEXT NOT NULL,
    gjahr TEXT NOT NULL,
    period TEXT NOT NULL,
    snapshot_type TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    
    -- Summary Balances
    total_assets REAL,
    total_liabilities REAL,
    total_equity REAL,
    total_debits REAL,
    total_credits REAL,
    balance_difference REAL,
    is_balanced INTEGER,
    
    -- P&L Summary
    total_revenue REAL,
    total_expenses REAL,
    net_income REAL,
    
    -- Snapshot Data
    snapshot_data TEXT,
    
    -- Metadata
    record_count INTEGER,
    calculation_time_ms INTEGER,
    data_quality_score REAL,
    
    -- Audit Fields
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    
    UNIQUE(mandt, rbukrs, gjahr, period, snapshot_type, snapshot_date)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tb_hist_company_period ON TB_BALANCE_HISTORY(mandt, rbukrs, gjahr, period);
CREATE INDEX IF NOT EXISTS idx_tb_hist_snapshot_type ON TB_BALANCE_HISTORY(snapshot_type);