-- ============================================================================
-- Trial Balance Views for SQLite
-- Simplified calculation views for local development
-- ============================================================================

-- ============================================================================
-- V_ACCOUNT_BALANCES - Account Balance Aggregation View
-- ============================================================================

CREATE VIEW IF NOT EXISTS V_ACCOUNT_BALANCES AS
SELECT
    je.mandt,
    je.rbukrs,
    je.gjahr,
    je.poper as period,
    je.racct,
    MAX(ga.txt50) as account_name,
    je.kostl,
    je.prctr,
    je.segment,
    je.rwcur as currency,
    SUM(CASE WHEN je.drcrk = 'S' THEN je.hsl ELSE 0 END) as total_debit,
    SUM(CASE WHEN je.drcrk = 'H' THEN je.hsl ELSE 0 END) as total_credit,
    SUM(CASE WHEN je.drcrk = 'S' THEN je.hsl ELSE -je.hsl END) as net_balance,
    COUNT(*) as transaction_count
FROM TB_JOURNAL_ENTRIES je
LEFT JOIN TB_GL_ACCOUNTS ga 
    ON je.mandt = ga.mandt 
    AND je.racct = ga.saknr
WHERE je.validated = 1
GROUP BY 
    je.mandt,
    je.rbukrs,
    je.gjahr,
    je.poper,
    je.racct,
    je.kostl,
    je.prctr,
    je.segment,
    je.rwcur;

-- ============================================================================
-- V_MULTICURRENCY_BALANCES - Multi-Currency Conversion View
-- ============================================================================

CREATE VIEW IF NOT EXISTS V_MULTICURRENCY_BALANCES AS
SELECT
    je.mandt,
    je.rbukrs,
    je.gjahr,
    je.poper as period,
    je.racct,
    ga.txt50 as account_name,
    ga.ifrs_schedule,
    je.rtcur as original_currency,
    je.hsl as local_amount,
    'USD' as target_currency,
    COALESCE(r.ukurs / CAST(r.ffact AS REAL) * CAST(r.tfact AS REAL), 1.0) as exchange_rate,
    je.hsl * COALESCE(r.ukurs / CAST(r.ffact AS REAL) * CAST(r.tfact AS REAL), 1.0) as converted_amount,
    r.gdatu as conversion_date
FROM TB_JOURNAL_ENTRIES je
LEFT JOIN TB_GL_ACCOUNTS ga 
    ON je.mandt = ga.mandt 
    AND je.racct = ga.saknr
LEFT JOIN TB_EXCHANGE_RATES r 
    ON je.mandt = r.mandt 
    AND je.rtcur = r.fcurr
    AND r.tcurr = 'USD'
    AND r.kurst = 'M'
    AND r.is_active = 1
    AND r.gdatu <= je.budat
    AND r.gdatu = (
        SELECT MAX(r2.gdatu)
        FROM TB_EXCHANGE_RATES r2
        WHERE r2.mandt = je.mandt
            AND r2.fcurr = je.rtcur
            AND r2.tcurr = 'USD'
            AND r2.kurst = 'M'
            AND r2.gdatu <= je.budat
            AND r2.is_active = 1
    )
WHERE je.validated = 1;

-- ============================================================================
-- V_IFRS_SUMMARY - IFRS Schedule Summary View
-- ============================================================================

CREATE VIEW IF NOT EXISTS V_IFRS_SUMMARY AS
SELECT
    t.mandt,
    t.rbukrs,
    t.gjahr,
    t.period,
    t.ifrs_schedule,
    t.ifrs_category,
    t.account_type,
    t.currency,
    SUM(t.closing_balance) as category_total,
    COUNT(DISTINCT t.racct) as account_count,
    SUM(t.debit_amount + t.credit_amount) as transaction_count
FROM TB_TRIAL_BALANCE t
WHERE t.calculation_status = 'APPROVED'
GROUP BY 
    t.mandt,
    t.rbukrs,
    t.gjahr,
    t.period,
    t.ifrs_schedule,
    t.ifrs_category,
    t.account_type,
    t.currency;

-- ============================================================================
-- V_EXCEPTION_SUMMARY - Exception Summary View
-- ============================================================================

CREATE VIEW IF NOT EXISTS V_EXCEPTION_SUMMARY AS
SELECT
    exception_type,
    severity,
    resolution_status,
    COUNT(*) as exception_count,
    SUM(CASE WHEN sla_breached = 1 THEN 1 ELSE 0 END) as sla_breached_count,
    AVG(occurrence_count) as avg_occurrences
FROM TB_EXCEPTIONS
GROUP BY exception_type, severity, resolution_status;

-- ============================================================================
-- V_WORKFLOW_STATISTICS - Workflow Statistics View
-- ============================================================================

CREATE VIEW IF NOT EXISTS V_WORKFLOW_STATISTICS AS
SELECT
    DATE(started_at) as execution_date,
    workflow_type,
    status,
    COUNT(*) as execution_count,
    AVG(total_duration_ms) as avg_duration_ms,
    MIN(total_duration_ms) as min_duration_ms,
    MAX(total_duration_ms) as max_duration_ms,
    AVG(progress_percentage) as avg_progress_pct
FROM TB_WORKFLOW_STATE
GROUP BY DATE(started_at), workflow_type, status;