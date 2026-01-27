-- ============================================================================
-- CV_IFRS_SUMMARY - IFRS Schedule Summary Calculation View
-- ============================================================================

SET SCHEMA TB_SCHEMA;

CREATE CALCULATION VIEW CV_IFRS_SUMMARY (
    mandt,
    rbukrs,
    gjahr,
    period,
    ifrs_schedule,
    ifrs_category,
    account_type,
    currency,
    category_total,
    account_count,
    transaction_count
)
AS SELECT
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
    t.currency
WITH READ ONLY;

COMMENT ON VIEW CV_IFRS_SUMMARY IS 'IFRS schedule summary with aggregated balances';

GRANT SELECT ON CV_IFRS_SUMMARY TO TB_APP_USER;