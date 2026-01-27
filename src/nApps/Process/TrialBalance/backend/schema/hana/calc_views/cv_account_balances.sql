-- ============================================================================
-- CV_ACCOUNT_BALANCES - Account Balance Aggregation Calculation View
-- ============================================================================

SET SCHEMA TB_SCHEMA;

CREATE CALCULATION VIEW CV_ACCOUNT_BALANCES (
    mandt,
    rbukrs,
    gjahr,
    period,
    racct,
    account_name,
    kostl,
    prctr,
    segment,
    currency,
    total_debit,
    total_credit,
    net_balance,
    transaction_count
)
AS SELECT
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
WHERE je.validated = TRUE
GROUP BY 
    je.mandt,
    je.rbukrs,
    je.gjahr,
    je.poper,
    je.racct,
    je.kostl,
    je.prctr,
    je.segment,
    je.rwcur
WITH READ ONLY;

COMMENT ON VIEW CV_ACCOUNT_BALANCES IS 'Aggregated account balances by period';

-- Grant access
GRANT SELECT ON CV_ACCOUNT_BALANCES TO TB_APP_USER;