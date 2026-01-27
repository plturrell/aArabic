-- ============================================================================
-- CV_MULTICURRENCY_BALANCES - Multi-Currency Conversion Calculation View
-- ============================================================================

SET SCHEMA TB_SCHEMA;

CREATE CALCULATION VIEW CV_MULTICURRENCY_BALANCES (
    mandt,
    rbukrs,
    gjahr,
    period,
    racct,
    account_name,
    ifrs_schedule,
    original_currency,
    local_amount,
    target_currency,
    exchange_rate,
    converted_amount,
    conversion_date
)
AS SELECT
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
    COALESCE(r.ukurs / r.ffact * r.tfact, 1.0) as exchange_rate,
    je.hsl * COALESCE(r.ukurs / r.ffact * r.tfact, 1.0) as converted_amount,
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
    AND r.gdatu = (
        SELECT MAX(r2.gdatu)
        FROM TB_EXCHANGE_RATES r2
        WHERE r2.mandt = je.mandt
            AND r2.fcurr = je.rtcur
            AND r2.tcurr = 'USD'
            AND r2.kurst = 'M'
            AND r2.gdatu <= je.budat
            AND r2.is_active = TRUE
    )
WHERE je.validated = TRUE
WITH READ ONLY;

COMMENT ON VIEW CV_MULTICURRENCY_BALANCES IS 'Multi-currency converted balances';

GRANT SELECT ON CV_MULTICURRENCY_BALANCES TO TB_APP_USER;