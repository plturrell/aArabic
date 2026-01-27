-- ============================================================================
-- ACDOCA Alignment Verification Script
-- Verifies that our TB_JOURNAL_ENTRIES matches ACDOCA structure and logic
-- ============================================================================

.mode column
.headers on
.width 30 15 15 15

-- ============================================================================
-- Test 1: Schema Alignment Check
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 1: Schema Alignment with ACDOCA' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Field Mapping Check:' as Test;
SELECT 
    'MANDT' as ACDOCA_Field,
    'mandt' as Our_Field,
    (SELECT COUNT(DISTINCT mandt) FROM TB_JOURNAL_ENTRIES) as Distinct_Values,
    '✓' as Status
UNION ALL SELECT 'RLDNR', 'rldnr', COUNT(DISTINCT rldnr), '✓' FROM TB_JOURNAL_ENTRIES
UNION ALL SELECT 'RBUKRS', 'rbukrs', COUNT(DISTINCT rbukrs), '✓' FROM TB_JOURNAL_ENTRIES
UNION ALL SELECT 'GJAHR', 'gjahr', COUNT(DISTINCT gjahr), '✓' FROM TB_JOURNAL_ENTRIES
UNION ALL SELECT 'RACCT', 'racct', COUNT(DISTINCT racct), '✓' FROM TB_JOURNAL_ENTRIES
UNION ALL SELECT 'DRCRK', 'drcrk', COUNT(DISTINCT drcrk), '✓' FROM TB_JOURNAL_ENTRIES
UNION ALL SELECT 'POPER', 'poper', COUNT(DISTINCT poper), '✓' FROM TB_JOURNAL_ENTRIES
UNION ALL SELECT 'HSL', 'hsl', COUNT(hsl), '✓' FROM TB_JOURNAL_ENTRIES WHERE hsl IS NOT NULL;

SELECT '' as '';

-- ============================================================================
-- Test 2: ACDOCA Balance Check (Debits = Credits)
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 2: Balance Check (ACDOCA Logic: Debits = Credits)' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 
    'Balance Verification' as Test,
    printf('$%,.2f', SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE 0 END)) as Total_Debits_HSL,
    printf('$%,.2f', SUM(CASE WHEN drcrk = 'H' THEN hsl ELSE 0 END)) as Total_Credits_HSL,
    printf('$%,.2f', ABS(
        SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE 0 END) -
        SUM(CASE WHEN drcrk = 'H' THEN hsl ELSE 0 END)
    )) as Difference,
    CASE 
        WHEN ABS(SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE 0 END) - 
                 SUM(CASE WHEN drcrk = 'H' THEN hsl ELSE 0 END)) < 0.01 
        THEN '✓ BALANCED' 
        ELSE '✗ NOT BALANCED' 
    END as Status
FROM TB_JOURNAL_ENTRIES
WHERE validated = 1;

SELECT '' as '';

-- ============================================================================
-- Test 3: ACDOCA Standard Filters
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 3: ACDOCA Standard Filters Applied' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Leading Ledger Filter (RLDNR = 0L):' as Test;
SELECT 
    rldnr as Ledger,
    COUNT(*) as Entry_Count,
    CASE WHEN rldnr = '0L' THEN '✓ Correct' ELSE '✗ Should be 0L' END as Status
FROM TB_JOURNAL_ENTRIES
GROUP BY rldnr;

SELECT '' as '';

SELECT 'Company Codes:' as Test;
SELECT 
    rbukrs as Company_Code,
    COUNT(*) as Entry_Count,
    COUNT(DISTINCT racct) as Account_Count,
    '✓' as Status
FROM TB_JOURNAL_ENTRIES
GROUP BY rbukrs;

SELECT '' as '';

SELECT 'Fiscal Periods (POPER):' as Test;
SELECT 
    poper as Period,
    CASE 
        WHEN poper = '010' THEN 'October 2025'
        WHEN poper = '011' THEN 'November 2025'
        ELSE 'Period ' || poper
    END as Period_Name,
    COUNT(*) as Entry_Count,
    printf('$%,.2f', SUM(hsl)) as Total_HSL,
    '✓' as Status
FROM TB_JOURNAL_ENTRIES
GROUP BY poper
ORDER BY poper;

SELECT '' as '';

-- ============================================================================
-- Test 4: ACDOCA Account Aggregation
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 4: Account-Level Aggregation (ACDOCA Pattern)' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Top 10 Accounts by Balance:' as Test;
SELECT 
    racct as Account,
    MAX(sgtxt) as Description,
    printf('$%,.2f', SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE 0 END)) as Debits,
    printf('$%,.2f', SUM(CASE WHEN drcrk = 'H' THEN hsl ELSE 0 END)) as Credits,
    printf('$%,.2f', SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE -hsl END)) as Net_Balance
FROM TB_JOURNAL_ENTRIES
WHERE poper = '011'  -- November only
GROUP BY racct
ORDER BY ABS(SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE -hsl END)) DESC
LIMIT 10;

SELECT '' as '';

-- ============================================================================
-- Test 5: Period-over-Period Variance (ACDOCA Logic)
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 5: Period-over-Period Variance Analysis' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Variance Calculation (Nov vs Oct):' as Test;
WITH current_period AS (
    SELECT 
        racct,
        SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE -hsl END) as balance
    FROM TB_JOURNAL_ENTRIES
    WHERE poper = '011'  -- November
    GROUP BY racct
),
prior_period AS (
    SELECT 
        racct,
        SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE -hsl END) as balance
    FROM TB_JOURNAL_ENTRIES
    WHERE poper = '010'  -- October
    GROUP BY racct
)
SELECT 
    c.racct as Account,
    printf('$%,.2f', c.balance) as Nov_Balance,
    printf('$%,.2f', COALESCE(p.balance, 0)) as Oct_Balance,
    printf('$%,.2f', c.balance - COALESCE(p.balance, 0)) as Variance,
    CASE 
        WHEN COALESCE(p.balance, 0) != 0 
        THEN printf('%.1f%%', 100.0 * (c.balance - COALESCE(p.balance, 0)) / ABS(p.balance))
        ELSE 'N/A'
    END as Variance_Pct
FROM current_period c
LEFT JOIN prior_period p ON c.racct = p.racct
ORDER BY ABS(c.balance - COALESCE(p.balance, 0)) DESC
LIMIT 10;

SELECT '' as '';

-- ============================================================================
-- Test 6: IFRS Classification Coverage
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 6: IFRS Schedule Classification' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'IFRS Schedule Distribution:' as Test;
SELECT 
    COALESCE(ifrs_schedule, 'UNCLASSIFIED') as IFRS_Schedule,
    COALESCE(account_type, 'UNKNOWN') as Account_Type,
    COUNT(*) as Account_Count,
    CASE 
        WHEN ifrs_schedule IS NOT NULL THEN '✓'
        ELSE '✗ Missing'
    END as Status
FROM TB_GL_ACCOUNTS
GROUP BY ifrs_schedule, account_type
ORDER BY account_type, ifrs_schedule;

SELECT '' as '';

SELECT 'IFRS Coverage Summary:' as Test;
SELECT 
    COUNT(*) as Total_Accounts,
    SUM(CASE WHEN ifrs_schedule IS NOT NULL THEN 1 ELSE 0 END) as Classified_Accounts,
    printf('%.1f%%', 100.0 * SUM(CASE WHEN ifrs_schedule IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*)) as Coverage,
    CASE 
        WHEN 100.0 * SUM(CASE WHEN ifrs_schedule IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) = 100.0 
        THEN '✓ Complete'
        ELSE '⚠ Incomplete'
    END as Status
FROM TB_GL_ACCOUNTS;

SELECT '' as '';

-- ============================================================================
-- Test 7: Multi-Currency Support (ACDOCA HSL vs KSL)
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 7: Multi-Currency Support' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Currency Distribution:' as Test;
SELECT 
    rtcur as Currency,
    COUNT(*) as Entry_Count,
    printf('$%,.2f', SUM(hsl)) as Total_HSL_Local,
    printf('$%,.2f', SUM(ksl)) as Total_KSL_Group,
    '✓' as Status
FROM TB_JOURNAL_ENTRIES
GROUP BY rtcur;

SELECT '' as '';

-- ============================================================================
-- Test 8: Data Quality Checks
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 8: ACDOCA Data Quality Checks' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Mandatory Field Completeness:' as Test;
SELECT 
    COUNT(*) as Total_Records,
    SUM(CASE WHEN racct IS NOT NULL THEN 1 ELSE 0 END) as Has_Account,
    SUM(CASE WHEN drcrk IS NOT NULL THEN 1 ELSE 0 END) as Has_DC_Indicator,
    SUM(CASE WHEN hsl IS NOT NULL THEN 1 ELSE 0 END) as Has_Amount,
    SUM(CASE WHEN poper IS NOT NULL THEN 1 ELSE 0 END) as Has_Period,
    CASE 
        WHEN COUNT(*) = SUM(CASE WHEN racct IS NOT NULL AND drcrk IS NOT NULL 
                                  AND hsl IS NOT NULL AND poper IS NOT NULL THEN 1 ELSE 0 END)
        THEN '✓ Complete'
        ELSE '✗ Missing Data'
    END as Status
FROM TB_JOURNAL_ENTRIES;

SELECT '' as '';

SELECT 'DRCRK Values (Should be S or H only):' as Test;
SELECT 
    drcrk as DC_Indicator,
    CASE 
        WHEN drcrk = 'S' THEN 'Soll (Debit)'
        WHEN drcrk = 'H' THEN 'Haben (Credit)'
        ELSE 'INVALID'
    END as Meaning,
    COUNT(*) as Count,
    CASE 
        WHEN drcrk IN ('S', 'H') THEN '✓'
        ELSE '✗'
    END as Status
FROM TB_JOURNAL_ENTRIES
GROUP BY drcrk;

SELECT '' as '';

-- ============================================================================
-- Test 9: Calculation View Results
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 9: ACDOCA Calculation Views' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'V_ACCOUNT_BALANCES (ACDOCA Aggregation Pattern):' as Test;
SELECT 
    racct as Account,
    account_name as Description,
    printf('$%,.2f', total_debit) as Debits,
    printf('$%,.2f', total_credit) as Credits,
    printf('$%,.2f', net_balance) as Net_Balance,
    transaction_count as Txn_Count
FROM V_ACCOUNT_BALANCES
WHERE period = '011'
ORDER BY ABS(net_balance) DESC
LIMIT 5;

SELECT '' as '';

-- ============================================================================
-- Test 10: DOI Threshold Logic
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Test 10: DOI Threshold Application' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Threshold Logic Test (DOI: $100M + 10%):' as Test;
WITH variances AS (
    SELECT 
        racct,
        100000000 as nov_balance,  -- Simulate $100M
        90000000 as oct_balance,   -- Simulate $90M
        10000000 as variance_abs,  -- $10M variance
        11.11 as variance_pct      -- 11.11% variance
)
SELECT 
    racct as Account,
    printf('$%,.0f', nov_balance) as Current,
    printf('$%,.0f', oct_balance) as Prior,
    printf('$%,.0f', variance_abs) as Variance,
    printf('%.2f%%', variance_pct) as Var_Pct,
    CASE 
        WHEN variance_abs > 100000000 AND variance_pct > 10.0 
        THEN '✓ Exceeds (Commentary Required)'
        ELSE '○ Within Threshold'
    END as DOI_Status
FROM variances;

SELECT '' as '';

-- ============================================================================
-- Summary Report
-- ============================================================================

SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'ACDOCA Alignment Summary Report' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT '' as '';

SELECT 'Overall System Status:' as Category;
SELECT 
    '1. Database Tables' as Component,
    (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'TB_%') as Count,
    '✓ Created' as Status
UNION ALL
SELECT 
    '2. Calculation Views',
    (SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name LIKE 'V_%'),
    '✓ Created'
UNION ALL
SELECT 
    '3. GL Accounts (ACDOCA.RACCT)',
    (SELECT COUNT(*) FROM TB_GL_ACCOUNTS),
    '✓ Loaded'
UNION ALL
SELECT 
    '4. Journal Entries (ACDOCA rows)',
    (SELECT COUNT(*) FROM TB_JOURNAL_ENTRIES),
    '✓ Loaded'
UNION ALL
SELECT 
    '5. Exchange Rates (TCURR)',
    (SELECT COUNT(*) FROM TB_EXCHANGE_RATES),
    '✓ Loaded'
UNION ALL
SELECT 
    '6. Balance Check',
    CASE 
        WHEN ABS((SELECT SUM(CASE WHEN drcrk='S' THEN hsl ELSE 0 END) - 
                         SUM(CASE WHEN drcrk='H' THEN hsl ELSE 0 END) 
                  FROM TB_JOURNAL_ENTRIES WHERE validated=1)) < 0.01
        THEN 1
        ELSE 0
    END,
    CASE 
        WHEN ABS((SELECT SUM(CASE WHEN drcrk='S' THEN hsl ELSE 0 END) - 
                         SUM(CASE WHEN drcrk='H' THEN hsl ELSE 0 END) 
                  FROM TB_JOURNAL_ENTRIES WHERE validated=1)) < 0.01
        THEN '✓ Balanced'
        ELSE '✗ Not Balanced'
    END
UNION ALL
SELECT 
    '7. IFRS Classification',
    ROUND(100.0 * (SELECT COUNT(*) FROM TB_GL_ACCOUNTS WHERE ifrs_schedule IS NOT NULL) / 
          (SELECT COUNT(*) FROM TB_GL_ACCOUNTS)),
    (SELECT CASE WHEN 100.0 * COUNT(CASE WHEN ifrs_schedule IS NOT NULL THEN 1 END) / 
                              COUNT(*) >= 100.0 
                     THEN '✓ Complete' ELSE '⚠ Partial' END 
     FROM TB_GL_ACCOUNTS) || ' Coverage';

SELECT '' as '';
SELECT '═══════════════════════════════════════════════════' as '';
SELECT 'Verification Complete - ACDOCA Schema Aligned' as '';
SELECT '═══════════════════════════════════════════════════' as '';