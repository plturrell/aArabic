# SAP S/4HANA ACDOCA Table Schema Reference

## Overview

**ACDOCA** (Accounting Documents - Actual) is the **Universal Journal** table in SAP S/4HANA that stores all accounting line items in a single unified table. This replaces multiple tables from SAP ECC (BSEG, BSIS, BSAS, etc.).

## Key Characteristics

- **Type**: Transparent table (database table)
- **Package**: FINS_ACDOC_REPORTING
- **Delivery Class**: A (Application table)
- **Primary Key**: MANDT, RCLNT, RLDNR, RBUKRS, GJAHR, BELNR, DOCLN
- **Size**: Can reach billions of rows in large enterprises
- **Storage**: HANA column store optimized

## Complete Field List (169 Fields)

### Key Fields (Primary Key Components)

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| MANDT | CLNT | 3 | Client | 100 |
| RCLNT | CLNT | 3 | Client of source document | 100 |
| RLDNR | CHAR | 2 | Ledger | 0L (Leading ledger) |
| RBUKRS | CHAR | 4 | Company Code | 1000, HKG |
| GJAHR | CHAR | 4 | Fiscal Year | 2025 |
| BELNR | CHAR | 10 | Document Number | 1900000001 |
| DOCLN | CHAR | 6 | Line Item (ACDOCA-specific) | 000001 |

### Document Header Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| BLART | CHAR | 2 | Document Type | SA, DR, DZ, KR |
| BLDAT | DATS | 8 | Document Date | 20250115 |
| BUDAT | DATS | 8 | **Posting Date** | 20250115 |
| CPUDT | DATS | 8 | Entry Date | 20250115 |
| CPUTM | TIMS | 6 | Entry Time | 143025 |
| USNAM | CHAR | 12 | User Name | FINANCEU01 |
| TCODE | CHAR | 20 | Transaction Code | FB50, FB01 |

### Accounting Document Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| RACCT | CHAR | 10 | **Account Number (GL)** | 100000, 200000 |
| DRCRK | CHAR | 1 | **Debit/Credit Indicator** | S (Debit), H (Credit) |
| POPER | CHAR | 3 | **Posting Period** | 001-012 |
| RYEAR | CHAR | 4 | Fiscal Year Variant | K4 |
| PERIO | CHAR | 7 | Reporting Period | 2025001 |

### Amount Fields (Critical for Trial Balance)

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| **TSL** | CURR | 23,2 | Amount in Transaction Currency | 100000.00 |
| **HSL** | CURR | 23,2 | **Amount in Local Currency** | 100000.00 |
| **WSL** | CURR | 23,2 | Amount in Document Currency | 100000.00 |
| **KSL** | CURR | 23,2 | **Amount in Group Currency** | 128300.00 |
| RTCUR | CUKY | 5 | Transaction Currency | HKD, USD, EUR |
| RWCUR | CUKY | 5 | Company Code Currency | HKD |
| RHCUR | CUKY | 5 | Local Currency | HKD |
| RKCUR | CUKY | 5 | Group Currency | USD |

### Organizational Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| **KOSTL** | CHAR | 10 | **Cost Center** | CC-1000 |
| **PRCTR** | CHAR | 10 | **Profit Center** | PC-HKG |
| **SEGMENT** | CHAR | 10 | **Segment** | RETAIL |
| PSEGMENT | CHAR | 10 | Partner Segment | WHOLESALE |
| GSBER | CHAR | 4 | Business Area | 1000 |
| KOKRS | CHAR | 4 | Controlling Area | A000 |

### Subledger Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| KUNNR | CHAR | 10 | Customer Number | 0000100001 |
| LIFNR | CHAR | 10 | Vendor Number | 0000200001 |
| BSCHL | CHAR | 2 | Posting Key | 01, 31, 40, 50 |
| KOART | CHAR | 1 | Account Type | S, D, K, A, M |

### Document Reference Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| AWTYP | CHAR | 5 | Reference Transaction | BKPF, VBRK, MKPF |
| AWREF | CHAR | 10 | Reference Number | 1900000001 |
| AWORG | CHAR | 10 | Reference Organization | 1000 |
| AWSYS | CHAR | 10 | Logical System | S4HCLNT100 |
| SGTXT | CHAR | 50 | **Line Item Text** | Payment to vendor |
| XREF1 | CHAR | 12 | Reference Key 1 | INV-2025-001 |
| XREF2 | CHAR | 12 | Reference Key 2 | PO-2025-123 |
| XREF3 | CHAR | 20 | Reference Key 3 | Contract-456 |

### Tax Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| MWSKZ | CHAR | 2 | Tax Code | V0, V1 |
| TXJCD | CHAR | 15 | Tax Jurisdiction Code | TXJCD01 |
| WMWST | CURR | 13,2 | Tax Amount in Document Currency | 100.00 |

### Payment Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| ZTERM | CHAR | 4 | Terms of Payment | 0001, Z030 |
| ZBD1T | DEC | 3 | Cash Discount Days 1 | 10 |
| ZBD1P | DEC | 5,3 | Cash Discount % 1 | 2.000 |
| SHKZG | CHAR | 1 | Debit/Credit Indicator (Legacy) | S, H |

### Asset Accounting Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| ANLN1 | CHAR | 12 | Main Asset Number | 100000 |
| ANLN2 | CHAR | 4 | Asset Sub-number | 0001 |
| BZDAT | DATS | 8 | Asset Value Date | 20250115 |

### Material Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| MATNR | CHAR | 40 | Material Number | 000000000100000001 |
| WERKS | CHAR | 4 | Plant | 1000 |
| MENGE | QUAN | 13,3 | Quantity | 100.000 |
| MEINS | UNIT | 3 | Base Unit of Measure | EA, KG |

### Sales/Distribution Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| VBELN | CHAR | 10 | Sales Document | 0000000001 |
| VBEL2 | CHAR | 10 | Delivery | 0000000001 |
| VBEL3 | CHAR | 10 | Billing Document | 0000000001 |
| POSN2 | CHAR | 6 | Sales Item | 000010 |

### Profitability Analysis Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| PAOBJNR | NUMC | 10 | Profitability Segment Number | 1234567890 |
| PASUBNR | NUMC | 5 | Profitability Subsegment | 12345 |

### Statistical Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| XFAKT | CHAR | 1 | Statistical Posting | X |
| XUMAN | CHAR | 1 | Manual Statistical Posting | X |

### Reversal/Clearing Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| XSTOV | CHAR | 1 | Reversal Posting | X |
| STBLG | CHAR | 10 | Reversal Document Number | 1900000002 |
| STJAH | CHAR | 4 | Reversal Fiscal Year | 2025 |
| AUGDT | DATS | 8 | Clearing Date | 20250131 |
| AUGBL | CHAR | 10 | Clearing Document | 1900000003 |

### Extension/Enhancement Fields

| Field | Type | Length | Description | Example |
|-------|------|--------|-------------|---------|
| ZZFIELD1 | CHAR | 10 | Custom Field 1 | Custom value |
| ZZFIELD2 | CHAR | 10 | Custom Field 2 | Custom value |

## Essential Fields for Trial Balance

### Minimum Required Fields

For basic Trial Balance calculation, you need these fields:

```sql
SELECT 
    -- Key fields
    MANDT,
    RLDNR,      -- Ledger (0L = Leading)
    RBUKRS,     -- Company Code
    GJAHR,      -- Fiscal Year
    BELNR,      -- Document Number
    DOCLN,      -- Line Item
    
    -- Posting information
    BUDAT,      -- Posting Date (CRITICAL)
    POPER,      -- Posting Period (CRITICAL)
    
    -- Account information
    RACCT,      -- GL Account (CRITICAL)
    DRCRK,      -- Debit/Credit Indicator (CRITICAL)
    
    -- Amounts (CRITICAL)
    HSL,        -- Amount in Local Currency
    KSL,        -- Amount in Group Currency
    RTCUR,      -- Transaction Currency
    RWCUR,      -- Company Code Currency
    
    -- Organizational units
    KOSTL,      -- Cost Center
    PRCTR,      -- Profit Center
    SEGMENT,    -- Segment
    
    -- Document reference
    BLART,      -- Document Type
    SGTXT       -- Line Item Text
    
FROM ACDOCA
WHERE RLDNR = '0L'  -- Leading ledger only
  AND RBUKRS IN ('1000', 'HKG')
  AND GJAHR = '2025'
  AND POPER = '011';
```

## ACDOCA vs TB_JOURNAL_ENTRIES Mapping

### Our Schema Design → ACDOCA Fields

Our `TB_JOURNAL_ENTRIES` table is modeled after ACDOCA:

| Our Field | ACDOCA Field | Type | Purpose |
|-----------|--------------|------|---------|
| entry_id | (Generated) | PK | Unique identifier |
| mandt | MANDT | CLNT | Client |
| rldnr | RLDNR | CHAR(2) | Ledger |
| rbukrs | RBUKRS | CHAR(4) | Company Code |
| gjahr | GJAHR | CHAR(4) | Fiscal Year |
| belnr | BELNR | CHAR(10) | Document Number |
| buzei | DOCLN | CHAR(6) | Line Item |
| budat | BUDAT | DATE | **Posting Date** |
| bldat | BLDAT | DATE | Document Date |
| cpudt | CPUDT | DATE | Entry Date |
| blart | BLART | CHAR(2) | Document Type |
| racct | RACCT | CHAR(10) | **GL Account** |
| rtcur | RTCUR | CHAR(5) | Transaction Currency |
| rwcur | RWCUR | CHAR(5) | Company Currency |
| drcrk | DRCRK | CHAR(1) | **Debit/Credit** |
| poper | POPER | CHAR(3) | **Posting Period** |
| tsl | TSL | DECIMAL | Txn Currency Amt |
| **hsl** | **HSL** | DECIMAL | **Local Currency Amt** |
| wsl | WSL | DECIMAL | Doc Currency Amt |
| **ksl** | **KSL** | DECIMAL | **Group Currency Amt** |
| kostl | KOSTL | CHAR(10) | Cost Center |
| prctr | PRCTR | CHAR(10) | Profit Center |
| segment | SEGMENT | CHAR(10) | Segment |
| sgtxt | SGTXT | CHAR(50) | Line Item Text |
| xref1 | XREF1 | CHAR(12) | Reference Key 1 |
| kunnr | KUNNR | CHAR(10) | Customer |
| lifnr | LIFNR | CHAR(10) | Vendor |

## Critical Fields for Trial Balance

### 1. **HSL (Amount in Local Currency)**

**Most Important Field** for Trial Balance calculation.

```sql
-- Trial Balance formula
SUM(CASE WHEN DRCRK = 'S' THEN HSL ELSE 0 END) as total_debits,
SUM(CASE WHEN DRCRK = 'H' THEN HSL ELSE 0 END) as total_credits
```

**Why HSL?**
- Represents amounts in company code currency
- Used for balance sheet and P&L reporting
- Consistent currency within company code
- Matches statutory reporting requirements

**Alternative: KSL (Group Currency)**
- Used for consolidation across company codes
- Typically USD for multinational groups
- Used in group reporting

### 2. **DRCRK (Debit/Credit Indicator)**

**Values**:
- `S` = Soll (Debit) - German for "Should"
- `H` = Haben (Credit) - German for "Have"

**Critical Logic**:
```zig
// In balance_engine.zig
if (entry.drcrk == 'S') {
    account.debit_amount += entry.hsl;
} else { // 'H'
    account.credit_amount += entry.hsl;
}

// Closing balance
closing = opening + debits - credits;
```

### 3. **POPER (Posting Period)**

**Format**: `001` to `012` (or `016` for special periods)

**Usage**:
```sql
-- Monthly close for November
WHERE POPER = '011'

-- Quarterly close for Q4
WHERE POPER IN ('010', '011', '012')

-- Year-to-date
WHERE POPER <= '011'
```

### 4. **RLDNR (Ledger)**

**Standard Values**:
- `0L` = Leading Ledger (statutory reporting)
- `2L` = Management Ledger
- `3L` = Tax Ledger
- `9L` = IFRS Ledger

**Best Practice**: Always filter on `RLDNR = '0L'` for Trial Balance

### 5. **RACCT (GL Account)**

**Format**: 10-character, right-padded with zeros
**Example**: `0000100000` for account 100000

**In our system**:
```sql
-- Store as VARCHAR without leading zeros
INSERT INTO TB_JOURNAL_ENTRIES (racct) VALUES ('100000');

-- Join to GL master
LEFT JOIN TB_GL_ACCOUNTS ga 
    ON je.racct = ga.saknr;
```

## ACDOCA Query Patterns

### Standard Trial Balance Extraction

```sql
-- Extract for Trial Balance calculation
SELECT 
    MANDT,
    RLDNR,
    RBUKRS,
    GJAHR,
    BELNR,
    DOCLN as BUZEI,
    BUDAT,
    BLDAT,
    BLART,
    RACCT,
    RTCUR,
    RWCUR,
    DRCRK,
    POPER,
    HSL,
    KSL,
    KOSTL,
    PRCTR,
    SEGMENT,
    SGTXT,
    KUNNR,
    LIFNR
FROM ACDOCA
WHERE RLDNR = '0L'                    -- Leading ledger only
  AND RBUKRS = 'HKG'                  -- Company code
  AND GJAHR = '2025'                  -- Fiscal year
  AND POPER = '011'                   -- November
  AND XSTOV = ''                      -- Not a reversal
  ORDER BY RACCT, BUDAT, BELNR, DOCLN;
```

### Account Balance Aggregation

```sql
-- Aggregate to account level (mimics our CV_ACCOUNT_BALANCES)
SELECT 
    RBUKRS,
    GJAHR,
    POPER,
    RACCT,
    SUM(CASE WHEN DRCRK = 'S' THEN HSL ELSE 0 END) as DEBIT_AMOUNT,
    SUM(CASE WHEN DRCRK = 'H' THEN HSL ELSE 0 END) as CREDIT_AMOUNT,
    SUM(CASE WHEN DRCRK = 'S' THEN HSL ELSE -HSL END) as NET_BALANCE,
    COUNT(*) as LINE_ITEM_COUNT
FROM ACDOCA
WHERE RLDNR = '0L'
  AND RBUKRS = 'HKG'
  AND GJAHR = '2025'
  AND POPER = '011'
GROUP BY RBUKRS, GJAHR, POPER, RACCT;
```

### Multi-Currency Balances

```sql
-- With FX conversion (mimics our CV_MULTICURRENCY_BALANCES)
SELECT 
    a.RBUKRS,
    a.GJAHR,
    a.POPER,
    a.RACCT,
    a.RTCUR as ORIGINAL_CURRENCY,
    SUM(a.HSL) as LOCAL_AMOUNT,
    'USD' as TARGET_CURRENCY,
    t.UKURS as EXCHANGE_RATE,
    SUM(a.KSL) as CONVERTED_AMOUNT
FROM ACDOCA a
LEFT JOIN TCURR t 
    ON a.RTCUR = t.FCURR
    AND t.TCURR = 'USD'
    AND t.KURST = 'M'
    AND t.GDATU = (
        SELECT MAX(t2.GDATU)
        FROM TCURR t2
        WHERE t2.FCURR = a.RTCUR
          AND t2.TCURR = 'USD'
          AND t2.GDATU <= a.BUDAT
    )
WHERE a.RLDNR = '0L'
GROUP BY a.RBUKRS, a.GJAHR, a.POPER, a.RACCT, a.RTCUR, t.UKURS;
```

### Period-over-Period Comparison

```sql
-- Nov vs Oct comparison (for variance analysis)
WITH current_period AS (
    SELECT RACCT, SUM(CASE WHEN DRCRK = 'S' THEN HSL ELSE -HSL END) as balance
    FROM ACDOCA
    WHERE RLDNR = '0L' AND RBUKRS = 'HKG' AND GJAHR = '2025' AND POPER = '011'
    GROUP BY RACCT
),
prior_period AS (
    SELECT RACCT, SUM(CASE WHEN DRCRK = 'S' THEN HSL ELSE -HSL END) as balance
    FROM ACDOCA
    WHERE RLDNR = '0L' AND RBUKRS = 'HKG' AND GJAHR = '2025' AND POPER = '010'
    GROUP BY RACCT
)
SELECT 
    c.RACCT,
    c.balance as nov_balance,
    p.balance as oct_balance,
    (c.balance - p.balance) as variance_absolute,
    ROUND(100.0 * (c.balance - p.balance) / NULLIF(p.balance, 0), 2) as variance_percentage
FROM current_period c
LEFT JOIN prior_period p ON c.RACCT = p.RACCT
WHERE ABS(c.balance - p.balance) > 100000000  -- $100M threshold
  AND ABS((c.balance - p.balance) / NULLIF(p.balance, 0)) > 0.10;  -- 10% threshold
```

## ACDOCA Performance Considerations

### Indexing Strategy

**SAP Standard Indexes**:
1. Primary Index: `MANDT, RLDNR, RBUKRS, GJAHR, BELNR, DOCLN`
2. Secondary Index: `RBUKRS, RACCT, GJAHR, POPER`
3. Additional: `BUDAT`, `PRCTR`, `KOSTL`

**Our Indexes** (matching ACDOCA strategy):
```sql
CREATE INDEX idx_tb_je_company_year ON TB_JOURNAL_ENTRIES(mandt, rbukrs, gjahr);
CREATE INDEX idx_tb_je_account ON TB_JOURNAL_ENTRIES(racct);
CREATE INDEX idx_tb_je_posting_date ON TB_JOURNAL_ENTRIES(budat);
```

### Query Optimization

**Best Practices**:
```sql
-- ✅ GOOD: Filter on indexed fields first
SELECT * FROM ACDOCA
WHERE RLDNR = '0L'      -- Indexed
  AND RBUKRS = 'HKG'    -- Indexed
  AND GJAHR = '2025'    -- Indexed
  AND POPER = '011';    -- Indexed

-- ❌ BAD: Missing key filters
SELECT * FROM ACDOCA
WHERE RACCT = '100000'  -- Slow without company/year filters
```

**Partition by Fiscal Year**: ACDOCA is typically partitioned by `GJAHR` for performance

## Data Extraction Best Practices

### 1. **Filter Early and Aggressively**

```sql
-- Extract only what's needed
WHERE RLDNR = '0L'              -- Leading ledger
  AND RBUKRS IN ('1000', 'HKG') -- Specific company codes
  AND GJAHR = '2025'            -- Current fiscal year
  AND POPER BETWEEN '001' AND '012'  -- Regular periods only
  AND XSTOV = ''                -- Exclude reversals
```

### 2. **Use Appropriate Ledger**

| Ledger | Use Case |
|--------|----------|
| 0L | Statutory reporting, Trial Balance |
| 2L | Management reporting |
| 3L | Tax calculations |
| 9L | IFRS adjustments |

### 3. **Handle Special Periods**

```sql
-- Regular periods: 001-012
-- Special periods: 013-016 (year-end adjustments)

-- Exclude special periods for monthly close
WHERE POPER BETWEEN '001' AND '012'

-- Include special periods for year-end
WHERE POPER <= '016'
```

### 4. **Currency Selection**

| Field | When to Use |
|-------|-------------|
| TSL | Transaction-level detail |
| **HSL** | **Company code reporting** |
| WSL | Document currency (rare) |
| **KSL** | **Group consolidation** |

## ACDOCA to Our Schema ETL

### Extraction SQL (SAP S/4HANA)

```sql
-- Run this in S/4HANA to extract data
SELECT 
    CONCAT(MANDT, '_', RLDNR, '_', RBUKRS, '_', GJAHR, '_', BELNR, '_', DOCLN) as entry_id,
    MANDT as mandt,
    RLDNR as rldnr,
    RBUKRS as rbukrs,
    GJAHR as gjahr,
    BELNR as belnr,
    DOCLN as buzei,
    BUDAT as budat,
    BLDAT as bldat,
    CPUDT as cpudt,
    BLART as blart,
    RACCT as racct,
    RTCUR as rtcur,
    RWCUR as rwcur,
    DRCRK as drcrk,
    POPER as poper,
    TSL as tsl,
    HSL as hsl,
    WSL as wsl,
    KSL as ksl,
    KOSTL as kostl,
    PRCTR as prctr,
    SEGMENT as segment,
    SGTXT as sgtxt,
    XREF1 as xref1,
    XREF2 as xref2,
    XREF3 as xref3,
    KUNNR as kunnr,
    LIFNR as lifnr
FROM ACDOCA
WHERE RLDNR = '0L'
  AND RBUKRS = 'HKG'
  AND GJAHR = '2025'
  AND POPER = '011'
INTO TABLE @DATA(lt_journal_entries);

-- Export to CSV or call REST API
```

### Transformation Logic

```python
# In our load_sample_to_sqlite.py (enhanced)
def transform_acdoca_row(row):
    """Transform ACDOCA row to TB_JOURNAL_ENTRIES format"""
    return {
        'entry_id': f"{row['MANDT']}_{row['RLDNR']}_{row['RBUKRS']}_{row['GJAHR']}_{row['BELNR']}_{row['DOCLN']}",
        'mandt': row['MANDT'],
        'rldnr': row['RLDNR'],
        'rbukrs': row['RBUKRS'],
        'gjahr': row['GJAHR'],
        'belnr': row['BELNR'],
        'buzei': row['DOCLN'],  # Note: ACDOCA uses DOCLN, not BUZEI
        'budat': format_sap_date(row['BUDAT']),
        'racct': row['RACCT'].lstrip('0'),  # Remove leading zeros
        'drcrk': row['DRCRK'],
        'hsl': float(row['HSL']),
        'ksl': float(row['KSL']),
        'poper': row['POPER'],
        # ... other fields
    }
```

## Sample Data Validation

### Verify ACDOCA Alignment

```sql
-- Check 1: Verify debit/credit balance
SELECT 
    'Balance Check' as test,
    SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE 0 END) as total_debits,
    SUM(CASE WHEN drcrk = 'H' THEN hsl ELSE 0 END) as total_credits,
    ABS(
        SUM(CASE WHEN drcrk = 'S' THEN hsl ELSE 0 END) -
        SUM(CASE WHEN drcrk = 'H' THEN hsl ELSE 0 END)
    ) as difference
FROM TB_JOURNAL_ENTRIES
WHERE rbukrs = 'HKG' AND gjahr = '2025' AND poper = '011';

-- Expected: difference < 0.01 (balanced)
```

```sql
-- Check 2: Verify period data exists
SELECT 
    poper as period,
    COUNT(*) as entry_count,
    COUNT(DISTINCT racct) as account_count,
    SUM(hsl) as total_amount
FROM TB_JOURNAL_ENTRIES
WHERE rbukrs = 'HKG' AND gjahr = '2025'
GROUP BY poper
ORDER BY poper;

-- Expected: Entries for periods 010 (Oct) and 011 (Nov)
```

```sql
-- Check 3: Verify key field population
SELECT 
    COUNT(*) as total_records,
    COUNT(racct) as has_account,
    COUNT(drcrk) as has_dr_cr,
    COUNT(hsl) as has_amount,
    COUNT(poper) as has_period,
    ROUND(100.0 * COUNT(racct) / COUNT(*), 2) as completeness_pct
FROM TB_JOURNAL_ENTRIES;

-- Expected: completeness_pct = 100.00
```

## IFRS Classification in ACDOCA

### Standard Practice

ACDOCA doesn't natively contain IFRS classifications. These are typically:

1. **Stored in Z-tables** (custom tables)
2. **Maintained in GL Account Master** (SKA1 extensions)
3. **Derived via mapping tables**

### Our Approach

```sql
-- Extend TB_GL_ACCOUNTS with IFRS mapping
CREATE TABLE TB_GL_ACCOUNTS (
    saknr VARCHAR(10),
    txt50 VARCHAR(50),
    ifrs_schedule VARCHAR(10),  -- Added for IFRS
    ifrs_category VARCHAR(50),  -- Added for IFRS
    account_type VARCHAR(20),   -- Asset/Liability/Equity/Revenue/Expense
    -- ...
);

-- Join during extraction
SELECT 
    a.RACCT,
    a.HSL,
    g.ifrs_schedule,  -- From extended master
    g.account_type
FROM ACDOCA a
LEFT JOIN SKA1 s ON a.RACCT = s.SAKNR
LEFT JOIN ZIFRS_MAPPING z ON s.SAKNR = z.ACCOUNT;  -- Custom Z-table
```

## Real-World ACDOCA Volumes

### Typical Sizes

| Company Size | Daily Entries | Monthly Entries | Annual Entries |
|-------------|---------------|-----------------|----------------|
| Small | 1,000 | 20,000 | 250,000 |
| Medium | 10,000 | 200,000 | 2.5M |
| Large | 100,000 | 2,000,000 | 25M |
| Enterprise | 1,000,000+ | 20,000,000+ | 250M+ |

**HKG Sample**: ~100-500 accounts × 12 periods = 1,200-6,000 line items (Small)

### Performance Targets

| Operation | Target Time | ACDOCA Approach |
|-----------|-------------|-----------------|
| Extract 1 month | < 30 sec | Filter on RBUKRS, GJAHR, POPER |
| Aggregate accounts | < 10 sec | Use HANA calculation view |
| Calculate TB | < 5 sec | Parallel aggregation |
| Variance analysis | < 5 sec | Use history table |

## ACDOCA Extensions

### Common Z-Fields

Many organizations extend ACDOCA with custom fields:

```abap
" In SE11 - Append Structure to ACDOCA
TYPES: BEGIN OF zz_acdoca_ext,
    zzifrs_schedule TYPE zifrs_schedule,    " IFRS Schedule
    zzfund          TYPE zfund,              " Fund
    zzproject       TYPE zproject,           " Project
    zzgrant         TYPE zgrant,             " Grant
END OF zz_acdoca_ext.
```

**Our Schema Accommodates This**:
```sql
-- Custom fields can be added
ALTER TABLE TB_JOURNAL_ENTRIES 
ADD COLUMN custom_field1 VARCHAR(50);
```

## Migration from ECC to S/4HANA

### ECC Legacy Tables → ACDOCA

| ECC Table | Description | ACDOCA Equivalent |
|-----------|-------------|-------------------|
| BSEG | Accounting Document Segment | ACDOCA (all fields) |
| BSIS | G/L Account Line Items | ACDOCA (where KOART='S') |
| BSAS | G/L Account Cleared Items | ACDOCA (where AUGDT <> '') |
| BSID | Customer Line Items | ACDOCA (where KOART='D') |
| BSAD | Customer Cleared Items | ACDOCA (where KOART='D' AND AUGDT <> '') |
| BSIK | Vendor Line Items | ACDOCA (where KOART='K') |
| BSAK | Vendor Cleared Items | ACDOCA (where KOART='K' AND AUGDT <> '') |

**Key Difference**: ACDOCA is a **single unified table** vs. multiple ECC tables

## Integration with Our System

### Data Flow from SAP S/4HANA

```
┌──────────────────────────────────────────────────────────────┐
│ SAP S/4HANA                                                  │
│                                                              │
│ ACDOCA Table (Billions of rows)                             │
│   ├─ All journal entries since S/4 go-live                  │
│   ├─ Real-time updates                                      │
│   └─ Ledger-specific views                                  │
└────────────┬─────────────────────────────────────────────────┘
             │
             │ SAP OData/RFC/CDS View
             ↓
┌──────────────────────────────────────────────────────────────┐
│ Extraction Service (s4_extractor component)                 │
│   ├─ Filter: Company, Year, Period                          │
│   ├─ Transform: Field mapping                               │
│   └─ Load: Insert into TB_JOURNAL_ENTRIES                   │
└────────────┬─────────────────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────────────────┐
│ Our Database (HANA Cloud / SQLite)                          │
│                                                              │
│ TB_JOURNAL_ENTRIES (ACDOCA subset)                          │
│   ├─ Only current year data                                 │
│   ├─ Only leading ledger (0L)                               │
│   └─ Enhanced with IFRS classification                      │
└──────────────────────────────────────────────────────────────┘
```

### Why We Mirror ACDOCA Structure

1. **Compatibility**: Easy data exchange with S/4HANA
2. **Familiarity**: SAP consultants recognize field names
3. **Standards**: Follows SAP best practices
4. **Future-proof**: Direct mapping for go-live
5. **Testing**: Can validate against ACDOCA data

## Reference SQL for Developers

### Get ACDOCA Structure

```sql
-- In S/4HANA system
SELECT 
    FIELDNAME,
    KEYFLAG,
    DATATYPE,
    LENG,
    DECIMALS,
    ROLLNAME,
    DDTEXT
FROM DD03L
WHERE TABNAME = 'ACDOCA'
  AND AS4LOCAL = 'A'
ORDER BY POSITION;
```

### Test Query Template

```sql
-- Template for testing ACDOCA extraction
SELECT TOP 100
    RBUKRS,
    GJAHR,
    POPER,
    RACCT,
    DRCRK,
    HSL,
    RTCUR
FROM ACDOCA
WHERE RLDNR = '0L'
  AND RBUKRS = :p_company_code
  AND GJAHR = :p_fiscal_year
  AND POPER = :p_period
ORDER BY RACCT, BUDAT;
```

## Additional Resources

- [SAP ACDOCA Documentation](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/6aa39f1ac05441e5a23f484f31e477e7/4ec0026c69e945d89e18725cd69e0a06.html)
- [Universal Journal Overview](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/6aa39f1ac05441e5a23f484f31e477e7/1b5c5b7e4d3f4f3c8e1c5d3f4f3c8e1c.html)
- [S/4HANA Finance Documentation](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/8308e6d301d54584a33cd04a9861bc52/4ec0026c69e945d89e18725cd69e0a06.html)

---

**Version**: 1.0  
**Last Updated**: January 26, 2026  
**Status**: ✅ Complete ACDOCA Reference