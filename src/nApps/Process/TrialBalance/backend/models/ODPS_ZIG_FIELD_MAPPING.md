# ODPS to Zig Field Mapping

**Last Updated:** 2026-01-27  
**Purpose:** Bidirectional mapping between ODPS specifications and Zig implementation

---

## 1. Trial Balance (AccountBalance Struct)

**ODPS Source:** `odps/primary/trial-balance-aggregated.odps.yaml`  
**Zig File:** `calculation/balance_engine.zig`

| ODPS Field | Zig Field | Type | Validation Rule |
|------------|-----------|------|-----------------|
| `account_code` | `account_id` | `[]const u8` | - |
| `business_unit` (rbukrs) | `company_code` | `[]const u8` | - |
| `fiscal_period.year` (gjahr) | `fiscal_year` | `[]const u8` | TB004 |
| `fiscal_period.period` (poper) | `period` | `[]const u8` | TB004 |
| `opening_balance` | `opening_balance` | `f64` | TB001 |
| `debit_amount` | `debit_amount` | `f64` | TB001, TB002 |
| `credit_amount` | `credit_amount` | `f64` | TB001, TB002 |
| `closing_balance` | `closing_balance` | `f64` | TB001 |
| `currency` (rtcur) | `currency` | `[]const u8` | - |
| `ifrs_category` | `ifrs_category` | `?[]const u8` | TB003 |
| `account_type` | `account_type` | `?[]const u8` | VAR003/VAR004 |
| `gcoa_mapping_status` | `gcoa_mapping_status` | `?[]const u8` | TB005 |

### Validation Methods

| ODPS Rule | Zig Method | Formula |
|-----------|------------|---------|
| TB001 | `validateTB001()` | `closing = opening + debit - credit` |
| TB003 | `validateTB003()` | `ifrs_category != null` |
| TB005 | `validateTB005()` | `gcoa_mapping_status == "mapped"` |

---

## 2. Journal Entries (JournalEntry Struct)

**ODPS Source:** `odps/primary/acdoca-journal-entries.odps.yaml`  
**Zig File:** `calculation/balance_engine.zig`

| ODPS Field | SAP Field | Zig Field | Type |
|------------|-----------|-----------|------|
| `company_code` | RBUKRS | `company_code` | `[]const u8` |
| `fiscal_year` | GJAHR | `fiscal_year` | `[]const u8` |
| `posting_period` | POPER | `period` | `[]const u8` |
| `document_number` | BELNR | `document_number` | `[]const u8` |
| `line_item` | BUZEI | `line_item` | `[]const u8` |
| `gl_account` | RACCT | `account` | `[]const u8` |
| `debit_credit_indicator` | DRCRK | `debit_credit_indicator` | `u8` |
| `amount_local_currency` | HSL | `amount` | `f64` |
| `transaction_currency` | RTCUR | `currency` | `[]const u8` |
| `posting_date` | BUDAT | `posting_date` | `[]const u8` |

---

## 3. Variance Analysis (VarianceAnalysis Struct)

**ODPS Source:** `odps/primary/variances.odps.yaml`  
**Zig File:** `calculation/balance_engine.zig`

| ODPS Field | Zig Field | Type | Validation Rule |
|------------|-----------|------|-----------------|
| `account_code` | `account_id` | `[]const u8` | - |
| `current_period_balance` | `current_balance` | `f64` | VAR001 |
| `comparative_period_balance` | `previous_balance` | `f64` | VAR001 |
| `variance_amount` | `variance_absolute` | `f64` | VAR001 |
| `variance_percentage` | `variance_percentage` | `f64` | VAR002 |
| `is_material` | `exceeds_threshold` | `bool` | VAR003/VAR004 |
| `materiality_threshold_amount` | `threshold_amount` | `f64` | VAR003/VAR004 |
| `materiality_threshold_percentage` | `threshold_percentage` | `f64` | VAR003/VAR004 |
| `account_type` | `account_type` | `?[]const u8` | VAR003/VAR004 |
| `has_commentary` | `has_commentary` | `bool` | VAR005 |
| `commentary_text` | `commentary` | `?[]const u8` | VAR005 |
| `major_driver` | `major_driver` | `?[]const u8` | VAR008 |
| `is_exception` | `is_exception` | `bool` | VAR007 |

### Variance Calculation Methods

| ODPS Rule | Zig Method | Formula |
|-----------|------------|---------|
| VAR001 | `calculate_variance()` | `variance = current - previous` |
| VAR002 | `calculate_variance()` | `pct = (variance / abs(previous)) × 100` |
| VAR003/VAR004 | `applyDOIThresholds()` | Apply $100M/$3M based on account type |
| VAR006 | `calculateCommentaryCoverage()` | `explained / material × 100 >= 90%` |

---

## 4. DOI Thresholds

**ODPS Source:** `odps/requirements/business-rules-thresholds.odps.yaml`  
**Zig File:** `calculation/balance_engine.zig` → `DOIThresholds` struct

| ODPS Requirement | Zig Constant | Value |
|------------------|--------------|-------|
| REQ-THRESH-001 | `BALANCE_SHEET_AMOUNT` | $100,000,000 |
| REQ-THRESH-003 | `PROFIT_LOSS_AMOUNT` | $3,000,000 |
| REQ-THRESH-002 | `VARIANCE_PERCENTAGE` | 10% (0.10) |
| MKR-CHK-001 | `COMMENTARY_COVERAGE` | 90% (0.90) |
| - | `BALANCE_TOLERANCE` | $0.01 |

---

## 5. Data Quality Rules

**ODPS Source:** `odps/primary/trial-balance-aggregated.odps.yaml#validationRules`  
**Zig File:** `calculation/balance_engine.zig` → `ODPSRuleID` struct

| Rule ID | Zig Constant | Control Reference |
|---------|--------------|-------------------|
| TB001 | `TB001_BALANCE_EQUATION` | VAL-001 |
| TB002 | `TB002_DEBIT_CREDIT_BALANCE` | VAL-001 |
| TB003 | `TB003_IFRS_CLASSIFICATION` | REC-002 |
| TB004 | `TB004_PERIOD_DATA_ACCURACY` | REC-001 |
| TB005 | `TB005_GCOA_MAPPING_COMPLETENESS` | REC-002 |
| TB006 | `TB006_GLOBAL_MAPPING_CURRENCY` | REC-004 |

**ODPS Source:** `odps/primary/variances.odps.yaml#validationRules`  
**Zig File:** `calculation/balance_engine.zig` → `ODPSVarianceRuleID` struct

| Rule ID | Zig Constant | Control Reference |
|---------|--------------|-------------------|
| VAR001 | `VAR001_VARIANCE_CALCULATION` | - |
| VAR002 | `VAR002_VARIANCE_PERCENT` | - |
| VAR003 | `VAR003_MATERIALITY_THRESHOLD_BS` | REQ-THRESH-001/002 |
| VAR004 | `VAR004_MATERIALITY_THRESHOLD_PL` | REQ-THRESH-003 |
| VAR005 | `VAR005_COMMENTARY_REQUIRED` | - |
| VAR006 | `VAR006_COMMENTARY_COVERAGE_90` | MKR-CHK-001 |
| VAR007 | `VAR007_EXCEPTION_FLAGGING` | MKR-CHK-002 |
| VAR008 | `VAR008_MAJOR_DRIVER_IDENTIFICATION` | REQ-THRESH-004 |

---

## 6. Schema Mapping

**ODPS Source:** Various ODPS primary products  
**Schema File:** `schema/sqlite/01_tb_core_tables.sql`

| ODPS Product | SQL Table | Key Fields |
|--------------|-----------|------------|
| acdoca-journal-entries | `TB_JOURNAL_ENTRIES` | entry_id, rbukrs, gjahr, racct |
| account-master | `TB_GL_ACCOUNTS` | account_id, saknr, ktopl |
| exchange-rates | `TB_EXCHANGE_RATES` | rate_id, fcurr, tcurr, gdatu |
| trial-balance-aggregated | `TB_TRIAL_BALANCE` | tb_id, rbukrs, gjahr, period, racct |
| - | `TB_BALANCE_HISTORY` | snapshot_id, snapshot_type |

---

## 7. Data Quality Dimensions

**ODPS Source:** `odps/primary/trial-balance-aggregated.odps.yaml#dataQuality`  
**Zig File:** `calculation/balance_engine.zig` → `DataQualityResult` struct

| ODPS Dimension | Zig Field | Target Score | Calculation |
|----------------|-----------|--------------|-------------|
| completeness | `completeness_score` | 95 | TB003 + TB005 pass rate |
| accuracy | `accuracy_score` | 98 | TB001 pass rate |
| consistency | `consistency_score` | 90 | TB002 pass (balanced) |
| timeliness | `timeliness_score` | 95 | Assumed current |
| - | `overall_score` | 92 | Weighted average |

---

## Update History

| Date | Change | Author |
|------|--------|--------|
| 2026-01-27 | Initial mapping document | System |
| 2026-01-27 | Added ODPS rule constants to balance_engine.zig | System |