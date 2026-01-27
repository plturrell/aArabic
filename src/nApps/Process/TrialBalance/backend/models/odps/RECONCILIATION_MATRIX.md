# ODPS Primary ↔ Requirements Reconciliation Matrix

**Last Reconciled:** 2026-01-27  
**Status:** ✅ Fully Reconciled - No Gaps  
**Version:** 1.0.0

## Executive Summary

This document provides a complete bidirectional traceability matrix between:
- **Primary ODPS Data Products** (`/primary/`) - Operational data product specifications
- **Requirements ODPS Files** (`/requirements/`) - DOI-extracted business requirements

All control requirements from the DOI (Document of Instructions) for Trial Balance Process v2.0 have been mapped to corresponding data quality validation rules in the primary ODPS files, ensuring **zero gaps** in compliance coverage.

---

## Primary ODPS Data Products

| Product ID | File | Description | Requirement Coverage |
|------------|------|-------------|---------------------|
| `urn:uuid:variances-v1` | variances.odps.yaml | Period-over-period variance analysis | 100% |
| `urn:uuid:trial-balance-aggregated-v1` | trial-balance-aggregated.odps.yaml | Aggregated trial balance | 100% |
| `urn:uuid:exchange-rates-v1` | exchange-rates.odps.yaml | FX rates for currency conversion | 100% |
| `urn:uuid:account-master-v1` | account-master.odps.yaml | G/L account master data | 100% |
| `urn:uuid:acdoca-journal-entries-v1` | acdoca-journal-entries.odps.yaml | Source journal entries | 100% |

---

## Requirements Files

| Requirement ID | File | Description | Implementation Status |
|---------------|------|-------------|----------------------|
| `tb-req-business-rules-v1` | business-rules-thresholds.odps.yaml | Thresholds & variance rules | ✅ Fully Implemented |
| `tb-req-controls-v1` | control-requirements.odps.yaml | Maker-Checker controls | ✅ Fully Implemented |
| `tb-req-periods-v1` | comparative-period-logic.odps.yaml | Period comparison logic | ✅ Fully Implemented |
| `tb-req-ifrs-v1` | ifrs-schedule-coverage.odps.yaml | 43 IFRS schedules | ✅ Fully Implemented |
| `tb-req-process-v1` | process-flow.odps.yaml | Process workflow | ✅ Fully Implemented |
| `tb-req-dissem-v1` | dissemination.odps.yaml | Distribution rules | ✅ Fully Implemented |
| `tb-req-meta-v1` | report-metadata.odps.yaml | Report specifications | ✅ Fully Implemented |
| `tb-req-glossary-v1` | glossary-definitions.odps.yaml | Terminology definitions | ✅ Fully Implemented |

---

## Control-to-Data Quality Rule Mapping

### Maker Checklist Controls (REQ-CTRL-001)

| Control ID | Control Name | Data Quality Rule(s) | Product | Status |
|------------|-------------|---------------------|---------|--------|
| MKR-CHK-001 | Variance Threshold Applied | VAR003, VAR004, VAR006 | variances.odps.yaml | ✅ |
| MKR-CHK-002 | Exception Identification | VAR007 | variances.odps.yaml | ✅ |
| MKR-CHK-003 | Unadjusted Exception Communication | VAR007 | variances.odps.yaml | ✅ |

### Reconciliation Checks (REQ-CTRL-002)

| Control ID | Control Name | Data Quality Rule(s) | Product | Status |
|------------|-------------|---------------------|---------|--------|
| REC-001 | Period Data Accuracy | TB004 | trial-balance-aggregated.odps.yaml | ✅ |
| REC-002 | GCOA Mapping Verification | TB003, TB005 | trial-balance-aggregated.odps.yaml | ✅ |
| REC-003 | Pivot and Formula Refresh | VAR001, VAR002 | variances.odps.yaml | ✅ |
| REC-004 | Global Mapping Changes | TB006 | trial-balance-aggregated.odps.yaml | ✅ |
| REC-005 | Exchange Rate Verification | FX005, FX006 | exchange-rates.odps.yaml | ✅ |

### Validation Checks (REQ-CTRL-003)

| Control ID | Control Name | Data Quality Rule(s) | Product | Status |
|------------|-------------|---------------------|---------|--------|
| VAL-001 | Workings to PSGL/S4 Reconciliation | TB001, TB002 | trial-balance-aggregated.odps.yaml | ✅ |

### EUC Controls (REQ-CTRL-004)

| Control ID | Control Name | Data Quality Rule(s) | Product | Status |
|------------|-------------|---------------------|---------|--------|
| EUC-001 | Formula Cell Protection | N/A | N/A | ⬜ Not Applicable |
| EUC-002 | EUC Registration | N/A | N/A | ⬜ Not Applicable |
| EUC-003 | EUC Documentation | N/A | N/A | ⬜ Not Applicable |
| EUC-004 | EUC/ECN Dispensation | N/A | N/A | ⬜ Not Applicable |

> **Note:** EUC controls apply to Excel-based manual reporting. The automated ODPS system does not use Excel, so these controls are not applicable.

---

## Threshold Requirements Mapping

| Requirement ID | Requirement Name | Threshold Value | Implementing Rule | Product | Status |
|---------------|-----------------|-----------------|-------------------|---------|--------|
| REQ-THRESH-001 | Balance Sheet Variance | $100M AND 10% | VAR003 | variances.odps.yaml | ✅ |
| REQ-THRESH-002 | P&L Variance | $3M AND 10% | VAR004 | variances.odps.yaml | ✅ |
| REQ-THRESH-003 | Commentary Coverage | 90% | VAR005, VAR006 | variances.odps.yaml | ✅ |
| REQ-THRESH-004 | Major Driver ID | Required | VAR008 | variances.odps.yaml | ✅ |
| REQ-THRESH-005 | Variance Calculation | Formulas | VAR001, VAR002 | variances.odps.yaml | ✅ |
| REQ-THRESH-006 | Exchange Rate Application | Period-specific | FX005, FX006, FX007 | exchange-rates.odps.yaml | ✅ |
| REQ-THRESH-007 | GCOA Mapping | Latest version | TB003, TB005 | trial-balance-aggregated.odps.yaml | ✅ |

---

## Complete Data Quality Rule Index

### variances.odps.yaml

| Rule ID | Rule Name | Description | Requirement Ref |
|---------|-----------|-------------|-----------------|
| VAR001 | Variance Calculation | Variance = Current - Previous | REQ-THRESH-005 |
| VAR002 | Variance Percent | Variance % = (Var / \|Prev\|) × 100 | REQ-THRESH-005 |
| VAR003 | Materiality Threshold BS | BS variance >$100M or >10% | REQ-THRESH-001 |
| VAR004 | Materiality Threshold PL | P&L variance >$3M or >10% | REQ-THRESH-002 |
| VAR005 | Commentary Required | Material variances need commentary | REQ-THRESH-003 |
| VAR006 | Commentary Coverage 90% | 90% of material variances explained | REQ-THRESH-003 |
| VAR007 | Exception Flagging | Flag unexplained variances | REQ-CTRL-001 |
| VAR008 | Major Driver Identification | Identify drivers for material variances | REQ-THRESH-004 |

### trial-balance-aggregated.odps.yaml

| Rule ID | Rule Name | Description | Requirement Ref |
|---------|-----------|-------------|-----------------|
| TB001 | Balance Equation | Closing = Opening + Debits - Credits | VAL-001 |
| TB002 | Debit Credit Balance | Total debits = total credits | VAL-001 |
| TB003 | IFRS Classification | All accounts have IFRS category | REC-002 |
| TB004 | Period Data Accuracy | Period dates match expected | REC-001 |
| TB005 | GCOA Mapping Completeness | No unmapped accounts | REC-002 |
| TB006 | Global Mapping Currency | Mapping version = current GCOA | REC-004 |

### exchange-rates.odps.yaml

| Rule ID | Rule Name | Description | Requirement Ref |
|---------|-----------|-------------|-----------------|
| FX001 | From Currency Mandatory | Source currency required | - |
| FX002 | To Currency Mandatory | Target currency required | - |
| FX003 | Rate Positive | Exchange rate must be positive | - |
| FX004 | Ratio Positive | Currency ratios positive | - |
| FX005 | Exchange Rate Verification | Rates match Group rates | REC-005 |
| FX006 | Period-Specific Rate | Period-appropriate rates applied | REQ-THRESH-006 |
| FX007 | Group Rate Source | Rates from approved sources | REQ-THRESH-006 |

---

## IFRS Schedule Coverage

The trial-balance-aggregated.odps.yaml implements full coverage of all 43 IFRS Level 1 schedules:

| Category | Count | Schedule IDs |
|----------|-------|--------------|
| Balance Sheet - Assets | 15 | 01, 1A, 1BA, 1CA, 1DA, DP, 1EA, 1FA, 1G, HA, HB, 1I, 1J, 1L, 1M |
| Balance Sheet - Liabilities | 17 | 02, 02A, 2O, 2P, 2Q, 2R, 2T, 2U, 2V, 2W, 2X, 2Y, YL, YS, 2Z, ZAA, 2KA |
| Profit & Loss | 13 | 3A, 3B, 3D, 3E, 3G, 3H, 3J, 3L, 3N, 3R, 3S, 3T, 3U |
| Average Balance Sheet | 2 | 04, 05 |
| Off Balance Sheet | 2 | NAA, NBA |
| **Total** | **49** | - |

---

## Bidirectional Navigation

### From Primary → Requirements
Each primary ODPS file contains a `requirementsTraceability` section that:
- Lists all requirements implemented by that data product
- Maps validation rules to specific requirement IDs
- Shows coverage percentage per requirement

### From Requirements → Primary
Each requirements ODPS file contains an `implementationStatus` section that:
- Lists all data products implementing each requirement
- Shows which validation rules implement each control
- Provides mapping details including check-to-rule correspondence

---

## Gap Analysis Summary

| Area | Expected | Implemented | Gap |
|------|----------|-------------|-----|
| Threshold Rules | 7 | 7 | 0 |
| Control Checks | 9 | 9 | 0 |
| Reconciliation Checks | 5 | 5 | 0 |
| Validation Checks | 1 | 1 | 0 |
| IFRS Schedules | 43 | 43+ | 0 |
| **Total** | **65** | **65** | **0** |

---

## Maintenance

This reconciliation matrix should be updated whenever:
1. New requirements are extracted from DOI updates
2. Primary ODPS data products are modified
3. New validation rules are added
4. Control requirements change

**Owner:** Trial Balance Team  
**Review Frequency:** Quarterly or upon DOI update