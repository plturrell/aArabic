# Trial Balance Requirements (ODPS Format)

## Overview

This directory contains the complete requirements extracted from the **DOI for Trial Balance Process v2.0** (BCBS 239 Report Specification), formatted in ODPS v4.1 YAML structure.

**Source Document:** DOI for Trial Balance Process .docx  
**Version:** 2.0  
**Approved:** 23rd October 2025  
**Extraction Date:** 27th January 2026

---

## Requirements Files

| # | File | Description | DOI Section |
|---|------|-------------|-------------|
| 1 | [report-metadata.odps.yaml](./report-metadata.odps.yaml) | Report identification, ownership, scope, frequency, timing, and system requirements | Section 2 |
| 2 | [business-rules-thresholds.odps.yaml](./business-rules-thresholds.odps.yaml) | Variance thresholds ($100M BS, $3M P&L), commentary coverage (90%), calculation rules | Section 3 |
| 3 | [comparative-period-logic.odps.yaml](./comparative-period-logic.odps.yaml) | Monthly/Quarterly comparative period rules for BS and P&L | Section 3 |
| 4 | [ifrs-schedule-coverage.odps.yaml](./ifrs-schedule-coverage.odps.yaml) | Complete list of 43 IFRS Level 1 schedules | Section 3 |
| 5 | [process-flow.odps.yaml](./process-flow.odps.yaml) | Detailed process flow steps for variance analysis | Sections 5 & 6 |
| 6 | [control-requirements.odps.yaml](./control-requirements.odps.yaml) | Maker-Checker controls, reconciliation checks, validation rules | Section 6 |
| 7 | [dissemination.odps.yaml](./dissemination.odps.yaml) | Sign-off, distribution, and confidentiality requirements | Section 7 |
| 8 | [glossary-definitions.odps.yaml](./glossary-definitions.odps.yaml) | Abbreviations and terminology definitions | Sections 4 & 10 |

---

## Requirements Summary

### Total Requirements Extracted

| Category | Count |
|----------|-------|
| Report Metadata | 5 requirements (REQ-META-001 to REQ-META-005) |
| Business Rules & Thresholds | 7 requirements (REQ-THRESH-001 to REQ-THRESH-007) |
| Comparative Period Logic | 6 requirements (REQ-PERIOD-001 to REQ-PERIOD-006) |
| IFRS Schedule Coverage | 5 requirement groups (43 schedules) |
| Process Flow | 9 requirements (REQ-FLOW-001 to REQ-FLOW-009) |
| Control Requirements | 7 requirements (REQ-CTRL-001 to REQ-CTRL-007) |
| Dissemination | 6 requirements (REQ-DISS-001 to REQ-DISS-006) |
| Glossary | 27 abbreviations, 8 definitions |

---

## Key Thresholds

| Type | Amount Threshold | Percentage Threshold | Condition |
|------|-----------------|---------------------|-----------|
| **Balance Sheet** | $100 million USD | 10% | AND (both must be met) |
| **P&L** | $3 million USD | 10% | AND (both must be met) |

### Commentary Coverage
- **Target:** 90% of material variances must have explanatory commentary
- **Requirement:** Major drivers must be identified for all material variances

---

## Comparative Period Matrix

### Monthly Review

| Report Type | Current Period | Comparative Period | Mandatory |
|-------------|---------------|-------------------|-----------|
| Balance Sheet | YTD Current Month | YTD Previous Month (same FY) | ✅ Yes |
| P&L | MTD Current Month | MTD Previous Month (same FY) | ❌ No (good practice by M+12) |

### Quarterly Review

| Report Type | Current Period | Comparative Period | Mandatory |
|-------------|---------------|-------------------|-----------|
| Balance Sheet | YTD Current Quarter | YTD Dec (Previous FY) | ✅ Yes |
| P&L | YTD Current Period | YTD Same Period Previous Year (YoY) | ✅ Yes |

---

## Timeline Requirements

| Timeline | Description | Applicability |
|----------|-------------|---------------|
| **M+7** | Target completion date | Quarterly TB review (accelerated from M+10) |
| **M+12** | Good practice completion | Monthly P&L comments (optional) |

---

## IFRS Schedule Coverage

### Summary by Category

| Category | Count | Schedule IDs |
|----------|-------|--------------|
| Balance Sheet - Assets | 15 | 01, 1A, 1BA, 1CA, 1DA, DP, 1EA, 1FA, 1G, HA, HB, 1I, 1J, 1L, 1M |
| Balance Sheet - Liabilities | 17 | 02, 02A, 2O, 2P, 2Q, 2R, 2T, 2U, 2V, 2W, 2X, 2Y, YL, YS, 2Z, ZAA, 2KA |
| Profit & Loss | 13 | 3A, 3B, 3D, 3E, 3G, 3H, 3J, 3L, 3N, 3R, 3S, 3T, 3U |
| Average Balance Sheet | 2 | 04, 05 |
| Off-Balance Sheet | 2 | NAA, NBA |
| **Total** | **43** | |

---

## Control Requirements Summary

### Maker Checklist
- [ ] Variance threshold applied
- [ ] Comments updated for balances beyond threshold
- [ ] 90% of variances explained
- [ ] Major drivers identified
- [ ] Exceptions identified and escalated to GPM

### Checker/Reconciliation Checks
- [ ] Current and prior period data accuracy
- [ ] Latest GCOA used, no unmapped accounts
- [ ] Pivots refreshed, variance formulas updated
- [ ] Global mapping changes incorporated
- [ ] Exchange rates correctly applied

### Validation
- [ ] Check cells equal zero (workings reconcile to PSGL/S4)

---

## Data Sources

| System | Full Name | Purpose |
|--------|-----------|---------|
| **PSGL** | PeopleSoft General Ledger | Primary GL data source |
| **S4** | SAP S/4HANA | Alternative/migrated GL system |
| **GCOA** | Global Chart of Accounts | Account mapping reference |
| **TP System** | Transaction Processing System | Deal/transaction data |

---

## Distribution & Dissemination

| Deliverable | Mode | Recipients | Frequency |
|-------------|------|------------|-----------|
| TB Review File | SharePoint | Financial Control Cluster Lead | Monthly/Quarterly |
| Exceptions | Email | GPM, Command Center | As needed |
| CFO Declaration Items | Formal | CFO Declaration Process | Quarterly |

**Sign-off Authority:** Grade 5 or above (LEC/Caption Lead)

---

## Document Change History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 8th May 2025 | DOI newly created |
| 2.0 | 23rd Oct 2025 | a) Updated TB review limit to $100m for BS; b) Accelerated timelines from M+10 to M+7 |

---

## Related ODPS Files

These requirement files complement the existing ODPS data products:

- `../primary/variances.odps.yaml` - Variance data product specification
- `../primary/trial-balance-aggregated.odps.yaml` - Aggregated trial balance data
- `../operational/checklist-items.odps.yaml` - Control checklist items
- `../metadata/data-lineage.odps.yaml` - Data lineage tracking

---

## Usage

These YAML files can be:
1. **Parsed programmatically** for automated requirement validation
2. **Used as configuration** for the Trial Balance application
3. **Referenced for compliance** during audits
4. **Maintained as living documentation** with version control

### Example: Loading Requirements in Zig

```zig
const yaml_parser = @import("../../backend/src/metadata/yaml_parser.zig");

pub fn loadRequirements() !void {
    const thresholds = try yaml_parser.parse("requirements/business-rules-thresholds.odps.yaml");
    // Use thresholds.requirement.requirements[0].rule.conditions for validation
}
```

### Example: Loading Requirements in JavaScript

```javascript
import YAML from 'yaml';

async function loadThresholds() {
    const response = await fetch('/api/requirements/business-rules-thresholds');
    const yaml = await response.text();
    return YAML.parse(yaml);
}