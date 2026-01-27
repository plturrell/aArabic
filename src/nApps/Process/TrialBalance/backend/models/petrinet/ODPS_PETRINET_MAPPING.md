# ODPS ↔ PNML Petri Net Mapping

**Last Updated:** 2026-01-27  
**Purpose:** Efficient bidirectional mapping between ODPS specification and executable PNML

## Overview

This document maps the ODPS Petri Net specification (`/odps/operational/trial-balance-petrinet.odps.yaml`) to the executable PNML format (`trial_balance.pnml`) and runtime configuration (`workflow_config.json`).

---

## Place Mapping (ODPS → PNML)

| ODPS Place ID | ODPS Name | PNML Place ID | PNML Name |
|---------------|-----------|---------------|-----------|
| P_INIT | Process Initialized | p1 | S4_Extraction_Queue |
| P_DATA_READY | Source Data Available | p2 | Validation_Buffer |
| P_GCOA_MAPPED | GCOA Mapping Complete | (merged into p2) | (part of validation) |
| P_FX_CONVERTED | Currency Conversion Complete | p3 | Currency_Conversion_Queue |
| P_TB_AGGREGATED | Trial Balance Aggregated | p4 | GL_Posting_Buffer |
| P_VARIANCE_CALC | Variances Calculated | p5→p6→p7 | TB_Calculation → IFRS_Compliance → Variance_Analysis |
| P_THRESH_APPLIED | Thresholds Applied | p8 | Threshold_Check_Queue |
| P_COMMENTARY_INIT | Commentary Collection Started | (merged into p9) | Commentary_Collection |
| P_COMMENTARY_COLLECTED | Business Commentary Collected | p9 | Commentary_Collection |
| P_DRIVERS_IDENTIFIED | Major Drivers Identified | (merged into p9) | Commentary_Collection |
| P_COVERAGE_MET | Commentary Coverage Met | (merged into p9→p10) | Commentary → Maker_Review |
| P_EXCEPTIONS_FLAGGED | Exceptions Identified | p14 | Exception_Queue |
| P_MAKER_COMPLETE | Maker Review Complete | p10 | Maker_Review |
| P_CHECKER_REVIEW | Awaiting Checker Review | p11 | Checker_Review |
| P_CHECKER_APPROVED | Checker Approved | (after t11) | Post-Checker_Review |
| P_REWORK_REQUIRED | Rework Required | (loop back) | (arc to earlier stage) |
| P_SUBMITTED | Submission Complete | p12 | Report_Generation |
| P_ARCHIVED | Process Archived | p13 | Audit_Archive |

---

## Transition Mapping (ODPS → PNML)

| ODPS Transition ID | ODPS Name | PNML Transition ID | PNML Name | Component |
|--------------------|-----------|-------------------|-----------|-----------|
| T_EXTRACT | Extract Source Data | t1 | Extract_S4_Data | s4_extractor |
| T_MAP_GCOA | Map to IFRS Categories | t2 | Validate_IFRS | ifrs_validator |
| T_CONVERT_FX | Apply Exchange Rates | t3 | Apply_FX_Conversion | fx_converter |
| T_AGGREGATE | Aggregate Trial Balance | t4 | Aggregate_Balances | balance_aggregator |
| T_CALC_VARIANCE | Calculate Variances | t5 + t6 + t7 | Calculate_TB + Check_IFRS + Analyze_Variances | trial_balance_calc |
| T_APPLY_THRESH | Apply Materiality Thresholds | t8 | Apply_Thresholds | variance_analyzer |
| T_INIT_COMMENTARY | Initiate Commentary Collection | (part of t9) | Collect_Commentaries | commentary_collector |
| T_COLLECT_COMMENTARY | Collect Business Commentary | t9 | Collect_Commentaries | commentary_collector |
| T_IDENTIFY_DRIVERS | Identify Major Drivers | (part of t9) | Collect_Commentaries | commentary_collector |
| T_VERIFY_COVERAGE | Verify 90% Commentary Coverage | (part of t9/t10) | Commentary/Maker | workflow_approver |
| T_FLAG_EXCEPTIONS | Flag Unexplained Variances | (feeds t14) | Handle_Exception | error_handler |
| T_MAKER_REVIEW | Complete Maker Checklist | t10 | Maker_Review | workflow_approver |
| T_SUBMIT_CHECKER | Submit to Checker | (part of t10→t11) | Maker → Checker | workflow_approver |
| T_CHECKER_REVIEW | Checker Review and Approval | t11 | Checker_Review | workflow_approver |
| T_REWORK | Rework Required | (new arc needed) | (loop back) | workflow_approver |
| T_SUBMIT | Submit to Group Reporting | t12 | Generate_Reports | report_generator |
| T_ARCHIVE | Archive for Audit | t13 | Archive_Audit | audit_archiver |

---

## Control Checkpoint → PNML Mapping

| ODPS Checkpoint | Control ID | PNML Transition | Validation Rules | Data Quality Rules |
|-----------------|------------|-----------------|------------------|-------------------|
| CP-MKR-001 | MKR-CHK-001 | t8 | threshold_applied | VAR003, VAR004 |
| CP-MKR-002 | MKR-CHK-001 | t9/t10 | commentary_coverage | VAR005, VAR006 |
| CP-MKR-003 | MKR-CHK-002 | t14 | exceptions_flagged | VAR007 |
| CP-REC-001 | REC-001 | t4/t5 | period_data_accuracy | TB004 |
| CP-REC-002 | REC-002 | t2 | gcoa_mapping_complete | TB003, TB005 |
| CP-REC-003 | REC-003 | t5/t7 | calculation_integrity | VAR001, VAR002 |
| CP-REC-004 | REC-004 | t4 | global_mapping_changes | TB006 |
| CP-REC-005 | REC-005 | t3 | fx_rate_verification | FX005, FX006 |
| CP-VAL-001 | VAL-001 | t11 | balance_reconciliation | TB001, TB002 |

---

## Stage Mapping (13-Stage ODPS → 13-Stage Config)

| ODPS Stage | Stage Name | Config Stage ID | Config Stage Name |
|------------|------------|-----------------|-------------------|
| S01 | Data Extraction | extract | S/4 Data Extraction |
| S02 | GCOA Mapping | validate | IFRS Validation |
| S03 | Currency Conversion | fx_conversion | Currency Conversion |
| S04 | Trial Balance Aggregation | aggregate | Balance Aggregation |
| S05 | Variance Calculation | calculate_tb + variance_analysis | TB Calculation + Variance Analysis |
| S06 | Threshold Application | threshold_check | Threshold Application |
| S07 | Commentary Initiation | (part of commentary) | Commentary Collection |
| S08 | Commentary Collection | commentary | Commentary Collection |
| S09 | Driver Analysis | (part of commentary) | Commentary Collection |
| S10 | Coverage Verification | (part of maker_review) | Maker Review |
| S11 | Maker Review | maker_review | Maker Review |
| S12 | Checker Review | checker_review | Checker Review |
| S13 | Submission & Archive | report_generation + audit_archive | Report Generation + Audit Archive |

---

## Implementation Notes

### Differences Between ODPS and PNML

1. **Granularity**: ODPS has 17 places vs PNML has 14 places
   - ODPS is more granular in commentary collection (4 places)
   - PNML consolidates into single Commentary_Collection place

2. **Rework Loop**: ODPS explicitly models T_REWORK
   - PNML needs additional arc from t11 back to p7 for rework

3. **Exception Handling**: Both model exceptions similarly
   - ODPS: P_EXCEPTIONS_FLAGGED + T_FLAG_EXCEPTIONS
   - PNML: p14 (Exception_Queue) + t14 (Handle_Exception)

### Synchronization Strategy

1. **ODPS is Source of Truth** for business requirements
2. **PNML is Execution Format** for Petri Net engine
3. **workflow_config.json** bridges ODPS definitions to PNML execution

### Required PNML Updates

To fully align with ODPS:
1. Add rework arc (a31) from t11 → p7
2. Add control checkpoint annotations to transitions
3. Add data quality rule references

---

## File References

| Type | File | Purpose |
|------|------|---------|
| ODPS Spec | `backend/models/odps/operational/trial-balance-petrinet.odps.yaml` | Business requirements |
| PNML Exec | `backend/models/petrinet/trial_balance.pnml` | Petri Net execution |
| Config | `backend/models/petrinet/workflow_config.json` | Runtime configuration |
| Checklist | `backend/models/odps/operational/checklist-items.odps.yaml` | Stage checklists |
| Execution Log | `backend/models/odps/operational/workflow-execution-log.odps.yaml` | Audit trail |
| CSN Schema | `backend/models/csn/trial-balance.csn.json` | SAP Cloud Application Programming Model |
| DOI Source | `BusDocs/requirements/DOI for Trial Balance Process .docx` | Original DOI document |

---

## Update History

| Date | Change | Author |
|------|--------|--------|
| 2026-01-27 | Initial mapping from ODPS to PNML | System |