# ODPS v4.1 Data Product Specifications

This directory contains **ODPS v4.1 (Open Data Product Specification)** compliant metadata for all Trial Balance data products.

## 📊 What is ODPS?

ODPS is a **vendor-neutral metadata specification** for describing data products. Unlike SAP-specific formats (ORD/CSN), ODPS enables interoperability with any data catalog or marketplace.

**Official Specification:** https://opendataproducts.org/v4.1/schema/odps.yaml

---

## 🗂️ Directory Structure

```
odps/
├── primary/              # Primary and derived data products (5 files)
│   ├── acdoca-journal-entries.odps.yaml    # Raw transactional data
│   ├── exchange-rates.odps.yaml            # FX master data
│   ├── trial-balance-aggregated.odps.yaml  # Aggregated analytical
│   ├── variances.odps.yaml                 # Variance analytical
│   └── account-master.odps.yaml            # Account master data
│
├── metadata/             # Metadata products (2 files)
│   ├── data-lineage.odps.yaml              # Lineage tracking
│   └── dataset-metadata.odps.yaml          # Quality metrics
│
└── operational/          # Operational products (1 file)
    └── checklist-items.odps.yaml           # Workflow tracking
```

---

## 🎯 8 Data Products

### Primary & Analytical (5)

1. **ACDOCA Journal Entries** (`primary/acdoca-journal-entries.odps.yaml`)
   - **Type:** Transactional dataset
   - **Source:** SAP S/4HANA ACDOCA table
   - **Quality:** 95% (10 validation rules)
   - **SLA:** 99.99% availability, <100ms response
   - **Features:** Real-time, multi-currency, IFRS-compliant

2. **Exchange Rates** (`primary/exchange-rates.odps.yaml`)
   - **Type:** Master data
   - **Source:** ECB, Federal Reserve, SAP
   - **Quality:** 98% (4 validation rules)
   - **SLA:** 99.9% availability, daily updates
   - **Features:** Multiple rate types, historical depth

3. **Trial Balance Aggregated** (`primary/trial-balance-aggregated.odps.yaml`)
   - **Type:** Analytical dataset (derived from ACDOCA)
   - **Source:** ACDOCA aggregated by account
   - **Quality:** 92% (3 validation rules)
   - **SLA:** 99.5% availability, <1s response
   - **Features:** MTD/YTD, account hierarchy, IFRS categories

4. **Period Variances** (`primary/variances.odps.yaml`)
   - **Type:** Analytical dataset (derived from Trial Balance)
   - **Source:** Period-over-period comparison
   - **Quality:** 90% (5 validation rules)
   - **SLA:** 99.5% availability, AI commentary
   - **Features:** Materiality thresholds, AI-generated explanations

5. **Account Master** (`primary/account-master.odps.yaml`)
   - **Type:** Master data
   - **Source:** SAP S/4HANA SKA1 table
   - **Quality:** 98% (5 validation rules)
   - **SLA:** 99.9% availability, on-change updates
   - **Features:** Chart of accounts, hierarchies

### Metadata (2)

6. **Data Lineage** (`metadata/data-lineage.odps.yaml`)
   - **Type:** Metadata
   - **Source:** System-generated during transformations
   - **Quality:** 100% (4 validation rules)
   - **SLA:** 99% availability, real-time
   - **Features:** SHA-256 hashes, transformation tracking

7. **Dataset Metadata** (`metadata/dataset-metadata.odps.yaml`)
   - **Type:** Metadata
   - **Source:** System-generated quality metrics
   - **Quality:** 100% (4 validation rules)
   - **SLA:** 95% availability
   - **Features:** Quality monitoring, dataset discovery

### Operational (1)

8. **Checklist Items** (`operational/checklist-items.odps.yaml`)
   - **Type:** Operational dataset
   - **Source:** Petri net workflow engine
   - **Quality:** 85% (4 validation rules)
   - **SLA:** 99% availability, real-time
   - **Features:** 13-stage workflow, maker-checker

---

## 🔗 Relationship to SAP Artifacts

Each ODPS file cross-references SAP ORD and CSN:

```yaml
extensions:
  sapORD: "../../ord/trial-balance-product.json#<ordId>"
  sapCSN: "../../csn/trial-balance.csn.json#<EntityName>"
```

This enables:
- ✅ **Dual compliance** - Works with both vendor-neutral and SAP catalogs
- ✅ **Single source of truth** - ODPS as primary, ORD/CSN as derived
- ✅ **Bidirectional navigation** - Jump between formats easily
- ✅ **Maximum interoperability** - Use in any data catalog

---

## 📐 ODPS v4.1 Structure

Every ODPS file follows this structure:

```yaml
product:
  productID: "urn:uuid:..."           # Unique identifier
  name: "..."                         # Display name
  description: "..."                  # Detailed description
  version: "1.0.0"                    # Semantic version
  status: "active|beta|deprecated"    # Lifecycle status
  visibility: "organization|private"  # Access level
  
  details:
    type: "dataset|metadata"          # Product type
    category: "..."                   # Business category
    tags: [...]                       # Searchable tags
    owner: {...}                      # Ownership info
    domain: "Finance"                 # Business domain
    updateFrequency: "..."            # Update cadence
    
  contract:
    dataFormat: "CSN"                 # Schema format
    contractRef: "../../csn/..."      # Schema reference
    dataModel: {...}                  # Data structure
    
  dataQuality:
    dataQualityScore: 95.0            # Overall score
    qualityDimensions: [...]          # Dimension scores
    validationRules: [...]            # Validation rules
    dataLineage: {...}                # Lineage info
    
  outputPort: [...]                   # API endpoints
  pricingPlans: [...]                 # Pricing (ODPS-only)
  SLA: {...}                          # SLA (ODPS-only)
  extensions: {...}                   # Custom fields
```

---

## 🛠️ Validation

To validate ODPS files against the official schema:

```bash
# Using yq or yamllint
yamllint -d relaxed primary/*.odps.yaml

# Using JSON Schema validator (convert YAML to JSON first)
yq eval -o=json primary/acdoca-journal-entries.odps.yaml | \
  ajv validate -s https://opendataproducts.org/v4.1/schema/odps.yaml
```

---

## 📊 Quick Reference

| Need | Use This File |
|------|---------------|
| Raw journal entries | `primary/acdoca-journal-entries.odps.yaml` |
| Currency conversion | `primary/exchange-rates.odps.yaml` |
| Account balances | `primary/trial-balance-aggregated.odps.yaml` |
| Variance analysis | `primary/variances.odps.yaml` |
| Account names | `primary/account-master.odps.yaml` |
| Data traceability | `metadata/data-lineage.odps.yaml` |
| Quality metrics | `metadata/dataset-metadata.odps.yaml` |
| Workflow status | `operational/checklist-items.odps.yaml` |

---

## 🔄 Update Process

When runtime quality scores change:

1. **Calculate new quality score** from `data_quality.zig`
2. **Update ODPS YAML** with new score
3. **Regenerate ORD** (if needed for SAP catalogs)
4. **Commit changes** to version control

---

## 📞 More Information

- **Mapping Guide:** `../docs/ODPS_METADATA_MAPPING.md`
- **ORD Document:** `../ord/trial-balance-product.json`
- **CSN Schema:** `../csn/trial-balance.csn.json`
- **Implementation Plan:** `../docs/IMPLEMENTATION_PLAN.md`

---

**Created:** January 27, 2026  
**ODPS Version:** 4.1  
**Product Count:** 8 data products  
**Standards:** Vendor-neutral, interoperable