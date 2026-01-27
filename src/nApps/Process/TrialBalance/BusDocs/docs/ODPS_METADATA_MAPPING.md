# ODPS v4.1 ↔ SAP ORD/CSN Metadata Mapping Guide

**Version:** 1.0.0  
**Date:** January 27, 2026  
**Purpose:** Document the mapping between vendor-neutral ODPS v4.1 and SAP-specific ORD/CSN metadata formats

---

## 📊 Overview

This Trial Balance application implements **dual metadata compliance**:

1. **ODPS v4.1** (Open Data Product Specification) - Vendor-neutral metadata
2. **SAP ORD v1.9** (Open Resource Discovery) - SAP ecosystem metadata
3. **SAP CSN v2.0** (Core Schema Notation) - SAP data structure schema

```
┌─────────────────────────────────────────┐
│   ODPS v4.1 (Source of Truth)          │
│   - Product details                     │
│   - Pricing/SLA                         │
│   - Data quality                        │
│   - Cross-references ORD/CSN            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       ↓                ↓
┌─────────────┐  ┌─────────────┐
│  SAP ORD    │  │  SAP CSN    │
│  (Product)  │  │  (Schema)   │
└─────────────┘  └─────────────┘
```

---

## 🗂️ File Structure

```
BusDocs/models/
├── odps/                    # ODPS v4.1 (Vendor-Neutral)
│   ├── primary/
│   │   ├── acdoca-journal-entries.odps.yaml
│   │   ├── exchange-rates.odps.yaml
│   │   ├── trial-balance-aggregated.odps.yaml
│   │   ├── variances.odps.yaml
│   │   └── account-master.odps.yaml
│   ├── metadata/
│   │   ├── data-lineage.odps.yaml
│   │   └── dataset-metadata.odps.yaml
│   └── operational/
│       └── checklist-items.odps.yaml
│
├── ord/                     # SAP ORD v1.9 (SAP-Specific)
│   └── trial-balance-product.json
│
└── csn/                     # SAP CSN v2.0 (SAP Schema)
    └── trial-balance.csn.json
```

---

## 🔗 Complete Mapping Table

| ODPS v4.1 Element | SAP ORD v1.9 | SAP CSN v2.0 | Purpose |
|-------------------|--------------|--------------|---------|
| **`product.productID`** | `dataProducts[].ordId` | N/A | Unique identifier |
| **`product.name`** | `dataProducts[].title` | `definitions[].@title` | Display name |
| **`product.description`** | `dataProducts[].description` | `definitions[].@description` | Detailed description |
| **`product.version`** | `dataProducts[].version` | `$version` | Version number |
| **`product.status`** | `releaseStatus` (active/beta/deprecated) | N/A | Product lifecycle |
| **`product.visibility`** | `visibility` (public/private/internal) | N/A | Access control |
| **`product.details.type`** | `type` (primary/derived) | `kind` (entity/type/service) | Product type |
| **`product.details.category`** | `category` (transactional/analytical) | N/A | Business category |
| **`product.details.tags`** | `tags[]` | N/A | Searchable tags |
| **`product.details.owner`** | `responsible` | N/A | Ownership |
| **`product.details.domain`** | `lineOfBusiness[]` | N/A | Business domain |
| **`product.contract.dataFormat`** | N/A | `namespace` + `definitions` | Schema format |
| **`product.contract.contractRef`** | N/A | File path to CSN | Schema reference |
| **`product.dataQuality.score`** | Custom extension | N/A | Quality score |
| **`product.dataQuality.validationRules`** | Custom extension | `@assert.*` annotations | Validation rules |
| **`product.dataQuality.dataLineage`** | `lineage.parents[]` | N/A | Lineage tracking |
| **`product.outputPort`** | `outputPorts[].ordId` → `apiResources[]` | N/A | API endpoints |
| **`product.pricingPlans`** | Custom extension | N/A | Pricing (ODPS-only) |
| **`product.SLA`** | Custom extension | N/A | SLA (ODPS-only) |
| **`product.extensions.sapORD`** | Back-reference to ORD | N/A | Cross-reference |
| **`product.extensions.sapCSN`** | N/A | Back-reference to CSN | Cross-reference |

---

## 📐 Detailed Mapping Examples

### Example 1: ACDOCA Journal Entries

**ODPS v4.1** (`odps/primary/acdoca-journal-entries.odps.yaml`):
```yaml
product:
  productID: "urn:uuid:acdoca-journal-entries-v1"
  name: "ACDOCA Universal Journal Entries"
  description: "Complete SAP S/4HANA ACDOCA..."
  dataQuality:
    dataQualityScore: 95.0
    validationRules:
      - ruleID: "R001"
        name: "RACCT Mandatory"
  outputPort:
    - portID: "acdoca-rest-api"
      endpoint: "/api/v1/journal-entries"
  extensions:
    sapORD: "../../ord/trial-balance-product.json#..."
    sapCSN: "../../csn/trial-balance.csn.json#JournalEntry"
```

**SAP ORD** (`ord/trial-balance-product.json`):
```json
{
  "dataProducts": [{
    "ordId": "sap.nApps.Process:dataProduct:ACDOCA:v1",
    "title": "ACDOCA Journal Entries",
    "description": "Complete ACDOCA universal journal entries...",
    "type": "primary",
    "category": "transactional",
    "outputPorts": [{
      "ordId": "sap.nApps.Process:apiResource:ACDOCA-API:v1"
    }]
  }]
}
```

**SAP CSN** (`csn/trial-balance.csn.json`):
```json
{
  "definitions": {
    "JournalEntry": {
      "kind": "entity",
      "@title": "ACDOCA Universal Journal Entry",
      "elements": {
        "racct": {
          "type": "cds.String",
          "@mandatory": true
        }
      },
      "keys": ["rldnr", "rbukrs", "gjahr", "belnr", "docln"]
    }
  }
}
```

---

## 🔄 Transformation Pipeline

### 1. ODPS → ORD Mapping

```
ODPS Field                    → ORD Field
─────────────────────────────────────────────
product.productID             → dataProducts[].ordId
product.name                  → dataProducts[].title
product.description           → dataProducts[].description
product.version               → dataProducts[].version
product.details.type          → dataProducts[].type
product.details.category      → dataProducts[].category
product.details.tags          → dataProducts[].tags
product.details.owner         → dataProducts[].responsible
product.details.domain        → dataProducts[].lineOfBusiness
product.outputPort[].endpoint → apiResources[].entryPoints
```

### 2. ODPS → CSN Mapping

```
ODPS Field                    → CSN Field
─────────────────────────────────────────────
product.name                  → definitions[].@title
product.description           → definitions[].@description
product.contract.dataModel    → definitions[].elements
validationRules[].severity    → elements[].@assert.*
contract.primaryKeys          → definitions[].keys
```

### 3. Quality Metrics Integration

ODPS quality metrics come from `data_quality.zig`:

```zig
// data_quality.zig → ODPS dataQuality
ValidationResult.quality_score → product.dataQuality.dataQualityScore
ValidationResult.error_count   → qualityDimensions.accuracy (penalty)
ValidationResult.warning_count → qualityDimensions.completeness (penalty)
RuleViolation[]               → dataQuality.validationRules[]
```

---

## 🎯 Usage Patterns

### Pattern 1: Publish to Data Catalog (Vendor-Neutral)

Use ODPS files for:
- Collibra Data Catalog
- Alation Data Catalog
- DataHub
- Amundsen
- Any ODPS-compatible catalog

**Example:**
```bash
# Register ACDOCA data product
curl -X POST https://catalog.company.com/api/v1/data-products \
  -H "Content-Type: application/yaml" \
  --data-binary @odps/primary/acdoca-journal-entries.odps.yaml
```

### Pattern 2: Publish to SAP Ecosystem

Use ORD+CSN for:
- SAP Business Data Cloud
- SAP Data Intelligence
- SAP BTP
- SAP API Business Hub

**Example:**
```bash
# Register to SAP BDC
curl -X POST https://bdc.sap.com/api/discovery \
  -H "Content-Type: application/json" \
  --data @ord/trial-balance-product.json
```

### Pattern 3: Runtime Conversion

Use `odps_mapper.zig` (to be built):
```zig
const odps = try loadODPS("odps/primary/acdoca.odps.yaml");
const ord = try ODPSMapper.toORD(odps);
const csn = try ODPSMapper.toCSN(odps);
```

---

## 📋 8 Data Products Summary

| # | Product Name | ODPS File | Type | Quality Score | Key Features |
|---|--------------|-----------|------|---------------|--------------|
| 1 | **ACDOCA Journal Entries** | primary/acdoca-journal-entries.odps.yaml | Transactional | 95% | 10 validation rules, real-time |
| 2 | **Exchange Rates** | primary/exchange-rates.odps.yaml | Master Data | 98% | Daily updates, multi-source |
| 3 | **Trial Balance Aggregated** | primary/trial-balance-aggregated.odps.yaml | Analytical | 92% | Daily aggregation, MTD/YTD |
| 4 | **Period Variances** | primary/variances.odps.yaml | Analytical | 90% | AI commentary, materiality |
| 5 | **Account Master** | primary/account-master.odps.yaml | Master Data | 98% | Chart of accounts, hierarchy |
| 6 | **Data Lineage** | metadata/data-lineage.odps.yaml | Metadata | 100% | Hash verification, audit |
| 7 | **Checklist Items** | operational/checklist-items.odps.yaml | Operational | 85% | 13-stage workflow, Petri net |
| 8 | **Dataset Metadata** | metadata/dataset-metadata.odps.yaml | Metadata | 100% | Quality monitoring, discovery |

---

## 🔍 ODPS-Specific Elements (Not in ORD/CSN)

These ODPS elements have no direct ORD/CSN equivalents and are stored only in ODPS files:

### 1. Pricing Plans
```yaml
pricingPlans:
  - planID: "internal-use"
    price: 0
    currency: "USD"
    limitations: [...]
```

**Storage:** ODPS YAML only  
**Usage:** Cost allocation, chargeback reporting  
**Alternative:** Could be stored in custom ORD extension field

### 2. SLA Definitions
```yaml
SLA:
  availability: 99.99
  responseTime:
    p50: "50ms"
    p95: "100ms"
  support: [...]
```

**Storage:** ODPS YAML only  
**Usage:** Service level monitoring, SLA reporting  
**Alternative:** Could be stored in separate SLA management system

### 3. Detailed Quality Dimensions
```yaml
dataQuality:
  qualityDimensions:
    - dimension: "completeness"
      score: 98
      measurementMethod: "..."
```

**Storage:** ODPS YAML (detailed) + CSN (validation rules)  
**Usage:** Quality monitoring dashboards  
**Mapping:** CSN `@assert.*` captures validation rules

---

## 🛠️ Implementation Guidelines

### When to Use Each Format

**Use ODPS when:**
- Publishing to vendor-neutral data catalogs
- Need pricing/SLA metadata
- Require detailed quality dimensions
- Want interoperability with non-SAP systems

**Use ORD when:**
- Integrating with SAP Business Data Cloud
- Publishing to SAP API Business Hub
- Need SAP-specific discovery features
- Want SAP BTP integration

**Use CSN when:**
- Need precise data structure definition
- Validating data against schema
- Generating OData services
- SAP CAP application development

### Cross-Reference Pattern

Every ODPS file includes `extensions` that point to ORD and CSN:

```yaml
extensions:
  sapORD: "../../ord/trial-balance-product.json#<ordId>"
  sapCSN: "../../csn/trial-balance.csn.json#<EntityName>"
```

This creates a **bidirectional link** between vendor-neutral (ODPS) and SAP-specific (ORD/CSN) metadata.

---

## 📈 Quality Metrics Integration

### From data_quality.zig to ODPS

```zig
// Calculate quality score
const validation = try data_quality.validateJournalEntry(&entry, allocator);

// Map to ODPS
odps.dataQuality.dataQualityScore = validation.quality_score;

// Map violations to validation rules
for (validation.violations.items) |violation| {
    odps.dataQuality.validationRules.append(.{
        .ruleID = violation.rule_id,
        .name = violation.rule_name,
        .severity = violation.severity.toString(),
        .field = violation.field,
        .description = violation.message,
    });
}
```

### Quality Dimension Calculation

```yaml
qualityDimensions:
  - dimension: "completeness"
    score: (records_with_all_mandatory_fields / total_records) × 100
  - dimension: "accuracy"
    score: (records_passing_validation / total_records) × 100
  - dimension: "consistency"
    score: 100 - (cross_field_violations / total_records) × 100
  - dimension: "timeliness"
    score: based on extraction_ts vs expected_time
```

---

## 🔄 Lineage Tracking

### ODPS Lineage Structure

```yaml
dataLineage:
  upstream:
    - productID: "urn:uuid:source-product-v1"
      name: "Source Product Name"
      type: "source-dataset|external-source|configuration"
  transformations:
    - transformationID: "transform-name"
      description: "What this transformation does"
      method: "group-by-sum|subtraction|aggregation"
      logic: "SQL or pseudocode"
  downstream:
    - productID: "urn:uuid:target-product-v1"
      name: "Target Product Name"
      relationship: "aggregates|calculates|enriches|enables"
```

### Example: Complete Lineage Chain

```
ACDOCA (Raw)
    ↓ [extract]
ACDOCA Table (Validated)
    ↓ [aggregate-by-account]
Trial Balance (Aggregated)
    ↓ [period-comparison]
Variances (Analytical)
```

Each arrow is a `DataLineage` entry with:
- Source/target dataset IDs
- Source/target SHA-256 hashes
- Transformation type
- Quality score at that stage

---

## 🎬 Usage Examples

### Example 1: Query ODPS Metadata

```python
import yaml

# Load ODPS file
with open('odps/primary/acdoca-journal-entries.odps.yaml') as f:
    product = yaml.safe_load(f)

print(f"Product: {product['product']['name']}")
print(f"Quality: {product['product']['dataQuality']['dataQualityScore']}%")
print(f"API: {product['product']['outputPort'][0]['endpoint']}")
```

### Example 2: Cross-Reference to SAP

```python
# Get SAP ORD reference
ord_ref = product['product']['extensions']['sapORD']
# ../../ord/trial-balance-product.json#sap.nApps.Process:dataProduct:ACDOCA:v1

# Get SAP CSN reference
csn_ref = product['product']['extensions']['sapCSN']
# ../../csn/trial-balance.csn.json#JournalEntry
```

### Example 3: Validate Data Quality

```zig
// Load validation rules from ODPS
const odps_rules = try loadODPSValidationRules("odps/primary/acdoca.odps.yaml");

// Apply rules to data
for (entries.items) |entry| {
    const validation = try data_quality.validateJournalEntry(&entry, allocator);
    
    // Update ODPS quality score
    if (validation.is_valid) {
        verified_count += 1;
    }
}

// Calculate ODPS quality score
const quality_score = (verified_count / total_count) × 100.0;
```

---

## 🚀 Future Enhancements

### Phase 1: ODPS Mapper (Day 2)
Build `backend/src/metadata/odps_mapper.zig`:
```zig
pub fn loadODPS(file_path: []const u8) !ODPSProduct
pub fn toORD(odps: ODPSProduct) !ORDDocument
pub fn toCSN(odps: ODPSProduct) !CSNSchema
pub fn validateODPS(odps: ODPSProduct) !bool
```

### Phase 2: Runtime Sync (Day 3)
Automatically sync quality scores from runtime to ODPS:
```zig
pub fn updateODPSQuality(
    odps_file: []const u8,
    quality_summary: QualitySummary,
) !void
```

### Phase 3: Catalog Integration (Week 2)
- REST API endpoint: `GET /api/v1/data-products` returns ODPS
- Data catalog registration scripts
- Automated ODPS generation from CSN

---

## 📚 Standards Compliance

### ODPS v4.1
- ✅ All required fields present
- ✅ Proper YAML structure
- ✅ Valid productID (URN format)
- ✅ Complete data quality section
- ✅ Output ports defined
- ✅ Pricing/SLA included

### SAP ORD v1.9
- ✅ Valid JSON schema
- ✅ Required fields (ordId, title, version, type)
- ✅ API resources linked
- ✅ Industry standards tagged

### SAP CSN v2.0
- ✅ Valid namespace
- ✅ Entity definitions with keys
- ✅ Type annotations
- ✅ Validation assertions

---

## 🎓 Best Practices

1. **ODPS as Source of Truth**
   - Maintain ODPS files as primary metadata source
   - Generate ORD/CSN as derived artifacts when possible
   - Update ODPS first, then sync to ORD/CSN

2. **Cross-References**
   - Always include `extensions.sapORD` and `extensions.sapCSN`
   - Use JSON Pointer syntax for precise references (#entityName)
   - Keep paths relative for portability

3. **Quality Metrics**
   - Calculate quality scores programmatically from data_quality.zig
   - Update ODPS files with actual runtime metrics
   - Include measurement methods for transparency

4. **Lineage Tracking**
   - Document upstream/downstream relationships
   - Include transformation logic for reproducibility
   - Use consistent productID references

5. **Versioning**
   - Bump version when schema changes
   - Use semantic versioning (major.minor.patch)
   - Document breaking changes in description

---

## 📞 Support

For questions about:
- **ODPS v4.1:** https://opendataproducts.org/
- **SAP ORD:** https://sap.github.io/open-resource-discovery/
- **SAP CSN:** https://cap.cloud.sap/docs/cds/csn

---

**Last Updated:** January 27, 2026  
**Maintained By:** Trial Balance Team  
**Review Cycle:** Quarterly