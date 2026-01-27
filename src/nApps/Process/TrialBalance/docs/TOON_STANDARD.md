# TOON - Token-Oriented Object Notation Standard

**Version:** 1.0  
**Purpose:** AI-readable semantic comments for cross-file traceability and vectorization

---

## Overview

TOON provides a structured comment format that enables:
- AI/LLM systems to extract relationships through tokenization
- IDE tools to navigate between related code and specifications
- Vector databases to correlate chunks across different file types
- Bidirectional traceability from DOI → ODPS → Code → Data

---

## Token Namespaces

| Namespace | Description | Example |
|-----------|-------------|---------|
| `[ODPS:...]` | ODPS product and rule references | `[ODPS:rule=TB001]` |
| `[DOI:...]` | DOI control references | `[DOI:control=VAL-001]` |
| `[PETRI:...]` | Petri net stage/transition references | `[PETRI:stage=S04]` |
| `[TABLE:...]` | Database table/column references | `[TABLE:name=TB_TRIAL_BALANCE]` |
| `[CODE:...]` | Source code file/function references | `[CODE:function=validateTB001]` |
| `[API:...]` | REST API endpoint references | `[API:endpoint=/api/v1/trial-balance]` |
| `[RELATION:...]` | Explicit relationships between entities | `[RELATION:implements=ODPS:TB001]` |

---

## Token Attributes

### ODPS Tokens
```
[ODPS:product=<id>]           - Product identifier
[ODPS:product.version=<ver>]  - Product version
[ODPS:rule=<id>]              - Rule identifier (TB001, VAR001, FX001, etc.)
[ODPS:rule.name=<name>]       - Human-readable rule name
[ODPS:rule.formula=<expr>]    - Business logic formula
[ODPS:rule.severity=<level>]  - error, warning, info
[ODPS:rules=<id1>,<id2>,...]  - Multiple rules (comma-separated)
```

### DOI Tokens
```
[DOI:control=<id>]            - Single control reference
[DOI:controls=<id1>,<id2>]    - Multiple controls (comma-separated)
[DOI:section=<num>]           - DOI document section
```

### PETRI Tokens
```
[PETRI:process=<file>]        - Petri net process definition file
[PETRI:stage=<id>]            - Workflow stage (S01, S02, etc.)
[PETRI:stages=<id1>,<id2>]    - Multiple stages
[PETRI:transition=<id>]       - Transition identifier
[PETRI:place=<id>]            - Place identifier
```

### TABLE Tokens
```
[TABLE:name=<table>]          - Table name
[TABLE:column=<table>.<col>]  - Specific column
[TABLE:reads=<t1>,<t2>]       - Tables this code reads
[TABLE:writes=<t1>,<t2>]      - Tables this code writes
[TABLE:stores=<t1>,<t2>]      - Tables that store this data
```

### CODE Tokens
```
[CODE:file=<filename>]        - Source file name
[CODE:module=<name>]          - Module/namespace
[CODE:function=<name>]        - Function/method name
[CODE:struct=<name>]          - Struct/class name
[CODE:language=<lang>]        - zig, javascript, sql, yaml
```

### API Tokens
```
[API:endpoint=<path>]         - REST endpoint path
[API:method=<verb>]           - HTTP method (GET, POST, etc.)
[API:consumes=<path>]         - API this code calls
[API:produces=<path>]         - API this code provides
```

### RELATION Tokens
```
[RELATION:implements=<ref>]       - This code implements that spec
[RELATION:implemented_by=<ref>]   - This spec is implemented by that code
[RELATION:validates=<ref>]        - This function validates that data
[RELATION:stores_result=<ref>]    - Validation result stored here
[RELATION:calls=<ref>]            - What this code calls
[RELATION:called_by=<ref>]        - What calls this code
[RELATION:reads=<ref>]            - Data sources read
[RELATION:writes=<ref>]           - Data targets written
[RELATION:displays=<ref>]         - UI displays this data
[RELATION:displayed_by=<ref>]     - Data displayed by this UI
```

---

## Comment Format by Language

### Zig
```zig
/// [CODE:file=balance_engine.zig]
/// [ODPS:product=trial-balance-aggregated]
/// [ODPS:rule=TB001]
/// [DOI:control=VAL-001]
/// [RELATION:implements=ODPS:TB001]
pub fn validateTB001() bool {
```

### JavaScript
```javascript
/**
 * [CODE:file=TrialBalance.controller.js]
 * [ODPS:product=trial-balance-aggregated]
 * [ODPS:rules=TB001,TB002,TB003]
 * [RELATION:displays=TABLE:TB_TRIAL_BALANCE]
 */
sap.ui.define([...], function() {
```

### YAML
```yaml
# [ODPS:product=trial-balance-aggregated]
# [RELATION:implemented_by=CODE:balance_engine.zig]
# [TABLE:stores=TB_TRIAL_BALANCE]
info:
  id: trial-balance-aggregated
```

### SQL
```sql
-- [TABLE:name=TB_TRIAL_BALANCE]
-- [ODPS:product=trial-balance-aggregated]
-- [RELATION:implements=ODPS:TB001,ODPS:TB002]
-- [CODE:reads=balance_engine.zig]
CREATE TABLE TB_TRIAL_BALANCE (
```

### XML (Petri Net)
```xml
<!-- [PETRI:process=TB_PROCESS_petrinet.pnml] -->
<!-- [ODPS:product=trial-balance-process] -->
<!-- [RELATION:orchestrates=CODE:odps_petrinet_bridge.zig] -->
<pnml>
```

---

## Reference Format

References can be fully qualified or shorthand:

```
# Fully qualified
[RELATION:implements=ODPS:trial-balance-aggregated#TB001]

# Shorthand (when context is clear)
[RELATION:implements=ODPS:TB001]

# Multiple references
[RELATION:implements=ODPS:TB001,ODPS:TB002,ODPS:TB003]

# Code reference with function
[RELATION:implemented_by=CODE:balance_engine.zig:validateTB001]

# Table column reference
[RELATION:stores_result=TABLE:TB_TRIAL_BALANCE.tb001_passed]
```

---

## File Header Template

Every file should have a header block with TOON tokens:

### Zig File Header
```zig
//! [CODE:file=<filename>.zig]
//! [CODE:module=<module_name>]
//! [CODE:language=zig]
//! [ODPS:product=<product_id>]
//! [ODPS:rules=<rule1>,<rule2>,...]
//! [DOI:controls=<control1>,<control2>,...]
//! [PETRI:stages=<stage1>,<stage2>,...]
//! [TABLE:reads=<table1>,<table2>,...]
//! [TABLE:writes=<table1>,<table2>,...]
//! [API:produces=<endpoint>]
//!
//! <Brief description of file purpose>
```

### JavaScript File Header
```javascript
/**
 * [CODE:file=<filename>.js]
 * [CODE:module=<module_name>]
 * [CODE:language=javascript]
 * [ODPS:product=<product_id>]
 * [ODPS:rules=<rule1>,<rule2>,...]
 * [DOI:controls=<control1>,<control2>,...]
 * [PETRI:stages=<stage1>,<stage2>,...]
 * [API:consumes=<endpoint1>,<endpoint2>,...]
 * [TABLE:displays=<table1>,<table2>,...]
 *
 * <Brief description of file purpose>
 */
```

---

## Function/Method Template

### Zig Function
```zig
/// [CODE:function=<function_name>]
/// [ODPS:rule=<rule_id>]
/// [ODPS:rule.name=<rule_name>]
/// [ODPS:rule.formula=<formula>]
/// [DOI:control=<control_id>]
/// [TABLE:column=<table>.<column>]
/// [RELATION:implements=ODPS:<rule_id>]
/// [RELATION:validates=TABLE:<table>.<column>]
/// [RELATION:stores_result=TABLE:<table>.<result_column>]
///
/// <Brief description of function purpose>
pub fn functionName() ReturnType {
```

### JavaScript Method
```javascript
/**
 * [CODE:function=<method_name>]
 * [ODPS:rules=<rule1>,<rule2>]
 * [API:consumes=<endpoint>]
 * [RELATION:displays=ODPS:<rule_id>]
 *
 * <Brief description of method purpose>
 */
methodName: function() {
```

---

## Traversal Queries

TOON enables these query patterns:

### Forward Trace (DOI → Code)
```
Query: "What code implements DOI control VAL-001?"
Search: [DOI:control=VAL-001] or [DOI:controls=...VAL-001...]
Find: Files containing these tokens
Extract: [CODE:file=...] and [CODE:function=...]
```

### Reverse Trace (Code → DOI)
```
Query: "What DOI control does validateTB001 satisfy?"
Search: [CODE:function=validateTB001]
Find: File containing this token
Extract: [DOI:control=...] in same block
```

### Cross-File Navigation
```
Query: "What UI displays TB001 validation results?"
Chain: 
  1. Find [ODPS:rule=TB001] with [TABLE:column=...tb001_passed]
  2. Find [RELATION:displays=TABLE:TB_TRIAL_BALANCE.tb001_passed]
  3. Return [CODE:file=...] of matching UI component
```

### Impact Analysis
```
Query: "What breaks if TB_TRIAL_BALANCE.closing_balance changes?"
Search: [TABLE:column=TB_TRIAL_BALANCE.closing_balance]
        or [RELATION:validates=TABLE:TB_TRIAL_BALANCE.closing_balance]
        or [RELATION:reads=TABLE:TB_TRIAL_BALANCE]
Return: All files with matching tokens
```

---

## Validation Rules

1. Every file MUST have a header with `[CODE:file=...]`
2. Every ODPS-related file MUST have `[ODPS:product=...]`
3. Every function implementing ODPS rules MUST have `[ODPS:rule=...]`
4. Every `[ODPS:rule=X]` in code SHOULD have matching `[RELATION:implemented_by=...]` in ODPS YAML
5. Every `[TABLE:writes=X]` SHOULD have corresponding `[TABLE:reads=X]` somewhere
6. Bidirectional relations SHOULD be maintained:
   - `implements` ↔ `implemented_by`
   - `calls` ↔ `called_by`
   - `displays` ↔ `displayed_by`

---

## Tools

### Grep-based validation
```bash
# Find all ODPS rules implemented
grep -rh "\[ODPS:rule=" src/ | sort -u

# Find all code-refs in ODPS files  
grep -rh "\[RELATION:implemented_by=" models/odps/ | sort -u

# Cross-reference validation
# (compare the two lists to find gaps)
```

### Parsing for vectorization
```python
import re

TOON_PATTERN = r'\[([A-Z]+):([^\]]+)\]'

def extract_toon_tokens(text):
    tokens = []
    for match in re.finditer(TOON_PATTERN, text):
        namespace = match.group(1)
        content = match.group(2)
        attrs = dict(item.split('=') for item in content.split(',') if '=' in item)
        tokens.append({'namespace': namespace, **attrs})
    return tokens
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial TOON standard |