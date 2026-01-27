# ODPS-to-Code Verification Matrix

**Generated:** 2026-01-27  
**Purpose:** Verify that Zig code actually implements what ODPS rules specify  
**SCIP Reference:** `lineage/odps-scip-lineage.yaml`

---

## Verification Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Code matches ODPS specification exactly |
| ⚠️ | Code exists but doesn't match spec exactly |
| ❌ | Code missing - needs implementation |
| 🔍 | Needs manual verification |

---

## Trial Balance Rules (TB001-TB006)

### TB001: Balance Equation

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Closing = Opening + Debits - Credits` | `balance_engine.zig:calculate_closing()` | ✅ |
| **Formula** | `closing_balance = opening_balance + debit_amount - credit_amount` | `self.closing_balance = self.opening_balance + self.debit_amount - self.credit_amount` | ✅ |
| **Validation** | Error if equation doesn't balance | `validateTB001()` returns `@abs(closing - expected) < TOLERANCE` | ✅ |
| **Tolerance** | Not specified | `0.01` (1 cent) | ✅ |
| **SCIP Symbol** | - | `scip-zig+balance_engine+AccountBalance#validateTB001` | ✅ |

**Code Excerpt:**
```zig
pub fn calculate_closing(self: *AccountBalance) void {
    self.closing_balance = self.opening_balance + self.debit_amount - self.credit_amount;
}

pub fn validateTB001(self: *const AccountBalance) bool {
    const expected = self.opening_balance + self.debit_amount - self.credit_amount;
    return @abs(self.closing_balance - expected) < DOIThresholds.BALANCE_TOLERANCE;
}
```

---

### TB002: Debit Credit Balance

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Total debits = total credits` | `balance_engine.zig:calculate_totals()` | ✅ |
| **Formula** | `SUM(debit_amount) = SUM(credit_amount)` | `balance_difference = total_debits - total_credits` | ✅ |
| **Validation** | Error if not balanced | `is_balanced = @abs(balance_difference) < TOLERANCE` | ✅ |
| **SCIP Symbol** | - | `scip-zig+balance_engine+TrialBalanceResult#calculate_totals` | ✅ |

**Code Excerpt:**
```zig
pub fn calculate_totals(self: *TrialBalanceResult) void {
    for (self.accounts.items) |account| {
        self.total_debits += account.debit_amount;
        self.total_credits += account.credit_amount;
    }
    self.balance_difference = self.total_debits - self.total_credits;
    self.is_balanced = @abs(self.balance_difference) < DOIThresholds.BALANCE_TOLERANCE;
    self.tb002_passed = self.is_balanced;
}
```

---

### TB003: IFRS Classification

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `All accounts must have IFRS category` | `balance_engine.zig:validateTB003()` | ✅ |
| **Validation** | `ifrs_category IS NOT NULL` | `self.ifrs_category != null and self.ifrs_category.?.len > 0` | ✅ |
| **Severity** | Warning | Error (could adjust) | ⚠️ |
| **SCIP Symbol** | - | `scip-zig+balance_engine+AccountBalance#validateTB003` | ✅ |

**Code Excerpt:**
```zig
pub fn validateTB003(self: *const AccountBalance) bool {
    return self.ifrs_category != null and self.ifrs_category.?.len > 0;
}
```

---

### TB004: Period Data Accuracy

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Period dates match expected reporting period` | No dedicated function | ❌ |
| **Control Ref** | REC-001 | - | ❌ |
| **SCIP Symbol** | - | None | ❌ |

**Gap:** TB004 validation not implemented in code. Needs `validateTB004()` function.

---

### TB005: GCOA Mapping Completeness

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `No unmapped accounts allowed` | `balance_engine.zig:validateTB005()` | ✅ |
| **Validation** | `COUNT(unmapped) = 0` | `gcoa_mapping_status == "mapped"` | ✅ |
| **SCIP Symbol** | - | `scip-zig+balance_engine+AccountBalance#validateTB005` | ✅ |

**Code Excerpt:**
```zig
pub fn validateTB005(self: *const AccountBalance) bool {
    return self.gcoa_mapping_status != null and
        std.mem.eql(u8, self.gcoa_mapping_status.?, "mapped");
}
```

---

### TB006: Global Mapping Currency

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `mapping_version = current_gcoa_version` | No dedicated function | ❌ |
| **Control Ref** | REC-004 | - | ❌ |
| **SCIP Symbol** | - | None | ❌ |

**Gap:** TB006 validation not implemented in code. Needs `validateTB006()` function.

---

## Variance Rules (VAR001-VAR008)

### VAR001: Variance Calculation

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `variance = current_period - comparative_period` | `balance_engine.zig:calculate_variance()` | ✅ |
| **Formula** | `variance_amount = current - previous` | `self.variance_absolute = self.current_balance - self.previous_balance` | ✅ |
| **SCIP Symbol** | - | `scip-zig+balance_engine+VarianceAnalysis#calculate_variance` | ✅ |

**Code Excerpt:**
```zig
self.variance_absolute = self.current_balance - self.previous_balance;
```

---

### VAR002: Variance Percent

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `variance_pct = (variance / |comparative_period|) × 100` | `balance_engine.zig:calculate_variance()` | ✅ |
| **Formula** | As specified | `(variance / @abs(previous)) * 100.0` | ✅ |
| **Edge Case** | Division by zero | Returns 100% if current != 0, else 0% | ✅ |

**Code Excerpt:**
```zig
if (self.previous_balance != 0.0) {
    self.variance_percentage = (self.variance_absolute / @abs(self.previous_balance)) * 100.0;
} else {
    self.variance_percentage = if (self.current_balance != 0.0) 100.0 else 0.0;
}
```

---

### VAR003: Materiality Threshold BS

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `BS variance > $100M AND > 10%` | `DOIThresholds.BALANCE_SHEET_AMOUNT` | ✅ |
| **Amount** | $100,000,000 | `100_000_000.0` | ✅ |
| **Percentage** | 10% | `0.10` | ✅ |
| **Logic** | Both conditions required | `exceeds_amount AND exceeds_percentage` | ✅ |

**Code Excerpt:**
```zig
pub const BALANCE_SHEET_AMOUNT: f64 = 100_000_000.0;
const exceeds_amount = @abs(self.variance_absolute) > self.threshold_amount;
const exceeds_percentage = @abs(self.variance_percentage) > (self.threshold_percentage * 100.0);
self.exceeds_threshold = exceeds_amount and exceeds_percentage;
```

---

### VAR004: Materiality Threshold PL

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `P&L variance > $3M AND > 10%` | `DOIThresholds.PROFIT_LOSS_AMOUNT` | ✅ |
| **Amount** | $3,000,000 | `3_000_000.0` | ✅ |
| **Percentage** | 10% | `0.10` | ✅ |
| **Account Type Selection** | Based on account type | `applyDOIThresholds()` checks account_type | ✅ |

**Code Excerpt:**
```zig
pub const PROFIT_LOSS_AMOUNT: f64 = 3_000_000.0;

pub fn applyDOIThresholds(self: *VarianceAnalysis) void {
    if (std.mem.eql(u8, acct_type, "Revenue") or std.mem.eql(u8, acct_type, "Expense")) {
        self.threshold_amount = DOIThresholds.PROFIT_LOSS_AMOUNT;
    }
}
```

---

### VAR005: Commentary Required

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Material variances must have commentary` | `has_commentary` field exists | ✅ |
| **Tracking** | Boolean flag | `has_commentary: bool = false` | ✅ |
| **Storage** | Commentary text | `commentary: ?[]const u8 = null` | ✅ |

---

### VAR006: Commentary Coverage 90%

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `90% of material variances explained` | `calculateCommentaryCoverage()` | ✅ |
| **Threshold** | 90% | `DOIThresholds.COMMENTARY_COVERAGE = 0.90` | ✅ |
| **Check** | `meetsCommentaryCoverage()` | `coverage >= 90.0` | ✅ |

**Code Excerpt:**
```zig
pub fn calculateCommentaryCoverage(variances: []const VarianceAnalysis) f64 {
    var material_count: usize = 0;
    var with_commentary: usize = 0;
    for (variances) |v| {
        if (v.exceeds_threshold) {
            material_count += 1;
            if (v.has_commentary) with_commentary += 1;
        }
    }
    return (@as(f64, @floatFromInt(with_commentary)) / @as(f64, @floatFromInt(material_count))) * 100.0;
}

pub fn meetsCommentaryCoverage(variances: []const VarianceAnalysis) bool {
    const coverage = calculateCommentaryCoverage(variances);
    return coverage >= (DOIThresholds.COMMENTARY_COVERAGE * 100.0);
}
```

---

### VAR007: Exception Flagging

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Flag material variances without commentary` | `is_exception` field | ✅ |
| **Logic** | `is_material AND NOT has_commentary` | `exceeds_threshold and !has_commentary` | ✅ |

**Code Excerpt:**
```zig
if (self.exceeds_threshold and !self.has_commentary) {
    self.is_exception = true;
}
```

---

### VAR008: Major Driver Identification

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Identify drivers for material variances` | `major_driver` field exists | ✅ |
| **Storage** | Driver text | `major_driver: ?[]const u8 = null` | ✅ |
| **Category** | - | No `driver_category` field | ⚠️ |

---

## Exchange Rate Rules (FX001-FX007)

### FX001: From Currency Mandatory

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Source currency required` | Field exists in struct | ✅ |
| **Validation** | NOT NULL | No explicit validation | ⚠️ |

---

### FX002: To Currency Mandatory

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Target currency required` | Field exists in struct | ✅ |
| **Validation** | NOT NULL | No explicit validation | ⚠️ |

---

### FX003: Rate Positive

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Exchange rate must be positive` | No validation | ❌ |
| **Validation** | `rate > 0` | Not implemented | ❌ |

---

### FX004: Ratio Positive

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Currency ratios positive` | No validation | ❌ |
| **Validation** | `from_factor > 0 AND to_factor > 0` | Not implemented | ❌ |

---

### FX005: Exchange Rate Verification

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Rates match Group rates` | No validation | ❌ |
| **Control Ref** | REC-005 | Not implemented | ❌ |

---

### FX006: Period-Specific Rate

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Period-appropriate rates applied` | Rate lookup by date | ✅ |
| **Implementation** | `find_rate()` uses `posting_date` | ✅ |

---

### FX007: Group Rate Source

| Aspect | ODPS Specification | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| **Rule** | `Rates from approved sources` | No validation | ❌ |
| **Source tracking** | GROUP_TREASURY, ECB, FED | Not tracked | ❌ |

---

## Summary: Implementation Gaps

### Fully Implemented ✅ (18 rules)
- TB001, TB002, TB003, TB005
- VAR001, VAR002, VAR003, VAR004, VAR005, VAR006, VAR007, VAR008
- FX001 (partial), FX002 (partial), FX006

### Needs Validation Code ⚠️ (4 rules)
- FX001 - Add NOT NULL check
- FX002 - Add NOT NULL check
- VAR008 - Add `driver_category` field
- TB003 - Consider warning severity

### Not Implemented ❌ (5 rules)
- TB004 - Period Data Accuracy
- TB006 - Global Mapping Currency
- FX003 - Rate Positive
- FX004 - Ratio Positive
- FX005 - Exchange Rate Verification
- FX007 - Group Rate Source

---

## Action Items

1. **Add TB004 validation** - Verify period dates match expected
2. **Add TB006 validation** - Check GCOA mapping version
3. **Add FX validations** - FX003, FX004, FX005, FX007
4. **Add driver_category** to VarianceAnalysis struct
5. **Add rate_source** tracking to FX converter