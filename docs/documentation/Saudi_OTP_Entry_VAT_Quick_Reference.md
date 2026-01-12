# Saudi OTP Entry Processing - VAT Quick Reference Guide

**Country:** Saudi Arabia | **Entity:** Standard Chartered Capital Saudi (SCSA)  
**Document:** OTP Entry Processing - VAT Q3'25  
**Year:** 2025

---

## Quick Decision Tree

```
Is it a VAT Payment Entry?
├─ YES → Complete GL Details → Verify Tax Invoice Compliance → Fill AP Instructions
└─ NO → Check if Withholding Tax applies
    ├─ YES → Vendor outside KSA → Complete WHT section
    └─ NO → Standard processing
```

---

## 1. GL Details (General Ledger Information)

### Required GL Account Details:

| Field | Value | Description |
|-------|-------|-------------|
| **Account** | 287961 | Default GL account |
| **Product** | 910 | Product code |
| **Department** | 8105 | Department code |
| **Operating Unit** | 800 | Operating unit |
| **Class** | 98 | Classification code |
| **Affiliate** | - | Leave blank |
| **Project ID** | - | Leave blank |

### Cost Centre Cost Breakdown:
- Provide detailed cost centre breakdown as required
- Ensure proper allocation across cost centres

**Responsibility:** SCCSA Finance Team

---

## 2. Tax Invoice Compliance Checklist

### Internal Checklist to Verify Tax Invoice/Tax Credit Note/Tax Debit Note Compliance

**Per KSA VAT Regulation:**

| S.No. | Checklist Point | Status | Notes |
|-------|----------------|--------|-------|
| 1 | Invoice in Arabic language (Y/N) | ✅ | **MANDATORY** |
| 2 | Name and address addressed to "Standard Chartered Capital" (Y/N) | ✅ | **MANDATORY** |
| 3 | Date of issue | ✅ | Required |
| 4 | Sequential number | ✅ | Required |
| 5 | Tax Identification Number (15 digits), name and address of Supplier | ✅ | Required |
| 6 | Nature/Description of Goods and Services | ✅ | Required |
| 7 | Rate of Tax applied | ✅ | Required (e.g., 0%, 5%) |
| 8 | Net Amount, VAT and Gross amount in SAR | ✅ | **MANDATORY** - Convert FCY using SAMA rates |
| 9 | Date of Supply (if different from issue date) | ✅ | If applicable |
| 10 | Tax treatment explanation (if not 5%) | ✅ | If applicable |
| 11 | Reference to original invoice (Credit/Debit Notes only) | ✅ | If applicable |

### Invoice Classification:

#### ✅ Simplified Tax Invoice (Local Purchase)
**Criteria:** If all **bold checklist items** are satisfied:
- Invoice in Arabic
- Addressed to Standard Chartered Capital
- Date of issue
- Sequential number
- Supplier TIN, name, address
- Description
- Tax rate
- Amounts in SAR

#### ✅ Full Tax Invoice (Local Purchase)
**Criteria:** If **ALL** checklist points are satisfied (including non-bold items)

#### ✅ Reverse Charge (Foreign Purchase)
**Criteria:** Purchases outside KSA
- Tag supply as reverse charge
- Must be supported with invoice copy

**Responsibility:** SCCSA Finance Team

---

## 3. Instructions to AP Team

### Required Fields to Complete:

| S.No. | Field | Details | Example |
|-------|-------|---------|---------|
| 1 | **Tax Type** | FI: Full Tax Invoice<br>SI: Simplified Tax Invoice<br>RC: Reverse Charge<br>TCN: Tax Credit Note<br>Other: Miscellaneous invoices | FI / SI / RC / TCN |
| 2 | **PSID** | Default | 2002976 |
| 3 | **Date of Issue** | Invoice issue date | 30 October 2025 |
| 4 | **Sequential Number** | Invoice sequential number | 3003093648243001 |
| 5 | **Supplier Name** | Full name of supplier | THE GENERAL AUTHORITY OF ZAKAT GAZT |
| 6 | **Tax Identification Number** | 15-digit TIN | 3003093648 |
| 7 | **Description** | Goods/services description | VAT Q3'25 |
| 8 | **Tax Rate** | Rate applied | 0% / 5% |
| 9 | **Net Amount** | Amount excluding VAT (in SAR) | SAR 00 |
| 10 | **Gross Amount** | Total amount including VAT (in SAR) | SAR 1,558,629.95 |
| 11 | **Tax Amount** | VAT amount (in SAR) | SAR 00 |

### Currency Conversion:
- ⚠️ **If invoice in foreign currency:** Convert to SAR using SAMA exchange rates as per date of supply
- ⚠️ All amounts must be shown in Saudi Riyals (SAR)

**Responsibility:** SCCSA Finance Team → AP Team

---

## 4. Withholding Tax (WHT) Applicability

### ⚠️ ONLY Applicable if Vendor Location is Outside Saudi Arabia

| S.No. | Field | Details | Example |
|-------|-------|---------|---------|
| 1 | **WHT Applicability** | Yes/No/NA | NA |
| 2 | **WHT Rate** | Rate based on nature of expense (use Excel calculator) | 15% |
| 3 | **Invoice Amount** | Original invoice amount | SAR 100 |
| 4 | **WHT Amount** | Calculated withholding tax | SAR 15 |
| 5 | **Net Amount Payable** | Invoice amount minus WHT | SAR 85 |

### WHT Calculation:
- Use Excel calculator based on nature of expense
- Apply appropriate WHT rate per KSA regulations
- Calculate: Net Payable = Invoice Amount - WHT Amount

**Responsibility:** SCCSA Finance Team

---

## 5. Step-by-Step Processing Workflow

### Process Flow:

1. ✅ **Receive Invoice/Document**
   - Verify it's a VAT-related payment
   - Check if vendor is local or foreign

2. ✅ **Complete GL Details**
   - Fill in account number: 287961
   - Complete product, department, operating unit, class
   - Provide cost centre breakdown

3. ✅ **Verify Tax Invoice Compliance**
   - Run through 11-point checklist
   - Classify as Simplified/Full/Reverse Charge
   - Verify Arabic language requirement
   - Check amounts are in SAR (convert if needed)

4. ✅ **Fill AP Team Instructions**
   - Select Tax Type (FI/SI/RC/TCN)
   - Enter PSID: 2002976
   - Complete all invoice details
   - Ensure amounts in SAR

5. ✅ **Check WHT Applicability**
   - If vendor outside KSA → Complete WHT section
   - Calculate WHT using Excel calculator
   - Determine net payable amount

6. ✅ **Submit to AP Team**
   - All sections completed
   - All compliance checks passed
   - Ready for processing

---

## 6. Control Checks Summary

### Pre-Processing Checks:
- [ ] Invoice in Arabic language
- [ ] Addressed to "Standard Chartered Capital"
- [ ] All mandatory fields completed
- [ ] Amounts converted to SAR (if FCY)
- [ ] SAMA exchange rate used (for FCY conversion)
- [ ] Tax classification correct (FI/SI/RC/TCN)
- [ ] Sequential number present
- [ ] TIN verified (15 digits)

### Tax Compliance Checks:
- [ ] Tax rate verified (0%, 5%, or other)
- [ ] Net, VAT, and Gross amounts calculated correctly
- [ ] Date of supply matches (if different from issue date)
- [ ] Tax treatment explained (if not standard 5%)

### WHT Checks (if applicable):
- [ ] Vendor location verified (outside KSA)
- [ ] WHT rate correct for expense type
- [ ] WHT amount calculated correctly
- [ ] Net payable amount accurate

---

## 7. Common Scenarios

### Scenario 1: Local Supplier - Full Tax Invoice
- ✅ All 11 checklist points satisfied
- ✅ Tax Type: **FI** (Full Tax Invoice)
- ✅ WHT: **NA** (local supplier)

### Scenario 2: Local Supplier - Simplified Tax Invoice
- ✅ Bold checklist items satisfied only
- ✅ Tax Type: **SI** (Simplified Tax Invoice)
- ✅ WHT: **NA** (local supplier)

### Scenario 3: Foreign Supplier
- ✅ Tag as Reverse Charge
- ✅ Tax Type: **RC** (Reverse Charge)
- ✅ WHT: **Complete WHT section** (vendor outside KSA)
- ✅ Attach invoice copy

### Scenario 4: Tax Credit Note
- ✅ Reference to original invoice number
- ✅ Tax Type: **TCN** (Tax Credit Note)
- ✅ Complete all invoice details

### Scenario 5: Zero-Rated Supply
- ✅ Tax Rate: **0%**
- ✅ Tax Amount: **SAR 00**
- ✅ Explanation provided (if required)

---

## 8. Currency Conversion Rules

### SAMA Exchange Rate Requirements:
- ⚠️ **Mandatory:** Use SAMA (Saudi Arabian Monetary Authority) exchange rates
- ⚠️ **Date:** Use exchange rate as of **date of supply** (not invoice date if different)
- ⚠️ **All amounts:** Must be shown in Saudi Riyals (SAR)

### Conversion Process:
1. Identify invoice currency
2. Identify date of supply
3. Obtain SAMA exchange rate for that date
4. Convert Net Amount to SAR
5. Calculate VAT in SAR
6. Calculate Gross Amount in SAR

---

## 9. Tax Classification Guide

| Tax Type | Code | When to Use |
|----------|------|-------------|
| **Full Tax Invoice** | FI | All 11 checklist points satisfied |
| **Simplified Tax Invoice** | SI | Bold checklist items only (local purchase) |
| **Reverse Charge** | RC | Purchase from outside KSA |
| **Tax Credit Note** | TCN | Credit note with reference to original invoice |
| **Other** | Other | Miscellaneous invoices |

---

## 10. Key Requirements Summary

### Mandatory Requirements:
- ✅ Invoice in Arabic language
- ✅ Addressed to "Standard Chartered Capital"
- ✅ Amounts in SAR (convert FCY using SAMA rates)
- ✅ Sequential invoice number
- ✅ Supplier TIN (15 digits)
- ✅ Tax rate clearly stated
- ✅ Net, VAT, and Gross amounts shown

### Optional/If Applicable:
- Date of supply (if different from issue date)
- Tax treatment explanation (if not standard rate)
- Original invoice reference (for credit/debit notes)
- WHT details (if vendor outside KSA)

---

## 11. Common Mistakes to Avoid

1. ❌ **Not converting FCY to SAR** - All amounts must be in SAR
2. ❌ **Using wrong exchange rate** - Must use SAMA rates as of supply date
3. ❌ **Missing Arabic language** - Invoice must be in Arabic
4. ❌ **Wrong addressee** - Must be "Standard Chartered Capital"
5. ❌ **Incorrect tax classification** - Verify FI vs SI vs RC
6. ❌ **Missing sequential number** - Required for all invoices
7. ❌ **Incomplete TIN** - Must be 15 digits
8. ❌ **Not checking WHT** - Must check if vendor outside KSA
9. ❌ **Wrong WHT rate** - Use Excel calculator based on expense type
10. ❌ **Missing invoice copy** - Required for reverse charge

---

## 12. Quick Reference Tables

### GL Account Details:
```
Account:      287961
Product:      910
Department:   8105
Operating Unit: 800
Class:        98
PSID:         2002976 (Default)
```

### Tax Type Codes:
- **FI** = Full Tax Invoice
- **SI** = Simplified Tax Invoice
- **RC** = Reverse Charge
- **TCN** = Tax Credit Note
- **Other** = Miscellaneous

### Standard Tax Rates:
- **0%** = Zero-rated supplies
- **5%** = Standard VAT rate
- **Other** = As per KSA regulations

---

## 13. Document Checklist

### For Each OTP Entry:
- [ ] GL Details completed
- [ ] Cost centre breakdown provided
- [ ] Tax compliance checklist completed (11 points)
- [ ] Tax classification determined (FI/SI/RC/TCN)
- [ ] AP instructions completed (all 11 fields)
- [ ] Currency converted to SAR (if FCY)
- [ ] SAMA exchange rate used
- [ ] WHT section completed (if vendor outside KSA)
- [ ] Invoice copy attached (if reverse charge)
- [ ] All amounts verified (Net, VAT, Gross)

---

## 14. Escalation Points

### Escalate if:
- ⚠️ Invoice not in Arabic language
- ⚠️ Not addressed to Standard Chartered Capital
- ⚠️ Missing mandatory fields
- ⚠️ Cannot determine tax classification
- ⚠️ Exchange rate not available for date of supply
- ⚠️ WHT rate unclear for expense type
- ⚠️ Amounts don't reconcile (Net + VAT ≠ Gross)

---

## 15. Key Contacts

| Issue | Contact |
|-------|---------|
| GL Account Questions | SCCSA Finance Team |
| Tax Classification | SCCSA Finance Team |
| AP Processing | AP Team |
| WHT Calculations | SCCSA Finance Team |
| SAMA Exchange Rates | Treasury / Finance |
| Compliance Questions | Tax Team / Finance Manager |

---

## 16. Example: Complete Entry

### GL Details:
- Account: 287961
- Product: 910
- Department: 8105
- Operating Unit: 800
- Class: 98

### Tax Compliance:
- ✅ Invoice in Arabic: Yes
- ✅ Addressed to SCSA: Yes
- ✅ Date: 30 October 2025
- ✅ Sequential No: 3003093648243001
- ✅ Supplier: THE GENERAL AUTHORITY OF ZAKAT GAZT
- ✅ TIN: 3003093648
- ✅ Description: VAT Q3'25
- ✅ Tax Rate: 0%
- ✅ Gross Amount: SAR 1,558,629.95

### AP Instructions:
- Tax Type: **Other** (miscellaneous)
- PSID: 2002976
- Net Amount: SAR 00
- VAT Amount: SAR 00
- Gross Amount: SAR 1,558,629.95

### WHT:
- Applicability: **NA** (local supplier)

---

**Last Updated:** Based on VAT Q3'25 Processing Template  
**Next Review:** As per KSA VAT regulation updates

