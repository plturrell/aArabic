# Saudi OTP Entry Processing - VAT Executive Summary

**Document Review Date:** [Current Date]  
**Source Document:** OTP Entry Processing - VAT Q3'25  
**Country:** Saudi Arabia | **Entity:** Standard Chartered Capital Saudi (SCSA)

---

## Overview

This document provides a high-level summary of the Saudi OTP Entry Processing procedures for VAT-related transactions. The full procedures are documented in:
- **Detailed Review:** `Saudi_OTP_Entry_VAT_Review.md`
- **Quick Reference:** `Saudi_OTP_Entry_VAT_Quick_Reference.md`

---

## Core Process Overview

### Purpose
Process One-Time Payment (OTP) entries for VAT-related transactions ensuring compliance with KSA VAT regulations.

### Key Components
1. **GL Details** - Account structure and cost allocation
2. **Tax Compliance** - 11-point checklist verification
3. **AP Instructions** - Complete processing information
4. **WHT Processing** - Withholding tax for foreign suppliers

---

## Critical Requirements

### Mandatory Requirements
- ✅ Invoice in **Arabic language**
- ✅ Addressed to **"Standard Chartered Capital"**
- ✅ All amounts in **Saudi Riyals (SAR)**
- ✅ Use **SAMA exchange rates** for FCY conversion
- ✅ **15-digit TIN** for suppliers
- ✅ **Sequential invoice number**

### Tax Classification
- **FI:** Full Tax Invoice (all 11 points satisfied)
- **SI:** Simplified Tax Invoice (bold points only)
- **RC:** Reverse Charge (foreign suppliers)
- **TCN:** Tax Credit Note

---

## Key Process Steps

### 1. GL Details
- Account: 287961
- Product: 910
- Department: 8105
- Operating Unit: 800
- Class: 98
- PSID: 2002976 (default)

### 2. Tax Compliance Checklist
- Verify 11 compliance points
- Classify invoice type
- Ensure Arabic language
- Verify addressee

### 3. Currency Conversion
- Identify invoice currency
- Use SAMA rates (as of supply date)
- Convert all amounts to SAR

### 4. AP Instructions
- Complete all 11 fields
- Select correct Tax Type
- Enter amounts in SAR

### 5. WHT (if applicable)
- Check vendor location
- Calculate WHT rate
- Determine net payable

---

## Tax Invoice Compliance Checklist (11 Points)

| # | Requirement | Status |
|---|------------|--------|
| 1 | Invoice in Arabic language | ✅ Mandatory |
| 2 | Addressed to Standard Chartered Capital | ✅ Mandatory |
| 3 | Date of issue | ✅ Required |
| 4 | Sequential number | ✅ Required |
| 5 | Supplier TIN (15 digits), name, address | ✅ Required |
| 6 | Description of goods/services | ✅ Required |
| 7 | Tax rate applied | ✅ Required |
| 8 | Net, VAT, Gross amounts in SAR | ✅ Mandatory |
| 9 | Date of supply (if different) | ⚠️ If applicable |
| 10 | Tax treatment explanation (if not 5%) | ⚠️ If applicable |
| 11 | Original invoice reference (credit/debit notes) | ⚠️ If applicable |

---

## Currency Conversion Rules

### SAMA Exchange Rate Requirements
- ⚠️ **Mandatory:** Use SAMA exchange rates
- ⚠️ **Date:** Use rate as of **date of supply**
- ⚠️ **All amounts:** Must be in SAR

### Conversion Process
1. Identify invoice currency
2. Identify date of supply
3. Obtain SAMA rate
4. Convert Net Amount
5. Calculate VAT in SAR
6. Calculate Gross Amount

---

## Withholding Tax (WHT)

### Applicability
- ✅ **ONLY** if vendor location is **outside Saudi Arabia**
- ❌ **NOT** applicable for local suppliers

### WHT Fields
1. WHT Applicability (Yes/No/NA)
2. WHT Rate (from Excel calculator)
3. Invoice Amount
4. WHT Amount (calculated)
5. Net Amount Payable (Invoice - WHT)

---

## Common Scenarios

### Scenario 1: Local Supplier - Full Tax Invoice
- ✅ All 11 points satisfied
- Tax Type: **FI**
- WHT: **NA**

### Scenario 2: Local Supplier - Simplified Tax Invoice
- ✅ Bold points only
- Tax Type: **SI**
- WHT: **NA**

### Scenario 3: Foreign Supplier
- ✅ Tag as Reverse Charge
- Tax Type: **RC**
- ✅ Complete WHT section
- ✅ Attach invoice copy

### Scenario 4: Tax Credit Note
- ✅ Reference original invoice
- Tax Type: **TCN**
- Complete all details

---

## Control Checks

### Pre-Processing
- [ ] Invoice in Arabic
- [ ] Correct addressee
- [ ] All mandatory fields
- [ ] Amounts in SAR

### Processing
- [ ] Tax classification correct
- [ ] Currency conversion accurate
- [ ] WHT calculated (if applicable)
- [ ] All AP fields completed

### Post-Processing
- [ ] All sections completed
- [ ] Calculations verified
- [ ] Documentation attached
- [ ] Ready for submission

---

## Common Mistakes to Avoid

1. ❌ Not converting FCY to SAR
2. ❌ Using wrong exchange rate source
3. ❌ Missing Arabic language requirement
4. ❌ Wrong addressee name
5. ❌ Incorrect tax classification
6. ❌ Missing sequential number
7. ❌ Incomplete TIN (not 15 digits)
8. ❌ Not checking WHT applicability
9. ❌ Wrong WHT rate
10. ❌ Missing invoice copy (reverse charge)

---

## Key Systems & Resources

### Systems
- **GL System** - Account 287961
- **AP System** - PSID 2002976
- **Excel Calculator** - WHT rate calculation
- **SAMA Rates** - Exchange rate source

### Contacts
- **SCCSA Finance Team** - GL details, tax classification
- **AP Team** - Processing and payment
- **Tax Team** - Compliance guidance
- **Treasury** - SAMA exchange rates

---

## Document Structure

### Full Documentation Available:

1. **Saudi_OTP_Entry_VAT_Review.md**
   - Complete detailed procedures
   - All compliance requirements explained
   - Currency conversion procedures
   - WHT calculation details
   - Error prevention and escalation

2. **Saudi_OTP_Entry_VAT_Quick_Reference.md**
   - Step-by-step checklists
   - Decision trees
   - Quick reference tables
   - Common scenarios
   - Contact reference

3. **Saudi_OTP_Entry_VAT_Summary.md** (this document)
   - Executive overview
   - Key points summary

---

## Quick Reference Tables

### GL Account Details:
```
Account:        287961
Product:        910
Department:     8105
Operating Unit: 800
Class:          98
PSID:           2002976 (Default)
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

## Escalation Points

### Escalate if:
- ⚠️ Invoice not in Arabic
- ⚠️ Wrong addressee
- ⚠️ Missing mandatory fields
- ⚠️ Cannot determine tax classification
- ⚠️ Exchange rate not available
- ⚠️ WHT rate unclear

---

## Next Steps

1. ✅ Review detailed procedures document
2. ✅ Use quick reference guide for daily operations
3. ✅ Ensure access to SAMA exchange rates
4. ✅ Verify Excel WHT calculator available
5. ✅ Train team on compliance checklist
6. ✅ Establish document retention procedures

---

**For detailed procedures, refer to:** `Saudi_OTP_Entry_VAT_Review.md`  
**For quick daily reference, use:** `Saudi_OTP_Entry_VAT_Quick_Reference.md`

