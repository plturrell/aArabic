# Saudi OTP Entry Processing - VAT Detailed Review

**Document:** OTP Entry Processing - VAT Q3'25  
**Country:** Saudi Arabia  
**Entity:** Standard Chartered Capital Saudi (SCSA)  
**Year:** 2025

---

## Executive Summary

This document provides detailed procedures for processing One-Time Payment (OTP) entries for VAT-related transactions in Saudi Arabia. The process ensures compliance with KSA VAT regulations and proper accounting treatment for tax invoices, credit notes, and foreign supplier transactions.

### Key Objectives
- Ensure compliance with KSA VAT regulations
- Proper classification of tax invoices (Full, Simplified, Reverse Charge)
- Accurate currency conversion using SAMA exchange rates
- Correct application of Withholding Tax (WHT) for foreign suppliers
- Proper GL accounting and cost centre allocation

---

## 1. Process Overview

### 1.1 Purpose
To document the standardized process for processing VAT-related OTP entries, ensuring:
- Tax invoice compliance verification
- Proper GL accounting
- Accurate tax classification
- Withholding tax application (where applicable)

### 1.2 Scope
- Local supplier invoices (Full and Simplified Tax Invoices)
- Foreign supplier invoices (Reverse Charge)
- Tax Credit Notes and Tax Debit Notes
- Withholding Tax calculations for foreign suppliers

### 1.3 Key Requirements
- All invoices must be in Arabic language
- All amounts must be in Saudi Riyals (SAR)
- SAMA exchange rates must be used for FCY conversion
- Proper tax classification is mandatory

---

## 2. GL Details Section

### 2.1 Standard GL Account Structure

| Field | Value | Description |
|-------|-------|-------------|
| **Account** | 287961 | Default GL account for VAT entries |
| **Product** | 910 | Product classification code |
| **Department** | 8105 | Department code |
| **Operating Unit** | 800 | Operating unit identifier |
| **Class** | 98 | Classification code |
| **Affiliate** | - | Not applicable (leave blank) |
| **Project ID** | - | Not applicable (leave blank) |

### 2.2 Cost Centre Cost Breakdown
- Provide detailed breakdown of costs across cost centres
- Ensure proper allocation based on expense nature
- Maintain audit trail for cost allocation

**Responsibility:** SCCSA Finance Team

---

## 3. Tax Invoice Compliance Checklist

### 3.1 KSA VAT Regulation Requirements

The following checklist must be completed for each tax invoice, tax credit note, or tax debit note to ensure compliance with KSA VAT regulations:

#### Checklist Items:

**1. Invoice Language**
- **Requirement:** Invoice must be in Arabic language
- **Status:** Mandatory (Y/N)
- **Verification:** Check if invoice contains Arabic text

**2. Addressee**
- **Requirement:** Name and address must be addressed to "Standard Chartered Capital"
- **Status:** Mandatory (Y/N)
- **Verification:** Verify invoice is addressed correctly

**3. Date of Issue**
- **Requirement:** Invoice must show date of issue
- **Status:** Required
- **Example:** 30 October 2025

**4. Sequential Number**
- **Requirement:** Invoice must have sequential number
- **Status:** Required
- **Example:** 3003093648243001
- **Note:** Must be unique and sequential

**5. Supplier Information**
- **Requirement:** Tax Identification Number (15 digits), name and address of supplier
- **Status:** Required
- **Verification:** TIN must be exactly 15 digits

**6. Description**
- **Requirement:** Nature/Description of goods and services supplied
- **Status:** Required
- **Example:** VAT Payment, VAT Q3'25

**7. Tax Rate**
- **Requirement:** Rate of tax applied
- **Status:** Required
- **Common Rates:** 0%, 5%
- **Note:** Must be clearly stated

**8. Amounts in SAR**
- **Requirement:** Net Amount, VAT and Gross amount shown in riyals
- **Status:** Mandatory
- **Important:** If invoice raised in FCY, amounts must be converted to SAR using SAMA exchange rates as per date of supply
- **Example:** Total Amount SAR 1,558,629.95

**9. Date of Supply**
- **Requirement:** Date on which supply took place (if different from issue date)
- **Status:** If applicable
- **Note:** Only required if different from date of issue

**10. Tax Treatment Explanation**
- **Requirement:** Narration explaining tax treatment (if tax not charged at basic rate of 5%)
- **Status:** If applicable
- **Note:** Required for zero-rated, exempt, or other non-standard rates

**11. Original Invoice Reference**
- **Requirement:** Reference to sequential number of original tax invoice (for credit/debit notes only)
- **Status:** If applicable
- **Note:** Only for Tax Credit Notes and Tax Debit Notes

---

### 3.2 Tax Invoice Classification

#### Simplified Tax Invoice (Local Purchase)
**Criteria:** Invoice satisfies all **bold checklist items** (items 1, 2, 3, 4, 5, 6, 7, 8)

**Characteristics:**
- Invoice in Arabic
- Addressed to Standard Chartered Capital
- Date of issue present
- Sequential number present
- Supplier TIN, name, address present
- Description present
- Tax rate shown
- Amounts in SAR

**Use Case:** Standard local purchases meeting minimum requirements

#### Full Tax Invoice (Local Purchase)
**Criteria:** Invoice satisfies **ALL** checklist points (items 1-11 as applicable)

**Characteristics:**
- All simplified tax invoice requirements PLUS
- Date of supply (if different)
- Tax treatment explanation (if applicable)
- Original invoice reference (for credit/debit notes)

**Use Case:** Complete tax-compliant invoices with all details

#### Reverse Charge (Foreign Purchase)
**Criteria:** Purchase from supplier outside Saudi Arabia

**Requirements:**
- Tag the supply as reverse charge
- Must be supported with invoice copy
- VAT calculation may be required locally

**Use Case:** Services or goods purchased from foreign suppliers

---

## 4. AP Team Instructions

### 4.1 Required Information Fields

The following information must be provided to the AP team for processing:

#### Field 1: Tax Type
- **Options:**
  - **FI:** Full Tax Invoice
  - **SI:** Simplified Tax Invoice
  - **RC:** Reverse Charge
  - **TCN:** Tax Credit Note
  - **Other:** Miscellaneous invoices
- **Selection:** Based on compliance checklist results

#### Field 2: PSID
- **Value:** Default 2002976
- **Note:** Standard PSID for processing

#### Field 3: Date of Issue
- **Format:** Date format
- **Example:** 30 October 2025
- **Source:** From invoice

#### Field 4: Sequential Number
- **Format:** Invoice sequential number
- **Example:** 3003093648243001
- **Source:** From invoice

#### Field 5: Supplier Name
- **Format:** Full legal name
- **Example:** THE GENERAL AUTHORITY OF ZAKAT GAZT
- **Source:** From invoice

#### Field 6: Tax Identification Number
- **Format:** 15-digit number
- **Example:** 3003093648
- **Verification:** Must be exactly 15 digits

#### Field 7: Description
- **Format:** Description of goods/services
- **Example:** VAT Q3'25
- **Source:** From invoice

#### Field 8: Tax Rate
- **Format:** Percentage
- **Examples:** 0%, 5%
- **Source:** From invoice

#### Field 9: Net Amount
- **Format:** Amount in SAR (excluding VAT)
- **Example:** SAR 00
- **Calculation:** Gross Amount - VAT Amount

#### Field 10: Gross Amount
- **Format:** Total amount in SAR (including VAT)
- **Example:** SAR 1,558,629.95
- **Note:** Must be in SAR

#### Field 11: Tax Amount
- **Format:** VAT amount in SAR
- **Example:** SAR 00
- **Calculation:** Based on tax rate and net amount

---

## 5. Currency Conversion Requirements

### 5.1 SAMA Exchange Rate Rules

**Mandatory Requirements:**
- Use SAMA (Saudi Arabian Monetary Authority) exchange rates
- Use exchange rate as of **date of supply** (not invoice date if different)
- All amounts must be converted to Saudi Riyals (SAR)

### 5.2 Conversion Process

**Step 1:** Identify Invoice Currency
- Check invoice currency (USD, EUR, GBP, etc.)

**Step 2:** Identify Date of Supply
- Use date of supply from invoice
- If not stated, use date of issue

**Step 3:** Obtain SAMA Exchange Rate
- Access SAMA exchange rate for the specific date
- Use official SAMA rates only

**Step 4:** Convert Amounts
- Convert Net Amount to SAR
- Calculate VAT in SAR
- Calculate Gross Amount in SAR

**Step 5:** Verify Calculations
- Net Amount + VAT Amount = Gross Amount
- All amounts in SAR

### 5.3 Example Conversion

**Original Invoice:**
- Currency: USD
- Net Amount: USD 100,000
- VAT Rate: 5%
- VAT Amount: USD 5,000
- Gross Amount: USD 105,000
- Date of Supply: 30 October 2025
- SAMA Rate: 3.75 SAR/USD

**Converted Amounts:**
- Net Amount: SAR 375,000 (100,000 × 3.75)
- VAT Amount: SAR 18,750 (5,000 × 3.75)
- Gross Amount: SAR 393,750 (105,000 × 3.75)

---

## 6. Withholding Tax (WHT) Processing

### 6.1 Applicability

**WHT is ONLY applicable if:**
- Vendor location is **outside Saudi Arabia**
- Transaction is subject to WHT per KSA regulations

**WHT is NOT applicable if:**
- Vendor is located in Saudi Arabia
- Transaction is exempt from WHT

### 6.2 WHT Calculation Fields

#### Field 1: WHT Applicability
- **Options:** Yes / No / NA
- **Determination:** Based on vendor location

#### Field 2: WHT Rate
- **Source:** Excel calculator based on nature of expense
- **Common Rates:** 5%, 15%, 20%
- **Note:** Rate varies by expense type

#### Field 3: Invoice Amount
- **Format:** Original invoice amount
- **Example:** SAR 100
- **Source:** From invoice

#### Field 4: WHT Amount
- **Format:** Calculated withholding tax
- **Calculation:** Invoice Amount × WHT Rate
- **Example:** SAR 15 (if rate is 15%)

#### Field 5: Net Amount Payable
- **Format:** Amount after WHT deduction
- **Calculation:** Invoice Amount - WHT Amount
- **Example:** SAR 85 (100 - 15)

### 6.3 WHT Calculation Process

**Step 1:** Verify Vendor Location
- Check if vendor is outside Saudi Arabia

**Step 2:** Determine Expense Type
- Identify nature of expense
- Refer to WHT rate schedule

**Step 3:** Calculate WHT
- Use Excel calculator
- Apply appropriate WHT rate
- Calculate WHT amount

**Step 4:** Calculate Net Payable
- Subtract WHT from invoice amount
- This is the amount to be paid

**Step 5:** Document
- Complete WHT section
- Maintain supporting documentation

---

## 7. Processing Workflow

### 7.1 Complete Workflow Steps

**Step 1: Receipt and Initial Review**
- Receive invoice/document
- Verify it's a VAT-related payment
- Check if vendor is local or foreign
- Determine processing requirements

**Step 2: Complete GL Details**
- Fill in account number: 287961
- Complete product code: 910
- Complete department: 8105
- Complete operating unit: 800
- Complete class: 98
- Provide cost centre breakdown

**Step 3: Tax Compliance Verification**
- Run through 11-point compliance checklist
- Verify Arabic language requirement
- Verify addressee is "Standard Chartered Capital"
- Check all mandatory fields
- Classify invoice type (FI/SI/RC/TCN)

**Step 4: Currency Conversion (if applicable)**
- Identify invoice currency
- Identify date of supply
- Obtain SAMA exchange rate
- Convert all amounts to SAR
- Verify calculations

**Step 5: Complete AP Instructions**
- Select Tax Type based on classification
- Enter PSID: 2002976
- Complete all invoice details
- Enter amounts in SAR
- Verify all fields completed

**Step 6: WHT Processing (if applicable)**
- Verify vendor location (outside KSA)
- Determine expense type
- Calculate WHT using Excel calculator
- Complete WHT section
- Calculate net payable amount

**Step 7: Final Review and Submission**
- Review all sections completed
- Verify compliance checks passed
- Verify calculations correct
- Submit to AP team for processing

---

## 8. Control Checks and Quality Assurance

### 8.1 Pre-Processing Controls

**Documentation Checks:**
- [ ] Invoice received and legible
- [ ] Invoice in Arabic language
- [ ] Addressed to Standard Chartered Capital
- [ ] All mandatory fields present

**Data Integrity Checks:**
- [ ] Sequential number present and valid
- [ ] TIN is 15 digits
- [ ] Dates are valid and logical
- [ ] Amounts are reasonable

**Compliance Checks:**
- [ ] Tax classification correct
- [ ] Tax rate appropriate
- [ ] Amounts in SAR (or converted)
- [ ] SAMA rate used (if FCY)

### 8.2 Processing Controls

**Calculation Checks:**
- [ ] Net + VAT = Gross (for standard invoices)
- [ ] WHT calculated correctly (if applicable)
- [ ] Net Payable = Invoice - WHT (if applicable)
- [ ] Currency conversion accurate

**Classification Checks:**
- [ ] Tax Type selection correct
- [ ] FI vs SI classification verified
- [ ] Reverse Charge tagged correctly
- [ ] WHT applicability verified

### 8.3 Post-Processing Controls

**Verification Checks:**
- [ ] All sections completed
- [ ] All amounts verified
- [ ] GL details correct
- [ ] Cost centre allocation appropriate
- [ ] Documentation attached (if required)

---

## 9. Common Scenarios and Examples

### Scenario 1: Local Supplier - Full Tax Invoice

**Invoice Details:**
- Supplier: Local Saudi company
- Language: Arabic
- Addressed to: Standard Chartered Capital
- All 11 checklist points satisfied

**Processing:**
- Tax Type: **FI** (Full Tax Invoice)
- Complete all AP instruction fields
- WHT: **NA** (local supplier)
- Process normally

### Scenario 2: Local Supplier - Simplified Tax Invoice

**Invoice Details:**
- Supplier: Local Saudi company
- Language: Arabic
- Only bold checklist items satisfied

**Processing:**
- Tax Type: **SI** (Simplified Tax Invoice)
- Complete AP instructions
- WHT: **NA** (local supplier)
- Process normally

### Scenario 3: Foreign Supplier - Reverse Charge

**Invoice Details:**
- Supplier: Company outside Saudi Arabia
- Invoice in foreign currency
- Services provided from abroad

**Processing:**
- Tax Type: **RC** (Reverse Charge)
- Convert amounts to SAR using SAMA rates
- Complete WHT section (vendor outside KSA)
- Attach invoice copy
- Calculate WHT based on expense type

### Scenario 4: Tax Credit Note

**Document Details:**
- Type: Tax Credit Note
- References original invoice number
- Issued by supplier

**Processing:**
- Tax Type: **TCN** (Tax Credit Note)
- Reference original invoice sequential number
- Complete all AP instruction fields
- Process as credit entry

### Scenario 5: Zero-Rated Supply

**Invoice Details:**
- Tax Rate: 0%
- Supply type: Zero-rated (e.g., exports, certain services)

**Processing:**
- Tax Type: **FI** or **SI** (depending on completeness)
- Tax Rate: 0%
- VAT Amount: SAR 00
- Provide tax treatment explanation
- Process normally

---

## 10. Error Prevention and Common Mistakes

### 10.1 Common Errors

**1. Currency Conversion Errors**
- ❌ Using wrong exchange rate source
- ❌ Using invoice date instead of supply date
- ❌ Not converting all amounts to SAR
- ✅ **Solution:** Always use SAMA rates as of supply date

**2. Tax Classification Errors**
- ❌ Wrong classification (FI vs SI)
- ❌ Missing reverse charge tag
- ❌ Incorrect tax type code
- ✅ **Solution:** Follow checklist systematically

**3. Language and Addressee Errors**
- ❌ Invoice not in Arabic
- ❌ Wrong addressee name
- ✅ **Solution:** Verify before processing

**4. WHT Calculation Errors**
- ❌ Not checking vendor location
- ❌ Wrong WHT rate
- ❌ Calculation errors
- ✅ **Solution:** Use Excel calculator, verify vendor location

**5. Data Entry Errors**
- ❌ Missing sequential number
- ❌ Incomplete TIN (not 15 digits)
- ❌ Amount calculation errors
- ✅ **Solution:** Double-check all fields

### 10.2 Prevention Measures

**Checklist Usage:**
- Always use the 11-point checklist
- Verify each point systematically
- Don't skip any mandatory items

**Calculation Verification:**
- Verify Net + VAT = Gross
- Verify currency conversion
- Verify WHT calculations

**Documentation:**
- Keep all supporting documents
- Maintain audit trail
- Document exceptions

---

## 11. Escalation and Exception Handling

### 11.1 Escalation Triggers

**Escalate to Finance Manager if:**
- Invoice not in Arabic language
- Not addressed to Standard Chartered Capital
- Cannot determine tax classification
- Exchange rate not available
- WHT rate unclear

**Escalate to Tax Team if:**
- Tax treatment unclear
- Zero-rated vs exempt determination needed
- Reverse charge applicability unclear
- WHT rate determination needed

**Escalate to CFO if:**
- Significant amount involved
- Compliance issues
- Policy interpretation needed

### 11.2 Exception Handling

**Missing Information:**
- Contact supplier for missing details
- Document attempts to obtain information
- Escalate if cannot obtain required information

**Non-Standard Situations:**
- Document the situation
- Seek guidance from tax team
- Maintain audit trail
- Don't process without clarification

---

## 12. Record Retention

### 12.1 Required Documentation

**For Each Entry:**
- Original invoice/document
- Compliance checklist (completed)
- AP instruction form (completed)
- WHT calculation (if applicable)
- Currency conversion documentation (if applicable)
- Supporting emails/correspondence

### 12.2 Retention Period

- Maintain records per KSA tax regulations
- Typically 5-7 years for tax-related documents
- Ensure accessibility for audit purposes

---

## 13. Key Contacts and Resources

### 13.1 Internal Contacts

| Role | Responsibility |
|------|----------------|
| SCCSA Finance Team | GL details, tax classification, WHT calculation |
| AP Team | Processing and payment execution |
| Tax Team | Tax treatment guidance, compliance questions |
| Treasury | SAMA exchange rates |
| Finance Manager | Escalations, exceptions |

### 13.2 External Resources

- **SAMA:** Exchange rates (official source)
- **GAZT:** Tax regulations and guidance
- **KSA VAT Regulations:** Official tax rules

---

## 14. Training Requirements

### 14.1 Required Knowledge

**KSA VAT Regulations:**
- Understanding of tax invoice requirements
- Knowledge of tax rates and classifications
- Understanding of reverse charge mechanism
- Knowledge of WHT regulations

**Systems and Processes:**
- GL account structure
- AP processing systems
- Excel calculators for WHT
- SAMA rate access

**Compliance:**
- Checklist completion
- Documentation requirements
- Record retention

### 14.2 Training Topics

1. KSA VAT regulation overview
2. Tax invoice compliance checklist
3. Tax classification (FI/SI/RC/TCN)
4. Currency conversion procedures
5. WHT calculation and application
6. GL accounting and cost allocation
7. Error prevention and quality checks
8. Escalation procedures

---

## 15. Continuous Improvement

### 15.1 Review Points

- Regular review of compliance checklist
- Update based on regulation changes
- Feedback from AP team
- Error analysis and prevention

### 15.2 Updates

- Monitor KSA VAT regulation changes
- Update procedures as needed
- Communicate changes to team
- Maintain version control

---

**Document Version:** 1.0  
**Last Updated:** Based on VAT Q3'25 Processing Template  
**Next Review:** As per KSA VAT regulation updates or process changes

