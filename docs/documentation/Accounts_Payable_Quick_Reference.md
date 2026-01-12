# Accounts Payable - Quick Reference Procedures Guide

**Country:** Bahrain | **Department:** Country Finance – Accounts Payable

---

## Quick Decision Tree

```
Is it a PO Invoice?
├─ YES → Process via eProcurement → Tax-Q → Classify VAT → Provide POP
└─ NO → Is it Manual Invoice?
    ├─ YES → Get approvals → Scan to APUeInvoice
    └─ NO → Is it Manual Payment?
        ├─ YES → One-off or Recurring?
        │   ├─ One-off → Get MD/CEO + CFO approval → Process
        │   └─ Recurring → Check dispensation → Get CFO approval → Process
        └─ NO → Is it CBR?
            └─ YES → Forward to APU CBR Team (GBS)
```

---

## 1. PO Invoice Processing (Standard Process)

### Step-by-Step:
1. ✅ Vendor scans invoice with PO number to AP team
2. ✅ Invoice appears in Tax-Q system (country finance queue)
3. ✅ Provide Purpose of Payment (POP) - See Annexure 2
4. ✅ Classify VAT category (see VAT Classification Guide below)
5. ✅ Process payment

**Timeline:** Standard processing timeline  
**System:** eProcurement → Tax-Q  
**Responsibility:** Country Finance

---

## 2. Manual Invoice Processing

### Required Approvals (ALL REQUIRED):
- [ ] Cost Centre Owner
- [ ] Country Supply Chain Manager
- [ ] CFO or Delegate

### Documents Required:
- [ ] Invoice
- [ ] Non e-proc memo (Annexure 1) - with all approvals
- [ ] Scan to: **BH, APUeInvoice**

### Control Checks:
- [ ] Payment to vendor as per invoice (unless vendor instruction)
- [ ] Amount within DOA limit (exception: GSAM legal/professional fees)

**Responsibility:** Country Finance / Payment Requester

---

## 3. Manual Payment Processing

### 3.1 One-Off Manual Payment

#### Required Approvals:
- [ ] Cost Centre Owner
- [ ] Country Supply Chain Manager
- [ ] **Managing Director (MD)** or CEO (if no MD)
- [ ] **CFO**

#### Required Documents:
- [ ] Invoice
- [ ] Non e-proc memo (approved)
- [ ] Statement of nature of expense (proving it's one-off)
- [ ] Supplier number OR Screening confirmation (valid 48 hours)
- [ ] Approval document (showing account to debit)
- [ ] OTT form (Annexure 3)

#### Process:
1. ✅ Get all approvals
2. ✅ Take documents to Operations
3. ✅ Print cheque
4. ✅ Record in eOPS
5. ✅ Mark invoice as PAID
6. ✅ Send to APU team (GBS) with AP standard memo (Annexure 4)

---

### 3.2 Recurring Manual Payment

#### Required Approvals:
- [ ] Cost Centre Owner
- [ ] Country Supply Chain Manager
- [ ] **CFO** (NO MD/CEO approval needed)
- [ ] **DISPENSATION** (mandatory - approved by policy owner)

#### Required Documents:
- [ ] Invoice
- [ ] Non e-proc memo (approved)
- [ ] Supplier number OR Screening confirmation (valid 48 hours)
- [ ] Approval document (showing account to debit)
- [ ] **Written dispensation** (must be held by department)
- [ ] OTT form (Annexure 3)

#### Process:
1. ✅ Verify dispensation is in place
2. ✅ Get all approvals (except MD/CEO)
3. ✅ Take documents to Operations
4. ✅ Print cheque
5. ✅ Record in eOPS
6. ✅ Mark invoice as PAID
7. ✅ Send to APU team (GBS) with AP standard memo (Annexure 4)

---

## 4. VAT Classification Quick Guide

| Category | Key Identifier | When to Use |
|----------|----------------|-------------|
| **Full Tax Invoice** | "Tax Invoice" label, full details | Standard compliant VAT invoice |
| **Simplified Tax Invoice** | Missing some details | Amount ≤ BHD 500 (inclusive) |
| **Unregistered Vendor** | Vendor not registered with NBR | Government/non-registered vendors |
| **Zero Rated** | 0% VAT, specific categories | Transport, food, healthcare, education, construction, oil/gas |
| **Exempt** | VAT exempt supply | Real estate, financial services |
| **Tax Credit Note** | Credit note with VAT | Vendor credit notes |
| **Reverse Charge** | Overseas supplier | Calculate VAT and write on invoice |
| **GCC Tax Invoice** | GCC-based supplier | Currently not in use |
| **Inter BU Transfer** | Interbranch transaction | CBRs |
| **Out of Scope** | Pre-2019 or MPV | Supplies before 1 Jan 2019, MPVs |
| **Other** | EWA bills | Electricity and Water Authority |

### VAT Calculation for Reverse Charge:
- Calculate VAT in FCY (foreign currency)
- Calculate VAT in LCY using central bank exchange rate (as of invoice date)
- Write VAT amount on invoice

---

## 5. Payment Suspense Account Clearing

### Accounts:
- **BHD:** 09-0906743-50
- **USD:** 09-0906743-50

### Process:
1. ✅ Extract PSGL account **140-288843-800** (for period)
2. ✅ Reconcile to trial balance (YTD balance 288843)
3. ✅ Extract eBBS transaction archival (payment suspense account)
4. ✅ Compare amounts (must match)
5. ✅ Post PSGL entry to clear suspense → cost centres/GL accounts

### Important:
- ⚠️ Clear ALL items in same month
- ⚠️ Escalate aged items per escalation grid
- ⚠️ Aging = Current Date - Value Date - Life Span

**Approval:** Finance Manager prepares → CFO approves

---

## 6. Post-Payment Actions (Manual Payments)

### Checklist:
- [ ] Record payment in eOPS
  - URL: https://eopsvip05.hk.standardchartered.com:9443/eopssec/index.jsp
- [ ] Mark invoice as PAID
- [ ] Send to APU team (GBS Bangalore/Chennai):
  - Invoice (marked PAID)
  - AP standard memo (Annexure 4)
  - For recording in PSAP and GL posting

### Quarterly Requirement:
- [ ] Retrieve CFO log from eOPS
- [ ] Get CFO approval
- [ ] Upload approval in eOPS (for quarterly manual payments CST check)

---

## 7. Control Checks Summary

### For ALL Payments:
- [ ] Correct approvals obtained
- [ ] Payment to correct vendor
- [ ] Amount within DOA limits (or exception documented)
- [ ] Vendor onboarded OR screened (within 48 hours)

### For Manual Payments:
- [ ] Dispensation in place (if recurring)
- [ ] All required documents attached
- [ ] Payment recorded in eOPS
- [ ] Sent to APU team for GL posting

### For Payment Suspense Clearing:
- [ ] GL extract reconciled to eBBS extract
- [ ] Payments properly approved
- [ ] Correct PSGL chart fields

---

## 8. Approval Matrix

| Payment Type | Cost Centre Owner | Supply Chain Mgr | CFO/Delegate | MD/CEO | Dispensation |
|--------------|-------------------|------------------|--------------|--------|--------------|
| **PO Invoice** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Manual Invoice** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **One-off Manual** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Recurring Manual** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **CBR** | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## 9. DOA Exceptions

### Not Covered by GDAM:
- ✅ GSAM legal and professional fees
- ✅ Tax/VAT payments

**Confirmed by:** Group Corporate Secretariat

---

## 10. Key Systems & Contacts

### Systems:
- **eProcurement** - PO and invoice processing
- **Tax-Q** - Invoice queue and processing
- **PSAP** - PeopleSoft Accounts Payable
- **eBBS** - Banking (payment suspense: 09-0906743-50)
- **eOPS** - Manual payment recording
- **PSGL** - General Ledger

### Teams:
- **APU** - GBS Bangalore/Chennai (invoice processing, GL posting)
- **APU CBR Team** - GBS (Cross Border Recharges)
- **Country Finance AP** - Bahrain (local processing)

### Email Addresses:
- **Manual Invoices:** BH, APUeInvoice
- **Manual Payments:** APU team (GBS)

---

## 11. Escalation Triggers

### Escalate to CFO if:
- ⚠️ Payment request not aligned with Group Business Expenditure Standards
- ⚠️ Suspected fraud or bribery
- ⚠️ Aged items in payment suspense account

---

## 12. Common Mistakes to Avoid

1. ❌ **Processing manual payment without MD/CEO approval** (for one-off)
2. ❌ **Processing recurring payment without dispensation**
3. ❌ **Payment after 48 hours of screening** (screening expires)
4. ❌ **Wrong VAT classification** (especially simplified vs full)
5. ❌ **Missing POP** (Purpose of Payment)
6. ❌ **Not recording in eOPS** (for manual payments)
7. ❌ **Not clearing payment suspense** (monthly requirement)
8. ❌ **Payment to wrong vendor** (not matching invoice)

---

## 13. Quick Contact Reference

| Issue | Contact |
|-------|---------|
| PO Invoice Processing | Country Finance AP Team |
| Manual Invoice | BH, APUeInvoice |
| Manual Payment Recording | eOPS system |
| GL Posting | APU Team (GBS) |
| CBR Processing | APU CBR Team (GBS) |
| Payment Suspense Clearing | Country Finance Manager → CFO |
| VAT Classification Questions | Country Finance |
| DOA Questions | Country Finance / CFO |

---

## 14. Document Checklist

### For Manual Invoice:
- [ ] Invoice
- [ ] Non e-proc memo (Annexure 1)
- [ ] Email approvals (cost centre, supply chain, CFO)

### For Manual Payment:
- [ ] Invoice
- [ ] Non e-proc memo (Annexure 1)
- [ ] Statement (one-off proof) OR Dispensation (recurring)
- [ ] Supplier number OR Screening confirmation
- [ ] Approval document (with account details)
- [ ] OTT form (Annexure 3)
- [ ] AP standard memo (Annexure 4) - for APU team

### For Payment Suspense Clearing:
- [ ] PSGL extract (140-288843-800)
- [ ] Trial balance reconciliation
- [ ] eBBS transaction archival
- [ ] PSGL entry for clearing

---

**Last Updated:** Based on DOI Version 11 (November 2025)  
**Next Review:** November 2026

