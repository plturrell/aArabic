"""
Additional methods for Saudi OTP VAT Workflow Adapter
Contains the detailed processing methods for compliance, currency conversion, and AP instructions
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .saudi_otp_vat_workflow import (
    TaxInvoiceCompliance, 
    TaxInvoiceType, 
    APTeamInstructions,
    CurrencyType
)

logger = logging.getLogger(__name__)


class SaudiOTPVATMethods:
    """Additional processing methods for Saudi OTP VAT workflow"""
    
    def __init__(self):
        # SAMA exchange rates cache (in production, fetch from SAMA API)
        self.sama_rates = {
            "USD": 3.75,  # USD to SAR
            "EUR": 4.10,  # EUR to SAR  
            "GBP": 4.75,  # GBP to SAR
            "SAR": 1.0    # SAR to SAR
        }
    
    async def extract_invoice_data(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure invoice data from OCR or manual input
        Simulates OCR extraction with Arabic language detection
        """
        extracted = {
            "invoice_language": "arabic" if self._detect_arabic_text(invoice_data.get("text", "")) else "other",
            "addressee": invoice_data.get("addressee", ""),
            "date_of_issue": invoice_data.get("date_of_issue"),
            "sequential_number": invoice_data.get("invoice_number"),
            "supplier_info": {
                "name": invoice_data.get("supplier_name", ""),
                "address": invoice_data.get("supplier_address", ""),
                "tin": invoice_data.get("supplier_tin", "")
            },
            "goods_services_description": invoice_data.get("description", ""),
            "tax_rate": invoice_data.get("tax_rate", 0.05),  # Default 5% VAT
            "amounts": {
                "net_amount": invoice_data.get("net_amount", 0.0),
                "vat_amount": invoice_data.get("vat_amount", 0.0),
                "gross_amount": invoice_data.get("gross_amount", 0.0),
                "currency": invoice_data.get("currency", "SAR")
            },
            "date_of_supply": invoice_data.get("date_of_supply"),
            "tax_treatment_explanation": invoice_data.get("tax_treatment_explanation"),
            "original_invoice_reference": invoice_data.get("original_invoice_reference")
        }
        
        return extracted
    
    def _detect_arabic_text(self, text: str) -> bool:
        """Detect if text contains Arabic characters"""
        if not text:
            return False
        
        arabic_chars = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        for char in text:
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                arabic_chars += 1
        
        return arabic_chars > 0 and (arabic_chars / max(total_chars, 1)) > 0.1
    
    async def verify_tax_compliance(self, extracted_data: Dict[str, Any]) -> TaxInvoiceCompliance:
        """
        Verify tax invoice compliance against KSA VAT regulations
        Returns compliance object with detailed checklist results
        """
        compliance = TaxInvoiceCompliance()
        
        # Point 1: Invoice in Arabic language (MANDATORY)
        compliance.invoice_in_arabic = extracted_data.get("invoice_language") == "arabic"
        
        # Point 2: Addressed to "Standard Chartered Capital" (MANDATORY)
        addressee = extracted_data.get("addressee", "").lower()
        compliance.addressed_to_scc = "standard chartered capital" in addressee
        
        # Point 3: Date of issue
        compliance.date_of_issue = extracted_data.get("date_of_issue")
        
        # Point 4: Sequential number
        compliance.sequential_number = extracted_data.get("sequential_number")
        
        # Point 5: Supplier information
        supplier_info = extracted_data.get("supplier_info", {})
        compliance.supplier_name = supplier_info.get("name")
        compliance.supplier_address = supplier_info.get("address")
        compliance.supplier_tin_15_digits = supplier_info.get("tin")
        
        # Validate TIN is 15 digits
        tin = compliance.supplier_tin_15_digits or ""
        if len(tin.replace(" ", "").replace("-", "")) != 15:
            compliance.supplier_tin_15_digits = None
        
        # Point 6: Description of goods/services
        compliance.goods_services_description = extracted_data.get("goods_services_description")
        
        # Point 7: Tax rate applied
        compliance.tax_rate_applied = extracted_data.get("tax_rate")
        
        # Point 8: Amounts in SAR (MANDATORY)
        amounts = extracted_data.get("amounts", {})
        currency = amounts.get("currency", "SAR")
        compliance.amounts_in_sar = currency.upper() == "SAR"
        
        if compliance.amounts_in_sar:
            compliance.net_amount_sar = amounts.get("net_amount")
            compliance.vat_amount_sar = amounts.get("vat_amount")
            compliance.gross_amount_sar = amounts.get("gross_amount")
        
        # Point 9: Date of supply (if different from issue date)
        compliance.date_of_supply = extracted_data.get("date_of_supply")
        
        # Point 10: Tax treatment explanation (if not 5%)
        if compliance.tax_rate_applied and compliance.tax_rate_applied != 0.05:
            compliance.tax_treatment_explanation = extracted_data.get("tax_treatment_explanation")
        
        # Point 11: Original invoice reference (for credit/debit notes)
        compliance.original_invoice_reference = extracted_data.get("original_invoice_reference")
        
        return compliance
    
    async def convert_currency_to_sar(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert foreign currency amounts to SAR using SAMA exchange rates
        """
        amounts = extracted_data.get("amounts", {})
        currency = amounts.get("currency", "SAR").upper()
        
        if currency == "SAR":
            return {
                "net_amount_sar": amounts.get("net_amount", 0.0),
                "vat_amount_sar": amounts.get("vat_amount", 0.0),
                "gross_amount_sar": amounts.get("gross_amount", 0.0),
                "conversion_rate": 1.0,
                "original_currency": currency,
                "conversion_date": datetime.now().strftime("%Y-%m-%d")
            }
        
        # Get SAMA exchange rate
        conversion_rate = self.sama_rates.get(currency, 1.0)
        
        converted = {
            "net_amount_sar": amounts.get("net_amount", 0.0) * conversion_rate,
            "vat_amount_sar": amounts.get("vat_amount", 0.0) * conversion_rate,
            "gross_amount_sar": amounts.get("gross_amount", 0.0) * conversion_rate,
            "conversion_rate": conversion_rate,
            "original_currency": currency,
            "original_amounts": {
                "net_amount": amounts.get("net_amount", 0.0),
                "vat_amount": amounts.get("vat_amount", 0.0),
                "gross_amount": amounts.get("gross_amount", 0.0)
            },
            "conversion_date": datetime.now().strftime("%Y-%m-%d"),
            "sama_rate_used": True
        }
        
        return converted
    
    async def classify_tax_invoice(
        self, 
        compliance: TaxInvoiceCompliance, 
        extracted_data: Dict[str, Any]
    ) -> TaxInvoiceType:
        """
        Classify tax invoice type based on compliance and supplier location
        """
        supplier_country = extracted_data.get("supplier_country", "KSA").upper()
        invoice_type = extracted_data.get("invoice_type", "").upper()
        
        # Check for Credit/Debit Notes first
        if "CREDIT" in invoice_type or compliance.original_invoice_reference:
            return TaxInvoiceType.TAX_CREDIT_NOTE
        elif "DEBIT" in invoice_type:
            return TaxInvoiceType.TAX_DEBIT_NOTE
        
        # Foreign supplier = Reverse Charge
        if supplier_country != "KSA":
            return TaxInvoiceType.REVERSE_CHARGE
        
        # Local supplier classification
        if compliance.is_full_compliant():
            return TaxInvoiceType.FULL_TAX_INVOICE
        elif compliance.is_simplified_compliant():
            return TaxInvoiceType.SIMPLIFIED_TAX_INVOICE
        else:
            # Default to simplified if basic requirements are met
            return TaxInvoiceType.SIMPLIFIED_TAX_INVOICE
