"""
Integration tests for Saudi OTP Entry VAT Processing Workflow
Tests the complete workflow alignment with documentation requirements
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from backend.adapters.saudi_otp_vat_workflow import (
    SaudiOTPVATWorkflowAdapter,
    TaxInvoiceType,
    TaxInvoiceCompliance,
    GLAccountDetails
)
from backend.adapters.saudi_otp_vat_methods import SaudiOTPVATMethods


class TestSaudiOTPVATIntegration:
    """Integration tests for Saudi OTP VAT workflow"""
    
    @pytest.fixture
    def workflow_adapter(self):
        return SaudiOTPVATWorkflowAdapter()
    
    @pytest.fixture
    def methods_adapter(self):
        return SaudiOTPVATMethods()
    
    @pytest.fixture
    def sample_arabic_invoice_data(self):
        """Sample invoice data that meets all compliance requirements"""
        return {
            "entry_type": "vat_payment",
            "supplier_name": "الهيئة العامة للزكاة والدخل",  # Arabic supplier name
            "supplier_address": "الرياض، المملكة العربية السعودية",  # Arabic address
            "supplier_tin": "300309364824300",  # 15-digit TIN
            "supplier_country": "KSA",
            "invoice_number": "3003093648243001",
            "date_of_issue": "2025-01-07",
            "date_of_supply": "2025-01-07",
            "description": "ضريبة القيمة المضافة للربع الثالث 2025",  # Arabic description
            "net_amount": 1000.0,
            "vat_amount": 50.0,  # 5% VAT
            "gross_amount": 1050.0,
            "currency": "SAR",
            "tax_rate": 0.05,
            "addressee": "Standard Chartered Capital Saudi Arabia",
            "text": "فاتورة ضريبية مبسطة - الهيئة العامة للزكاة والدخل",  # Arabic text
            "cost_centres": {"8105": 1050.0}
        }
    
    @pytest.fixture
    def sample_foreign_invoice_data(self):
        """Sample foreign supplier invoice requiring reverse charge"""
        return {
            "entry_type": "vat_payment",
            "supplier_name": "International Tech Solutions LLC",
            "supplier_address": "Dubai, United Arab Emirates",
            "supplier_tin": "123456789012345",
            "supplier_country": "UAE",
            "invoice_number": "INT-2025-001",
            "date_of_issue": "2025-01-07",
            "description": "Software licensing services",
            "net_amount": 2000.0,
            "vat_amount": 0.0,  # No VAT for foreign supplier
            "gross_amount": 2000.0,
            "currency": "USD",
            "tax_rate": 0.0,
            "addressee": "Standard Chartered Capital Saudi Arabia",
            "text": "Invoice for software services",
            "cost_centres": {"8105": 2000.0}
        }
    
    @pytest.mark.asyncio
    async def test_workflow_definition_creation(self, workflow_adapter):
        """Test that workflow definition is created correctly"""
        workflow_def = await workflow_adapter.create_saudi_otp_vat_workflow()
        
        # Verify workflow structure
        assert workflow_def.id == "saudi-otp-vat-processing"
        assert workflow_def.name == "Saudi OTP Entry VAT Processing"
        assert len(workflow_def.nodes) > 10  # Should have multiple processing nodes
        assert len(workflow_def.edges) > 5   # Should have multiple connections
        
        # Verify metadata
        assert workflow_def.metadata["country"] == "Saudi Arabia"
        assert workflow_def.metadata["entity"] == "Standard Chartered Capital Saudi (SCSA)"
        assert workflow_def.metadata["gl_account_default"] == "287961"
        assert workflow_def.metadata["supports_arabic"] is True
        
        # Verify inputs and outputs
        assert "invoice_document" in workflow_def.inputs
        assert "entry_type" in workflow_def.inputs
        assert "ap_instructions" in workflow_def.outputs
        assert "compliance_report" in workflow_def.outputs
    
    @pytest.mark.asyncio
    async def test_arabic_invoice_full_workflow(self, workflow_adapter, sample_arabic_invoice_data):
        """Test complete workflow with Arabic invoice meeting all requirements"""
        result = await workflow_adapter.execute_saudi_otp_workflow(
            sample_arabic_invoice_data,
            {"confidence_threshold": 0.85}
        )
        
        # Verify successful execution
        assert result["success"] is True
        assert "workflow_id" in result
        assert result["execution_time"] > 0
        
        workflow_result = result["result"]
        
        # Verify GL details
        gl_details = workflow_result["gl_details"]
        assert gl_details["account"] == "287961"
        assert gl_details["product"] == "910"
        assert gl_details["department"] == "8105"
        assert gl_details["psid"] == "2002976"
        
        # Verify compliance status
        compliance = workflow_result["compliance_status"]
        assert compliance.invoice_in_arabic is True
        assert compliance.addressed_to_scc is True
        assert compliance.amounts_in_sar is True
        assert compliance.is_simplified_compliant() is True
        
        # Verify tax classification
        tax_classification = workflow_result["tax_classification"]
        assert tax_classification in [TaxInvoiceType.FULL_TAX_INVOICE, TaxInvoiceType.SIMPLIFIED_TAX_INVOICE]
        
        # Verify AP instructions
        ap_instructions = workflow_result["ap_instructions"]
        assert ap_instructions.tax_type in [TaxInvoiceType.FULL_TAX_INVOICE, TaxInvoiceType.SIMPLIFIED_TAX_INVOICE]
        assert ap_instructions.psid == "2002976"
        assert ap_instructions.supplier_name == "الهيئة العامة للزكاة والدخل"
        assert ap_instructions.net_amount_sar == 1000.0
        assert ap_instructions.vat_amount_sar == 50.0
        assert ap_instructions.gross_amount_sar == 1050.0
    
    @pytest.mark.asyncio
    async def test_foreign_supplier_reverse_charge(self, workflow_adapter, sample_foreign_invoice_data):
        """Test workflow with foreign supplier requiring reverse charge"""
        result = await workflow_adapter.execute_saudi_otp_workflow(
            sample_foreign_invoice_data,
            {"confidence_threshold": 0.85}
        )
        
        # Verify successful execution
        assert result["success"] is True
        
        workflow_result = result["result"]
        
        # Should be classified as reverse charge
        tax_classification = workflow_result["tax_classification"]
        assert tax_classification == TaxInvoiceType.REVERSE_CHARGE
        
        # Should have withholding tax calculation
        assert "withholding_tax" in workflow_result
        wht_info = workflow_result["withholding_tax"]
        assert wht_info["rate"] == 0.05  # 5% WHT
        assert wht_info["amount"] > 0
        
        # Should require review
        assert workflow_result["requires_review"] is True
        
        # Should have currency conversion
        converted_amounts = workflow_result["converted_amounts"]
        assert converted_amounts["original_currency"] == "USD"
        assert converted_amounts["conversion_rate"] > 1.0  # USD to SAR
        assert converted_amounts["net_amount_sar"] > 2000.0  # Converted amount
    
    @pytest.mark.asyncio
    async def test_compliance_scoring(self, methods_adapter):
        """Test tax invoice compliance scoring system"""
        
        # Test fully compliant invoice
        compliant_data = {
            "invoice_language": "arabic",
            "addressee": "Standard Chartered Capital Saudi Arabia",
            "date_of_issue": "2025-01-07",
            "sequential_number": "3003093648243001",
            "supplier_info": {
                "name": "Test Supplier",
                "address": "Riyadh, KSA",
                "tin": "300309364824300"
            },
            "goods_services_description": "Test services",
            "tax_rate": 0.05,
            "amounts": {
                "net_amount": 1000.0,
                "vat_amount": 50.0,
                "gross_amount": 1050.0,
                "currency": "SAR"
            }
        }
        
        compliance = await methods_adapter.verify_tax_compliance(compliant_data)
        score = compliance.get_compliance_score()
        
        assert score >= 0.9  # Should be highly compliant
        assert compliance.is_simplified_compliant() is True
        assert compliance.is_full_compliant() is True
        
        # Test non-compliant invoice
        non_compliant_data = {
            "invoice_language": "english",  # Not Arabic
            "addressee": "Wrong Company",    # Wrong addressee
            "amounts": {
                "currency": "USD"  # Not SAR
            }
        }
        
        compliance_bad = await methods_adapter.verify_tax_compliance(non_compliant_data)
        score_bad = compliance_bad.get_compliance_score()
        
        assert score_bad < 0.5  # Should be poorly compliant
        assert compliance_bad.is_simplified_compliant() is False
        assert compliance_bad.is_full_compliant() is False
    
    @pytest.mark.asyncio
    async def test_currency_conversion_sama_rates(self, methods_adapter):
        """Test currency conversion using SAMA exchange rates"""
        
        usd_data = {
            "amounts": {
                "net_amount": 1000.0,
                "vat_amount": 50.0,
                "gross_amount": 1050.0,
                "currency": "USD"
            }
        }
        
        converted = await methods_adapter.convert_currency_to_sar(usd_data)
        
        # Verify conversion
        assert converted["original_currency"] == "USD"
        assert converted["conversion_rate"] == 3.75  # USD to SAR rate
        assert converted["net_amount_sar"] == 3750.0  # 1000 * 3.75
        assert converted["vat_amount_sar"] == 187.5   # 50 * 3.75
        assert converted["gross_amount_sar"] == 3937.5 # 1050 * 3.75
        assert converted["sama_rate_used"] is True
        
        # Test SAR (no conversion needed)
        sar_data = {
            "amounts": {
                "net_amount": 1000.0,
                "vat_amount": 50.0,
                "gross_amount": 1050.0,
                "currency": "SAR"
            }
        }
        
        sar_converted = await methods_adapter.convert_currency_to_sar(sar_data)
        assert sar_converted["conversion_rate"] == 1.0
        assert sar_converted["net_amount_sar"] == 1000.0
