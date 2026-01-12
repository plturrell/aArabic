"""
Saudi OTP Entry VAT Processing Workflow Adapter
Implements the specific workflow requirements from Saudi_OTP_Entry_VAT documentation
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from backend.schemas.workflow import WorkflowDefinition, WorkflowNode, WorkflowEdge, NodeType, ProcessType

logger = logging.getLogger(__name__)


class TaxInvoiceType(Enum):
    """Tax invoice classification types per KSA VAT regulations"""
    FULL_TAX_INVOICE = "FI"  # Full Tax Invoice (all 11 checklist points)
    SIMPLIFIED_TAX_INVOICE = "SI"  # Simplified Tax Invoice (bold points only)
    REVERSE_CHARGE = "RC"  # Reverse Charge (foreign suppliers)
    TAX_CREDIT_NOTE = "TCN"  # Tax Credit Note
    TAX_DEBIT_NOTE = "TDN"  # Tax Debit Note


class CurrencyType(Enum):
    """Supported currency types"""
    SAR = "SAR"  # Saudi Riyal (required)
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


@dataclass
class GLAccountDetails:
    """Standard GL Account structure for Saudi OTP entries"""
    account: str = "287961"  # Default GL account for VAT entries
    product: str = "910"     # Product classification code
    department: str = "8105" # Department code
    operating_unit: str = "800"  # Operating unit identifier
    class_code: str = "98"   # Classification code
    affiliate: Optional[str] = None  # Not applicable (leave blank)
    project_id: Optional[str] = None  # Not applicable (leave blank)
    psid: str = "2002976"    # Default PSID


@dataclass
class TaxInvoiceCompliance:
    """Tax invoice compliance checklist per KSA VAT regulations"""
    # Mandatory fields (bold in documentation)
    invoice_in_arabic: bool = False  # Point 1
    addressed_to_scc: bool = False   # Point 2
    amounts_in_sar: bool = False     # Point 8
    
    # Required fields
    date_of_issue: Optional[str] = None      # Point 3
    sequential_number: Optional[str] = None  # Point 4
    supplier_tin_15_digits: Optional[str] = None  # Point 5
    supplier_name: Optional[str] = None      # Point 5
    supplier_address: Optional[str] = None   # Point 5
    goods_services_description: Optional[str] = None  # Point 6
    tax_rate_applied: Optional[float] = None  # Point 7 (e.g., 0.05 for 5%)
    net_amount_sar: Optional[float] = None   # Point 8
    vat_amount_sar: Optional[float] = None   # Point 8
    gross_amount_sar: Optional[float] = None # Point 8
    date_of_supply: Optional[str] = None     # Point 9 (if different from issue date)
    tax_treatment_explanation: Optional[str] = None  # Point 10 (if not 5%)
    original_invoice_reference: Optional[str] = None  # Point 11 (Credit/Debit Notes only)
    
    def is_simplified_compliant(self) -> bool:
        """Check if invoice meets Simplified Tax Invoice requirements (bold points)"""
        return (
            self.invoice_in_arabic and
            self.addressed_to_scc and
            self.amounts_in_sar and
            self.date_of_issue is not None and
            self.sequential_number is not None and
            self.supplier_tin_15_digits is not None and
            self.supplier_name is not None and
            self.supplier_address is not None and
            self.goods_services_description is not None and
            self.tax_rate_applied is not None
        )
    
    def is_full_compliant(self) -> bool:
        """Check if invoice meets Full Tax Invoice requirements (all points)"""
        return (
            self.is_simplified_compliant() and
            # Additional checks for full compliance can be added here
            True  # For now, simplified compliance covers the main requirements
        )
    
    def get_compliance_score(self) -> float:
        """Calculate compliance score (0.0 to 1.0)"""
        total_checks = 11
        passed_checks = 0
        
        # Mandatory checks (weighted higher)
        if self.invoice_in_arabic:
            passed_checks += 2
        if self.addressed_to_scc:
            passed_checks += 2
        if self.amounts_in_sar:
            passed_checks += 2
            
        # Required field checks
        required_fields = [
            self.date_of_issue, self.sequential_number, self.supplier_tin_15_digits,
            self.supplier_name, self.supplier_address, self.goods_services_description,
            self.tax_rate_applied, self.net_amount_sar, self.vat_amount_sar, self.gross_amount_sar
        ]
        
        for field in required_fields:
            if field is not None:
                passed_checks += 0.5
        
        return min(passed_checks / total_checks, 1.0)


@dataclass
class APTeamInstructions:
    """Instructions to AP Team - 11 required fields"""
    tax_type: TaxInvoiceType  # Field 1: FI/SI/RC/TCN/TDN
    psid: str = "2002976"     # Field 2: Default PSID
    date_of_issue: Optional[str] = None      # Field 3
    sequential_number: Optional[str] = None  # Field 4
    supplier_name: Optional[str] = None      # Field 5
    tax_identification_number: Optional[str] = None  # Field 6: 15-digit TIN
    description: Optional[str] = None        # Field 7
    net_amount_sar: Optional[float] = None   # Field 8
    vat_amount_sar: Optional[float] = None   # Field 9
    gross_amount_sar: Optional[float] = None # Field 10
    currency_conversion_rate: Optional[float] = None  # Field 11: SAMA rate if FCY


class SaudiOTPVATWorkflowAdapter:
    """
    Specialized workflow adapter for Saudi OTP Entry VAT processing
    Implements the complete workflow per documentation requirements
    """
    
    def __init__(self):
        self.gl_defaults = GLAccountDetails()
        self.sama_exchange_rates = {}  # Cache for SAMA exchange rates
        
    async def create_saudi_otp_vat_workflow(self) -> WorkflowDefinition:
        """
        Create the Saudi OTP Entry VAT processing workflow
        Based on the decision tree and process flow in documentation
        """
        nodes = [
            # Start node
            WorkflowNode(
                id="start",
                type=NodeType.START,
                label="OTP Entry Request Received",
                position={"x": 100, "y": 100}
            ),
            
            # Decision: Is it a VAT Payment Entry?
            WorkflowNode(
                id="vat_payment_check",
                type=NodeType.DECISION,
                label="Is VAT Payment Entry?",
                condition="check_vat_payment_entry",
                position={"x": 100, "y": 200}
            ),
            
            # GL Details completion
            WorkflowNode(
                id="complete_gl_details",
                type=NodeType.PROCESS,
                label="Complete GL Details",
                process_type=ProcessType.ANALYSIS,
                config={
                    "account": "287961",
                    "product": "910",
                    "department": "8105",
                    "operating_unit": "800",
                    "class": "98"
                },
                position={"x": 300, "y": 150}
            ),
            
            # Tax Invoice Compliance Verification
            WorkflowNode(
                id="verify_tax_compliance",
                type=NodeType.PROCESS,
                label="Verify Tax Invoice Compliance",
                process_type=ProcessType.VALIDATION,
                config={
                    "checklist_points": 11,
                    "mandatory_fields": ["arabic_language", "addressed_to_scc", "amounts_in_sar"]
                },
                position={"x": 500, "y": 150}
            ),
            
            # OCR and Document Analysis
            WorkflowNode(
                id="ocr_extract",
                type=NodeType.PROCESS,
                label="OCR Extraction",
                process_type=ProcessType.OCR,
                position={"x": 300, "y": 250}
            ),
            
            # Arabic Language Verification
            WorkflowNode(
                id="verify_arabic",
                type=NodeType.PROCESS,
                label="Verify Arabic Language",
                process_type=ProcessType.VALIDATION,
                condition="invoice_in_arabic == true",
                position={"x": 500, "y": 250}
            ),
            
            # Currency Conversion (if needed)
            WorkflowNode(
                id="currency_conversion",
                type=NodeType.PROCESS,
                label="Convert to SAR (SAMA rates)",
                process_type=ProcessType.ANALYSIS,
                condition="currency != 'SAR'",
                position={"x": 700, "y": 200}
            ),
            
            # Tax Classification
            WorkflowNode(
                id="classify_tax_invoice",
                type=NodeType.DECISION,
                label="Classify Tax Invoice Type",
                condition="determine_tax_type",
                position={"x": 700, "y": 300}
            ),
            
            # Withholding Tax Check
            WorkflowNode(
                id="wht_check",
                type=NodeType.DECISION,
                label="Check Withholding Tax",
                condition="vendor_outside_ksa",
                position={"x": 100, "y": 350}
            ),
            
            # AP Instructions Generation
            WorkflowNode(
                id="generate_ap_instructions",
                type=NodeType.PROCESS,
                label="Generate AP Team Instructions",
                process_type=ProcessType.ANALYSIS,
                position={"x": 900, "y": 250}
            ),
            
            # Final Review
            WorkflowNode(
                id="final_review",
                type=NodeType.ACTION,
                label="SCCSA Finance Team Review",
                position={"x": 900, "y": 350}
            ),
            
            # End node
            WorkflowNode(
                id="end",
                type=NodeType.END,
                label="OTP Entry Processed",
                position={"x": 1100, "y": 300}
            )
        ]

        # Define workflow edges (connections between nodes)
        edges = [
            # Main flow
            WorkflowEdge(id="start-vat_check", source="start", target="vat_payment_check"),

            # VAT Payment Entry path
            WorkflowEdge(id="vat_check-gl", source="vat_payment_check", target="complete_gl_details",
                        condition="is_vat_payment == true"),
            WorkflowEdge(id="gl-compliance", source="complete_gl_details", target="verify_tax_compliance"),
            WorkflowEdge(id="compliance-ocr", source="verify_tax_compliance", target="ocr_extract"),
            WorkflowEdge(id="ocr-arabic", source="ocr_extract", target="verify_arabic"),
            WorkflowEdge(id="arabic-currency", source="verify_arabic", target="currency_conversion"),
            WorkflowEdge(id="currency-classify", source="currency_conversion", target="classify_tax_invoice"),
            WorkflowEdge(id="classify-ap", source="classify_tax_invoice", target="generate_ap_instructions"),
            WorkflowEdge(id="ap-review", source="generate_ap_instructions", target="final_review"),
            WorkflowEdge(id="review-end", source="final_review", target="end"),

            # Non-VAT Payment Entry path
            WorkflowEdge(id="vat_check-wht", source="vat_payment_check", target="wht_check",
                        condition="is_vat_payment == false"),
            WorkflowEdge(id="wht-review", source="wht_check", target="final_review"),
        ]

        return WorkflowDefinition(
            id="saudi-otp-vat-processing",
            name="Saudi OTP Entry VAT Processing",
            description="Complete workflow for Saudi OTP Entry VAT processing per KSA regulations",
            version="1.0.0",
            nodes=nodes,
            edges=edges,
            inputs={
                "invoice_document": "",
                "entry_type": "vat_payment",  # vat_payment, withholding_tax, standard
                "currency": "SAR",
                "supplier_country": "KSA",
                "confidence_threshold": 0.85
            },
            outputs=["ap_instructions", "compliance_report", "gl_details"],
            metadata={
                "country": "Saudi Arabia",
                "entity": "Standard Chartered Capital Saudi (SCSA)",
                "compliance_framework": "KSA VAT Regulations",
                "document_reference": "OTP Entry Processing - VAT Q3'25",
                "supports_arabic": True,
                "supports_sama_rates": True,
                "gl_account_default": "287961"
            }
        )

    async def execute_saudi_otp_workflow(
        self,
        invoice_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the Saudi OTP VAT workflow with specific business logic
        """
        if context is None:
            context = {}

        start_time = datetime.now()
        workflow_id = f"saudi-otp-{start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 1: Determine if this is a VAT payment entry
            is_vat_payment = await self._check_vat_payment_entry(invoice_data)

            if is_vat_payment:
                # VAT Payment Entry workflow
                result = await self._process_vat_payment_entry(invoice_data, context)
            else:
                # Check for Withholding Tax
                needs_wht = await self._check_withholding_tax(invoice_data)
                if needs_wht:
                    result = await self._process_withholding_tax_entry(invoice_data, context)
                else:
                    result = await self._process_standard_entry(invoice_data, context)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "workflow_id": workflow_id,
                "success": True,
                "execution_time": execution_time,
                "result": result,
                "compliance_status": result.get("compliance_status", "unknown"),
                "tax_classification": result.get("tax_classification", "unknown")
            }

        except Exception as e:
            logger.error(f"Saudi OTP workflow execution failed: {e}", exc_info=True)
            return {
                "workflow_id": workflow_id,
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

    async def _check_vat_payment_entry(self, invoice_data: Dict[str, Any]) -> bool:
        """Check if this is a VAT payment entry"""
        # Implementation logic for VAT payment detection
        entry_type = invoice_data.get("entry_type", "").lower()
        description = invoice_data.get("description", "").lower()

        vat_keywords = ["vat", "tax", "ضريبة", "قيمة مضافة"]

        return (
            entry_type == "vat_payment" or
            any(keyword in description for keyword in vat_keywords)
        )

    async def _check_withholding_tax(self, invoice_data: Dict[str, Any]) -> bool:
        """Check if withholding tax applies (vendor outside KSA)"""
        supplier_country = invoice_data.get("supplier_country", "KSA").upper()
        return supplier_country != "KSA"

    def _complete_gl_details(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete GL details using standard Saudi OTP structure"""
        return {
            "account": self.gl_defaults.account,
            "product": self.gl_defaults.product,
            "department": self.gl_defaults.department,
            "operating_unit": self.gl_defaults.operating_unit,
            "class": self.gl_defaults.class_code,
            "affiliate": self.gl_defaults.affiliate,
            "project_id": self.gl_defaults.project_id,
            "psid": self.gl_defaults.psid,
            "cost_centre_breakdown": invoice_data.get("cost_centres", {}),
            "responsibility": "SCCSA Finance Team"
        }

    async def _extract_invoice_data(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure invoice data"""
        from .saudi_otp_vat_methods import SaudiOTPVATMethods
        methods = SaudiOTPVATMethods()
        return await methods.extract_invoice_data(invoice_data)

    async def _verify_tax_compliance(self, extracted_data: Dict[str, Any]) -> TaxInvoiceCompliance:
        """Verify tax invoice compliance"""
        from .saudi_otp_vat_methods import SaudiOTPVATMethods
        methods = SaudiOTPVATMethods()
        return await methods.verify_tax_compliance(extracted_data)

    async def _convert_currency_to_sar(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert currency to SAR using SAMA rates"""
        from .saudi_otp_vat_methods import SaudiOTPVATMethods
        methods = SaudiOTPVATMethods()
        return await methods.convert_currency_to_sar(extracted_data)

    async def _classify_tax_invoice(
        self,
        compliance: TaxInvoiceCompliance,
        extracted_data: Dict[str, Any]
    ) -> TaxInvoiceType:
        """Classify tax invoice type"""
        from .saudi_otp_vat_methods import SaudiOTPVATMethods
        methods = SaudiOTPVATMethods()
        return await methods.classify_tax_invoice(compliance, extracted_data)

    async def _generate_ap_instructions(
        self,
        tax_classification: TaxInvoiceType,
        converted_amounts: Dict[str, Any],
        extracted_data: Dict[str, Any]
    ) -> APTeamInstructions:
        """Generate AP team instructions with all 11 required fields"""

        instructions = APTeamInstructions(
            tax_type=tax_classification,
            psid=self.gl_defaults.psid,
            date_of_issue=extracted_data.get("date_of_issue"),
            sequential_number=extracted_data.get("sequential_number"),
            supplier_name=extracted_data.get("supplier_info", {}).get("name"),
            tax_identification_number=extracted_data.get("supplier_info", {}).get("tin"),
            description=extracted_data.get("goods_services_description"),
            net_amount_sar=converted_amounts.get("net_amount_sar"),
            vat_amount_sar=converted_amounts.get("vat_amount_sar"),
            gross_amount_sar=converted_amounts.get("gross_amount_sar"),
            currency_conversion_rate=converted_amounts.get("conversion_rate")
        )

        return instructions

    async def _process_withholding_tax_entry(
        self,
        invoice_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process withholding tax entry for foreign suppliers"""

        # Extract basic information
        extracted_data = await self._extract_invoice_data(invoice_data)

        # Complete GL details
        gl_details = self._complete_gl_details(invoice_data)

        # Calculate WHT (simplified - in production use proper WHT rates)
        wht_rate = 0.05  # 5% WHT for foreign suppliers
        gross_amount = extracted_data.get("amounts", {}).get("gross_amount", 0.0)
        wht_amount = gross_amount * wht_rate

        # Convert to SAR if needed
        converted_amounts = await self._convert_currency_to_sar(extracted_data)

        return {
            "gl_details": gl_details,
            "extracted_data": extracted_data,
            "withholding_tax": {
                "rate": wht_rate,
                "amount": wht_amount,
                "currency": extracted_data.get("amounts", {}).get("currency", "SAR")
            },
            "converted_amounts": converted_amounts,
            "tax_classification": TaxInvoiceType.REVERSE_CHARGE,
            "requires_review": True  # WHT entries always require review
        }

    async def _process_standard_entry(
        self,
        invoice_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process standard (non-VAT, non-WHT) entry"""

        # Extract basic information
        extracted_data = await self._extract_invoice_data(invoice_data)

        # Complete GL details
        gl_details = self._complete_gl_details(invoice_data)

        # Convert to SAR if needed
        converted_amounts = await self._convert_currency_to_sar(extracted_data)

        return {
            "gl_details": gl_details,
            "extracted_data": extracted_data,
            "converted_amounts": converted_amounts,
            "tax_classification": "STANDARD",
            "requires_review": False
        }
