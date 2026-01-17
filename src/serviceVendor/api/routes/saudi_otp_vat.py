"""
Saudi OTP Entry VAT Processing API Routes
Provides endpoints for Saudi-specific VAT processing workflows
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging

from backend.adapters.saudi_otp_vat_workflow import (
    SaudiOTPVATWorkflowAdapter, 
    TaxInvoiceType, 
    GLAccountDetails,
    TaxInvoiceCompliance,
    APTeamInstructions
)
from backend.adapters.saudi_otp_vat_methods import SaudiOTPVATMethods
from backend.services.rules_engine import (
    list_ksa_tax_types,
    get_ksa_gl_template,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/saudi-otp-vat", tags=["Saudi OTP VAT"])

# Initialize adapters
saudi_workflow_adapter = SaudiOTPVATWorkflowAdapter()
saudi_methods = SaudiOTPVATMethods()


class OTPEntryRequest(BaseModel):
    """Request model for OTP entry processing"""
    invoice_document: Optional[str] = Field(None, description="Base64 encoded invoice document")
    entry_type: str = Field("vat_payment", description="Entry type: vat_payment, withholding_tax, standard")
    
    # Invoice details
    supplier_name: Optional[str] = None
    supplier_address: Optional[str] = None
    supplier_tin: Optional[str] = Field(None, description="15-digit Tax Identification Number")
    supplier_country: str = Field("KSA", description="Supplier country code")
    
    # Invoice information
    invoice_number: Optional[str] = Field(None, description="Sequential invoice number")
    date_of_issue: Optional[str] = Field(None, description="Invoice issue date")
    date_of_supply: Optional[str] = Field(None, description="Supply date if different from issue")
    description: Optional[str] = Field(None, description="Goods/services description")
    
    # Financial details
    net_amount: float = Field(0.0, description="Net amount")
    vat_amount: float = Field(0.0, description="VAT amount")
    gross_amount: float = Field(0.0, description="Gross amount")
    currency: str = Field("SAR", description="Currency code")
    tax_rate: float = Field(0.05, description="Tax rate (e.g., 0.05 for 5%)")
    
    # Additional fields
    addressee: Optional[str] = Field(None, description="Invoice addressee")
    text: Optional[str] = Field(None, description="Invoice text content for Arabic detection")
    cost_centres: Dict[str, float] = Field(default_factory=dict, description="Cost centre breakdown")
    
    # Context
    confidence_threshold: float = Field(0.85, description="Confidence threshold for auto-processing")


class OTPEntryResponse(BaseModel):
    """Response model for OTP entry processing"""
    workflow_id: str
    success: bool
    execution_time: float
    
    # Processing results
    gl_details: Optional[Dict[str, Any]] = None
    compliance_status: Optional[Dict[str, Any]] = None
    tax_classification: Optional[str] = None
    ap_instructions: Optional[Dict[str, Any]] = None
    converted_amounts: Optional[Dict[str, Any]] = None
    
    # Decision flags
    requires_review: bool = False
    compliance_score: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ComplianceCheckRequest(BaseModel):
    """Request model for tax invoice compliance check"""
    invoice_data: Dict[str, Any]
    check_type: str = Field("full", description="Compliance check type: full, simplified, basic")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check"""
    compliance_score: float
    is_simplified_compliant: bool
    is_full_compliant: bool
    missing_requirements: List[str]
    compliance_details: Dict[str, Any]
    recommendations: List[str]


@router.post("/process", response_model=OTPEntryResponse)
async def process_otp_entry(request: OTPEntryRequest):
    """
    Process Saudi OTP Entry with VAT compliance checking
    
    This endpoint implements the complete Saudi OTP Entry VAT workflow:
    1. Determines entry type (VAT payment, WHT, standard)
    2. Completes GL details with Saudi-specific defaults
    3. Verifies tax invoice compliance (11-point checklist)
    4. Converts currency to SAR using SAMA rates
    5. Classifies tax invoice type (FI/SI/RC/TCN)
    6. Generates AP team instructions
    """
    try:
        # Convert request to invoice data format
        invoice_data = {
            "entry_type": request.entry_type,
            "supplier_name": request.supplier_name,
            "supplier_address": request.supplier_address,
            "supplier_tin": request.supplier_tin,
            "supplier_country": request.supplier_country,
            "invoice_number": request.invoice_number,
            "date_of_issue": request.date_of_issue,
            "date_of_supply": request.date_of_supply,
            "description": request.description,
            "net_amount": request.net_amount,
            "vat_amount": request.vat_amount,
            "gross_amount": request.gross_amount,
            "currency": request.currency,
            "tax_rate": request.tax_rate,
            "addressee": request.addressee,
            "text": request.text,
            "cost_centres": request.cost_centres
        }
        
        # Execute Saudi OTP workflow
        result = await saudi_workflow_adapter.execute_saudi_otp_workflow(
            invoice_data, 
            {"confidence_threshold": request.confidence_threshold}
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Workflow execution failed"))
        
        workflow_result = result["result"]
        
        # Calculate compliance score
        compliance_score = None
        if "compliance_status" in workflow_result:
            compliance_obj = workflow_result["compliance_status"]
            if hasattr(compliance_obj, 'get_compliance_score'):
                compliance_score = compliance_obj.get_compliance_score()
        
        return OTPEntryResponse(
            workflow_id=result["workflow_id"],
            success=result["success"],
            execution_time=result["execution_time"],
            gl_details=workflow_result.get("gl_details"),
            compliance_status=workflow_result.get("compliance_status").__dict__ if workflow_result.get("compliance_status") else None,
            tax_classification=workflow_result.get("tax_classification").value if workflow_result.get("tax_classification") else None,
            ap_instructions=workflow_result.get("ap_instructions").__dict__ if workflow_result.get("ap_instructions") else None,
            converted_amounts=workflow_result.get("converted_amounts"),
            requires_review=workflow_result.get("requires_review", False),
            compliance_score=compliance_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP entry processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.post("/compliance-check", response_model=ComplianceCheckResponse)
async def check_tax_invoice_compliance(request: ComplianceCheckRequest):
    """
    Check tax invoice compliance against KSA VAT regulations
    
    Performs the 11-point compliance checklist:
    1. Invoice in Arabic language (Mandatory)
    2. Addressed to "Standard Chartered Capital" (Mandatory)  
    3. Date of issue
    4. Sequential number
    5. Supplier TIN (15 digits), name and address
    6. Description of goods/services
    7. Tax rate applied
    8. Amounts in SAR (Mandatory)
    9. Date of supply (if different)
    10. Tax treatment explanation (if not 5%)
    11. Original invoice reference (Credit/Debit notes)
    """
    try:
        # Extract invoice data
        extracted_data = await saudi_methods.extract_invoice_data(request.invoice_data)
        
        # Verify compliance
        compliance = await saudi_methods.verify_tax_compliance(extracted_data)
        
        # Generate recommendations
        recommendations = []
        missing_requirements = []
        
        if not compliance.invoice_in_arabic:
            missing_requirements.append("Invoice must be in Arabic language")
            recommendations.append("Ensure invoice contains Arabic text")
        
        if not compliance.addressed_to_scc:
            missing_requirements.append("Invoice must be addressed to 'Standard Chartered Capital'")
            recommendations.append("Verify invoice addressee is correct")
        
        if not compliance.amounts_in_sar:
            missing_requirements.append("All amounts must be in Saudi Riyals (SAR)")
            recommendations.append("Convert foreign currency amounts using SAMA exchange rates")
        
        if not compliance.supplier_tin_15_digits:
            missing_requirements.append("Supplier Tax Identification Number must be 15 digits")
            recommendations.append("Verify supplier TIN format and completeness")
        
        return ComplianceCheckResponse(
            compliance_score=compliance.get_compliance_score(),
            is_simplified_compliant=compliance.is_simplified_compliant(),
            is_full_compliant=compliance.is_full_compliant(),
            missing_requirements=missing_requirements,
            compliance_details=compliance.__dict__,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Compliance check error: {str(e)}")


@router.get("/workflow-definition")
async def get_saudi_otp_workflow_definition():
    """
    Get the Saudi OTP Entry VAT workflow definition

    Returns the complete workflow structure with nodes, edges, and metadata
    for visualization and execution planning.
    """
    try:
        workflow_def = await saudi_workflow_adapter.create_saudi_otp_vat_workflow()
        return workflow_def.dict()
    except Exception as e:
        logger.error(f"Failed to get workflow definition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow definition error: {str(e)}")


@router.get("/gl-defaults")
async def get_gl_defaults():
    """
    Get default GL account details for Saudi OTP entries

    Returns the standard GL account structure used for VAT entries,
    sourced from the shared KSA VAT rules config to stay aligned with
    Saudi_OTP_Entry_VAT_Quick_Reference.md.
    """
    gl_template = get_ksa_gl_template()
    return {
        "gl_defaults": gl_template or saudi_workflow_adapter.gl_defaults.__dict__,
        "description": "Standard GL account structure for Saudi OTP VAT entries",
        "responsibility": "SCCSA Finance Team"
    }


@router.get("/sama-rates")
async def get_sama_exchange_rates():
    """
    Get current SAMA exchange rates for currency conversion

    Returns exchange rates used for converting foreign currency amounts to SAR.
    In production, this would fetch real-time rates from SAMA API.
    """
    return {
        "rates": saudi_methods.sama_rates,
        "base_currency": "SAR",
        "last_updated": "2025-01-07",  # In production, use actual update timestamp
        "source": "Saudi Arabian Monetary Authority (SAMA)",
        "note": "All foreign currency amounts must be converted to SAR using SAMA rates"
    }


@router.get("/tax-types")
async def get_tax_invoice_types():
    """
    Get available tax invoice classification types

    Returns the tax invoice types used in Saudi VAT processing, sourced
    from the shared KSA VAT rules config so that the API stays aligned
    with the documentation taxonomy.
    """
    tax_types = list_ksa_tax_types()
    return {
        "tax_types": tax_types,
        "classification_logic": "Based on supplier location, KSA VAT 11‑point checklist, and rules_ksa_vat.json"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for Saudi OTP VAT processing service"""
    try:
        # Test workflow creation
        workflow_def = await saudi_workflow_adapter.create_saudi_otp_vat_workflow()

        return {
            "status": "healthy",
            "service": "Saudi OTP Entry VAT Processing",
            "workflow_nodes": len(workflow_def.nodes),
            "workflow_edges": len(workflow_def.edges),
            "gl_account_default": saudi_workflow_adapter.gl_defaults.account,
            "supported_currencies": list(saudi_methods.sama_rates.keys()),
            "compliance_framework": "KSA VAT Regulations",
            "entity": "Standard Chartered Capital Saudi (SCSA)"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
