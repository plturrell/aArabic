"""
Shared rules engine for VAT and AP DOI logic.

This module centralizes rule loading for:
- Saudi OTP VAT processing (KSA)
- Bahrain AP DOI approvals

It is intentionally thin and delegates detailed KSA logic to the
existing SaudiOTPVATMethods / workflow adapter where appropriate,
while sourcing taxonomy / metadata from JSON config files.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from backend.adapters.saudi_otp_vat_methods import SaudiOTPVATMethods
from backend.adapters.saudi_otp_vat_workflow import TaxInvoiceType


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


class KsaClassificationResult(TypedDict, total=False):
    tax_type: str
    checklist_score: float
    is_full_compliant: bool
    is_simplified_compliant: bool
    compliance_details: Dict[str, Any]


@lru_cache()
def _load_json_config(name: str) -> Dict[str, Any]:
    """Load a JSON rules config from backend/config."""
    path = CONFIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Rules config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_ksa_rules() -> Dict[str, Any]:
    """Return parsed Saudi VAT rules config."""
    return _load_json_config("rules_ksa_vat.json")


def get_bahrain_ap_rules() -> Dict[str, Any]:
    """Return parsed Bahrain AP DOI rules config."""
    return _load_json_config("rules_bh_ap_doi.json")


async def classify_ksa_invoice(invoice_data: Dict[str, Any]) -> KsaClassificationResult:
    """
    Classify a Saudi invoice using the existing methods adapter, while
    sourcing taxonomy / metadata from the KSA rules config.

    This is a light wrapper that:
    - runs OCR/field extraction
    - verifies ZATCA 11‑point compliance
    - classifies FI / SI / RC / TCN
    """
    methods = SaudiOTPVATMethods()

    # 1) Normalize / extract
    extracted = await methods.extract_invoice_data(invoice_data)

    # 2) Compliance object (encodes 11‑point checklist)
    compliance = await methods.verify_tax_compliance(extracted)

    # 3) Tax type classification
    tax_type_enum: TaxInvoiceType = await methods.classify_tax_invoice(
        compliance, extracted
    )

    # 4) Build normalized result
    result: KsaClassificationResult = {
        "tax_type": tax_type_enum.value,
        "checklist_score": compliance.get_compliance_score(),
        "is_full_compliant": compliance.is_full_compliant(),
        "is_simplified_compliant": compliance.is_simplified_compliant(),
        "compliance_details": compliance.__dict__,
    }

    return result


def get_bahrain_approval_requirements(payment_type: str) -> Optional[Dict[str, Any]]:
    """
    Given a Bahrain payment type (e.g. MANUAL_ONE_OFF), return the
    approval matrix entry from the AP DOI rules config.
    """
    rules = get_bahrain_ap_rules()
    for row in rules.get("approvalMatrix", []):
        if row.get("paymentType") == payment_type:
            return row.get("approvals", {})
    return None


def list_ksa_tax_types() -> List[Dict[str, Any]]:
    """Expose KSA tax types (FI / SI / RC / TCN) from config for APIs/UI."""
    rules = get_ksa_rules()
    return rules.get("taxTypes", [])


def get_ksa_gl_template() -> Dict[str, Any]:
    """Expose default GL account template for Saudi VAT entries."""
    rules = get_ksa_rules()
    return rules.get("glTemplate", {})



