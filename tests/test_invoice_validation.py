import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.extraction.schemas import (
    InvoiceExtraction,
    InvoiceLineItem,
)
from app.validation.validator import validate_invoice


def make_invoice(**overrides):
    data = {
        "invoice_number": "INV-001",
        "invoice_date": "2026-10-15",
        "supplier_name": "ABC Technologies",
        "customer_name": "XYZ Trading",
        "subtotal": 1200.00,
        "tax": 180.00,
        "total": 1380.00,
        "currency": "USD",
        "line_items": [
            InvoiceLineItem(
                description="Software License",
                quantity=2,
                unit_price=500,
                amount=1000,
            ),
            InvoiceLineItem(
                description="Support Service",
                quantity=1,
                unit_price=200,
                amount=200,
            ),
        ],
    }

    data.update(overrides)
    return InvoiceExtraction(**data)


def test_valid_invoice():
    result = validate_invoice(make_invoice())

    assert result["valid"] is True
    assert result["review_required"] is False


def test_total_mismatch():
    invoice = make_invoice(total=1500)

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["total"] is False


def test_line_item_mismatch():
    invoice = make_invoice(
        line_items=[
            InvoiceLineItem(
                description="Software License",
                quantity=2,
                unit_price=500,
                amount=900,
            )
        ]
    )

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["line_items"] is False


def test_missing_required_field():
    invoice = make_invoice(invoice_number=None)

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["required_fields"] is False


def test_duplicate_invoice():
    invoice = make_invoice()

    existing = [
        {
            "invoice_number": "INV-001",
            "supplier_name": "ABC Technologies",
        }
    ]

    result = validate_invoice(invoice, existing)

    assert result["valid"] is False
    assert result["checks"]["duplicate"] is False


def test_uncertain_fields_require_review():
    invoice = make_invoice(
        uncertain_fields=["invoice_date"]
    )

    result = validate_invoice(invoice)

    assert result["valid"] is True
    assert result["review_required"] is True


def test_discount_is_included_in_financial_validation():
    invoice = make_invoice(
        discount=50.00,
        total=1330.00,
    )

    result = validate_invoice(invoice)

    assert result["valid"] is True
    assert result["checks"]["total"] is True


def test_discount_causes_total_mismatch():
    invoice = make_invoice(
        discount=50.00,
        total=1380.00,
    )

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["total"] is False


def test_probable_duplicate_requires_review():
    invoice = make_invoice(
        invoice_number="INV-NEW",
    )

    existing = [
        {
            "invoice_number": "INV-OLD",
            "supplier_name": "ABC Technologies",
            "invoice_date": "2026-10-15",
            "total": 1380.00,
            "currency": "USD",
        }
    ]

    result = validate_invoice(invoice, existing)

    assert result["valid"] is True
    assert result["review_required"] is True
    assert result["checks"]["duplicate"] is False
    assert result["duplicate_details"]["type"] == "probable"


def test_exact_duplicate_is_detected():
    invoice = make_invoice()

    existing = [
        {
            "invoice_number": "INV 001",
            "supplier_name": "ABC Technologies",
        }
    ]

    result = validate_invoice(invoice, existing)

    assert result["valid"] is False
    assert result["checks"]["duplicate"] is False
    assert result["duplicate_details"]["type"] == "exact"


def test_low_confidence_field_requires_review():
    invoice = make_invoice(
        confidence_scores={
            "invoice_number": 0.95,
            "invoice_date": 0.55,
            "total": 0.98,
        }
    )

    result = validate_invoice(invoice)

    assert result["valid"] is True
    assert result["review_required"] is True
    assert "invoice_date" in result["low_confidence_fields"]


def test_invalid_confidence_score_is_rejected():
    invoice = make_invoice(
        confidence_scores={
            "invoice_number": 1.5,
        }
    )

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["confidence_scores"] is False


def test_non_finite_total_is_rejected():
    invoice = make_invoice(total=float("nan"))

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["numeric_fields"] is False


def test_invalid_invoice_date_is_rejected():
    invoice = make_invoice(
        invoice_date="2026-02-30",
    )

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert result["checks"]["invoice_date_format"] is False


def test_due_date_before_invoice_date_is_rejected():
    invoice = make_invoice(
        due_date="2026-10-10",
    )

    result = validate_invoice(invoice)

    assert result["valid"] is False
    assert any(
        "Due date cannot be earlier" in error
        for error in result["errors"]
    )


def test_field_evidence_is_preserved():
    invoice = make_invoice(
        field_evidence={
            "invoice_number": "Invoice No: INV-001",
            "total": "Grand Total: USD 1380.00",
        }
    )

    assert invoice.field_evidence["invoice_number"] == (
        "Invoice No: INV-001"
    )
