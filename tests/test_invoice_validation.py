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
