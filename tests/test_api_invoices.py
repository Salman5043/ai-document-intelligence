
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.api.routes import invoices as invoices_module
from app.database.models import Document, Invoice, ProcessingEvent

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def create_test_document(db_session):
    document = Document(
        filename="invoice_test.pdf",
        file_type="pdf",
        status="processed",
        storage_path="invoice_test.pdf",
        extracted_text="Sample invoice",
        page_count=1,
        tables_data=[],
        metadata_json={},
        errors=[],
        warnings=[],
    )

    db_session.add(document)
    db_session.flush()

    return document


def create_test_invoice(db_session, document_id, **overrides):
    invoice_data = {
        "document_id": document_id,
        "invoice_number": "INV-001",
        "invoice_date": "2026-11-01",
        "supplier_name": "ABC Supplies",
        "customer_name": "XYZ Company",
        "subtotal": Decimal("100.00"),
        "tax": Decimal("15.00"),
        "total": Decimal("115.00"),
        "currency": "BDT",
        "line_items": [
            {
                "description": "Office supplies",
                "quantity": 1,
                "unit_price": 100.00,
                "amount": 100.00,
            }
        ],
        "missing_fields": [],
        "uncertain_fields": [],
        "extraction_notes": [],
        "valid": True,
        "review_required": False,
        "validation_errors": [],
        "validation_warnings": [],
    }

    invoice_data.update(overrides)

    invoice = Invoice(**invoice_data)
    db_session.add(invoice)
    db_session.flush()

    return invoice


def successful_validation(**overrides):
    result = {
        "valid": True,
        "review_required": False,
        "errors": [],
        "warnings": [],
    }
    result.update(overrides)
    return result


# -------------------------------------------------------------------
# Invoice correction tests
# -------------------------------------------------------------------


def test_correct_invoice_returns_404_when_invoice_does_not_exist(
    client,
):
    response = client.patch(
        "/invoices/999999",
        json={"invoice_number": "INV-UPDATED"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Invoice not found."


def test_correct_invoice_successfully_updates_partial_fields(
    client,
    db_session,
    monkeypatch,
):
    document = create_test_document(db_session)
    invoice = create_test_invoice(db_session, document.id)

    invoice_id = invoice.id

    validation_calls = {}

    def mock_validate_invoice(extraction, existing_invoices=None):
        validation_calls["extraction"] = extraction
        validation_calls["existing_invoices"] = existing_invoices

        return successful_validation()

    monkeypatch.setattr(
        invoices_module,
        "validate_invoice",
        mock_validate_invoice,
    )

    response = client.patch(
        f"/invoices/{invoice_id}",
        json={
            "invoice_number": "INV-UPDATED-001",
            "total": 150.00,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "invoice_id": invoice_id,
        "message": "Invoice corrected and revalidated.",
        "valid": True,
        "review_required": False,
        "errors": [],
        "warnings": [],
    }

    db_session.refresh(invoice)

    assert invoice.invoice_number == "INV-UPDATED-001"
    assert invoice.total == Decimal("150.00")

    # Fields not supplied by the client must remain unchanged.
    assert invoice.supplier_name == "ABC Supplies"
    assert invoice.customer_name == "XYZ Company"
    assert invoice.subtotal == Decimal("100.00")

    assert validation_calls["extraction"].invoice_number == (
        "INV-UPDATED-001"
    )
    assert validation_calls["extraction"].total == 150.00
    assert validation_calls["existing_invoices"] == []

    events = db_session.scalars(
        select(ProcessingEvent).where(
            ProcessingEvent.document_id == document.id,
            ProcessingEvent.stage == "manual_correction",
        )
    ).all()

    assert len(events) == 1
    assert events[0].status == "completed"
    assert events[0].details["invoice_id"] == invoice_id
    assert events[0].details["updated_fields"] == [
        "invoice_number",
        "total",
    ]
    assert events[0].details["valid"] is True


def test_correct_invoice_passes_existing_duplicates_to_validator(
    client,
    db_session,
    monkeypatch,
):
    document = create_test_document(db_session)

    invoice = create_test_invoice(
        db_session,
        document.id,
        invoice_number="INV-DUPLICATE",
        supplier_name="ABC Supplies",
    )

    other_document = create_test_document(db_session)

    create_test_invoice(
        db_session,
        other_document.id,
        invoice_number="INV-DUPLICATE",
        supplier_name="ABC Supplies",
    )

    captured = {}

    def mock_validate_invoice(extraction, existing_invoices=None):
        captured["existing_invoices"] = existing_invoices

        return successful_validation(
            valid=False,
            review_required=True,
            errors=["Duplicate invoice detected."],
            warnings=[],
        )

    monkeypatch.setattr(
        invoices_module,
        "validate_invoice",
        mock_validate_invoice,
    )

    response = client.patch(
        f"/invoices/{invoice.id}",
        json={"customer_name": "Updated Customer"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["valid"] is False
    assert data["review_required"] is True
    assert data["errors"] == ["Duplicate invoice detected."]

    assert captured["existing_invoices"] == [
        {
            "supplier_name": "ABC Supplies",
            "invoice_number": "INV-DUPLICATE",
        }
    ]

    db_session.refresh(invoice)

    assert invoice.customer_name == "Updated Customer"
    assert invoice.valid is False
    assert invoice.review_required is True
    assert invoice.validation_errors == [
        "Duplicate invoice detected."
    ]


def test_correct_invoice_returns_422_when_validation_raises(
    client,
    db_session,
    monkeypatch,
):
    document = create_test_document(db_session)
    invoice = create_test_invoice(db_session, document.id)

    def failing_validator(*args, **kwargs):
        raise ValueError("Invalid invoice data")

    monkeypatch.setattr(
        invoices_module,
        "validate_invoice",
        failing_validator,
    )

    response = client.patch(
        f"/invoices/{invoice.id}",
        json={"invoice_number": "INV-INVALID"},
    )

    assert response.status_code == 422

    assert response.json()["detail"] == (
        "Invoice correction could not be validated: "
        "Invalid invoice data"
    )


def test_correct_invoice_rolls_back_when_commit_fails(
    client,
    db_session,
    monkeypatch,
):
    document = create_test_document(db_session)
    invoice = create_test_invoice(db_session, document.id)

    invoice_id = invoice.id
    original_invoice_number = invoice.invoice_number

    # Persist test setup before simulating the request's failed commit.
    db_session.commit()

    monkeypatch.setattr(
        invoices_module,
        "validate_invoice",
        lambda *args, **kwargs: successful_validation(),
    )

    def failing_commit():
        raise RuntimeError("Database commit failed")

    monkeypatch.setattr(db_session, "commit", failing_commit)

    with pytest.raises(
        RuntimeError,
        match="Database commit failed",
    ):
        client.patch(
            f"/invoices/{invoice_id}",
            json={"invoice_number": "INV-COMMIT-FAIL"},
        )

    # Query again instead of refreshing the original ORM instance.
    persisted_invoice = db_session.execute(
        select(Invoice).where(Invoice.id == invoice_id)
    ).scalar_one()

    assert persisted_invoice.invoice_number == original_invoice_number

    events = db_session.scalars(
        select(ProcessingEvent).where(
            ProcessingEvent.document_id == document.id,
            ProcessingEvent.stage == "manual_correction",
        )
    ).all()

    assert events == []
