
from decimal import Decimal

from app.database.models import Document, Invoice
from app.database.repository import find_duplicate_invoice


def create_test_document(db, filename):
    document = Document(
        filename=filename,
        file_type="pdf",
        status="processed",
    )
    db.add(document)
    db.flush()
    return document


def create_test_invoice(db, document_id):
    invoice = Invoice(
        document_id=document_id,
        invoice_number="INV-1001",
        supplier_name="ABC Supplies",
        total=Decimal("1000.00"),
    )
    db.add(invoice)
    db.flush()
    return invoice


def test_find_duplicate_invoice_detects_other_document(db_session):
    document_one = create_test_document(
        db_session,
        "invoice_one.pdf",
    )
    document_two = create_test_document(
        db_session,
        "invoice_two.pdf",
    )

    existing_invoice = create_test_invoice(
        db_session,
        document_one.id,
    )

    result = find_duplicate_invoice(
        db=db_session,
        supplier_name="ABC Supplies",
        invoice_number="INV-1001",
        exclude_document_id=document_two.id,
    )

    assert result is not None
    assert result.id == existing_invoice.id


def test_find_duplicate_invoice_excludes_current_document(db_session):
    document = create_test_document(
        db_session,
        "invoice.pdf",
    )

    create_test_invoice(db_session, document.id)

    result = find_duplicate_invoice(
        db=db_session,
        supplier_name="ABC Supplies",
        invoice_number="INV-1001",
        exclude_document_id=document.id,
    )

    assert result is None


def test_find_duplicate_invoice_is_case_insensitive(db_session):
    document = create_test_document(
        db_session,
        "invoice.pdf",
    )

    existing_invoice = create_test_invoice(
        db_session,
        document.id,
    )

    result = find_duplicate_invoice(
        db=db_session,
        supplier_name="  abc supplies  ",
        invoice_number=" inv-1001 ",
    )

    assert result is not None
    assert result.id == existing_invoice.id


def test_find_duplicate_invoice_returns_none_without_match(db_session):
    result = find_duplicate_invoice(
        db=db_session,
        supplier_name="Unknown Supplier",
        invoice_number="INV-9999",
    )

    assert result is None


def test_find_duplicate_invoice_handles_missing_values(db_session):
    assert find_duplicate_invoice(
        db=db_session,
        supplier_name="",
        invoice_number="INV-1001",
    ) is None

    assert find_duplicate_invoice(
        db=db_session,
        supplier_name="ABC Supplies",
        invoice_number="",
    ) is None
