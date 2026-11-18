
from decimal import Decimal, InvalidOperation

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.embeddings import generate_embedding
from app.database.models import Document, Invoice, ProcessingEvent


def to_decimal(value):
    """Convert a numeric value to Decimal safely."""
    if value is None:
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid monetary value: {value}"
        ) from exc


def add_processing_event(
    db: Session,
    document_id: int,
    stage: str,
    status: str,
    message: str | None = None,
    details: dict | None = None,
):
    """Add a processing event to the database."""

    event = ProcessingEvent(
        document_id=document_id,
        stage=stage,
        status=status,
        message=message,
        details=details or {},
    )

    db.add(event)
    db.flush()

    return event


def save_document(
    db: Session,
    filename: str,
    file_type: str,
    extracted_text: str,
    page_count: int = 0,
    tables: list | None = None,
    metadata: dict | None = None,
    commit: bool = True,
):
    """
    Save document information.

    Set commit=False when saving multiple related records
    inside one transaction.
    """

    embedding = None

    if extracted_text and extracted_text.strip():
        embedding = generate_embedding(extracted_text)

    document = Document(
        filename=filename,
        file_type=file_type,
        status="processed",
        extracted_text=extracted_text,
        page_count=page_count,
        tables_data=tables or [],
        metadata_json=metadata or {},
        embedding=embedding,
    )

    db.add(document)
    db.flush()

    add_processing_event(
        db=db,
        document_id=document.id,
        stage="document_processing",
        status="completed",
        message="Document content saved successfully.",
        details={
            "filename": filename,
            "file_type": file_type,
            "page_count": page_count,
        },
    )

    if commit:
        db.commit()
        db.refresh(document)

    return document



def find_duplicate_invoice(
    db: Session,
    supplier_name: str,
    invoice_number: str,
    exclude_document_id: int | None = None,
):
    """
    Find an existing invoice using supplier name
    and invoice number.

    Comparison ignores leading/trailing whitespace
    and letter casing.

    Optionally exclude invoices belonging to the
    document currently being processed.
    """

    if not supplier_name or not invoice_number:
        return None

    supplier = supplier_name.strip().lower()
    number = invoice_number.strip().lower()

    statement = select(Invoice).where(
        func.lower(func.trim(Invoice.supplier_name)) == supplier,
        func.lower(func.trim(Invoice.invoice_number)) == number,
    )

    if exclude_document_id is not None:
        statement = statement.where(
            Invoice.document_id != exclude_document_id
        )

    return db.scalar(statement)


def save_invoice(
    db: Session,
    document_id: int,
    extraction: dict,
    validation: dict,
    commit: bool = True,
):
    """
    Save extracted invoice information and validation results.

    Set commit=False when saving the invoice together
    with its document in a single transaction.
    """

    invoice = Invoice(
        document_id=document_id,

        invoice_number=extraction.get("invoice_number"),
        invoice_date=extraction.get("invoice_date"),

        supplier_name=extraction.get("supplier_name"),
        customer_name=extraction.get("customer_name"),

        subtotal=to_decimal(extraction.get("subtotal")),
        tax=to_decimal(extraction.get("tax")),
        total=to_decimal(extraction.get("total")),

        currency=extraction.get("currency"),

        line_items=extraction.get("line_items", []),

        missing_fields=extraction.get("missing_fields", []),
        uncertain_fields=extraction.get("uncertain_fields", []),
        extraction_notes=extraction.get("extraction_notes", []),

        valid=validation.get("valid", False),
        review_required=validation.get("review_required", True),

        validation_errors=validation.get("errors", []),
        validation_warnings=validation.get("warnings", []),
    )

    db.add(invoice)
    db.flush()

    add_processing_event(
        db=db,
        document_id=document_id,
        stage="invoice_storage",
        status="completed",
        message="Invoice extraction and validation saved successfully.",
        details={
            "invoice_id": invoice.id,
            "valid": invoice.valid,
            "review_required": invoice.review_required,
        },
    )

    if commit:
        db.commit()
        db.refresh(invoice)

    return invoice
