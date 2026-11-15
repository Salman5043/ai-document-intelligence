
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.state import InvoiceExtractionState
from app.api.schemas import InvoiceCorrection
from app.database.models import Invoice
from app.database.repository import add_processing_event
from app.database.session import get_db
from app.extraction.schemas import InvoiceExtraction
from app.validation.validator import validate_invoice


router = APIRouter(
    prefix="/invoices",
    tags=["Invoices"],
)


@router.patch("/{invoice_id}")
def correct_invoice(
    invoice_id: int,
    correction: InvoiceCorrection,
    db: Session = Depends(get_db),
):
    invoice = db.get(Invoice, invoice_id)

    if invoice is None:
        raise HTTPException(
            status_code=404,
            detail="Invoice not found.",
        )

    # Only update fields explicitly supplied by the client.
    updates = correction.model_dump(exclude_unset=True)

    for field, value in updates.items():
        setattr(invoice, field, value)

    # Revalidate the corrected invoice.
    invoice_data = {
        "invoice_number": invoice.invoice_number,
        "invoice_date": invoice.invoice_date,
        "supplier_name": invoice.supplier_name,
        "customer_name": invoice.customer_name,
        "subtotal": (
            float(invoice.subtotal)
            if invoice.subtotal is not None else None
        ),
        "tax": float(invoice.tax) if invoice.tax is not None else None,
        "total": (
            float(invoice.total)
            if invoice.total is not None else None
        ),
        "currency": invoice.currency,
        "line_items": invoice.line_items or [],
        "missing_fields": invoice.missing_fields or [],
        "uncertain_fields": [],
        "extraction_notes": invoice.extraction_notes or [],
    }

    try:
        extraction = InvoiceExtraction(**invoice_data)

        existing_invoices = db.scalars(
            select(Invoice).where(
                Invoice.id != invoice.id,
                Invoice.supplier_name == invoice.supplier_name,
                Invoice.invoice_number == invoice.invoice_number,
            )
        ).all()

        validation = validate_invoice(
            extraction,
            existing_invoices=[
                {
                    "supplier_name": item.supplier_name,
                    "invoice_number": item.invoice_number,
                }
                for item in existing_invoices
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invoice correction could not be validated: {exc}",
        ) from exc

    invoice.valid = validation["valid"]
    invoice.review_required = validation["review_required"]
    invoice.validation_errors = validation["errors"]
    invoice.validation_warnings = validation["warnings"]

    add_processing_event(
        db=db,
        document_id=invoice.document_id,
        stage="manual_correction",
        status="completed",
        message="Invoice was manually corrected and revalidated.",
        details={
            "invoice_id": invoice.id,
            "updated_fields": list(updates.keys()),
            "valid": invoice.valid,
            "review_required": invoice.review_required,
        },
    )

    try:
        db.commit()
        db.refresh(invoice)
    except Exception:
        db.rollback()
        raise

    return {
        "invoice_id": invoice.id,
        "message": "Invoice corrected and revalidated.",
        "valid": invoice.valid,
        "review_required": invoice.review_required,
        "errors": invoice.validation_errors,
        "warnings": invoice.validation_warnings,
    }
