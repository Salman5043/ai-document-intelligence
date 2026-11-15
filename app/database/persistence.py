
from sqlalchemy.orm import Session

from app.database.repository import (
    add_processing_event,
    find_duplicate_invoice,
    save_document,
    save_invoice,
)


def persist_extraction_result(
    db: Session,
    document_content: dict,
    graph_result: dict,
):
    """
    Persist document, invoice extraction, validation,
    and processing events in a single transaction.

    If any operation fails, roll back the entire transaction.
    """

    try:
        # -----------------------------------------
        # 1. Get extraction and validation results
        # -----------------------------------------

        extraction = graph_result.get("extraction")
        validation = graph_result.get("validation")

        if extraction is None:
            raise ValueError(
                "Cannot persist document: invoice extraction is empty."
            )

        if validation is None:
            raise ValueError(
                "Cannot persist document: validation result is missing."
            )

        # Convert Pydantic model into a JSON-compatible dictionary.
        if hasattr(extraction, "model_dump"):
            invoice_data = extraction.model_dump(mode="json")
        elif isinstance(extraction, dict):
            invoice_data = extraction
        else:
            raise TypeError(
                "Extraction must be a Pydantic model or dictionary."
            )

        validation_data = dict(validation)

        # -----------------------------------------
        # 2. Save document
        # -----------------------------------------

        document = save_document(
            db=db,
            filename=document_content.get("filename", "unknown"),
            file_type=document_content.get("file_type", "unknown"),
            extracted_text=document_content.get("extracted_text", ""),
            page_count=len(document_content.get("pages", [])),
            tables=document_content.get("tables", []),
            metadata=document_content.get("metadata", {}),
            commit=False,
        )

        # -----------------------------------------
        # 3. Check for duplicate invoice
        # -----------------------------------------

        supplier_name = invoice_data.get("supplier_name")
        invoice_number = invoice_data.get("invoice_number")

        if supplier_name and invoice_number:
            duplicate = find_duplicate_invoice(
                db=db,
                supplier_name=supplier_name,
                invoice_number=invoice_number,
            )

            if duplicate:
                duplicate_message = (
                    "Potential duplicate invoice detected. "
                    f"Matches existing invoice ID {duplicate.id}."
                )

                validation_data["valid"] = False
                validation_data["review_required"] = True

                validation_data["errors"] = [
                    *validation_data.get("errors", []),
                    duplicate_message,
                ]

                validation_data["checks"] = {
                    **validation_data.get("checks", {}),
                    "duplicate": False,
                }

        # -----------------------------------------
        # 4. Save invoice and validation results
        # -----------------------------------------

        invoice = save_invoice(
            db=db,
            document_id=document.id,
            extraction=invoice_data,
            validation=validation_data,
            commit=False,
        )

        # -----------------------------------------
        # 5. Save graph warnings and errors
        # -----------------------------------------

        document.errors = graph_result.get("errors", [])
        document.warnings = graph_result.get("warnings", [])

        # -----------------------------------------
        # 6. Add final processing event
        # -----------------------------------------

        add_processing_event(
            db=db,
            document_id=document.id,
            stage="pipeline",
            status="completed",
            message="Document extraction and persistence completed.",
            details={
                "invoice_id": invoice.id,
                "valid": invoice.valid,
                "review_required": invoice.review_required,
            },
        )

        # -----------------------------------------
        # 7. Commit everything together
        # -----------------------------------------

        db.commit()

        db.refresh(document)
        db.refresh(invoice)

        return {
            "document_id": document.id,
            "invoice_id": invoice.id,
            "filename": document.filename,
            "status": document.status,
            "valid": invoice.valid,
            "review_required": invoice.review_required,
            "errors": invoice.validation_errors,
            "warnings": invoice.validation_warnings,
        }

    except Exception:
        db.rollback()
        raise
