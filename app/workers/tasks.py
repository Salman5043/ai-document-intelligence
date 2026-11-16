
from pathlib import Path

from app.agent.graph import invoice_extraction_graph
from app.database.models import Document
from app.database.repository import (
    add_processing_event,
    find_duplicate_invoice,
    save_invoice,
)
from app.database.session import SessionLocal
from app.document_processing.dispatcher import process_document
from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="process_uploaded_document",
    max_retries=2,
    default_retry_delay=10,
)
def process_uploaded_document(self, document_id: int):
    """
    Process an uploaded document through:
    document processing -> LangGraph extraction ->
    validation -> database persistence.
    """

    with SessionLocal() as db:
        document = db.get(Document, document_id)

        if document is None:
            raise ValueError(
                f"Document {document_id} was not found."
            )

        try:
            document.status = "processing"

            add_processing_event(
                db,
                document.id,
                stage="pipeline",
                status="started",
                message="Document processing started.",
            )

            db.commit()

            # 1. Check uploaded file
            file_path = Path(document.storage_path)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Uploaded file not found: {file_path}"
                )

            # 2. Extract text, pages and tables
            content = process_document(str(file_path))

            document_data = content.to_dict()

            # Keep original filename and extension.
            document_data["filename"] = document.filename
            document_data["file_type"] = document.file_type

            # 3. Run LangGraph extraction and validation
            graph_result = invoice_extraction_graph.invoke({
                "document": document_data,
                "document_text": "",
                "extraction": None,
                "existing_invoices": [],
                "validation": None,
                "errors": [],
                "warnings": [],
            })

            extraction = graph_result.get("extraction")
            validation = graph_result.get("validation")

            if extraction is None:
                raise RuntimeError(
                    "Invoice extraction returned no result: "
                    + "; ".join(graph_result.get("errors", []))
                )

            if validation is None:
                raise RuntimeError(
                    "Invoice validation returned no result."
                )

            invoice_data = extraction.model_dump(mode="json")
            validation_data = dict(validation)

            # 4. Check duplicate invoices in PostgreSQL
            supplier = invoice_data.get("supplier_name")
            invoice_number = invoice_data.get("invoice_number")

            if supplier and invoice_number:
                duplicate = find_duplicate_invoice(
                    db,
                    supplier,
                    invoice_number,
                )

                if duplicate:
                    validation_data["valid"] = False
                    validation_data["review_required"] = True
                    validation_data["errors"] = [
                        *validation_data.get("errors", []),
                        (
                            "Potential duplicate invoice detected. "
                            f"Existing invoice ID: {duplicate.id}"
                        ),
                    ]

                    validation_data["checks"] = {
                        **validation_data.get("checks", {}),
                        "duplicate": False,
                    }

            # 5. Update existing document record
            document.extracted_text = document_data.get(
                "extracted_text", ""
            )

            document.page_count = len(
                document_data.get("pages", [])
            )

            document.tables_data = document_data.get(
                "tables", []
            )

            document.metadata_json = document_data.get(
                "metadata", {}
            )

            document.errors = graph_result.get("errors", [])
            document.warnings = graph_result.get("warnings", [])

            # 6. Generate document embedding
            if document.extracted_text.strip():
                from app.database.embeddings import generate_embedding

                document.embedding = generate_embedding(
                    document.extracted_text
                )

            # 7. Save invoice under the existing document
            invoice = save_invoice(
                db=db,
                document_id=document.id,
                extraction=invoice_data,
                validation=validation_data,
                commit=False,
            )

            document.status = "processed"

            add_processing_event(
                db,
                document.id,
                stage="pipeline",
                status="completed",
                message="Document processing completed.",
                details={
                    "invoice_id": invoice.id,
                    "valid": invoice.valid,
                    "review_required": invoice.review_required,
                },
            )

            db.commit()

            return {
                "document_id": document.id,
                "invoice_id": invoice.id,
                "status": document.status,
                "valid": invoice.valid,
                "review_required": invoice.review_required,
            }

        except Exception as exc:
            db.rollback()

            # Update failure status in a separate transaction.
            failed_document = db.get(Document, document_id)

            if failed_document:
                failed_document.status = "failed"
                failed_document.errors = [
                    *(
                        failed_document.errors
                        if isinstance(failed_document.errors, list)
                        else []
                    ),
                    str(exc),
                ]

                add_processing_event(
                    db,
                    document_id,
                    stage="pipeline",
                    status="failed",
                    message=str(exc),
                )

                db.commit()

            raise
