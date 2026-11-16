
import os
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from kombu.exceptions import KombuError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import Document, Invoice
from app.database.search import keyword_search, semantic_search
from app.database.session import get_db
from app.workers.tasks import process_uploaded_document

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".xlsx",
    ".xls",
    ".csv",
}

MAX_UPLOAD_SIZE = (
    int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
    * 1024
    * 1024
)

STORAGE_DIR = Path(
    os.getenv("STORAGE_DIR", "storage/uploads")
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    original_filename = Path(file.filename or "").name

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="A valid filename is required.",
        )

    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = STORAGE_DIR / stored_filename

    size = 0

    # Save the uploaded file in chunks.
    try:
        with file_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the maximum upload size.",
                    )

                destination.write(chunk)

    except Exception:
        # Cleanup boundary: remove partially uploaded files.
        file_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    # Create the document database record.
    document = Document(
        filename=original_filename,
        file_type=extension.lstrip("."),
        storage_path=str(file_path),
        status="uploaded",
    )

    # Step 1: Save the document record.
    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except SQLAlchemyError as exc:
        db.rollback()
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=503,
            detail="Could not save document information.",
        ) from exc

    # Step 2: Submit document processing task to Celery.
    try:
        process_uploaded_document.delay(document.id)

    except KombuError as exc:
        document.status = "failed"
        document.errors = [
            "Could not queue document processing."
        ]
        db.commit()

        raise HTTPException(
            status_code=503,
            detail="Could not queue document processing.",
        ) from exc

    return {
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status,
        "message": "Document uploaded and queued for processing.",
    }


@router.get("/{document_id}/status")
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status,
        "errors": document.errors,
        "warnings": document.warnings,
    }


@router.get("/{document_id}/result")
def get_document_result(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    invoices = db.scalars(
        select(Invoice).where(
            Invoice.document_id == document_id
        )
    ).all()

    return {
        "document": {
            "id": document.id,
            "filename": document.filename,
            "file_type": document.file_type,
            "status": document.status,
            "extracted_text": document.extracted_text,
            "page_count": document.page_count,
            "errors": document.errors,
            "warnings": document.warnings,
        },
        "invoices": [
            {
                "id": invoice.id,
                "invoice_number": invoice.invoice_number,
                "invoice_date": invoice.invoice_date,
                "supplier_name": invoice.supplier_name,
                "customer_name": invoice.customer_name,
                "subtotal": invoice.subtotal,
                "tax": invoice.tax,
                "total": invoice.total,
                "currency": invoice.currency,
                "line_items": invoice.line_items,
                "valid": invoice.valid,
                "review_required": invoice.review_required,
                "validation_errors": invoice.validation_errors,
                "validation_warnings": invoice.validation_warnings,
            }
            for invoice in invoices
        ],
    }


@router.get("/search")
def search_documents(
    q: str = Query(..., min_length=1),
    mode: str = Query(
        default="keyword",
        pattern="^(keyword|semantic)$",
    ),
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    if mode == "semantic":
        results = semantic_search(db, q, limit)

        return [
            {
                "document_id": item["document"].id,
                "filename": item["document"].filename,
                "status": item["document"].status,
                "distance": item["similarity_distance"],
            }
            for item in results
        ]

    results = keyword_search(db, q, limit)

    return [
        {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
        }
        for document in results
    ]


@router.post("/{document_id}/reprocess")
def reprocess_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if (
        not document.storage_path
        or not Path(document.storage_path).exists()
    ):
        raise HTTPException(
            status_code=404,
            detail="Original uploaded file is unavailable.",
        )

    if document.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Document is already being processed.",
        )

    # Remove old invoice results before reprocessing.
    for invoice in list(document.invoices):
        db.delete(invoice)

    document.status = "uploaded"
    document.errors = []
    document.warnings = []

    db.commit()

    # Submit reprocessing task to Celery.
    try:
        process_uploaded_document.delay(document.id)

    except KombuError as exc:
        document.status = "failed"
        document.errors = [
            "Could not queue document for reprocessing."
        ]
        db.commit()

        raise HTTPException(
            status_code=503,
            detail="Could not queue document for reprocessing.",
        ) from exc

    return {
        "document_id": document.id,
        "status": "uploaded",
        "message": "Document has been queued for reprocessing.",
    }
