
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
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.agent.qa_graph import document_qa_graph
from app.api.schemas import (
    DocumentQuestionRequest,
    DocumentQuestionResponse,
)
from app.database.models import Document, Invoice
from app.database.search import (
    keyword_search,
    semantic_search,
    semantic_chunk_search,
)
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
        file_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    document = Document(
        filename=original_filename,
        file_type=extension.lstrip("."),
        storage_path=str(file_path),
        status="uploaded",
    )

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

    try:
        process_uploaded_document.delay(document.id)

    except KombuError as exc:
        document.status = "failed"
        document.errors = [
            "Could not queue document processing."
        ]

        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()

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
        pattern="^(keyword|semantic|chunks)$",
    ),
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Search uploaded documents.

    Modes:
    - keyword: Search filenames and extracted text.
    - semantic: Search document-level embeddings.
    - chunks: Search individual document chunks.
    """

    if mode == "semantic":
        results = semantic_search(
            db=db,
            query=q,
            limit=limit,
        )

        return [
            {
                "document_id": item["document"].id,
                "filename": item["document"].filename,
                "status": item["document"].status,
                "distance": item["similarity_distance"],
                "similarity_score": item.get("similarity_score"),
            }
            for item in results
        ]

    if mode == "chunks":
        results = semantic_chunk_search(
            db=db,
            query=q,
            limit=limit,
        )

        return [
            {
                "chunk_id": item["chunk"].id,
                "document_id": item["document"].id,
                "filename": item["document"].filename,
                "status": item["document"].status,
                "chunk_index": item["chunk"].chunk_index,
                "content": item["chunk"].content,
                "distance": item["similarity_distance"],
                "similarity_score": item.get("similarity_score"),
                "source": item["source"],
                "metadata": item["metadata"],
            }
            for item in results
        ]

    results = keyword_search(
        db=db,
        query=q,
        limit=limit,
    )

    return [
        {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
        }
        for document in results
    ]


@router.post(
    "/{document_id}/ask",
    response_model=DocumentQuestionResponse,
)
def ask_document_question(
    document_id: int,
    request: DocumentQuestionRequest,
    db: Session = Depends(get_db),
):
    """
    Ask a question about a processed document.

    Retrieves relevant chunks from the specified document
    and generates an evidence-grounded answer using Groq.

    Returns source references for supported answers.
    """

    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if document.status != "processed":
        raise HTTPException(
            status_code=409,
            detail=(
                "Document must be successfully processed "
                "before asking questions."
            ),
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    initial_state = {
        "document_id": document_id,
        "question": question,
        "retrieval_limit": request.retrieval_limit,
        "retrieved_chunks": [],
        "context": "",
        "answer": None,
        "sources": [],
        "insufficient_evidence": False,
        "errors": [],
        "warnings": [],
    }

    try:
        result = document_qa_graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "db": db,
                }
            },
        )

    except Exception as exc:
        # Log the underlying exception in the application logger.
        import logging

        logging.getLogger(__name__).exception(
            "Document question answering failed for document %s",
            document_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Could not process the document question.",
        ) from exc

    if result.get("errors"):
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve document evidence.",
        )

    return {
        "document_id": document_id,
        "question": question,
        "answer": result.get("answer") or (
            "I couldn't generate an answer from this document."
        ),
        "insufficient_evidence": result.get(
            "insufficient_evidence",
            True,
        ),
        "sources": result.get("sources", []),
    }


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

    previous_status = document.status
    previous_errors = list(document.errors or [])
    previous_warnings = list(document.warnings or [])

    try:
        result = db.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.status.notin_(
                    ["processing", "queued"]
                ),
            )
            .values(
                status="queued",
                errors=[],
                warnings=[],
            )
            .execution_options(synchronize_session=False)
        )

        if result.rowcount != 1:
            db.rollback()

            raise HTTPException(
                status_code=409,
                detail=(
                    "Document is already queued "
                    "or being processed."
                ),
            )

        db.commit()

    except HTTPException:
        raise

    except SQLAlchemyError as exc:
        db.rollback()

        raise HTTPException(
            status_code=503,
            detail="Could not prepare document for reprocessing.",
        ) from exc

    try:
        process_uploaded_document.delay(document_id)

    except KombuError as exc:
        try:
            db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.status == "queued",
                )
                .values(
                    status=previous_status,
                    errors=[
                        *previous_errors,
                        "Could not queue document for reprocessing.",
                    ],
                    warnings=previous_warnings,
                )
                .execution_options(synchronize_session=False)
            )

            db.commit()

        except SQLAlchemyError:
            db.rollback()

        raise HTTPException(
            status_code=503,
            detail="Could not queue document for reprocessing.",
        ) from exc

    return {
        "document_id": document_id,
        "status": "queued",
        "message": "Document has been queued for reprocessing.",
    }
