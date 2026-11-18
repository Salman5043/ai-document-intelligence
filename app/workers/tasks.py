
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from kombu.exceptions import KombuError
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import InterfaceError, OperationalError

from app.agent.graph import invoice_extraction_graph
from app.database.models import Document, Invoice
from app.database.repository import (
    add_processing_event,
    find_duplicate_invoice,
    save_document_chunks,
    save_invoice,
)
from app.database.session import SessionLocal
from app.document_processing.chunker import chunk_document
from app.document_processing.dispatcher import process_document
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


STALE_PROCESSING_MINUTES = int(
    os.getenv("STALE_PROCESSING_MINUTES", "30")
)

PROCESSING_HEARTBEAT_INTERVAL_SECONDS = int(
    os.getenv("PROCESSING_HEARTBEAT_INTERVAL_SECONDS", "60")
)


class ProcessingLeaseLostError(Exception):
    """Raised when a worker no longer owns a document."""


class ProcessingLeaseHeartbeat:
    """
    Renew a document processing lease in a background thread.

    Each heartbeat uses an independent database session.
    A worker can renew a lease only while its processing token
    remains the current token for that document.
    """

    def __init__(
        self,
        document_id: int,
        processing_token: str,
        interval_seconds: int = PROCESSING_HEARTBEAT_INTERVAL_SECONDS,
    ):
        if interval_seconds < 1:
            raise ValueError(
                "Heartbeat interval must be at least one second."
            )

        self.document_id = document_id
        self.processing_token = processing_token
        self.interval_seconds = interval_seconds

        self._stop_event = Event()
        self._lease_lost_event = Event()

        self._thread = Thread(
            target=self._run,
            name=f"document-heartbeat-{document_id}",
            daemon=True,
        )

    @property
    def lease_lost(self) -> bool:
        return self._lease_lost_event.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(
                timeout=max(5, self.interval_seconds + 1)
            )

    def _renew_lease(self) -> bool:
        """
        Renew the lease only if this worker still owns it.

        Returns False when the document is no longer owned
        by this worker.
        """

        with SessionLocal() as db:
            result = db.execute(
                update(Document)
                .where(
                    Document.id == self.document_id,
                    Document.status == "processing",
                    Document.processing_token
                    == self.processing_token,
                )
                .values(
                    processing_heartbeat_at=datetime.now(UTC),
                )
                .execution_options(synchronize_session=False)
            )

            if result.rowcount != 1:
                db.rollback()
                return False

            db.commit()

        return True

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                renewed = self._renew_lease()

                if not renewed:
                    self._lease_lost_event.set()

                    logger.warning(
                        "Processing lease lost for document %s.",
                        self.document_id,
                    )

                    return

            except Exception:
                # A temporary heartbeat database failure does not
                # immediately prove that ownership has been lost.
                # Continue attempting renewal on the next interval.
                logger.exception(
                    "Heartbeat renewal failed for document %s.",
                    self.document_id,
                )


def is_retryable_error(exc: Exception) -> bool:
    """
    Identify failures that may succeed on another attempt.

    Permanent document and extraction errors are not retried.
    """

    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            OperationalError,
            InterfaceError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None)

    if status_code == 429:
        return True

    if isinstance(status_code, int) and status_code >= 500:
        return True

    retryable_exception_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }

    return type(exc).__name__ in retryable_exception_names


def verify_processing_ownership(
    db,
    document_id: int,
    processing_token: str,
):
    """
    Lock the document and verify that this worker still owns it.

    The row lock prevents stale recovery from taking ownership
    while this worker is committing its final changes.
    """

    document = db.scalar(
        select(Document)
        .where(
            Document.id == document_id,
            Document.status == "processing",
            Document.processing_token == processing_token,
        )
        .with_for_update()
    )

    if document is None:
        raise ProcessingLeaseLostError(
            f"Processing lease lost for document {document_id}."
        )

    return document


@celery_app.task(
    bind=True,
    name="process_uploaded_document",
    max_retries=2,
    default_retry_delay=10,
)
def process_uploaded_document(self, document_id: int):
    """
    Process an uploaded document with lease-based ownership.

    Pipeline:
    document processing -> chunk generation -> invoice extraction
    -> validation -> embeddings -> transactional persistence.

    A background heartbeat keeps long-running processing leases
    alive. A worker may commit results only while it owns the
    document's current processing token.
    """

    processing_token = str(uuid4())
    heartbeat = None

    with SessionLocal() as db:
        document = db.get(Document, document_id)

        if document is None:
            raise ValueError(
                f"Document {document_id} was not found."
            )

        try:
            # -------------------------------------------------
            # 1. Atomically claim the document
            # -------------------------------------------------

            now = datetime.now(UTC)

            claim = db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.status.in_(["uploaded", "queued"]),
                )
                .values(
                    status="processing",
                    processing_started_at=now,
                    processing_heartbeat_at=now,
                    processing_token=processing_token,
                )
                .execution_options(synchronize_session=False)
            )

            if claim.rowcount != 1:
                db.rollback()
                db.refresh(document)

                return {
                    "document_id": document_id,
                    "status": document.status,
                    "skipped": True,
                    "message": (
                        "Document is already processing "
                        "or is not eligible for processing."
                    ),
                }

            add_processing_event(
                db,
                document_id,
                stage="pipeline",
                status="started",
                message="Document processing started.",
            )

            db.commit()

            # -------------------------------------------------
            # 2. Refresh and verify current ownership
            # -------------------------------------------------

            db.refresh(document)

            if (
                document.processing_token != processing_token
                or document.status != "processing"
            ):
                raise ProcessingLeaseLostError(
                    "Processing lease was lost after claiming."
                )

            # -------------------------------------------------
            # 3. Start independent background heartbeat
            # -------------------------------------------------

            heartbeat = ProcessingLeaseHeartbeat(
                document_id=document_id,
                processing_token=processing_token,
            )

            heartbeat.start()

            if not document.storage_path:
                raise FileNotFoundError(
                    "Document storage path is missing."
                )

            file_path = Path(document.storage_path)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Uploaded file not found: {file_path}"
                )

            # -------------------------------------------------
            # 4. Extract document content
            # -------------------------------------------------

            content = process_document(str(file_path))
            document_data = content.to_dict()

            document_data["filename"] = document.filename
            document_data["file_type"] = document.file_type

            # -------------------------------------------------
            # 5. Generate source-aware document chunks
            # -------------------------------------------------

            chunks = chunk_document(document_data)

            logger.info(
                "Generated %s chunks for document %s.",
                len(chunks),
                document_id,
            )

            # -------------------------------------------------
            # 6. Run invoice extraction and validation
            # -------------------------------------------------

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

            # -------------------------------------------------
            # 7. Check duplicate invoices across documents
            # -------------------------------------------------

            supplier = invoice_data.get("supplier_name")
            invoice_number = invoice_data.get("invoice_number")

            if supplier and invoice_number:
                duplicate = find_duplicate_invoice(
                    db=db,
                    supplier_name=supplier,
                    invoice_number=invoice_number,
                    exclude_document_id=document_id,
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

            # -------------------------------------------------
            # 8. Verify heartbeat and ownership before persistence
            # -------------------------------------------------

            if heartbeat.lease_lost:
                raise ProcessingLeaseLostError(
                    f"Heartbeat detected lease loss for "
                    f"document {document_id}."
                )

            document = verify_processing_ownership(
                db,
                document_id,
                processing_token,
            )

            # -------------------------------------------------
            # 9. Update extracted document data
            # -------------------------------------------------

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

            # -------------------------------------------------
            # 10. Generate document-level embedding
            # -------------------------------------------------

            if document.extracted_text.strip():
                from app.database.embeddings import generate_embedding

                document.embedding = generate_embedding(
                    document.extracted_text
                )
            else:
                document.embedding = None

            # -------------------------------------------------
            # 11. Persist chunks and chunk embeddings
            # -------------------------------------------------

            saved_chunks = save_document_chunks(
                db=db,
                document_id=document_id,
                chunks=chunks,
                commit=False,
            )

            chunk_count = len(saved_chunks)

            logger.info(
                "Persisted %s chunks for document %s.",
                chunk_count,
                document_id,
            )

            # -------------------------------------------------
            # 12. Replace previous invoice transactionally
            # -------------------------------------------------

            db.execute(
                delete(Invoice).where(
                    Invoice.document_id == document_id
                )
            )

            invoice = save_invoice(
                db=db,
                document_id=document_id,
                extraction=invoice_data,
                validation=validation_data,
                commit=False,
            )

            # -------------------------------------------------
            # 13. Complete processing and release lease
            # -------------------------------------------------

            if heartbeat.lease_lost:
                raise ProcessingLeaseLostError(
                    f"Heartbeat detected lease loss for "
                    f"document {document_id}."
                )

            document.status = "processed"
            document.processing_started_at = None
            document.processing_heartbeat_at = None
            document.processing_token = None

            add_processing_event(
                db,
                document_id,
                stage="pipeline",
                status="completed",
                message="Document processing completed.",
                details={
                    "invoice_id": invoice.id,
                    "chunk_count": chunk_count,
                    "valid": invoice.valid,
                    "review_required": invoice.review_required,
                },
            )

            # Document data, document embedding, chunks, invoice,
            # and completion event are committed together.
            db.commit()

            return {
                "document_id": document.id,
                "invoice_id": invoice.id,
                "chunk_count": chunk_count,
                "status": document.status,
                "valid": invoice.valid,
                "review_required": invoice.review_required,
            }

        except ProcessingLeaseLostError:
            db.rollback()
            raise

        except Exception as exc:
            db.rollback()

            # -------------------------------------------------
            # 14. Confirm this worker still owns the document
            # -------------------------------------------------

            failed_document = db.scalar(
                select(Document)
                .where(
                    Document.id == document_id,
                    Document.status == "processing",
                    Document.processing_token == processing_token,
                )
                .with_for_update()
            )

            if failed_document is None:
                raise ProcessingLeaseLostError(
                    f"Worker no longer owns document {document_id}."
                ) from exc

            retryable = is_retryable_error(exc)

            will_retry = (
                retryable
                and self.request.retries < self.max_retries
            )

            # -------------------------------------------------
            # 15. Record retry or permanent failure
            # -------------------------------------------------

            if will_retry:
                failed_document.status = "queued"
                failed_document.processing_started_at = None
                failed_document.processing_heartbeat_at = None
                failed_document.processing_token = None

                add_processing_event(
                    db,
                    document_id,
                    stage="pipeline",
                    status="retrying",
                    message=(
                        f"Transient failure. Retry "
                        f"{self.request.retries + 1} of "
                        f"{self.max_retries} scheduled: {exc}"
                    ),
                )

            else:
                failed_document.status = "failed"
                failed_document.processing_started_at = None
                failed_document.processing_heartbeat_at = None
                failed_document.processing_token = None

                existing_errors = (
                    failed_document.errors
                    if isinstance(failed_document.errors, list)
                    else []
                )

                failed_document.errors = [
                    *existing_errors,
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

            # -------------------------------------------------
            # 16. Ask Celery to retry transient failures
            # -------------------------------------------------

            if will_retry:
                raise self.retry(
                    exc=exc,
                    countdown=10,
                )

            raise

        finally:
            # Stop the heartbeat on success, failure, retry,
            # or lease loss.
            if heartbeat is not None:
                heartbeat.stop()


@celery_app.task(
    bind=True,
    name="recover_stale_documents",
    max_retries=2,
    default_retry_delay=30,
)
def recover_stale_documents(self):
    """
    Requeue documents whose processing heartbeats have expired.

    Recovery is conditional on the original processing token.
    """

    cutoff = datetime.now(UTC) - timedelta(
        minutes=STALE_PROCESSING_MINUTES
    )

    # Use processing_started_at as a fallback for older records
    # that might not have a heartbeat timestamp.
    last_activity = func.coalesce(
        Document.processing_heartbeat_at,
        Document.processing_started_at,
    )

    recovered_ids = []

    with SessionLocal() as db:
        stale_documents = db.execute(
            select(
                Document.id,
                Document.processing_token,
            ).where(
                Document.status == "processing",
                last_activity < cutoff,
                Document.processing_token.is_not(None),
            )
        ).all()

        for document_id, old_token in stale_documents:
            recovery = db.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.status == "processing",
                    Document.processing_token == old_token,
                    last_activity < cutoff,
                )
                .values(
                    status="queued",
                    processing_started_at=None,
                    processing_heartbeat_at=None,
                    processing_token=None,
                )
                .execution_options(synchronize_session=False)
            )

            if recovery.rowcount != 1:
                continue

            add_processing_event(
                db,
                document_id,
                stage="recovery",
                status="recovered",
                message=(
                    "Expired processing heartbeat detected. "
                    "Document queued for recovery."
                ),
                details={
                    "previous_processing_token": old_token,
                },
            )

            recovered_ids.append(document_id)

        db.commit()

    dispatched = []
    dispatch_failed = []

    for document_id in recovered_ids:
        try:
            process_uploaded_document.delay(document_id)
            dispatched.append(document_id)

        except KombuError as exc:
            dispatch_failed.append(document_id)

            with SessionLocal() as db:
                document = db.get(Document, document_id)

                if document and document.status == "queued":
                    document.status = "failed"

                    existing_errors = (
                        document.errors
                        if isinstance(document.errors, list)
                        else []
                    )

                    document.errors = [
                        *existing_errors,
                        f"Recovery dispatch failed: {exc}",
                    ]

                    add_processing_event(
                        db,
                        document_id,
                        stage="recovery",
                        status="failed",
                        message=(
                            f"Could not dispatch recovered task: {exc}"
                        ),
                    )

                    db.commit()

    return {
        "recovered": len(recovered_ids),
        "dispatched": dispatched,
        "dispatch_failed": dispatch_failed,
    }
