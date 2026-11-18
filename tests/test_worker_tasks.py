
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from kombu.exceptions import KombuError
from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from app.database.models import Document, DocumentChunk, Invoice
from app.workers import tasks as tasks_module


# ---------------------------------------------------------
# Test doubles
# ---------------------------------------------------------


class FakeExtraction:
    def model_dump(self, mode="json"):
        return {
            "invoice_number": "INV-NEW-1002",
            "invoice_date": "2026-11-17",
            "supplier_name": "ABC Supplies",
            "customer_name": "XYZ Company",
            "subtotal": "1500.00",
            "tax": "150.00",
            "total": "1650.00",
            "currency": "BDT",
            "line_items": [
                {
                    "description": "Test Product",
                    "quantity": 1,
                    "unit_price": "1500.00",
                    "total": "1500.00",
                }
            ],
            "missing_fields": [],
            "uncertain_fields": [],
            "extraction_notes": [],
        }


class FakeContent:
    def to_dict(self):
        return {
            "extracted_text": "",
            "pages": [],
            "tables": [],
            "metadata": {},
        }


class FakeGraph:
    def __init__(self, extraction=None, validation=None, error=None):
        self.extraction = extraction or FakeExtraction()
        self.validation = validation or {
            "valid": True,
            "review_required": False,
            "errors": [],
            "warnings": [],
            "checks": {},
        }
        self.error = error

    def invoke(self, state):
        if self.error:
            raise self.error

        return {
            "extraction": self.extraction,
            "validation": self.validation,
            "errors": [],
            "warnings": [],
        }


class RetryRequested(Exception):
    """Simulate Celery retry scheduling in tests."""


class RateLimitError(Exception):
    """Simulated AI provider rate-limit error."""

    status_code = 429


# ---------------------------------------------------------
# Database helpers
# ---------------------------------------------------------


def create_test_document(db, storage_path, status="uploaded"):
    document = Document(
        filename="test_invoice.pdf",
        file_type="pdf",
        status=status,
        storage_path=str(storage_path),
        extracted_text="",
        page_count=0,
        tables_data=[],
        metadata_json={},
        errors=[],
        warnings=[],
    )

    db.add(document)
    db.flush()

    return document


def create_test_invoice(db, document_id, invoice_number="INV-OLD-1001"):
    invoice = Invoice(
        document_id=document_id,
        invoice_number=invoice_number,
        invoice_date="2026-10-01",
        supplier_name="ABC Supplies",
        customer_name="XYZ Company",
        subtotal=Decimal("1000.00"),
        tax=Decimal("100.00"),
        total=Decimal("1100.00"),
        currency="BDT",
        line_items=[],
        missing_fields=[],
        uncertain_fields=[],
        extraction_notes=[],
        valid=True,
        review_required=False,
        validation_errors=[],
        validation_warnings=[],
    )

    db.add(invoice)
    db.flush()

    return invoice


def create_test_chunk(
    db,
    document_id,
    content="Previously stored document content",
    chunk_index=0,
):
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        source={
            "filename": "test_invoice.pdf",
            "page_number": 1,
        },
        metadata_json={},
        embedding=[0.2] * 384,
    )

    db.add(chunk)
    db.flush()

    return chunk


def create_processing_document(
    db,
    *,
    processing_started_at,
    processing_heartbeat_at,
    processing_token="test-processing-token",
):
    document = Document(
        filename="stale_invoice.pdf",
        file_type="pdf",
        status="processing",
        storage_path="stale_invoice.pdf",
        extracted_text="",
        page_count=0,
        tables_data=[],
        metadata_json={},
        errors=[],
        warnings=[],
        processing_started_at=processing_started_at,
        processing_heartbeat_at=processing_heartbeat_at,
        processing_token=processing_token,
    )

    db.add(document)
    db.flush()

    return document


# ---------------------------------------------------------
# Worker test fixture
# ---------------------------------------------------------


@pytest.fixture
def worker_environment(db_session, monkeypatch, tmp_path):
    """
    Run the worker against the existing test database.

    External document processing and AI extraction are mocked.
    Chunk generation and embedding generation are controlled.
    Database operations remain real.
    """

    connection = db_session.connection()

    test_session_factory = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    monkeypatch.setattr(
        tasks_module,
        "SessionLocal",
        test_session_factory,
    )

    monkeypatch.setattr(
        tasks_module,
        "process_document",
        lambda file_path: FakeContent(),
    )

    monkeypatch.setattr(
        tasks_module,
        "invoice_extraction_graph",
        FakeGraph(),
    )

    # Provide deterministic chunks for worker tests.
    monkeypatch.setattr(
        tasks_module,
        "chunk_document",
        lambda document: [
            {
                "chunk_id": "chunk_000001",
                "content": (
                    "Invoice INV-NEW-1002. "
                    "Total: 1650.00 BDT."
                ),
                "source": {
                    "filename": "test_invoice.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                },
                "metadata": {
                    "extraction_method": "digital",
                },
            }
        ],
    )

    # Avoid loading the real embedding model in unit tests.
    monkeypatch.setattr(
        "app.database.repository.generate_embeddings",
        lambda texts: [
            [0.1] * 384 for _ in texts
        ],
    )

    uploaded_file = tmp_path / "test_invoice.pdf"
    uploaded_file.write_bytes(b"test invoice content")

    return uploaded_file


# ---------------------------------------------------------
# Basic worker tests
# ---------------------------------------------------------


def test_worker_successfully_processes_document(
    db_session,
    worker_environment,
):
    document = create_test_document(
        db_session,
        worker_environment,
    )

    result = tasks_module.process_uploaded_document.run(
        document.id
    )

    assert result["document_id"] == document.id
    assert result["status"] == "processed"
    assert result["valid"] is True
    assert result["chunk_count"] == 1

    invoices = db_session.scalars(
        select(Invoice).where(
            Invoice.document_id == document.id
        )
    ).all()

    assert len(invoices) == 1
    assert invoices[0].invoice_number == "INV-NEW-1002"
    assert invoices[0].total == Decimal("1650.00")

    chunks = db_session.scalars(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    ).all()

    assert len(chunks) == 1
    assert chunks[0].content == (
        "Invoice INV-NEW-1002. Total: 1650.00 BDT."
    )
    assert chunks[0].chunk_index == 0
    assert chunks[0].source["page_number"] == 1
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == 384


def test_worker_reprocessing_replaces_existing_invoice(
    db_session,
    worker_environment,
):
    document = create_test_document(
        db_session,
        worker_environment,
        status="queued",
    )

    create_test_invoice(
        db_session,
        document.id,
    )

    result = tasks_module.process_uploaded_document.run(
        document.id
    )

    assert result["status"] == "processed"

    invoices = db_session.scalars(
        select(Invoice).where(
            Invoice.document_id == document.id
        )
    ).all()

    assert len(invoices) == 1
    assert invoices[0].invoice_number == "INV-NEW-1002"
    assert invoices[0].total == Decimal("1650.00")
    assert invoices[0].supplier_name == "ABC Supplies"


def test_worker_reprocessing_replaces_existing_chunks(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
        status="queued",
    )

    old_chunk = create_test_chunk(
        db_session,
        document.id,
    )
    old_chunk_content = old_chunk.content

    db_session.flush()

    new_chunk_content = "Newly processed document content"

    monkeypatch.setattr(
        tasks_module,
        "chunk_document",
        lambda document_data: [
            {
                "chunk_id": "chunk_000001",
                "content": new_chunk_content,
                "source": {
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "page_number": 1,
                },
                "metadata": {},
            }
        ],
    )

    result = tasks_module.process_uploaded_document.run(
        document.id
    )

    assert result["status"] == "processed"
    assert result["chunk_count"] == 1

    # Clear SQLAlchemy's cached objects before querying again.
    db_session.expire_all()

    chunks = db_session.scalars(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    ).all()

    assert len(chunks) == 1

    new_chunk = chunks[0]

    assert new_chunk.content == new_chunk_content
    assert new_chunk.content != old_chunk_content
    assert new_chunk.chunk_index == 0

def test_worker_failure_preserves_previous_invoice(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
        status="queued",
    )

    old_invoice = create_test_invoice(
        db_session,
        document.id,
    )

    old_invoice_id = old_invoice.id

    monkeypatch.setattr(
        tasks_module,
        "invoice_extraction_graph",
        FakeGraph(
            error=RuntimeError("Simulated extraction failure")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated extraction failure",
    ):
        tasks_module.process_uploaded_document.run(
            document.id
        )

    invoices = db_session.scalars(
        select(Invoice).where(
            Invoice.document_id == document.id
        )
    ).all()

    assert len(invoices) == 1
    assert invoices[0].id == old_invoice_id
    assert invoices[0].invoice_number == "INV-OLD-1001"

    db_session.refresh(document)

    assert document.status == "failed"
    assert any(
        "Simulated extraction failure" in error
        for error in document.errors
    )


def test_worker_skips_already_processing_document(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
        status="processing",
    )

    def unexpected_processing(file_path):
        pytest.fail(
            "Document processing should not run for an "
            "already processing document."
        )

    monkeypatch.setattr(
        tasks_module,
        "process_document",
        unexpected_processing,
    )

    result = tasks_module.process_uploaded_document.run(
        document.id
    )

    assert result["document_id"] == document.id
    assert result["status"] == "processing"
    assert result["skipped"] is True


def test_worker_transaction_rollback_preserves_previous_invoice(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
        status="queued",
    )

    old_invoice = create_test_invoice(
        db_session,
        document.id,
    )

    old_invoice_id = old_invoice.id

    original_add_processing_event = (
        tasks_module.add_processing_event
    )

    def failing_completion_event(
        db,
        document_id,
        stage,
        status,
        message=None,
        details=None,
    ):
        if stage == "pipeline" and status == "completed":
            raise RuntimeError(
                "Simulated completion transaction failure"
            )

        return original_add_processing_event(
            db=db,
            document_id=document_id,
            stage=stage,
            status=status,
            message=message,
            details=details,
        )

    monkeypatch.setattr(
        tasks_module,
        "add_processing_event",
        failing_completion_event,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated completion transaction failure",
    ):
        tasks_module.process_uploaded_document.run(
            document.id
        )

    invoices = db_session.scalars(
        select(Invoice).where(
            Invoice.document_id == document.id
        )
    ).all()

    assert len(invoices) == 1
    assert invoices[0].id == old_invoice_id
    assert invoices[0].invoice_number == "INV-OLD-1001"

    db_session.refresh(document)

    assert document.status == "failed"


# ---------------------------------------------------------
# Chunk persistence and transaction tests
# ---------------------------------------------------------


def test_worker_transaction_rollback_preserves_previous_chunks(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
        status="queued",
    )

    old_chunk = create_test_chunk(
        db_session,
        document.id,
    )

    old_chunk_id = old_chunk.id

    original_add_processing_event = (
        tasks_module.add_processing_event
    )

    def failing_completion_event(
        db,
        document_id,
        stage,
        status,
        message=None,
        details=None,
    ):
        if stage == "pipeline" and status == "completed":
            raise RuntimeError(
                "Simulated completion transaction failure"
            )

        return original_add_processing_event(
            db=db,
            document_id=document_id,
            stage=stage,
            status=status,
            message=message,
            details=details,
        )

    monkeypatch.setattr(
        tasks_module,
        "add_processing_event",
        failing_completion_event,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated completion transaction failure",
    ):
        tasks_module.process_uploaded_document.run(
            document.id
        )

    chunks = db_session.scalars(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    ).all()

    assert len(chunks) == 1
    assert chunks[0].id == old_chunk_id
    assert chunks[0].content == (
        "Previously stored document content"
    )

    db_session.refresh(document)

    assert document.status == "failed"


def test_worker_records_chunk_count_in_completion_event(
    db_session,
    worker_environment,
):
    document = create_test_document(
        db_session,
        worker_environment,
    )

    result = tasks_module.process_uploaded_document.run(
        document.id
    )

    assert result["chunk_count"] == 1

    db_session.refresh(document)

    completion_events = [
        event
        for event in document.events
        if event.stage == "pipeline"
        and event.status == "completed"
    ]

    assert len(completion_events) == 1
    assert completion_events[0].details["chunk_count"] == 1


# ---------------------------------------------------------
# Retry tests
# ---------------------------------------------------------


def test_worker_retries_connection_failure(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
    )

    retry_calls = []

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise RetryRequested("Retry scheduled")

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "retry",
        fake_retry,
    )

    def failing_process(file_path):
        raise ConnectionError("Temporary connection failure")

    monkeypatch.setattr(
        tasks_module,
        "process_document",
        failing_process,
    )

    with pytest.raises(RetryRequested):
        tasks_module.process_uploaded_document.run(
            document.id
        )

    db_session.refresh(document)

    assert document.status == "queued"
    assert len(retry_calls) == 1
    assert retry_calls[0]["countdown"] == 10

    assert any(
        event.status == "retrying"
        for event in document.events
    )


def test_worker_retries_ai_rate_limit(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
    )

    retry_calls = []

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise RetryRequested("Retry scheduled")

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "retry",
        fake_retry,
    )

    monkeypatch.setattr(
        tasks_module,
        "invoice_extraction_graph",
        FakeGraph(
            error=RateLimitError("Rate limit exceeded")
        ),
    )

    with pytest.raises(RetryRequested):
        tasks_module.process_uploaded_document.run(
            document.id
        )

    db_session.refresh(document)

    assert document.status == "queued"
    assert len(retry_calls) == 1
    assert retry_calls[0]["countdown"] == 10


def test_worker_does_not_retry_permanent_error(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
    )

    def unexpected_retry(**kwargs):
        pytest.fail("Permanent errors must not be retried.")

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "retry",
        unexpected_retry,
    )

    def fail_with_missing_file(file_path):
        raise FileNotFoundError("Invalid document file")

    monkeypatch.setattr(
        tasks_module,
        "process_document",
        fail_with_missing_file,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Invalid document file",
    ):
        tasks_module.process_uploaded_document.run(
            document.id
        )

    db_session.refresh(document)

    assert document.status == "failed"

    assert any(
        "Invalid document file" in error
        for error in document.errors
    )


def test_worker_marks_document_failed_after_retries_exhausted(
    db_session,
    worker_environment,
    monkeypatch,
):
    document = create_test_document(
        db_session,
        worker_environment,
    )

    retry_calls = []

    def unexpected_retry(**kwargs):
        retry_calls.append(kwargs)
        raise RetryRequested("Unexpected retry")

    task = tasks_module.process_uploaded_document

    monkeypatch.setattr(
        task,
        "retry",
        unexpected_retry,
    )

    def failing_process(file_path):
        raise ConnectionError(
            "Database temporarily unavailable"
        )

    monkeypatch.setattr(
        tasks_module,
        "process_document",
        failing_process,
    )

    # Simulate the final permitted attempt.
    task.push_request(retries=2)

    try:
        with pytest.raises(
            ConnectionError,
            match="Database temporarily unavailable",
        ):
            task.run(document.id)
    finally:
        task.pop_request()

    db_session.refresh(document)

    assert document.status == "failed"
    assert retry_calls == []

    assert any(
        "Database temporarily unavailable" in error
        for error in document.errors
    )


# ---------------------------------------------------------
# Stale document recovery tests
# ---------------------------------------------------------


def test_recovery_requeues_document_with_expired_heartbeat(
    db_session,
    worker_environment,
    monkeypatch,
):
    old_time = datetime.now(UTC) - timedelta(hours=2)

    document = create_processing_document(
        db_session,
        processing_started_at=old_time,
        processing_heartbeat_at=old_time,
    )

    dispatched_ids = []

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "delay",
        lambda document_id: dispatched_ids.append(document_id),
    )

    result = tasks_module.recover_stale_documents.run()

    db_session.refresh(document)

    assert result["recovered"] == 1
    assert result["dispatched"] == [document.id]
    assert result["dispatch_failed"] == []

    assert dispatched_ids == [document.id]

    assert document.status == "queued"
    assert document.processing_started_at is None
    assert document.processing_heartbeat_at is None
    assert document.processing_token is None

    assert any(
        event.stage == "recovery"
        and event.status == "recovered"
        for event in document.events
    )


def test_recovery_does_not_requeue_document_with_recent_heartbeat(
    db_session,
    worker_environment,
    monkeypatch,
):
    old_time = datetime.now(UTC) - timedelta(hours=2)
    recent_time = datetime.now(UTC)

    document = create_processing_document(
        db_session,
        processing_started_at=old_time,
        processing_heartbeat_at=recent_time,
    )

    dispatched_ids = []

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "delay",
        lambda document_id: dispatched_ids.append(document_id),
    )

    result = tasks_module.recover_stale_documents.run()

    db_session.refresh(document)

    assert result["recovered"] == 0
    assert result["dispatched"] == []
    assert result["dispatch_failed"] == []

    assert dispatched_ids == []
    assert document.status == "processing"
    assert document.processing_token == "test-processing-token"


def test_recovery_uses_processing_started_at_when_heartbeat_is_missing(
    db_session,
    worker_environment,
    monkeypatch,
):
    old_time = datetime.now(UTC) - timedelta(hours=2)

    document = create_processing_document(
        db_session,
        processing_started_at=old_time,
        processing_heartbeat_at=None,
    )

    dispatched_ids = []

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "delay",
        lambda document_id: dispatched_ids.append(document_id),
    )

    result = tasks_module.recover_stale_documents.run()

    db_session.refresh(document)

    assert result["recovered"] == 1
    assert result["dispatched"] == [document.id]
    assert result["dispatch_failed"] == []

    assert dispatched_ids == [document.id]
    assert document.status == "queued"
    assert document.processing_token is None


def test_recovery_marks_document_failed_when_dispatch_fails(
    db_session,
    worker_environment,
    monkeypatch,
):
    old_time = datetime.now(UTC) - timedelta(hours=2)

    document = create_processing_document(
        db_session,
        processing_started_at=old_time,
        processing_heartbeat_at=old_time,
    )

    def failing_dispatch(document_id):
        raise KombuError("Broker unavailable")

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "delay",
        failing_dispatch,
    )

    result = tasks_module.recover_stale_documents.run()

    db_session.refresh(document)

    assert result["recovered"] == 1
    assert result["dispatched"] == []
    assert result["dispatch_failed"] == [document.id]

    assert document.status == "failed"

    assert any(
        "Recovery dispatch failed: Broker unavailable" in error
        for error in document.errors
    )

    assert any(
        event.stage == "recovery"
        and event.status == "failed"
        for event in document.events
    )


def test_recovery_does_not_recover_document_with_changed_lease_token(
    db_session,
    worker_environment,
    monkeypatch,
):
    old_time = datetime.now(UTC) - timedelta(hours=2)

    document = create_processing_document(
        db_session,
        processing_started_at=old_time,
        processing_heartbeat_at=old_time,
        processing_token="original-token",
    )

    real_session_factory = tasks_module.SessionLocal

    class RacingSession:
        def __init__(self, session):
            self.session = session
            self.first_select = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.session.__exit__(
                exc_type,
                exc_value,
                traceback,
            )

        def __getattr__(self, name):
            return getattr(self.session, name)

        def execute(self, statement, *args, **kwargs):
            result = self.session.execute(
                statement,
                *args,
                **kwargs,
            )

            if self.first_select:
                self.first_select = False

                # Materialize the selected stale document before
                # simulating another worker acquiring a new lease.
                rows = result.all()

                self.session.execute(
                    update(Document)
                    .where(Document.id == document.id)
                    .values(
                        processing_token="new-worker-token",
                    )
                    .execution_options(synchronize_session=False)
                )

                class MaterializedResult:
                    def all(self):
                        return rows

                return MaterializedResult()

            return result

    def racing_session_factory():
        return RacingSession(real_session_factory())

    monkeypatch.setattr(
        tasks_module,
        "SessionLocal",
        racing_session_factory,
    )

    dispatched_ids = []

    monkeypatch.setattr(
        tasks_module.process_uploaded_document,
        "delay",
        lambda document_id: dispatched_ids.append(document_id),
    )

    result = tasks_module.recover_stale_documents.run()

    db_session.refresh(document)

    assert result["recovered"] == 0
    assert result["dispatched"] == []
    assert result["dispatch_failed"] == []

    assert dispatched_ids == []

    assert document.status == "processing"
    assert document.processing_token == "new-worker-token"


# ---------------------------------------------------------
# Processing heartbeat tests
# ---------------------------------------------------------


def test_heartbeat_renews_lease_when_worker_owns_document(
    db_session,
    worker_environment,
):
    now = datetime.now(UTC)

    document = create_processing_document(
        db_session,
        processing_started_at=now,
        processing_heartbeat_at=now,
    )

    heartbeat = tasks_module.ProcessingLeaseHeartbeat(
        document_id=document.id,
        processing_token=document.processing_token,
        interval_seconds=60,
    )

    assert heartbeat._renew_lease() is True

    db_session.refresh(document)

    assert document.processing_heartbeat_at is not None
    assert document.processing_heartbeat_at != now
    assert document.status == "processing"
    assert document.processing_token == "test-processing-token"


def test_heartbeat_does_not_renew_lease_after_token_changes(
    db_session,
    worker_environment,
):
    now = datetime.now(UTC)

    document = create_processing_document(
        db_session,
        processing_started_at=now,
        processing_heartbeat_at=now,
    )

    heartbeat = tasks_module.ProcessingLeaseHeartbeat(
        document_id=document.id,
        processing_token=document.processing_token,
        interval_seconds=60,
    )

    document.processing_token = "new-worker-token"
    db_session.flush()

    assert heartbeat._renew_lease() is False

    db_session.refresh(document)

    assert document.processing_token == "new-worker-token"
    assert document.processing_heartbeat_at.replace(tzinfo=UTC) == now


def test_heartbeat_marks_lease_lost_when_renewal_fails(
    monkeypatch,
):
    heartbeat = tasks_module.ProcessingLeaseHeartbeat(
        document_id=123,
        processing_token="test-token",
        interval_seconds=60,
    )

    class ControlledEvent:
        def __init__(self):
            self.results = iter([False])

        def wait(self, timeout=None):
            return next(self.results, True)

    monkeypatch.setattr(
        heartbeat,
        "_stop_event",
        ControlledEvent(),
    )

    monkeypatch.setattr(
        heartbeat,
        "_renew_lease",
        lambda: False,
    )

    heartbeat._run()

    assert heartbeat.lease_lost is True


def test_heartbeat_retries_after_temporary_database_error(
    monkeypatch,
    caplog,
):
    heartbeat = tasks_module.ProcessingLeaseHeartbeat(
        document_id=123,
        processing_token="test-token",
        interval_seconds=60,
    )

    class ControlledEvent:
        def __init__(self):
            self.results = iter([False, False, True])

        def wait(self, timeout=None):
            return next(self.results, True)

    attempts = {"count": 0}

    def renew_with_temporary_failure():
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise RuntimeError("Temporary database failure")

        return True

    monkeypatch.setattr(
        heartbeat,
        "_stop_event",
        ControlledEvent(),
    )

    monkeypatch.setattr(
        heartbeat,
        "_renew_lease",
        renew_with_temporary_failure,
    )

    heartbeat._run()

    assert attempts["count"] == 2
    assert heartbeat.lease_lost is False
    assert "Heartbeat renewal failed" in caplog.text


def test_heartbeat_start_and_stop_manage_thread():
    heartbeat = tasks_module.ProcessingLeaseHeartbeat(
        document_id=123,
        processing_token="test-token",
        interval_seconds=60,
    )

    heartbeat.start()

    assert heartbeat._thread.is_alive()

    heartbeat.stop()

    assert heartbeat._stop_event.is_set()
    assert not heartbeat._thread.is_alive()


def test_heartbeat_rejects_interval_below_one_second():
    with pytest.raises(ValueError, match="at least one second"):
        tasks_module.ProcessingLeaseHeartbeat(
            document_id=123,
            processing_token="test-token",
            interval_seconds=0,
        )
