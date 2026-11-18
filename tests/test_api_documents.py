
from decimal import Decimal
from pathlib import Path

import pytest
from kombu.exceptions import KombuError
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import documents as documents_module
from app.database.models import Document, Invoice, DocumentChunk

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def create_document(
    db_session,
    *,
    filename="sample.pdf",
    file_type="pdf",
    status="uploaded",
    storage_path=None,
):
    document = Document(
        filename=filename,
        file_type=file_type,
        status=status,
        storage_path=storage_path,
        extracted_text="Sample extracted invoice text",
        page_count=1,
        errors=[],
        warnings=[],
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    return document


def create_invoice(
    db_session,
    document_id,
    **overrides,
):
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
    db_session.commit()
    db_session.refresh(invoice)

    return invoice


# -------------------------------------------------------------------
# Upload tests
# -------------------------------------------------------------------

def test_upload_document_success(
    client,
    db_session,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_module,
        "STORAGE_DIR",
        tmp_path,
    )

    queued_tasks = []

    monkeypatch.setattr(
        documents_module.process_uploaded_document,
        "delay",
        lambda document_id: queued_tasks.append(document_id),
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"sample invoice content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] is not None
    assert data["filename"] == "sample.pdf"
    assert data["status"] == "uploaded"
    assert data["message"] == (
        "Document uploaded and queued for processing."
    )

    document = db_session.get(Document, data["document_id"])

    assert document is not None
    assert document.filename == "sample.pdf"
    assert document.file_type == "pdf"
    assert document.status == "uploaded"
    assert Path(document.storage_path).exists()

    assert queued_tasks == [document.id]


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"unsupported content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type."


def test_upload_rejects_empty_filename(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "",
                b"sample content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422


def test_upload_rejects_oversized_file(
    client,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_module,
        "STORAGE_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        documents_module,
        "MAX_UPLOAD_SIZE",
        5,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large.pdf",
                b"this content exceeds five bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "File exceeds the maximum upload size."
    )

    assert list(tmp_path.iterdir()) == []


def test_upload_removes_partial_file_after_write_failure(
    client_no_raise,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_module,
        "STORAGE_DIR",
        tmp_path,
    )

    original_open = Path.open

    def failing_open(path, mode="r", *args, **kwargs):
        if path.parent == tmp_path and mode == "wb":
            raise OSError("Simulated storage failure")

        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    response = client_no_raise.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"sample content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500
    assert list(tmp_path.iterdir()) == []


def test_upload_database_failure_removes_stored_file(
    client,
    db_session,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_module,
        "STORAGE_DIR",
        tmp_path,
    )

    def failing_commit():
        raise SQLAlchemyError("Simulated database failure")

    monkeypatch.setattr(db_session, "commit", failing_commit)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"sample content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Could not save document information."
    )

    assert list(tmp_path.iterdir()) == []


def test_upload_queue_failure_marks_document_failed(
    client,
    db_session,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_module,
        "STORAGE_DIR",
        tmp_path,
    )

    def failing_delay(document_id):
        raise KombuError("Simulated broker failure")

    monkeypatch.setattr(
        documents_module.process_uploaded_document,
        "delay",
        failing_delay,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"sample invoice content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Could not queue document processing."
    )

    document = (
        db_session.query(Document)
        .filter_by(filename="sample.pdf")
        .first()
    )

    assert document is not None
    assert document.status == "failed"
    assert document.errors == [
        "Could not queue document processing."
    ]

    # The uploaded file is retained for a possible retry.
    assert Path(document.storage_path).exists()


# -------------------------------------------------------------------
# Document status tests
# -------------------------------------------------------------------

def test_get_document_status_success(
    client,
    db_session,
):
    document = create_document(
        db_session,
        status="completed",
    )

    response = client.get(
        f"/documents/{document.id}/status"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == document.id
    assert data["filename"] == "sample.pdf"
    assert data["status"] == "completed"
    assert data["errors"] == []
    assert data["warnings"] == []


def test_get_document_status_not_found(client):
    response = client.get("/documents/9999/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


# -------------------------------------------------------------------
# Document result tests
# -------------------------------------------------------------------

def test_get_document_result_success(
    client,
    db_session,
):
    document = create_document(
        db_session,
        status="completed",
    )

    invoice = create_invoice(
        db_session,
        document.id,
    )

    response = client.get(
        f"/documents/{document.id}/result"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document"]["id"] == document.id
    assert data["document"]["filename"] == "sample.pdf"
    assert data["document"]["file_type"] == "pdf"
    assert data["document"]["status"] == "completed"
    assert data["document"]["extracted_text"] == (
        "Sample extracted invoice text"
    )

    assert len(data["invoices"]) == 1

    result_invoice = data["invoices"][0]

    assert result_invoice["id"] == invoice.id
    assert result_invoice["invoice_number"] == "INV-001"
    assert result_invoice["supplier_name"] == "ABC Supplies"
    assert result_invoice["total"] == 115.00
    assert result_invoice["currency"] == "BDT"
    assert result_invoice["valid"] is True
    assert result_invoice["review_required"] is False


def test_get_document_result_without_invoices(
    client,
    db_session,
):
    document = create_document(db_session)

    response = client.get(
        f"/documents/{document.id}/result"
    )

    assert response.status_code == 200
    assert response.json()["invoices"] == []


def test_get_document_result_not_found(client):
    response = client.get("/documents/9999/result")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


# -------------------------------------------------------------------
# Document search tests
# -------------------------------------------------------------------

def test_keyword_search_success(
    client,
    db_session,
    monkeypatch,
):
    document = create_document(db_session)

    monkeypatch.setattr(
        documents_module,
        "keyword_search",
        lambda db, query, limit: [document],
    )

    response = client.get(
        "/documents/search",
        params={
            "q": "invoice",
            "mode": "keyword",
            "limit": 10,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["document_id"] == document.id
    assert data[0]["filename"] == "sample.pdf"
    assert data[0]["status"] == "uploaded"


def test_semantic_search_success(
    client,
    db_session,
    monkeypatch,
):
    document = create_document(db_session)

    monkeypatch.setattr(
        documents_module,
        "semantic_search",
        lambda db, query, limit: [
            {
                "document": document,
                "similarity_distance": 0.15,
            }
        ],
    )

    response = client.get(
        "/documents/search",
        params={
            "q": "office supplies",
            "mode": "semantic",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["document_id"] == document.id
    assert data[0]["filename"] == "sample.pdf"
    assert data[0]["distance"] == pytest.approx(0.15)


def test_search_rejects_empty_query(client):
    response = client.get(
        "/documents/search",
        params={"q": ""},
    )

    assert response.status_code == 422


def test_search_rejects_invalid_mode(client):
    response = client.get(
        "/documents/search",
        params={
            "q": "invoice",
            "mode": "invalid",
        },
    )

    assert response.status_code == 422


def test_search_rejects_limit_above_maximum(client):
    response = client.get(
        "/documents/search",
        params={
            "q": "invoice",
            "limit": 101,
        },
    )

    assert response.status_code == 422


# -------------------------------------------------------------------
# Document reprocessing tests
# -------------------------------------------------------------------

def test_reprocess_document_success(
    client,
    db_session,
    tmp_path,
    monkeypatch,
):
    original_file = tmp_path / "original.pdf"
    original_file.write_bytes(b"original document content")

    document = create_document(
        db_session,
        status="failed",
        storage_path=str(original_file),
    )

    original_invoice = create_invoice(
        db_session,
        document.id,
    )

    queued_tasks = []

    monkeypatch.setattr(
        documents_module.process_uploaded_document,
        "delay",
        lambda document_id: queued_tasks.append(document_id),
    )

    response = client.post(
        f"/documents/{document.id}/reprocess"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == document.id
    assert data["status"] == "queued"
    assert data["message"] == (
        "Document has been queued for reprocessing."
    )

    db_session.refresh(document)

    assert document.status == "queued"
    assert document.errors == []
    assert document.warnings == []
    assert queued_tasks == [document.id]

    # Existing invoice results must remain available
    # until the worker successfully processes replacements.
    existing_invoices = (
        db_session.query(Invoice)
        .filter_by(document_id=document.id)
        .all()
    )

    assert len(existing_invoices) == 1
    assert existing_invoices[0].id == original_invoice.id


def test_reprocess_document_not_found(client):
    response = client.post("/documents/9999/reprocess")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_reprocess_document_original_file_missing(
    client,
    db_session,
    tmp_path,
):
    missing_file = tmp_path / "missing.pdf"

    document = create_document(
        db_session,
        storage_path=str(missing_file),
    )

    response = client.post(
        f"/documents/{document.id}/reprocess"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Original uploaded file is unavailable."
    )


def test_reprocess_document_already_processing(
    client,
    db_session,
    tmp_path,
):
    original_file = tmp_path / "original.pdf"
    original_file.write_bytes(b"document content")

    document = create_document(
        db_session,
        status="processing",
        storage_path=str(original_file),
    )

    response = client.post(
        f"/documents/{document.id}/reprocess"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Document is already queued or being processed."
    )


def test_reprocess_document_already_queued(
    client,
    db_session,
    tmp_path,
):
    original_file = tmp_path / "original.pdf"
    original_file.write_bytes(b"document content")

    document = create_document(
        db_session,
        status="queued",
        storage_path=str(original_file),
    )

    response = client.post(
        f"/documents/{document.id}/reprocess"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Document is already queued or being processed."
    )


def test_reprocess_queue_failure(
    client,
    db_session,
    tmp_path,
    monkeypatch,
):
    original_file = tmp_path / "original.pdf"
    original_file.write_bytes(b"document content")

    document = create_document(
        db_session,
        status="failed",
        storage_path=str(original_file),
    )

    original_invoice = create_invoice(
        db_session,
        document.id,
    )

    def failing_delay(document_id):
        raise KombuError("Simulated broker failure")

    monkeypatch.setattr(
        documents_module.process_uploaded_document,
        "delay",
        failing_delay,
    )

    response = client.post(
        f"/documents/{document.id}/reprocess"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Could not queue document for reprocessing."
    )

    db_session.refresh(document)

    # Previous status is restored.
    assert document.status == "failed"

    # Queue failure is recorded.
    assert document.errors == [
        "Could not queue document for reprocessing."
    ]

    # Existing invoice must not be deleted.
    existing_invoices = (
        db_session.query(Invoice)
        .filter_by(document_id=document.id)
        .all()
    )

    assert len(existing_invoices) == 1
    assert existing_invoices[0].id == original_invoice.id


# -------------------------------------------------------------------
# Chunk-level document search tests
# -------------------------------------------------------------------

def test_chunk_search_success(
    client,
    db_session,
    monkeypatch,
):
    document = create_document(
        db_session,
        status="processed",
    )

    chunk = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        content="Invoice total amount is BDT 115.00",
        source={
            "filename": "sample.pdf",
            "page_number": 2,
        },
        metadata_json={
            "extraction_method": "digital",
        },
        embedding=[0.1] * 384,
    )

    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(chunk)

    monkeypatch.setattr(
        documents_module,
        "semantic_chunk_search",
        lambda db, query, limit: [
            {
                "chunk": chunk,
                "document": document,
                "similarity_distance": 0.12,
                "similarity_score": 0.88,
                "source": chunk.source,
                "metadata": chunk.metadata_json,
            }
        ],
    )

    response = client.get(
        "/documents/search",
        params={
            "q": "What is the invoice total?",
            "mode": "chunks",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1

    result = data[0]

    assert result["chunk_id"] == chunk.id
    assert result["document_id"] == document.id
    assert result["filename"] == "sample.pdf"
    assert result["status"] == "processed"
    assert result["chunk_index"] == 0

    assert result["content"] == (
        "Invoice total amount is BDT 115.00"
    )

    assert result["distance"] == pytest.approx(0.12)
    assert result["similarity_score"] == pytest.approx(0.88)

    assert result["source"]["page_number"] == 2
    assert result["metadata"]["extraction_method"] == "digital"


def test_chunk_search_returns_empty_results(
    client,
    db_session,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_module,
        "semantic_chunk_search",
        lambda db, query, limit: [],
    )

    response = client.get(
        "/documents/search",
        params={
            "q": "nonexistent information",
            "mode": "chunks",
        },
    )

    assert response.status_code == 200
    assert response.json() == []


def test_chunk_search_respects_limit(
    client,
    monkeypatch,
):
    captured = {}

    def fake_chunk_search(db, query, limit):
        captured["query"] = query
        captured["limit"] = limit
        return []

    monkeypatch.setattr(
        documents_module,
        "semantic_chunk_search",
        fake_chunk_search,
    )

    response = client.get(
        "/documents/search",
        params={
            "q": "invoice",
            "mode": "chunks",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    assert captured["query"] == "invoice"
    assert captured["limit"] == 5


def test_search_rejects_invalid_chunk_search_limit(
    client,
):
    response = client.get(
        "/documents/search",
        params={
            "q": "invoice",
            "mode": "chunks",
            "limit": 101,
        },
    )

    assert response.status_code == 422
