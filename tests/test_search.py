
from unittest.mock import patch

import pytest
from sqlalchemy import literal
from pgvector.sqlalchemy import Vector

from app.database.models import Document, DocumentChunk
from app.database.search import (
    keyword_search,
    semantic_chunk_search,
    semantic_search,
)


@pytest.fixture
def search_document(db_session):
    """Create a document for search tests."""
    document = Document(
        filename="invoice_test.pdf",
        file_type="pdf",
        status="processed",
        extracted_text="Invoice number INV-1001. Total amount 5000.",
        page_count=2,
        tables_data=[],
        metadata_json={},
        embedding=[0.1] * 384,
    )

    db_session.add(document)
    db_session.flush()

    return document


@pytest.fixture
def search_chunks(db_session, search_document):
    """Create document chunks with embeddings."""
    chunks = [
        DocumentChunk(
            document_id=search_document.id,
            chunk_index=0,
            content="Invoice number INV-1001",
            source={
                "filename": "invoice_test.pdf",
                "page_number": 1,
            },
            metadata_json={
                "extraction_method": "digital",
            },
            embedding=[0.1] * 384,
        ),
        DocumentChunk(
            document_id=search_document.id,
            chunk_index=1,
            content="Total amount: 5000",
            source={
                "filename": "invoice_test.pdf",
                "page_number": 2,
            },
            metadata_json={
                "extraction_method": "digital",
            },
            embedding=[0.2] * 384,
        ),
        DocumentChunk(
            document_id=search_document.id,
            chunk_index=2,
            content="Unembedded chunk",
            source={
                "filename": "invoice_test.pdf",
                "page_number": 2,
            },
            metadata_json={},
            embedding=None,
        ),
    ]

    db_session.add_all(chunks)
    db_session.flush()

    return chunks


@pytest.fixture
def mock_cosine_distance(monkeypatch):
    """
    Replace pgvector cosine distance with a SQL literal.

    SQLite cannot execute PostgreSQL's vector distance operator.
    This mock allows testing query construction and result handling
    without performing actual vector calculations.
    """

    monkeypatch.setattr(
        Vector.comparator_factory,
        "cosine_distance",
        lambda self, other: literal(0.1),
    )


# ---------------------------------------------------------
# Keyword search tests
# ---------------------------------------------------------

def test_keyword_search_returns_matching_document(
    db_session,
    search_document,
):
    results = keyword_search(
        db_session,
        "INV-1001",
    )

    assert len(results) == 1
    assert results[0].id == search_document.id


def test_keyword_search_returns_empty_for_blank_query(
    db_session,
):
    assert keyword_search(db_session, "") == []
    assert keyword_search(db_session, "   ") == []


# ---------------------------------------------------------
# Document-level semantic search tests
# ---------------------------------------------------------

@patch("app.database.search.generate_embedding")
def test_semantic_search_returns_document(
    mock_embedding,
    mock_cosine_distance,
    db_session,
    search_document,
):
    mock_embedding.return_value = [0.1] * 384

    results = semantic_search(
        db_session,
        "Find invoice INV-1001",
    )

    assert len(results) == 1
    assert results[0]["document"].id == search_document.id
    assert results[0]["similarity_distance"] == pytest.approx(0.1)


def test_semantic_search_returns_empty_for_blank_query(
    db_session,
):
    assert semantic_search(db_session, "") == []
    assert semantic_search(db_session, "   ") == []


# ---------------------------------------------------------
# Chunk-level semantic search tests
# ---------------------------------------------------------

@patch("app.database.search.generate_embedding")
def test_semantic_chunk_search_returns_matching_chunks(
    mock_embedding,
    mock_cosine_distance,
    db_session,
    search_document,
    search_chunks,
):
    mock_embedding.return_value = [0.1] * 384

    results = semantic_chunk_search(
        db_session,
        "What is the invoice number?",
    )

    assert len(results) == 2

    for result in results:
        assert result["document"].id == search_document.id
        assert isinstance(result["chunk"], DocumentChunk)
        assert result["similarity_distance"] == pytest.approx(0.1)


@patch("app.database.search.generate_embedding")
def test_semantic_chunk_search_returns_source_references(
    mock_embedding,
    mock_cosine_distance,
    db_session,
    search_document,
    search_chunks,
):
    mock_embedding.return_value = [0.1] * 384

    results = semantic_chunk_search(
        db_session,
        "Find invoice information",
    )

    assert results

    first_result = results[0]

    assert first_result["source"]["filename"] == "invoice_test.pdf"
    assert "page_number" in first_result["source"]
    assert "metadata" in first_result


@patch("app.database.search.generate_embedding")
def test_semantic_chunk_search_excludes_chunks_without_embeddings(
    mock_embedding,
    mock_cosine_distance,
    db_session,
    search_document,
    search_chunks,
):
    mock_embedding.return_value = [0.1] * 384

    results = semantic_chunk_search(
        db_session,
        "Find invoice information",
    )

    assert len(results) == 2

    assert all(
        result["chunk"].embedding is not None
        for result in results
    )


@patch("app.database.search.generate_embedding")
def test_semantic_chunk_search_respects_limit(
    mock_embedding,
    mock_cosine_distance,
    db_session,
    search_document,
    search_chunks,
):
    mock_embedding.return_value = [0.1] * 384

    results = semantic_chunk_search(
        db_session,
        "Find invoice information",
        limit=1,
    )

    assert len(results) == 1


def test_semantic_chunk_search_returns_empty_for_blank_query(
    db_session,
):
    assert semantic_chunk_search(db_session, "") == []
    assert semantic_chunk_search(db_session, "   ") == []


def test_semantic_chunk_search_rejects_invalid_limit(
    db_session,
):
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        semantic_chunk_search(
            db_session,
            "Find invoice information",
            limit=0,
        )
