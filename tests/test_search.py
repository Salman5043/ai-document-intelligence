
from unittest.mock import patch

import pytest
from sqlalchemy import literal
from pgvector.sqlalchemy import Vector

from app.database.models import Document, DocumentChunk
from app.database.search import (
    keyword_search,
    semantic_chunk_search,
    semantic_search,
    _validate_max_distance,
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
    This mock allows testing query construction and result handling.
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


def test_keyword_search_strips_query_whitespace(
    db_session,
    search_document,
):
    results = keyword_search(
        db_session,
        "  INV-1001  ",
    )

    assert len(results) == 1
    assert results[0].id == search_document.id


@pytest.mark.parametrize(
    "query",
    ["", "   ", None],
)
def test_keyword_search_returns_empty_for_blank_query(
    db_session,
    query,
):
    assert keyword_search(db_session, query) == []


def test_keyword_search_respects_limit(
    db_session,
    search_document,
):
    results = keyword_search(
        db_session,
        "invoice",
        limit=1,
    )

    assert len(results) <= 1


def test_keyword_search_rejects_invalid_limit(
    db_session,
):
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        keyword_search(
            db_session,
            "invoice",
            limit=0,
        )


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

    result = results[0]

    assert result["document"].id == search_document.id
    assert result["similarity_distance"] == pytest.approx(0.1)
    assert result["similarity_score"] == pytest.approx(0.9)

    mock_embedding.assert_called_once_with(
        "Find invoice INV-1001"
    )


@patch("app.database.search.generate_embedding")
def test_semantic_search_strips_query_whitespace(
    mock_embedding,
    mock_cosine_distance,
    db_session,
    search_document,
):
    mock_embedding.return_value = [0.1] * 384

    semantic_search(
        db_session,
        "  Find invoice INV-1001  ",
    )

    mock_embedding.assert_called_once_with(
        "Find invoice INV-1001"
    )


@pytest.mark.parametrize(
    "query",
    ["", "   ", None],
)
def test_semantic_search_returns_empty_for_blank_query(
    db_session,
    query,
):
    assert semantic_search(db_session, query) == []


def test_semantic_search_rejects_invalid_limit(
    db_session,
):
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        semantic_search(
            db_session,
            "Find invoice",
            limit=0,
        )


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
        assert result["similarity_score"] == pytest.approx(0.9)


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

    assert first_result["metadata"]["extraction_method"] == "digital"


@patch("app.database.search.generate_embedding")
def test_semantic_chunk_search_returns_document_information(
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

    assert first_result["document_id"] == search_document.id
    assert first_result["filename"] == "invoice_test.pdf"
    assert first_result["chunk_index"] == 0
    assert first_result["content"] == "Invoice number INV-1001"


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


@pytest.mark.parametrize(
    "query",
    ["", "   ", None],
)
def test_semantic_chunk_search_returns_empty_for_blank_query(
    db_session,
    query,
):
    assert semantic_chunk_search(db_session, query) == []


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


def test_semantic_chunk_search_rejects_negative_limit(
    db_session,
):
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        semantic_chunk_search(
            db_session,
            "Find invoice information",
            limit=-1,
        )


def test_semantic_chunk_search_rejects_boolean_limit(
    db_session,
):
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        semantic_chunk_search(
            db_session,
            "Find invoice information",
            limit=True,
        )


# ---------------------------------------------------------
# Invalid input tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "search_function",
    [
        keyword_search,
        semantic_search,
        semantic_chunk_search,
    ],
)
def test_search_rejects_non_string_query(
    db_session,
    search_function,
):
    with pytest.raises(
        TypeError,
        match="query must be a string or None",
    ):
        search_function(
            db_session,
            123,
        )

@pytest.mark.parametrize(
    "value",
    [0, 0.3, 1, 1.5, 2],
)
def test_validate_max_distance_accepts_valid_values(value):
    _validate_max_distance(value)


def test_validate_max_distance_accepts_none():
    _validate_max_distance(None)


@pytest.mark.parametrize(
    "value",
    [-0.1, 2.1, float("inf"), float("-inf"), float("nan"), True, "0.5"],
)
def test_validate_max_distance_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _validate_max_distance(value)