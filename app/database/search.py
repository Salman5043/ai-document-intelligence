
import math

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.database.embeddings import generate_embedding
from app.database.models import Document, DocumentChunk


def _normalize_query(query: str | None) -> str | None:
    if query is None:
        return None

    if not isinstance(query, str):
        raise TypeError("query must be a string or None")

    query = query.strip()
    return query or None


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be greater than zero")


def _validate_document_id(document_id: int | None) -> None:
    """Validate an optional document ID."""

    if document_id is None:
        return

    if (
        isinstance(document_id, bool)
        or not isinstance(document_id, int)
        or document_id < 1
    ):
        raise ValueError("document_id must be a positive integer.")


def _validate_max_distance(max_distance: float | None) -> None:
    """Validate an optional cosine-distance threshold."""

    if max_distance is None:
        return

    if (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, (int, float))
        or not math.isfinite(max_distance)
        or not 0 <= max_distance <= 2
    ):
        raise ValueError(
            "max_distance must be a finite number between 0 and 2."
        )


def _safe_metadata(value) -> dict:
    """Return a dictionary for JSON metadata fields."""

    return value if isinstance(value, dict) else {}


def keyword_search(
    db: Session,
    query: str,
    limit: int = 10,
):
    """
    Search document filenames and extracted text using keywords.

    Results are ordered by creation date, newest first.
    """

    normalized_query = _normalize_query(query)
    _validate_limit(limit)

    if normalized_query is None:
        return []

    pattern = f"%{normalized_query}%"

    statement = (
        select(Document)
        .where(
            or_(
                Document.filename.ilike(pattern),
                Document.extracted_text.ilike(pattern),
            )
        )
        .order_by(
            Document.created_at.desc(),
            Document.id.desc(),
        )
        .limit(limit)
    )

    return list(db.scalars(statement).all())


def semantic_search(
    db: Session,
    query: str,
    limit: int = 10,
):
    """
    Search complete documents using vector similarity.

    Returns document ORM objects, cosine distance,
    and a similarity indicator.
    """

    normalized_query = _normalize_query(query)
    _validate_limit(limit)

    if normalized_query is None:
        return []

    query_embedding = generate_embedding(normalized_query)

    distance = Document.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(
            Document,
            distance,
        )
        .where(
            Document.embedding.is_not(None)
        )
        .order_by(
            distance,
            Document.id,
        )
        .limit(limit)
    )

    results = db.execute(statement).all()

    return [
        {
            "document": document,
            "similarity_distance": float(distance_value),
            "similarity_score": 1.0 - float(distance_value),
        }
        for document, distance_value in results
    ]


def semantic_chunk_search(
    db: Session,
    query: str,
    limit: int = 10,
    document_id: int | None = None,
    max_distance: float | None = None,
):
    """
    Search document chunks using vector similarity.

    Args:
        db: SQLAlchemy database session.
        query: User's search query.
        limit: Maximum number of results.
        document_id: Optional document ID to restrict retrieval.
        max_distance: Optional maximum cosine distance. Results
            above this threshold are excluded. Defaults to None,
            which disables distance filtering.

    Returns:
        A list containing chunk and document ORM objects,
        flattened document/chunk fields, similarity information,
        source references, and metadata.
    """

    normalized_query = _normalize_query(query)
    _validate_limit(limit)
    _validate_document_id(document_id)
    _validate_max_distance(max_distance)

    if normalized_query is None:
        return []

    query_embedding = generate_embedding(normalized_query)

    distance_expression = DocumentChunk.embedding.cosine_distance(
        query_embedding
    )

    distance = distance_expression.label("distance")

    statement = (
        select(
            DocumentChunk,
            Document,
            distance,
        )
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .where(
            DocumentChunk.embedding.is_not(None)
        )
    )

    # Restrict retrieval to one document when requested.
    if document_id is not None:
        statement = statement.where(
            DocumentChunk.document_id == document_id
        )

    # Exclude chunks whose cosine distance exceeds the threshold.
    if max_distance is not None:
        statement = statement.where(
            distance_expression <= max_distance
        )

    statement = (
        statement
        .order_by(
            distance_expression,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
        )
        .limit(limit)
    )

    results = db.execute(statement).all()

    search_results = []

    for chunk, document, distance_value in results:
        distance_value = float(distance_value)

        source = _safe_metadata(chunk.source)
        metadata = _safe_metadata(chunk.metadata_json)

        search_results.append(
            {
                # ORM objects for existing consumers.
                "chunk": chunk,
                "document": document,

                # Similarity information.
                "similarity_distance": distance_value,
                "similarity_score": 1.0 - distance_value,

                # Flattened fields for API/search consumers.
                "chunk_id": chunk.id,
                "document_id": document.id,
                "filename": document.filename,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "distance": distance_value,

                # Original source references.
                "source": source,
                "metadata": metadata,
            }
        )

    return search_results
