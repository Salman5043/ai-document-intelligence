
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.database.embeddings import generate_embedding
from app.database.models import Document, DocumentChunk


def keyword_search(
    db: Session,
    query: str,
    limit: int = 10,
):
    """Search document filenames and extracted text."""
    if not query.strip():
        return []

    pattern = f"%{query.strip()}%"

    statement = (
        select(Document)
        .where(
            or_(
                Document.filename.ilike(pattern),
                Document.extracted_text.ilike(pattern),
            )
        )
        .order_by(Document.created_at.desc())
        .limit(limit)
    )

    return list(db.scalars(statement).all())


def semantic_search(
    db: Session,
    query: str,
    limit: int = 10,
):
    """Search using document-level embeddings."""
    if not query.strip():
        return []

    query_embedding = generate_embedding(query)

    distance = Document.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(Document, distance)
        .where(Document.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )

    results = db.execute(statement).all()

    return [
        {
            "document": document,
            "similarity_distance": float(distance_value),
        }
        for document, distance_value in results
    ]


def semantic_chunk_search(
    db: Session,
    query: str,
    limit: int = 10,
):
    """
    Search document chunks using vector similarity.

    Returns matching chunks together with their parent documents
    and source references such as page numbers or spreadsheet rows.
    """
    if not query or not query.strip():
        return []

    if limit < 1:
        raise ValueError("limit must be greater than zero.")

    query_embedding = generate_embedding(query)

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    ).label("distance")

    statement = (
        select(DocumentChunk, Document, distance)
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )

    results = db.execute(statement).all()

    return [
        {
            "chunk": chunk,
            "document": document,
            "similarity_distance": float(distance_value),
            "source": chunk.source or {},
            "metadata": chunk.metadata_json or {},
        }
        for chunk, document, distance_value in results
    ]
