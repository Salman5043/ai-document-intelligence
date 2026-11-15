
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.database.embeddings import generate_embedding
from app.database.models import Document


def keyword_search(
    db: Session,
    query: str,
    limit: int = 10,
):
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
