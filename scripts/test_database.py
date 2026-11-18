import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from app.database.models import Document
from app.database.search import keyword_search, semantic_search
from app.database.session import SessionLocal, init_db


def main():
    init_db()

    with SessionLocal() as db:
        document = Document(
            filename="sample_invoice.pdf",
            file_type="pdf",
            status="processed",
            extracted_text=(
                "ABC Technologies invoice INV-001. "
                "Software license total 1380 USD."
            ),
            page_count=1,
            tables_data=[],
            metadata_json={"source": "test"},
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        print("Saved document:", document.id)

        keyword_results = keyword_search(
            db,
            "ABC Technologies",
        )

        print("\nKeyword search results:")
        for item in keyword_results:
            print(item.id, item.filename)

        # This test also exercises the embedding model.
        # It requires the document to have an embedding.
        print("\nSemantic search results:")
        for item in semantic_search(
            db,
            "software license invoice",
        ):
            print(
                item["document"].filename,
                item["similarity_distance"],
            )


if __name__ == "__main__":
    main()
