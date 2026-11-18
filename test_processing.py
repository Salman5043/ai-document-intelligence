
from pathlib import Path

from app.document_processing.dispatcher import process_document
from app.document_processing.models import DocumentProcessingError

folder = Path("sample_documents")

if not folder.exists():
    raise FileNotFoundError(f"Sample documents folder not found: {folder}")

for file_path in folder.iterdir():
    if not file_path.is_file():
        continue

    print("\n" + "=" * 60)
    print("Processing:", file_path.name)

    try:
        result = process_document(file_path)

        print("File type:", result.file_type)
        print("Extracted text length:", len(result.extracted_text))
        print("Pages:", len(result.pages))
        print("Tables:", len(result.tables))
        print("Warnings:", result.warnings)

        print("\nExtracted text preview:")
        print(result.extracted_text[:1000])

    except DocumentProcessingError as exc:
        print("Document processing failed:", exc)
