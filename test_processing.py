
from pathlib import Path

from app.document_processing.dispatcher import process_document


folder = Path("sample_documents")

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

    except Exception as exc:
        print("Processing failed:", exc)
