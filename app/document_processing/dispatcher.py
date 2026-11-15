
from pathlib import Path

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)
from app.document_processing.pdf_processor import process_pdf
from app.document_processing.image_processor import process_image
from app.document_processing.excel_processor import process_excel
from app.document_processing.csv_processor import process_csv


def process_document(
    file_path: str | Path
) -> DocumentContent:
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise DocumentProcessingError(
            f"Invalid document path: {path}"
        )

    extension = path.suffix.lower()

    processors = {
        ".pdf": process_pdf,
        ".jpg": process_image,
        ".jpeg": process_image,
        ".png": process_image,
        ".tiff": process_image,
        ".tif": process_image,
        ".xlsx": process_excel,
        ".xls": process_excel,
        ".csv": process_csv,
    }

    processor = processors.get(extension)

    if processor is None:
        raise DocumentProcessingError(
            f"Unsupported document format: {extension}"
        )

    return processor(path)
