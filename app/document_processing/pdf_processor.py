
from pathlib import Path

import pymupdf

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)
from app.document_processing.ocr_engine import extract_image_text

OCR_MIN_CHARACTERS = 40


def process_pdf(file_path: str | Path) -> DocumentContent:
    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise DocumentProcessingError(
            "Invalid PDF file extension"
        )

    pages = []
    warnings = []
    all_text = []

    try:
        with pymupdf.open(path) as document:

            if document.is_encrypted:
                raise DocumentProcessingError(
                    "Encrypted PDF cannot be processed"
                )

            if len(document) == 0:
                raise DocumentProcessingError(
                    "PDF contains no pages"
                )

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()
                method = "digital_pdf"

                if len(text) < OCR_MIN_CHARACTERS:
                    try:
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2),
                            alpha=False,
                        )

                        image_bytes = pixmap.tobytes("png")
                        ocr_text = extract_image_text(image_bytes)

                        if ocr_text.strip():
                            text = ocr_text.strip()
                            method = "ocr"

                        else:
                            warnings.append(
                                f"Page {page_index + 1}: "
                                "No readable text found"
                            )

                    except (
                        RuntimeError,
                        ValueError,
                        OSError,
                    ) as exc:
                        warnings.append(
                            f"Page {page_index + 1}: "
                            f"OCR failed: {exc}"
                        )

                pages.append({
                    "page_number": page_index + 1,
                    "text": text,
                    "extraction_method": method,
                    "character_count": len(text),
                })

                all_text.append(text)

            metadata = {
                "page_count": len(document),
                "title": document.metadata.get("title"),
                "author": document.metadata.get("author"),
            }

        return DocumentContent(
            filename=path.name,
            file_type="pdf",
            extracted_text="\n\n".join(all_text),
            pages=pages,
            metadata=metadata,
            warnings=warnings,
        )

    except DocumentProcessingError:
        raise

    except Exception as exc:  
        raise DocumentProcessingError(
            f"Failed to process PDF: {exc}"
        ) from exc
