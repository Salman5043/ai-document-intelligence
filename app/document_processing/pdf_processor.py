
from pathlib import Path
from typing import Any

import pymupdf

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)
from app.document_processing.ocr_engine import (
    extract_image_data,
    extract_image_text,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OCR_MIN_CHARACTERS = 40
OCR_DPI = 150
MAX_RENDER_DIMENSION = 2500

# Keep references to the original functions.
# These allow compatibility with existing tests that monkeypatch
# extract_image_text instead of extract_image_data.
_DEFAULT_EXTRACT_IMAGE_DATA = extract_image_data
_DEFAULT_EXTRACT_IMAGE_TEXT = extract_image_text


# ---------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------

def _extract_ocr_data(image_bytes: bytes) -> dict[str, Any]:
    """
    Extract structured OCR information from image bytes.

    Supports both:
        1. New structured extract_image_data() implementation.
        2. Legacy extract_image_text() implementation.

    The legacy interface is retained for backward compatibility
    with existing tests and other application modules.
    """

    # Support tests or existing code that monkeypatches the
    # legacy extract_image_text function.
    if extract_image_text is not _DEFAULT_EXTRACT_IMAGE_TEXT:
        text = extract_image_text(image_bytes)

        return {
            "text": text or "",
            "results": [],
            "average_confidence": 0.0,
            "total_detections": 0,
            "low_confidence_count": 0,
            "extraction_status": (
                "success" if text and text.strip()
                else "no_text_detected"
            ),
        }

    # Use the structured OCR implementation by default.
    return extract_image_data(image_bytes)


def _get_render_scale(page: pymupdf.Page) -> float:
    """
    Calculate a rendering scale based on the configured DPI.

    Prevents unnecessarily large rendered images for oversized
    PDF pages.
    """

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    if page_width <= 0 or page_height <= 0:
        raise ValueError("Invalid PDF page dimensions")

    dpi_scale = OCR_DPI / 72.0

    max_dimension = max(page_width, page_height)

    dimension_scale = (
        MAX_RENDER_DIMENSION / max_dimension
    )

    return min(dpi_scale, dimension_scale)


def _extract_page_ocr(page: pymupdf.Page) -> dict[str, Any]:
    """
    Render one PDF page and extract structured OCR information.

    Only one page is rendered at a time to avoid unnecessarily
    high memory consumption when processing large PDFs.
    """

    scale = _get_render_scale(page)

    matrix = pymupdf.Matrix(scale, scale)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

    return _extract_ocr_data(image_bytes)


# ---------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------

def process_pdf(file_path: str | Path) -> DocumentContent:
    """
    Process a PDF document.

    Features:
        - Digital PDF text extraction.
        - OCR fallback for scanned or image-based pages.
        - Mixed digital and scanned PDF support.
        - Structured OCR confidence and bounding-box metadata.
        - Page-level extraction warnings.
        - PDF metadata extraction.
        - Encrypted and corrupted PDF handling.
        - Sequential page processing for large documents.

    Args:
        file_path: Path to the PDF document.

    Returns:
        DocumentContent containing extracted text, page data,
        metadata, and warnings.

    Raises:
        DocumentProcessingError: If the PDF cannot be processed.
    """

    path = Path(file_path)

    # -------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise DocumentProcessingError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise DocumentProcessingError(
            f"Invalid PDF file extension: {path.suffix}"
        )

    if path.stat().st_size == 0:
        raise DocumentProcessingError(
            "PDF file is empty"
        )

    document = None

    try:
        # ---------------------------------------------------------
        # Open PDF
        # ---------------------------------------------------------

        document = pymupdf.open(str(path))

        if document.needs_pass or document.is_encrypted:
            raise DocumentProcessingError(
                "Encrypted PDF cannot be processed"
                )

        page_count = document.page_count

        if page_count == 0:
            raise DocumentProcessingError(
                "PDF document contains no pages"
            )

        # ---------------------------------------------------------
        # Initialize extraction results
        # ---------------------------------------------------------

        pages: list[dict[str, Any]] = []
        all_text: list[str] = []
        warnings: list[str] = []

        digital_page_count = 0
        ocr_page_count = 0
        empty_page_count = 0
        failed_ocr_page_count = 0

        total_ocr_detections = 0
        total_low_confidence_detections = 0

        # ---------------------------------------------------------
        # Process pages sequentially
        # ---------------------------------------------------------

        for page_index in range(page_count):

            page_number = page_index + 1
            page = document.load_page(page_index)

            # Extract embedded digital text first.
            digital_text = page.get_text("text") or ""
            text = digital_text

            extraction_method = "digital_pdf"

            ocr_results: list[dict[str, Any]] = []
            average_confidence = 0.0
            total_detections = 0
            low_confidence_count = 0

            ocr_attempted = False
            ocr_failed = False
            ocr_error = None

            # -----------------------------------------------------
            # OCR fallback
            # -----------------------------------------------------

            if len(digital_text.strip()) < OCR_MIN_CHARACTERS:

                ocr_attempted = True

                try:
                    ocr_data = _extract_page_ocr(page)

                    ocr_text = (
                        ocr_data.get("text", "") or ""
                    )

                    ocr_results = (
                        ocr_data.get("results", []) or []
                    )

                    average_confidence = float(
                        ocr_data.get(
                            "average_confidence",
                            0.0,
                        ) or 0.0
                    )

                    total_detections = int(
                        ocr_data.get(
                            "total_detections",
                            len(ocr_results),
                        ) or 0
                    )

                    low_confidence_count = int(
                        ocr_data.get(
                            "low_confidence_count",
                            0,
                        ) or 0
                    )

                    # Prefer OCR text when available.
                    # Preserve any existing digital text if OCR
                    # does not find additional readable text.
                    if ocr_text.strip():
                        text = ocr_text
                        extraction_method = "ocr"

                    else:
                        text = digital_text
                        extraction_method = "digital_pdf"

                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                ) as exc:

                    # Expected OCR failures should not stop the
                    # processing of the remaining PDF pages.
                    ocr_failed = True
                    ocr_error = str(exc)

                    warnings.append(
                        f"Page {page_number}: OCR failed: {exc}"
                    )

                    # Preserve any digital text already extracted.
                    text = digital_text
                    extraction_method = "digital_pdf"

                # Unexpected exceptions, such as TypeError,
                # intentionally propagate to the outer handler.

            # -----------------------------------------------------
            # Update extraction statistics
            # -----------------------------------------------------

            if extraction_method == "ocr":
                ocr_page_count += 1
            else:
                digital_page_count += 1

            if ocr_failed:
                failed_ocr_page_count += 1

            if not text.strip():
                empty_page_count += 1

                # Avoid adding a second warning when OCR itself
                # has already failed on this page.
                if not ocr_failed:
                    warnings.append(
                        f"Page {page_number}: No readable text found"
                    )

            total_ocr_detections += total_detections

            total_low_confidence_detections += (
                low_confidence_count
            )

            # -----------------------------------------------------
            # Store page data
            # -----------------------------------------------------

            page_data = {
                "page_number": page_number,
                "text": text,
                "extraction_method": extraction_method,
                "character_count": len(text),
                "ocr_attempted": ocr_attempted,
                "ocr_results": ocr_results,
                "average_confidence": average_confidence,
                "total_detections": total_detections,
                "low_confidence_count": low_confidence_count,
            }

            if ocr_error is not None:
                page_data["ocr_error"] = ocr_error

            pages.append(page_data)
            all_text.append(text)

        # ---------------------------------------------------------
        # Extract PDF metadata
        # ---------------------------------------------------------

        pdf_metadata = document.metadata or {}

        metadata = {
            "page_count": page_count,
            "title": pdf_metadata.get("title") or "",
            "author": pdf_metadata.get("author") or "",
            "subject": pdf_metadata.get("subject") or "",
            "creator": pdf_metadata.get("creator") or "",
            "producer": pdf_metadata.get("producer") or "",
            "format": pdf_metadata.get("format") or "PDF",
            "digital_page_count": digital_page_count,
            "ocr_page_count": ocr_page_count,
            "empty_page_count": empty_page_count,
            "failed_ocr_page_count": failed_ocr_page_count,
            "total_characters": sum(
                len(text) for text in all_text
            ),
            "total_ocr_detections": total_ocr_detections,
            "total_low_confidence_detections": (
                total_low_confidence_detections
            ),
            "extraction_method": (
                "mixed"
                if ocr_page_count > 0 and digital_page_count > 0
                else "ocr"
                if ocr_page_count > 0
                else "digital_pdf"
            ),
        }

        # ---------------------------------------------------------
        # Build final document result
        # ---------------------------------------------------------

        return DocumentContent(
            filename=path.name,
            file_type="pdf",
            extracted_text="\n\n".join(all_text),
            pages=pages,
            tables=[],
            metadata=metadata,
            warnings=warnings,
        )

    except DocumentProcessingError:
        raise

    except Exception as exc:
        raise DocumentProcessingError(
            f"Failed to process PDF: {exc}"
        ) from exc

    finally:
        if document is not None:
            document.close()
