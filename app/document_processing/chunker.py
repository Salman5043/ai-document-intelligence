
import json
from typing import Any


DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 300


def _split_text(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping chunks without losing content."""
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Prefer a nearby whitespace boundary.
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + max_chars // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)

        # Avoid beginning the next chunk in the middle of a word.
        if next_start < len(text):
            boundary = text.find(" ", next_start, end)

            if boundary != -1:
                next_start = boundary + 1

        start = next_start

    return chunks


def _make_chunk(
    *,
    chunk_number: int,
    content: str,
    filename: str,
    file_type: str,
    source: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a consistently structured source-aware chunk."""
    return {
        "chunk_id": f"chunk_{chunk_number:06d}",
        "content": content,
        "source": {
            "filename": filename,
            "file_type": file_type,
            **source,
        },
        "metadata": metadata or {},
    }


def _chunk_pages(
    document: dict[str, Any],
    filename: str,
    file_type: str,
    max_chars: int,
    overlap: int,
    starting_number: int,
) -> list[dict[str, Any]]:
    """Chunk PDF pages and image frames while retaining page metadata."""
    chunks = []

    for page in document.get("pages", []):
        text = page.get("text") or ""

        if not text.strip():
            continue

        page_number = page.get("page_number")

        page_metadata = {
            "extraction_method": page.get("extraction_method"),
            "average_confidence": page.get("average_confidence"),
            "ocr_attempted": page.get("ocr_attempted"),
            "total_detections": page.get("total_detections"),
            "low_confidence_count": page.get("low_confidence_count"),
            "extraction_status": page.get("extraction_status"),
            "ocr_results": page.get("ocr_results", []),
        }

        for part_number, part in enumerate(
            _split_text(text, max_chars, overlap),
            start=1,
        ):
            chunks.append(
                _make_chunk(
                    chunk_number=starting_number + len(chunks),
                    content=part,
                    filename=filename,
                    file_type=file_type,
                    source={
                        "page_number": page_number,
                        "sheet_name": None,
                        "start_row": None,
                        "end_row": None,
                        "part_number": part_number,
                    },
                    metadata=page_metadata,
                )
            )

    return chunks


def _format_record(record: dict[str, Any]) -> str:
    """Serialize a spreadsheet record without discarding its values."""
    return json.dumps(
        record,
        ensure_ascii=False,
        default=str,
    )


def _chunk_tables(
    document: dict[str, Any],
    filename: str,
    file_type: str,
    max_chars: int,
    starting_number: int,
) -> list[dict[str, Any]]:
    """Chunk spreadsheet records while preserving their original row ranges."""
    chunks = []

    for table in document.get("tables", []):
        records = table.get("records") or []

        if not records:
            continue

        sheet_name = table.get("sheet_name")
        columns = table.get("columns", [])

        current_lines = []
        current_size = 0
        current_start_row = None
        current_end_row = None

        def flush():
            nonlocal current_lines
            nonlocal current_size
            nonlocal current_start_row
            nonlocal current_end_row

            if not current_lines:
                return

            content = (
                f"Columns: {', '.join(columns)}\n"
                + "\n".join(current_lines)
            )

            chunks.append(
                _make_chunk(
                    chunk_number=starting_number + len(chunks),
                    content=content,
                    filename=filename,
                    file_type=file_type,
                    source={
                        "page_number": None,
                        "sheet_name": sheet_name,
                        "start_row": current_start_row,
                        "end_row": current_end_row,
                        "part_number": 1,
                    },
                    metadata={
                        "columns": columns,
                        "record_count": len(current_lines),
                    },
                )
            )

            current_lines = []
            current_size = 0
            current_start_row = None
            current_end_row = None

        for index, record in enumerate(records):
            # The processors use the first row as the header.
            source_row = index + 2

            record_text = (
                f"Row {source_row}: {_format_record(record)}"
            )

            # Keep a complete row together where possible.
            if len(record_text) > max_chars:
                flush()

                chunks.append(
                    _make_chunk(
                        chunk_number=starting_number + len(chunks),
                        content=record_text,
                        filename=filename,
                        file_type=file_type,
                        source={
                            "page_number": None,
                            "sheet_name": sheet_name,
                            "start_row": source_row,
                            "end_row": source_row,
                            "part_number": 1,
                        },
                        metadata={
                            "columns": columns,
                            "record_count": 1,
                            "oversized_record": True,
                        },
                    )
                )
                continue

            additional_size = len(record_text) + 1

            if (
                current_lines
                and current_size + additional_size > max_chars
            ):
                flush()

            if current_start_row is None:
                current_start_row = source_row

            current_end_row = source_row
            current_lines.append(record_text)
            current_size += additional_size

        flush()

    return chunks


def chunk_document(
    document: dict[str, Any],
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Convert processed document content into source-aware chunks.

    Supports:
        - PDF pages
        - Image and TIFF frames
        - Excel worksheets
        - CSV records
        - Generic extracted text

    Spreadsheet rows are kept together whenever possible.
    Text chunks retain page references and OCR metadata.

    Args:
        document: Dictionary returned by DocumentContent.to_dict().
        max_chars: Maximum target characters per chunk.
        overlap: Character overlap for page and generic text chunks.

    Returns:
        A list of structured chunks with source references.

    Raises:
        ValueError: If chunk configuration or document input is invalid.
    """
    if not isinstance(document, dict):
        raise ValueError("Document must be a dictionary")

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    if overlap < 0 or overlap >= max_chars:
        raise ValueError(
            "overlap must be non-negative and smaller than max_chars"
        )

    filename = document.get("filename") or "unknown"
    file_type = document.get("file_type") or "unknown"

    chunks = []

    # Page-based documents: PDFs and images.
    if document.get("pages"):
        chunks.extend(
            _chunk_pages(
                document,
                filename,
                file_type,
                max_chars,
                overlap,
                starting_number=1,
            )
        )

    # Table-based documents: Excel and CSV.
    if document.get("tables"):
        chunks.extend(
            _chunk_tables(
                document,
                filename,
                file_type,
                max_chars,
                starting_number=len(chunks) + 1,
            )
        )

    # Fallback for documents that contain only extracted text.
    if not chunks:
        text = document.get("extracted_text") or ""

        for part_number, part in enumerate(
            _split_text(text, max_chars, overlap),
            start=1,
        ):
            chunks.append(
                _make_chunk(
                    chunk_number=len(chunks) + 1,
                    content=part,
                    filename=filename,
                    file_type=file_type,
                    source={
                        "page_number": None,
                        "sheet_name": None,
                        "start_row": None,
                        "end_row": None,
                        "part_number": part_number,
                    },
                    metadata={
                        "extraction_method": "text_fallback",
                    },
                )
            )

    return chunks
