
import pytest

from app.document_processing.chunker import chunk_document


def test_pdf_chunks_preserve_page_references():
    document = {
        "filename": "invoice.pdf",
        "file_type": "pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Invoice number: INV-1001",
                "extraction_method": "digital_pdf",
            },
            {
                "page_number": 2,
                "text": "Total amount: 1500",
                "extraction_method": "ocr",
                "average_confidence": 0.92,
                "total_detections": 5,
            },
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0]["source"]["page_number"] == 1
    assert chunks[1]["source"]["page_number"] == 2
    assert chunks[1]["metadata"]["average_confidence"] == 0.92


def test_long_pdf_page_is_split_with_page_reference():
    document = {
        "filename": "large.pdf",
        "file_type": "pdf",
        "pages": [
            {
                "page_number": 4,
                "text": "invoice data " * 100,
                "extraction_method": "digital_pdf",
            }
        ],
    }

    chunks = chunk_document(
        document,
        max_chars=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert all(
        chunk["source"]["page_number"] == 4
        for chunk in chunks
    )
    assert all(
        len(chunk["content"]) <= 100
        for chunk in chunks
    )


def test_excel_chunks_preserve_sheet_and_row_references():
    document = {
        "filename": "invoices.xlsx",
        "file_type": "excel",
        "tables": [
            {
                "sheet_name": "Invoices",
                "columns": ["Invoice", "Amount"],
                "records": [
                    {"Invoice": "INV-001", "Amount": 100},
                    {"Invoice": "INV-002", "Amount": 200},
                ],
            }
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1

    source = chunks[0]["source"]

    assert source["sheet_name"] == "Invoices"
    assert source["start_row"] == 2
    assert source["end_row"] == 3
    assert "INV-001" in chunks[0]["content"]
    assert "INV-002" in chunks[0]["content"]


def test_csv_chunks_preserve_original_row_numbers():
    document = {
        "filename": "invoices.csv",
        "file_type": "csv",
        "tables": [
            {
                "columns": ["Invoice", "Amount"],
                "records": [
                    {"Invoice": "INV-001", "Amount": 100},
                    {"Invoice": "INV-002", "Amount": 200},
                ],
            }
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["source"]["start_row"] == 2
    assert chunks[0]["source"]["end_row"] == 3
    assert chunks[0]["source"]["sheet_name"] is None


def test_multiple_excel_sheets_create_separate_source_chunks():
    document = {
        "filename": "financial_report.xlsx",
        "file_type": "excel",
        "tables": [
            {
                "sheet_name": "Sales",
                "columns": ["Amount"],
                "records": [{"Amount": 100}],
            },
            {
                "sheet_name": "Expenses",
                "columns": ["Amount"],
                "records": [{"Amount": 50}],
            },
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0]["source"]["sheet_name"] == "Sales"
    assert chunks[1]["source"]["sheet_name"] == "Expenses"
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]


def test_oversized_spreadsheet_record_is_preserved():
    document = {
        "filename": "large.xlsx",
        "file_type": "excel",
        "tables": [
            {
                "sheet_name": "Details",
                "columns": ["Description"],
                "records": [
                    {"Description": "Product description " * 30}
                ],
            }
        ],
    }

    chunks = chunk_document(
        document,
        max_chars=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["oversized_record"] is True
    assert chunks[0]["source"]["start_row"] == 2
    assert chunks[0]["source"]["end_row"] == 2


def test_image_chunks_preserve_frame_number_and_ocr_metadata():
    document = {
        "filename": "scan.tiff",
        "file_type": "image",
        "pages": [
            {
                "page_number": 2,
                "text": "Invoice total: 5000",
                "extraction_method": "ocr",
                "average_confidence": 0.88,
                "total_detections": 4,
                "low_confidence_count": 1,
                "extraction_status": "success",
                "ocr_results": [{"text": "5000", "confidence": 0.88}],
            }
        ],
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["source"]["page_number"] == 2
    assert chunks[0]["metadata"]["average_confidence"] == 0.88
    assert chunks[0]["metadata"]["total_detections"] == 4


def test_generic_text_fallback():
    document = {
        "filename": "document.txt",
        "file_type": "text",
        "extracted_text": "This is a test document.",
    }

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["content"] == "This is a test document."
    assert chunks[0]["metadata"]["extraction_method"] == "text_fallback"


@pytest.mark.parametrize(
    "max_chars, overlap",
    [
        (0, 0),
        (100, -1),
        (100, 100),
        (100, 120),
    ],
)
def test_invalid_chunk_configuration(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_document(
            {"filename": "test.pdf", "file_type": "pdf"},
            max_chars=max_chars,
            overlap=overlap,
        )


def test_non_dictionary_document_is_rejected():
    with pytest.raises(ValueError):
        chunk_document("not a document")
