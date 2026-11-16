import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import pandas as pd
import pymupdf
import pytest

from app.document_processing.dispatcher import process_document
from app.document_processing.models import DocumentProcessingError


def test_csv_processing(tmp_path):
    file_path = tmp_path / "invoice.csv"

    pd.DataFrame({
        "item": ["Meter", "Sensor"],
        "amount": [5000, 3000],
    }).to_csv(file_path, index=False)

    result = process_document(file_path)

    assert result.file_type == "csv"
    assert len(result.tables) == 1
    assert result.tables[0]["row_count"] == 2


def test_excel_multiple_sheets(tmp_path):
    file_path = tmp_path / "invoice.xlsx"

    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({
            "item": ["Meter"]
        }).to_excel(
            writer,
            sheet_name="Items",
            index=False
        )

        pd.DataFrame({
            "tax": [500]
        }).to_excel(
            writer,
            sheet_name="Tax",
            index=False
        )

    result = process_document(file_path)

    assert result.file_type == "excel"
    assert result.metadata["sheet_count"] == 2


def test_digital_pdf_processing(tmp_path):
    file_path = tmp_path / "invoice.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (50, 50),
        "Invoice Number: INV-001. "
        "Supplier: ABC Technologies. "
        "Total: 10000 BDT."
    )

    document.save(file_path)
    document.close()

    result = process_document(file_path)

    assert result.file_type == "pdf"
    assert "INV-001" in result.extracted_text
    assert result.metadata["page_count"] == 1


def test_unsupported_file(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("Sample document")

    with pytest.raises(DocumentProcessingError):
        process_document(file_path)


def test_missing_file():
    with pytest.raises(DocumentProcessingError):
        process_document("missing.pdf")
