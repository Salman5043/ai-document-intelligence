
import sys
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pymupdf
import pytest
from PIL import Image

from app.document_processing import image_processor, ocr_engine, pdf_processor
from app.document_processing.dispatcher import process_document
from app.document_processing.models import DocumentProcessingError

# ---------------------------------------------------------
# Existing document processing tests
# ---------------------------------------------------------

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
            index=False,
        )

        pd.DataFrame({
            "tax": [500]
        }).to_excel(
            writer,
            sheet_name="Tax",
            index=False,
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
        "Total: 10000 BDT.",
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


# ---------------------------------------------------------
# Image processor tests
# ---------------------------------------------------------

def test_image_processing(tmp_path, monkeypatch):
    file_path = tmp_path / "invoice.png"

    image = Image.new("RGB", (200, 100), "white")
    image.save(file_path)
    image.close()

    monkeypatch.setattr(
        image_processor,
        "extract_image_text",
        lambda image: "Invoice Number: INV-100",
    )

    result = image_processor.process_image(file_path)

    assert result.file_type == "image"
    assert result.filename == "invoice.png"
    assert result.extracted_text == "Invoice Number: INV-100"
    assert result.metadata["frame_count"] == 1
    assert result.metadata["format"] == ".png"

    assert len(result.pages) == 1
    assert result.pages[0]["page_number"] == 1
    assert result.pages[0]["text"] == "Invoice Number: INV-100"
    assert result.pages[0]["extraction_method"] == "ocr"
    assert result.pages[0]["character_count"] == len(
        "Invoice Number: INV-100"
    )


def test_image_processing_multiple_tiff_frames(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "multipage.tiff"

    first = Image.new("RGB", (100, 100), "white")
    second = Image.new("RGB", (100, 100), "white")

    first.save(
        file_path,
        save_all=True,
        append_images=[second],
    )

    first.close()
    second.close()

    calls = []

    def fake_ocr(image):
        calls.append(image.size)
        return f"Page {len(calls)}"

    monkeypatch.setattr(
        image_processor,
        "extract_image_text",
        fake_ocr,
    )

    result = image_processor.process_image(file_path)

    assert result.metadata["frame_count"] == 2
    assert len(result.pages) == 2
    assert result.extracted_text == "Page 1\n\nPage 2"
    assert result.pages[0]["page_number"] == 1
    assert result.pages[1]["page_number"] == 2
    assert len(calls) == 2


def test_image_processor_resizes_large_image():
    image = Image.new("RGB", (3000, 1500), "white")

    try:
        prepared = image_processor.prepare_image_for_ocr(image)

        try:
            assert prepared.size == (2500, 1250)
            assert prepared.mode == "RGB"
        finally:
            prepared.close()

    finally:
        image.close()


def test_image_processor_converts_rgba_to_rgb():
    image = Image.new("RGBA", (100, 100), (255, 255, 255, 128))

    try:
        prepared = image_processor.prepare_image_for_ocr(image)

        try:
            assert prepared.mode == "RGB"
            assert prepared.size == (100, 100)
        finally:
            prepared.close()

    finally:
        image.close()


def test_image_processing_missing_file():
    with pytest.raises(
        DocumentProcessingError,
        match="File not found",
    ):
        image_processor.process_image("missing.png")


def test_image_processing_unsupported_format(tmp_path):
    file_path = tmp_path / "document.bmp"
    file_path.write_bytes(b"not an image")

    with pytest.raises(
        DocumentProcessingError,
        match="Unsupported image format",
    ):
        image_processor.process_image(file_path)


def test_image_processing_ocr_failure(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "invoice.png"

    image = Image.new("RGB", (100, 100), "white")
    image.save(file_path)
    image.close()

    def failing_ocr(image):
        raise RuntimeError("OCR engine failed")

    monkeypatch.setattr(
        image_processor,
        "extract_image_text",
        failing_ocr,
    )

    with pytest.raises(
        DocumentProcessingError,
        match="Image processing failed",
    ):
        image_processor.process_image(file_path)


def test_image_processing_corrupt_image(tmp_path):
    file_path = tmp_path / "corrupt.png"
    file_path.write_bytes(b"this is not a valid image")

    with pytest.raises(DocumentProcessingError):
        image_processor.process_image(file_path)


def test_image_processing_empty_ocr_result(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "empty.png"

    image = Image.new("RGB", (100, 100), "white")
    image.save(file_path)
    image.close()

    monkeypatch.setattr(
        image_processor,
        "extract_image_text",
        lambda image: "",
    )

    result = image_processor.process_image(file_path)

    assert result.extracted_text == ""
    assert result.pages[0]["text"] == ""
    assert result.pages[0]["character_count"] == 0


# ---------------------------------------------------------
# OCR engine preparation tests
# ---------------------------------------------------------

def test_prepare_ocr_image_resizes_large_image():
    image = Image.new("RGB", (2000, 1000), "white")

    try:
        prepared = ocr_engine.prepare_ocr_image(image)

        try:
            assert prepared.size == (1600, 800)
            assert prepared.mode == "RGB"
        finally:
            prepared.close()

    finally:
        image.close()


def test_prepare_ocr_image_converts_rgba_to_rgb():
    image = Image.new("RGBA", (100, 100), "white")

    try:
        prepared = ocr_engine.prepare_ocr_image(image)

        try:
            assert prepared.mode == "RGB"
            assert prepared.size == (100, 100)
        finally:
            prepared.close()

    finally:
        image.close()


def test_prepare_ocr_image_accepts_bytes():
    image = Image.new("RGB", (100, 80), "white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.close()

    prepared = ocr_engine.prepare_ocr_image(buffer.getvalue())

    try:
        assert prepared.size == (100, 80)
        assert prepared.mode == "RGB"
    finally:
        prepared.close()


def test_prepare_ocr_image_accepts_file_path(tmp_path):
    file_path = tmp_path / "source.png"

    image = Image.new("RGB", (120, 90), "white")
    image.save(file_path)
    image.close()

    prepared = ocr_engine.prepare_ocr_image(file_path)

    try:
        assert prepared.size == (120, 90)
        assert prepared.mode == "RGB"
    finally:
        prepared.close()


# ---------------------------------------------------------
# OCR inference tests - EasyOCR is mocked
# ---------------------------------------------------------

def test_extract_image_text_filters_low_confidence(
    monkeypatch,
):
    class FakeReader:
        def readtext(self, image_array, **kwargs):
            assert isinstance(image_array, np.ndarray)
            assert kwargs["batch_size"] == 1
            assert kwargs["workers"] == 0
            assert kwargs["canvas_size"] == 1280
            assert kwargs["mag_ratio"] == 1.0

            return [
                (None, "Invoice Number", 0.95),
                (None, "INV-100", 0.90),
                (None, "Unreliable Text", 0.10),
                (None, "   ", 0.99),
            ]

    monkeypatch.setattr(
        ocr_engine,
        "get_ocr_reader",
        lambda: FakeReader(),
    )

    image = Image.new("RGB", (100, 100), "white")

    try:
        result = ocr_engine.extract_image_text(image)
    finally:
        image.close()

    assert result == "Invoice Number\nINV-100"


def test_extract_image_text_empty_result(monkeypatch):
    class FakeReader:
        def readtext(self, image_array, **kwargs):
            return []

    monkeypatch.setattr(
        ocr_engine,
        "get_ocr_reader",
        lambda: FakeReader(),
    )

    image = Image.new("RGB", (100, 100), "white")

    try:
        result = ocr_engine.extract_image_text(image)
    finally:
        image.close()

    assert result == ""


def test_extract_image_text_wraps_ocr_exception(monkeypatch):
    class FakeReader:
        def readtext(self, image_array, **kwargs):
            raise ValueError("Simulated OCR failure")

    monkeypatch.setattr(
        ocr_engine,
        "get_ocr_reader",
        lambda: FakeReader(),
    )

    image = Image.new("RGB", (100, 100), "white")

    try:
        with pytest.raises(
            RuntimeError,
            match="EasyOCR extraction failed",
        ):
            ocr_engine.extract_image_text(image)
    finally:
        image.close()


def test_get_ocr_reader_is_cached(monkeypatch):
    calls = []

    class FakeEasyOCR:
        @staticmethod
        def Reader(languages, gpu):
            calls.append((languages, gpu))
            return object()

    monkeypatch.setattr(
        ocr_engine,
        "easyocr",
        FakeEasyOCR,
    )

    ocr_engine.get_ocr_reader.cache_clear()

    try:
        first = ocr_engine.get_ocr_reader()
        second = ocr_engine.get_ocr_reader()

        assert first is second
        assert len(calls) == 1
        assert calls[0] == (["en"], False)

    finally:
        ocr_engine.get_ocr_reader.cache_clear()



# ---------------------------------------------------------
# PDF processor tests
# ---------------------------------------------------------



def create_test_pdf(file_path, page_texts):
    """Create a PDF with the supplied text on each page."""
    document = pymupdf.open()

    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((50, 50), text)

    document.save(file_path)
    document.close()


def test_pdf_multiple_digital_pages(tmp_path, monkeypatch):
    file_path = tmp_path / "multiple.pdf"

    create_test_pdf(
        file_path,
        [
            "Invoice Number: INV-001. Supplier: ABC Technologies. Total: 10000 BDT.",
            "Invoice Number: INV-002. Supplier: XYZ Technologies. Total: 20000 BDT.",
        ],
    )

    def unexpected_ocr(image_bytes):
        pytest.fail("OCR should not run for digital PDF pages")

    monkeypatch.setattr(
        pdf_processor,
        "extract_image_text",
        unexpected_ocr,
    )

    result = pdf_processor.process_pdf(file_path)

    assert result.file_type == "pdf"
    assert result.metadata["page_count"] == 2
    assert len(result.pages) == 2
    assert result.pages[0]["extraction_method"] == "digital_pdf"
    assert result.pages[1]["extraction_method"] == "digital_pdf"
    assert "INV-001" in result.extracted_text
    assert "INV-002" in result.extracted_text


def test_pdf_scanned_page_uses_ocr(tmp_path, monkeypatch):
    file_path = tmp_path / "scanned.pdf"

    create_test_pdf(file_path, [""])

    monkeypatch.setattr(
        pdf_processor,
        "extract_image_text",
        lambda image_bytes: "Invoice Number: INV-500",
    )

    result = pdf_processor.process_pdf(file_path)

    assert result.pages[0]["extraction_method"] == "ocr"
    assert result.pages[0]["text"] == "Invoice Number: INV-500"
    assert result.pages[0]["character_count"] == len(
        "Invoice Number: INV-500"
    )
    assert result.extracted_text == "Invoice Number: INV-500"
    assert result.warnings == []


def test_pdf_ocr_returns_empty_text(tmp_path, monkeypatch):
    file_path = tmp_path / "unreadable.pdf"

    create_test_pdf(file_path, [""])

    monkeypatch.setattr(
        pdf_processor,
        "extract_image_text",
        lambda image_bytes: "",
    )

    result = pdf_processor.process_pdf(file_path)

    assert result.pages[0]["extraction_method"] == "digital_pdf"
    assert result.pages[0]["text"] == ""
    assert result.warnings == [
        "Page 1: No readable text found"
    ]


def test_pdf_ocr_failure_adds_warning(tmp_path, monkeypatch):
    file_path = tmp_path / "ocr_failure.pdf"

    create_test_pdf(file_path, [""])

    def failing_ocr(image_bytes):
        raise RuntimeError("Simulated OCR failure")

    monkeypatch.setattr(
        pdf_processor,
        "extract_image_text",
        failing_ocr,
    )

    result = pdf_processor.process_pdf(file_path)

    assert result.metadata["page_count"] == 1
    assert result.pages[0]["text"] == ""
    assert result.warnings == [
        "Page 1: OCR failed: Simulated OCR failure"
    ]


def test_pdf_continues_after_one_page_ocr_failure(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "mixed.pdf"

    create_test_pdf(
        file_path,
        [
            "",
            "Invoice Number: INV-200. Supplier: ABC Technologies. Total: 5000 BDT.",
        ],
    )

    calls = []

    def fake_ocr(image_bytes):
        calls.append(image_bytes)
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr(
        pdf_processor,
        "extract_image_text",
        fake_ocr,
    )

    result = pdf_processor.process_pdf(file_path)

    assert result.metadata["page_count"] == 2
    assert len(result.pages) == 2
    assert len(calls) == 1

    assert result.pages[0]["extraction_method"] == "digital_pdf"
    assert result.pages[1]["extraction_method"] == "digital_pdf"

    assert result.pages[1]["text"].startswith("Invoice Number")
    assert len(result.warnings) == 1
    assert "Page 1: OCR failed" in result.warnings[0]


def test_pdf_preserves_metadata(tmp_path):
    file_path = tmp_path / "metadata.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (50, 50),
        "Invoice Number: INV-300. Supplier: ABC Technologies. Total: 9000 BDT.",
    )

    document.set_metadata({
        "title": "Sample Invoice",
        "author": "ABC Technologies",
    })

    document.save(file_path)
    document.close()

    result = pdf_processor.process_pdf(file_path)

    assert result.metadata["title"] == "Sample Invoice"
    assert result.metadata["author"] == "ABC Technologies"


def test_pdf_missing_file():
    with pytest.raises(
        DocumentProcessingError,
        match="File not found",
    ):
        pdf_processor.process_pdf("missing.pdf")


def test_pdf_invalid_extension(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("Not a PDF")

    with pytest.raises(
        DocumentProcessingError,
        match="Invalid PDF file extension",
    ):
        pdf_processor.process_pdf(file_path)


def test_pdf_corrupt_file(tmp_path):
    file_path = tmp_path / "corrupt.pdf"
    file_path.write_bytes(b"This is not a valid PDF")

    with pytest.raises(
        DocumentProcessingError,
        match="Failed to process PDF",
    ):
        pdf_processor.process_pdf(file_path)


def test_pdf_encrypted_document(tmp_path):
    file_path = tmp_path / "encrypted.pdf"

    document = pymupdf.open()
    document.new_page()

    document.save(
        file_path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )

    document.close()

    with pytest.raises(
        DocumentProcessingError,
        match="Encrypted PDF cannot be processed",
    ):
        pdf_processor.process_pdf(file_path)

def test_pdf_unexpected_ocr_exception_propagates(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "unexpected_error.pdf"

    create_test_pdf(file_path, [""])

    def failing_ocr(image_bytes):
        raise TypeError("Unexpected OCR error")

    monkeypatch.setattr(
        pdf_processor,
        "extract_image_text",
        failing_ocr,
    )

    with pytest.raises(
        DocumentProcessingError,
        match="Failed to process PDF",
    ):
        pdf_processor.process_pdf(file_path)