
from unittest.mock import patch

import pytest
from PIL import Image

from app.document_processing.image_processor import process_image
from app.document_processing.models import DocumentProcessingError


def test_process_png_image(tmp_path):
    image_path = tmp_path / "sample.png"

    Image.new("RGB", (200, 100), "white").save(image_path)

    with patch(
        "app.document_processing.image_processor.extract_image_data"
    ) as mock_ocr:

        mock_ocr.return_value = {
            "text": "Invoice No: INV-001",
            "results": [],
            "average_confidence": 0.95,
            "total_detections": 1,
            "low_confidence_count": 0,
            "extraction_status": "success",
        }

        result = process_image(image_path)

    assert result.filename == "sample.png"
    assert result.file_type == "image"
    assert "INV-001" in result.extracted_text
    assert len(result.pages) == 1


def test_process_multipage_tiff(tmp_path):
    image_path = tmp_path / "sample.tiff"

    first = Image.new("RGB", (100, 100), "white")
    second = Image.new("RGB", (100, 100), "white")

    first.save(
        image_path,
        save_all=True,
        append_images=[second],
    )

    with patch(
        "app.document_processing.image_processor.extract_image_data"
    ) as mock_ocr:

        mock_ocr.return_value = {
            "text": "Sample text",
            "results": [],
            "average_confidence": 0.9,
            "total_detections": 1,
            "low_confidence_count": 0,
            "extraction_status": "success",
        }

        result = process_image(image_path)

    assert len(result.pages) == 2
    assert result.metadata["frame_count"] == 2


def test_missing_image_raises_error(tmp_path):
    image_path = tmp_path / "missing.png"

    with pytest.raises(DocumentProcessingError):
        process_image(image_path)


def test_unsupported_image_raises_error(tmp_path):
    image_path = tmp_path / "sample.txt"
    image_path.write_text("Not an image")

    with pytest.raises(DocumentProcessingError):
        process_image(image_path)
