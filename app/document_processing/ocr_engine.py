
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Any

import easyocr
import numpy as np
from PIL import Image, ImageOps


MAX_OCR_DIMENSION = 1600
OCR_MIN_CONFIDENCE = 0.25

OCR_LOCK = Lock()


@lru_cache(maxsize=1)
def get_ocr_reader():
    """Initialize EasyOCR once and reuse the model."""
    return easyocr.Reader(
        ["en"],
        gpu=False,
    )


def prepare_ocr_image(image_source) -> Image.Image:
    """Load and prepare an image for OCR."""

    image = None

    try:
        if isinstance(image_source, Image.Image):
            image = image_source.copy()

        elif isinstance(image_source, bytes):
            with Image.open(BytesIO(image_source)) as opened:
                image = opened.copy()

        else:
            with Image.open(image_source) as opened:
                image = opened.copy()

        corrected = ImageOps.exif_transpose(image)

        if corrected is not image:
            image.close()

        image = corrected

        if image.mode != "RGB":
            converted = image.convert("RGB")
            image.close()
            image = converted

        image.thumbnail(
            (MAX_OCR_DIMENSION, MAX_OCR_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        return image

    except Exception:
        if image is not None:
            image.close()
        raise


def extract_image_data(
    image_source,
    min_confidence: float = OCR_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """
    Extract OCR text, confidence scores and bounding boxes.

    All non-empty detections are retained.
    Low-confidence detections are flagged for review.
    """

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            "min_confidence must be between 0 and 1"
        )

    image = None

    try:
        image = prepare_ocr_image(image_source)
        image_array = np.asarray(image)

        reader = get_ocr_reader()

        with OCR_LOCK:
            results = reader.readtext(
                image_array,
                detail=1,
                paragraph=False,
                batch_size=1,
                workers=0,
                canvas_size=1280,
                mag_ratio=1.0,
            )

        extracted_items = []

        for bbox, text, confidence in results:
            text = text.strip()

            if not text:
                continue

            confidence = float(confidence)

            # EasyOCR normally returns a bounding box.
            # Handle None for mocks or malformed results.
            safe_bbox = []

            if bbox is not None:
                safe_bbox = [
                    [
                        float(point[0]),
                        float(point[1]),
                    ]
                    for point in bbox
                    if point is not None and len(point) >= 2
                ]

            extracted_items.append({
                "text": text,
                "confidence": confidence,
                "bbox": safe_bbox,
                "needs_review": confidence < min_confidence,
            })

        average_confidence = (
            sum(item["confidence"] for item in extracted_items)
            / len(extracted_items)
            if extracted_items
            else 0.0
        )

        return {
            "text": "\n".join(
                item["text"] for item in extracted_items
            ),
            "results": extracted_items,
            "average_confidence": average_confidence,
            "total_detections": len(extracted_items),
            "low_confidence_count": sum(
                item["needs_review"]
                for item in extracted_items
            ),
            "extraction_status": (
                "success"
                if extracted_items
                else "no_text_detected"
            ),
        }

    except Exception as exc:
        raise RuntimeError(
            f"EasyOCR extraction failed: {exc}"
        ) from exc

    finally:
        if image is not None:
            image.close()


def extract_image_text(image_source) -> str:
    """
    Backward-compatible OCR interface.

    Returns only detections meeting the confidence threshold.
    Detailed OCR results remain available through extract_image_data().
    """

    data = extract_image_data(image_source)

    return "\n".join(
        item["text"]
        for item in data["results"]
        if item["confidence"] >= OCR_MIN_CONFIDENCE
    )
