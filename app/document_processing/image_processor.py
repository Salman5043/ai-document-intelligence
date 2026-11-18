
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageSequence

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)

from app.document_processing.ocr_engine import (
    extract_image_data,
    extract_image_text,
)


SUPPORTED_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
}

MAX_OCR_DIMENSION = 2500


# Keep references to the original functions.
# This allows compatibility with older integrations that
# replace extract_image_text() at the image_processor module level.
_DEFAULT_EXTRACT_IMAGE_DATA = extract_image_data
_DEFAULT_EXTRACT_IMAGE_TEXT = extract_image_text


def prepare_image_for_ocr(frame: Image.Image) -> Image.Image:
    """
    Prepare an image frame for OCR.

    - Correct image orientation using EXIF metadata.
    - Preserve aspect ratio.
    - Avoid unnecessary upscaling.
    - Convert the image to RGB.
    """
    image = frame.copy()

    try:
        corrected = ImageOps.exif_transpose(image)

        if corrected is not image:
            image.close()

        image = corrected

        image.thumbnail(
            (MAX_OCR_DIMENSION, MAX_OCR_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        if image.mode != "RGB":
            converted = image.convert("RGB")
            image.close()
            image = converted

        return image

    except Exception:
        image.close()
        raise


def _extract_ocr_data(image: Image.Image) -> dict[str, Any]:
    """
    Extract structured OCR data.

    The detailed OCR interface is the default.

    If an older integration replaces extract_image_text(),
    use that implementation and adapt its result to the
    structured OCR response format.
    """

    # Backward compatibility for older tests and integrations.
    if extract_image_text is not _DEFAULT_EXTRACT_IMAGE_TEXT:
        text = extract_image_text(image)

        if not isinstance(text, str):
            text = str(text)

        return {
            "text": text,
            "results": [],
            "average_confidence": 0.0,
            "total_detections": 0,
            "low_confidence_count": 0,
            "extraction_status": (
                "success" if text.strip() else "no_text_detected"
            ),
        }

    # Normal production path: retain detailed OCR information.
    return extract_image_data(image)


def process_image(file_path: str | Path) -> DocumentContent:
    """
    Process JPG, JPEG, PNG, TIFF and multi-frame TIFF images.

    Extracts:
    - Text
    - OCR confidence scores
    - Text bounding boxes
    - Low-confidence detection flags
    - Per-frame extraction metadata
    """

    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise DocumentProcessingError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise DocumentProcessingError(
            f"Unsupported image format: {path.suffix}"
        )

    pages: list[dict[str, Any]] = []
    all_text: list[str] = []

    try:
        with Image.open(path) as image:
            frame_count = getattr(image, "n_frames", 1)
            original_format = image.format
            original_size = image.size

            for index, frame in enumerate(
                ImageSequence.Iterator(image)
            ):
                ocr_image = None

                try:
                    ocr_image = prepare_image_for_ocr(frame)

                    ocr_data = _extract_ocr_data(ocr_image)

                    text = ocr_data.get("text", "")

                    if not isinstance(text, str):
                        text = str(text)

                    pages.append({
                        "page_number": index + 1,
                        "text": text,
                        "extraction_method": "ocr",
                        "character_count": len(text),
                        "ocr_results": ocr_data.get(
                            "results", []
                        ),
                        "average_confidence": ocr_data.get(
                            "average_confidence", 0.0
                        ),
                        "total_detections": ocr_data.get(
                            "total_detections", 0
                        ),
                        "low_confidence_count": ocr_data.get(
                            "low_confidence_count", 0
                        ),
                        "extraction_status": ocr_data.get(
                            "extraction_status",
                            (
                                "success"
                                if text.strip()
                                else "no_text_detected"
                            ),
                        ),
                    })

                    all_text.append(text)

                finally:
                    if ocr_image is not None:
                        ocr_image.close()

        warnings = []

        if not any(text.strip() for text in all_text):
            warnings.append(
                "No text could be extracted from the image."
            )

        return DocumentContent(
            filename=path.name,
            file_type="image",
            extracted_text="\n\n".join(all_text),
            pages=pages,
            metadata={
                "frame_count": frame_count,
                "format": path.suffix.lower(),
                "original_format": original_format,
                "original_width": original_size[0],
                "original_height": original_size[1],
                "extraction_method": "easyocr",
                "total_pages": len(pages),
                "total_characters": sum(
                    len(text) for text in all_text
                ),
                "total_detections": sum(
                    page["total_detections"]
                    for page in pages
                ),
                "total_low_confidence_detections": sum(
                    page["low_confidence_count"]
                    for page in pages
                ),
            },
            warnings=warnings,
        )

    except DocumentProcessingError:
        raise

    except Exception as exc:
        raise DocumentProcessingError(
            f"Image processing failed: {exc}"
        ) from exc
