
from functools import lru_cache
from io import BytesIO
from threading import Lock

import easyocr
import numpy as np
from PIL import Image, ImageOps

MAX_OCR_DIMENSION = 1600

# Prevent concurrent OCR inference within this process.
OCR_LOCK = Lock()


@lru_cache(maxsize=1)
def get_ocr_reader():
    """
    Initialize EasyOCR once and reuse the model.
    """
    return easyocr.Reader(
        ["en"],
        gpu=False,
    )


def prepare_ocr_image(image_source) -> Image.Image:
    """
    Load and prepare an image for OCR.
    Supports file paths, bytes, and PIL Image objects.
    """

    if isinstance(image_source, Image.Image):
        image = image_source.copy()

    elif isinstance(image_source, bytes):
        with Image.open(BytesIO(image_source)) as opened:
            image = opened.copy()

    else:
        with Image.open(image_source) as opened:
            image = opened.copy()

    try:
        image = ImageOps.exif_transpose(image)

        if image.mode != "RGB":
            converted = image.convert("RGB")
            image.close()
            image = converted

        # Resize only when the image exceeds the limit.
        image.thumbnail(
            (MAX_OCR_DIMENSION, MAX_OCR_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        return image

    except Exception:
        image.close()
        raise


def extract_image_text(image_source) -> str:
    """
    Extract text using EasyOCR.
    """

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

        extracted_lines = []

        for bbox, text, confidence in results:
            if confidence >= 0.25 and text.strip():
                extracted_lines.append(text.strip())

        return "\n".join(extracted_lines)

    except Exception as exc:
        raise RuntimeError(
            f"EasyOCR extraction failed: {exc}"
        ) from exc

    finally:
        if image is not None:
            image.close()
