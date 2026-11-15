
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image, ImageOps


@lru_cache(maxsize=1)
def get_ocr_reader():
    """
    Initialize EasyOCR only once.
    Reuse the model across document processing requests.
    """
    return easyocr.Reader(
        ["en"],
        gpu=False
    )


def extract_image_text(image_source) -> str:
    """
    Extract text using EasyOCR.

    Supported inputs:
    - Image file path
    - Image bytes
    - PIL Image object
    """

    try:
        if isinstance(image_source, bytes):
            image = Image.open(
                BytesIO(image_source)
            )

        elif isinstance(image_source, Image.Image):
            image = image_source

        else:
            image = Image.open(image_source)

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image_array = np.asarray(image)

        reader = get_ocr_reader()

        results = reader.readtext(
            image_array,
            detail=1,
            paragraph=False,
            batch_size=1
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
