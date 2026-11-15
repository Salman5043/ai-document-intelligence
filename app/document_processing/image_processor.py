
from pathlib import Path

from PIL import Image, ImageSequence

from app.document_processing.models import (
    DocumentContent,
    DocumentProcessingError,
)
from app.document_processing.ocr_engine import extract_image_text


SUPPORTED_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
}


def process_image(file_path: str | Path) -> DocumentContent:
    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise DocumentProcessingError(
            "Unsupported image format"
        )

    pages = []
    all_text = []

    try:
        with Image.open(path) as image:
            frame_count = getattr(image, "n_frames", 1)

            for index, frame in enumerate(
                ImageSequence.Iterator(image)
            ):
                text = extract_image_text(frame.copy())

                pages.append({
                    "page_number": index + 1,
                    "text": text,
                    "extraction_method": "ocr",
                    "character_count": len(text),
                })

                all_text.append(text)

        return DocumentContent(
            filename=path.name,
            file_type="image",
            extracted_text="\n\n".join(all_text),
            pages=pages,
            metadata={
                "frame_count": frame_count,
                "format": path.suffix.lower(),
            },
        )

    except Exception as exc:
        raise DocumentProcessingError(
            f"Image processing failed: {exc}"
        ) from exc
