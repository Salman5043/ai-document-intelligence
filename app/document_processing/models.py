
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentContent:
    filename: str
    file_type: str
    extracted_text: str = ""
    pages: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "extracted_text": self.extracted_text,
            "pages": self.pages,
            "tables": self.tables,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
