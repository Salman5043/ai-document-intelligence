
from typing import Any, Optional, TypedDict

from app.extraction.schemas import InvoiceExtraction


class InvoiceExtractionState(TypedDict):
    document: dict[str, Any]
    document_text: str
    extraction: Optional[InvoiceExtraction]

    # Previously processed invoices for duplicate checking
    existing_invoices: list[dict[str, Any]]

    # Validation result
    validation: Optional[dict[str, Any]]

    errors: list[str]
    warnings: list[str]
