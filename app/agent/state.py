
from typing import Any, TypedDict

from app.extraction.schemas import InvoiceExtraction


class InvoiceExtractionState(TypedDict):
    document: dict[str, Any]
    document_text: str
    extraction: InvoiceExtraction | None

    # Previously processed invoices for duplicate checking
    existing_invoices: list[dict[str, Any]]

    # Validation result
    validation: dict[str, Any] | None

    errors: list[str]
    warnings: list[str]
