
from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    description: str | None = Field(
        default=None,
        description="Description of the product or service",
    )

    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class InvoiceExtraction(BaseModel):
    invoice_number: str | None = None

    invoice_date: str | None = Field(
        default=None,
        description="Invoice date in YYYY-MM-DD format when unambiguous",
    )

    due_date: str | None = Field(
        default=None,
        description="Payment due date in YYYY-MM-DD format when available",
    )

    supplier_name: str | None = None
    customer_name: str | None = None

    subtotal: float | None = None
    tax: float | None = None

    discount: float | None = Field(
        default=None,
        description="Invoice discount amount, if explicitly stated",
    )

    total: float | None = None
    currency: str | None = None

    line_items: list[InvoiceLineItem] = Field(
        default_factory=list
    )

    # Extraction quality information.
    # Values should be between 0.0 and 1.0.
    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Field-level extraction confidence scores",
    )

    # Supporting text copied from the source document.
    # Do not generate evidence that is not present in the document.
    field_evidence: dict[str, str] = Field(
        default_factory=dict,
        description="Source text supporting extracted field values",
    )

    missing_fields: list[str] = Field(
        default_factory=list
    )

    uncertain_fields: list[str] = Field(
        default_factory=list
    )

    extraction_notes: list[str] = Field(
        default_factory=list
    )
