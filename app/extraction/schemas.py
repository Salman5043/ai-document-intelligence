

from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    description: str | None = Field(
        default=None,
        description="Description of the product or service"
    )
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class InvoiceExtraction(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = Field(
        default=None,
        description="Invoice date in YYYY-MM-DD format when unambiguous"
    )
    supplier_name: str | None = None
    customer_name: str | None = None

    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None

    line_items: list[InvoiceLineItem] = Field(default_factory=list)

    missing_fields: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
