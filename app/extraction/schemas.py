
from typing import Optional

from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    description: Optional[str] = Field(
        default=None,
        description="Description of the product or service"
    )
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class InvoiceExtraction(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = Field(
        default=None,
        description="Invoice date in YYYY-MM-DD format when unambiguous"
    )
    supplier_name: Optional[str] = None
    customer_name: Optional[str] = None

    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None

    line_items: list[InvoiceLineItem] = Field(default_factory=list)

    missing_fields: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
