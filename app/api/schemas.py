
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class InvoiceCorrection(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    supplier_name: Optional[str] = None
    customer_name: Optional[str] = None

    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None

    line_items: Optional[list[dict]] = None


class DocumentStatusResponse(BaseModel):
    id: int
    filename: str
    status: str
    errors: list = Field(default_factory=list)
    warnings: list = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
