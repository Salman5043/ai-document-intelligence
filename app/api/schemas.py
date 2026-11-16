

from pydantic import BaseModel, ConfigDict, Field


class InvoiceCorrection(BaseModel):
    invoice_number: str | None = None
    invoice_date: str | None = None
    supplier_name: str | None = None
    customer_name: str | None = None

    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None

    line_items: list[dict] | None = None


class DocumentStatusResponse(BaseModel):
    id: int
    filename: str
    status: str
    errors: list = Field(default_factory=list)
    warnings: list = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
