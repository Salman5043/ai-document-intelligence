
from typing import Any

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


class DocumentSearchResult(BaseModel):
    document_id: int
    filename: str
    status: str


class SemanticDocumentSearchResult(DocumentSearchResult):
    distance: float
    similarity_score: float | None = None


class ChunkSearchResult(SemanticDocumentSearchResult):
    chunk_id: int
    chunk_index: int
    content: str
    source: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class DocumentQuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    retrieval_limit: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class DocumentQuestionResponse(BaseModel):
    document_id: int
    question: str
    answer: str
    insufficient_evidence: bool
    sources: list[dict[str, Any]] = Field(default_factory=list)