
from datetime import UTC, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def utc_now():
    return datetime.now(UTC)


class Document(Base):
    __tablename__ = "documents"

    __table_args__ = (
        Index(
            "ix_documents_status_processing_started_at",
            "status",
            "processing_started_at",
        ),
        Index(
            "ix_documents_status_processing_heartbeat_at",
            "status",
            "processing_heartbeat_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="uploaded",
        nullable=False,
    )

    extracted_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    page_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    tables_data: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    errors: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    warnings: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    # all-MiniLM-L6-v2 produces 384-dimensional vectors.
    embedding: Mapped[list | None] = mapped_column(
        Vector(384),
        nullable=True,
    )

    storage_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # Processing lease ownership.
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Last successful processing lease renewal.
    processing_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    processing_token: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    invoices: Mapped[list["Invoice"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    events: Mapped[list["ProcessingEvent"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
        Index(
            "ix_document_chunks_document_index",
            "document_id",
            "chunk_index",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    embedding: Mapped[list | None] = mapped_column(
        Vector(384),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    document: Mapped["Document"] = relationship(
        back_populates="chunks",
    )


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    invoice_number: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    invoice_date: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    supplier_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    customer_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    subtotal: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    tax: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    total: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    currency: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    line_items: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    missing_fields: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    uncertain_fields: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    extraction_notes: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    valid: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    review_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    validation_errors: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    validation_warnings: Mapped[list] = mapped_column(
        JSON,
        default=list,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    document: Mapped["Document"] = relationship(
        back_populates="invoices",
    )

    due_date: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    discount: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    confidence_scores: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    field_evidence: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_invoice_supplier_number",
            "supplier_name",
            "invoice_number",
        ),
    )


class ProcessingEvent(Base):
    __tablename__ = "processing_events"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="uploaded",
        nullable=False,
    )

    message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    details: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    document: Mapped["Document"] = relationship(
        back_populates="events",
    )
