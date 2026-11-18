
"""add document processing lease

Revision ID: 6c4a2f8d91b3
Revises: 741fb32bc809
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6c4a2f8d91b3"
down_revision: str | Sequence[str] | None = "741fb32bc809"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "processing_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "processing_token",
            sa.String(length=36),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_documents_status_processing_started_at",
        "documents",
        ["status", "processing_started_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_status_processing_started_at",
        table_name="documents",
    )

    op.drop_column("documents", "processing_token")
    op.drop_column("documents", "processing_started_at")
