
"""add document processing heartbeat

Revision ID: 15e34abf93f1
Revises: 6c4a2f8d91b3
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "15e34abf93f1"
down_revision: str | Sequence[str] | None = "6c4a2f8d91b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "processing_heartbeat_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Preserve the lease age for documents already processing.
    op.execute(
        """
        UPDATE documents
        SET processing_heartbeat_at = processing_started_at
        WHERE status = 'processing'
          AND processing_started_at IS NOT NULL
        """
    )

    op.create_index(
        "ix_documents_status_processing_heartbeat_at",
        "documents",
        ["status", "processing_heartbeat_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_status_processing_heartbeat_at",
        table_name="documents",
    )

    op.drop_column("documents", "processing_heartbeat_at")
