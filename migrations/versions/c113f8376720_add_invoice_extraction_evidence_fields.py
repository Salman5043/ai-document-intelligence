
"""add invoice extraction evidence fields

Revision ID: c113f8376720
Revises: 15e34abf93f1
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c113f8376720"
down_revision: Union[str, Sequence[str], None] = "15e34abf93f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable fields can safely be added to existing records.
    op.add_column(
        "invoices",
        sa.Column("due_date", sa.String(length=20), nullable=True),
    )

    op.add_column(
        "invoices",
        sa.Column(
            "discount",
            sa.Numeric(precision=14, scale=2),
            nullable=True,
        ),
    )

    # Initialize existing records with empty JSON objects.
    op.add_column(
        "invoices",
        sa.Column(
            "confidence_scores",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )

    op.add_column(
        "invoices",
        sa.Column(
            "field_evidence",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )

    # Remove database defaults after existing rows are populated.
    op.alter_column(
        "invoices",
        "confidence_scores",
        server_default=None,
    )

    op.alter_column(
        "invoices",
        "field_evidence",
        server_default=None,
    )


def downgrade() -> None:
    op.drop_column("invoices", "field_evidence")
    op.drop_column("invoices", "confidence_scores")
    op.drop_column("invoices", "discount")
    op.drop_column("invoices", "due_date")
