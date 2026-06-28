"""add_silver_gold_promotion_fields_to_dataset_versions

Revision ID: b4e7d1f93a20
Revises: a8f3c2e91d04
Create Date: 2026-06-27 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b4e7d1f93a20"
down_revision: Union[str, Sequence[str], None] = "a8f3c2e91d04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "dataset_versions",
        sa.Column("silver_path", sa.Text(), nullable=True),
    )
    op.add_column(
        "dataset_versions",
        sa.Column("gold_path", sa.Text(), nullable=True),
    )
    op.add_column(
        "dataset_versions",
        sa.Column("promoted_to_silver_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "dataset_versions",
        sa.Column("promoted_to_gold_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("dataset_versions", "promoted_to_gold_at")
    op.drop_column("dataset_versions", "promoted_to_silver_at")
    op.drop_column("dataset_versions", "gold_path")
    op.drop_column("dataset_versions", "silver_path")
