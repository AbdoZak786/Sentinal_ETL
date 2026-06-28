"""widen_dataset_versions_status

Revision ID: d9a4e6b12c78
Revises: c3f8a2b41d05
Create Date: 2026-06-28 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "d9a4e6b12c78"
down_revision: Union[str, Sequence[str], None] = "c3f8a2b41d05"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "dataset_versions",
        "status",
        existing_type=sa.String(length=10),
        type_=sa.String(length=30),
        existing_nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "dataset_versions",
        "status",
        existing_type=sa.String(length=30),
        type_=sa.String(length=10),
        existing_nullable=False,
    )
