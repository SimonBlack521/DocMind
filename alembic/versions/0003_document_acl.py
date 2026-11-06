"""documents acl_groups

Revision ID: 0003_document_acl
Revises: 0002_audit_logs
Create Date: 2026-11-02

"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_document_acl"
down_revision = "0002_audit_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # empty array = public (visible to all groups), keeps existing rows backward compatible
    op.add_column(
        "documents",
        sa.Column(
            "acl_groups",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "acl_groups")
