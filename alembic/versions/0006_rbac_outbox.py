"""v1.4 加固：Transactional Outbox

Revision ID: 0006_rbac_outbox
Revises: 0005_incremental_indexing
Create Date: 2026-11-04

- outbox_events：与 ingest_tasks 同事务写入的待投递事件，dispatcher 周期补投，
  解决"PG 已建任务但 broker 消息未发出"导致的永久 pending。

outbox_events 为新增表，存量数据无需回填。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_rbac_outbox"
down_revision = "0005_incremental_indexing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "outbox_events",
        sa.Column("event_id", sa.String(length=64), primary_key=True),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("kb_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_outbox_events_event_type", "outbox_events", ["event_type"])
    op.create_index("ix_outbox_events_kb_id", "outbox_events", ["kb_id"])
    op.create_index(
        "ix_outbox_status_created", "outbox_events", ["status", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_outbox_status_created", table_name="outbox_events")
    op.drop_index("ix_outbox_events_kb_id", table_name="outbox_events")
    op.drop_index("ix_outbox_events_event_type", table_name="outbox_events")
    op.drop_table("outbox_events")
