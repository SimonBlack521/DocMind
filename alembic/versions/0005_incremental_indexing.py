"""增量索引元数据：chunk_manifest + dead_letter_events

Revision ID: 0005_incremental_indexing
Revises: 0004_knowledge_governance
Create Date: 2026-11-04

- documents.chunk_manifest：该文档当前版本全部 chunk 的清单
  （chunk_id / chunk_index / content_hash / section_path / role），
  增量入库时与新版清单做 Diff，未变 chunk 复用旧向量、不重算 Embedding；
  对账任务（reconciler）也以它为"应该存在"的权威清单。
- dead_letter_events：入库重试耗尽后的死信记录（event_id 幂等），
  人工排查后可按 payload 重放，避免阻塞消费链路又杜绝静默丢数据。

所有新增列都带 server_default，存量数据无需回填。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_incremental_indexing"
down_revision = "0004_knowledge_governance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "chunk_manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
    )
    op.create_table(
        "dead_letter_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("kb_id", sa.String(64), nullable=False, index=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("doc_id", sa.String(64), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("dead_letter_events")
    op.drop_column("documents", "chunk_manifest")
