"""document_chunks 补 kb_id / index_status / created_at（清单第二优先级第 10 项）

Revision ID: 0010_document_chunks_kb
Revises: 0009_task_created_by
Create Date: 2026-11-05

- kb_id：冗余所属知识库，对账/大规模重建按 kb_id 流式分页时不必回 join documents；
  既有行从 documents 表回填后置 NOT NULL。
- index_status：索引状态（indexed / missing），对账任务发现缺失时置 missing，
  修复重灌（清单先删后插）自动回到 indexed。
- created_at：行创建时间，支撑按时间排查与留存审计。
- 新增 (kb_id, version) 索引：按版本核对索引、按知识库流式扫描都走它。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa

revision = "0010_document_chunks_kb"
down_revision = "0009_task_created_by"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_chunks", sa.Column("kb_id", sa.String(length=64), nullable=True)
    )
    # 既有行回填：从 documents 主表按 doc_id 取所属知识库
    op.execute(
        "UPDATE document_chunks SET kb_id = "
        "(SELECT kb_id FROM documents WHERE documents.doc_id = document_chunks.doc_id)"
    )
    op.alter_column("document_chunks", "kb_id", nullable=False)
    op.add_column(
        "document_chunks",
        sa.Column(
            "index_status",
            sa.String(length=16),
            nullable=False,
            server_default="indexed",
        ),
    )
    op.add_column(
        "document_chunks",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_document_chunks_kb_version", "document_chunks", ["kb_id", "version"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_kb_version", table_name="document_chunks")
    op.drop_column("document_chunks", "created_at")
    op.drop_column("document_chunks", "index_status")
    op.drop_column("document_chunks", "kb_id")
