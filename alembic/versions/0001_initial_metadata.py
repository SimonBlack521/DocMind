"""初始元数据表：documents 与 ingest_tasks

Revision ID: 0001_initial_metadata
Revises:
Create Date: 2026-10-31

"""

from __future__ import annotations

import sqlalchemy as sa
import alembic.op as op

revision = "0001_initial_metadata"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(length=64), primary_key=True),
        sa.Column("kb_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("source_uri", sa.String(length=1024), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("collection", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("es_index", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("raw_object_key", sa.String(length=1024), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="ingesting"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("kb_id", "source_uri", name="uq_documents_kb_source"),
    )
    op.create_index("ix_documents_kb_id", "documents", ["kb_id"])
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    op.create_table(
        "ingest_tasks",
        sa.Column("task_id", sa.String(length=64), primary_key=True),
        sa.Column("doc_id", sa.String(length=64), nullable=True),
        sa.Column("kb_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ingest_tasks_doc_id", "ingest_tasks", ["doc_id"])
    op.create_index("ix_ingest_tasks_kb_id", "ingest_tasks", ["kb_id"])


def downgrade() -> None:
    op.drop_table("ingest_tasks")
    op.drop_table("documents")
