"""Manifest 拆分：document_chunks 独立表

Revision ID: 0007_document_chunks
Revises: 0006_rbac_outbox
Create Date: 2026-11-04

documents.chunk_manifest（JSONB 整列）在大文档 / 百万级文档场景下会被整行
读取放大。改为独立表只存定位与校验字段（chunk_id / content_hash /
section_path / role / embedding_version），不存正文：

- 增量入库按 doc_id 读取清单做 Diff，不必反序列化整列 JSONB；
- 对账任务按 (kb_id, status=ready) 流式扫描，避免一次性加载全部清单。

documents.chunk_manifest 保留（旧数据兼容读取），新写入不再更新该列。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa

revision = "0007_document_chunks"
down_revision = "0006_rbac_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("doc_id", sa.String(length=64), primary_key=True),
        sa.Column("chunk_id", sa.String(length=64), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("section_path", sa.String(length=512), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=True),
        sa.Column("embedding_version", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_document_chunks_doc_index", "document_chunks", ["doc_id", "chunk_index"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_doc_index", table_name="document_chunks")
    op.drop_table("document_chunks")
