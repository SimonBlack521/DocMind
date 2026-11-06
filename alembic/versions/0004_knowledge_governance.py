"""知识治理元数据

Revision ID: 0004_knowledge_governance
Revises: 0003_document_acl
Create Date: 2026-11-03

给 documents 增加治理字段：
- 身份/责任：source_id, source_type, owner
- 一致性：content_hash（归一化正文 sha256）
- 时效：source_version, collected_at, valid_from, valid_to
- 权威与血缘：authority_level, superseded_by, duplicate_of
- 可复现：parser_name, parser_version, parse_config_hash, parse_report

所有新增列都带 server_default 或允许 NULL，存量数据无需回填即可上线。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_knowledge_governance"
down_revision = "0003_document_acl"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_id", sa.String(200), nullable=True))
    op.add_column(
        "documents",
        sa.Column("source_type", sa.String(32), nullable=False, server_default="file"),
    )
    op.add_column("documents", sa.Column("owner", sa.String(200), nullable=True))
    op.add_column("documents", sa.Column("source_version", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("content_hash", sa.String(64), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "collected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.add_column(
        "documents",
        sa.Column("authority_level", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("documents", sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("superseded_by", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("duplicate_of", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("parser_name", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("parser_version", sa.String(32), nullable=True))
    op.add_column("documents", sa.Column("parse_config_hash", sa.String(64), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "parse_report",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )
    # 去重查询：同库按内容指纹定位
    op.create_index(
        "ix_documents_kb_content_hash", "documents", ["kb_id", "content_hash"], unique=False
    )
    op.create_index("ix_documents_superseded_by", "documents", ["superseded_by"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_documents_superseded_by", table_name="documents")
    op.drop_index("ix_documents_kb_content_hash", table_name="documents")
    op.drop_column("documents", "parse_report")
    op.drop_column("documents", "parse_config_hash")
    op.drop_column("documents", "parser_version")
    op.drop_column("documents", "parser_name")
    op.drop_column("documents", "duplicate_of")
    op.drop_column("documents", "superseded_by")
    op.drop_column("documents", "valid_to")
    op.drop_column("documents", "valid_from")
    op.drop_column("documents", "authority_level")
    op.drop_column("documents", "collected_at")
    op.drop_column("documents", "content_hash")
    op.drop_column("documents", "source_version")
    op.drop_column("documents", "owner")
    op.drop_column("documents", "source_type")
    op.drop_column("documents", "source_id")
