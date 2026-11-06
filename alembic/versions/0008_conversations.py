"""多轮会话表（v2.0）

Revision ID: 0008_conversations
Revises: 0007_document_chunks
Create Date: 2026-11-04

- conversations：会话容器，按 租户 + 知识库 + 用户 绑定，是上下文隔离边界；
- conversation_messages：问答消息，追问改写只读取最近 N 条，限制 prompt 长度。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa

revision = "0008_conversations"
down_revision = "0007_document_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(length=64), primary_key=True),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("kb_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_conversations_tenant_id", "conversations", ["tenant_id"])
    op.create_index("ix_conversations_kb_id", "conversations", ["kb_id"])
    op.create_index("ix_conversations_user_kb", "conversations", ["user_id", "kb_id"])

    op.create_table(
        "conversation_messages",
        sa.Column("message_id", sa.String(length=64), primary_key=True),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_conversation_messages_conv_created",
        "conversation_messages",
        ["conversation_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_conv_created", table_name="conversation_messages"
    )
    op.drop_table("conversation_messages")
    op.drop_index("ix_conversations_user_kb", table_name="conversations")
    op.drop_index("ix_conversations_kb_id", table_name="conversations")
    op.drop_index("ix_conversations_tenant_id", table_name="conversations")
    op.drop_table("conversations")
