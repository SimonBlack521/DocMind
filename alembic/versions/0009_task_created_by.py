"""入库任务表增加创建者列（RBAC 任务属主校验）

Revision ID: 0009_task_created_by
Revises: 0008_conversations
Create Date: 2026-11-05

- ingest_tasks.created_by：任务创建者 user_id（连接器同步等系统任务为 NULL），
  任务详情接口据此做属主校验——非创建者需具备贡献者及以上角色才可查看，
  防止普通用户通过任务详情窥探他人上传的 doc_id / 错误信息。
"""

from __future__ import annotations

import alembic.op as op
import sqlalchemy as sa

revision = "0009_task_created_by"
down_revision = "0008_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingest_tasks",
        sa.Column("created_by", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingest_tasks", "created_by")
