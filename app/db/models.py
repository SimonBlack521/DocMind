"""元数据表模型：documents、ingest_tasks 与 audit_logs。

三张表的角色（PostgreSQL，检索数据本身在 Milvus/ES，这里只存"档案"）：
- documents    文档元数据主表：doc_id 主键（稳定哈希），记录标题/块数/指向
               Milvus collection 与 ES index 的名字、原始文件在对象存储的 key、
               入库状态机（ingesting -> ready/failed）以及文档级 ACL 字段
               acl_groups（JSONB 数组，空数组 = 公开，全员可见，兼容存量数据）。
               (kb_id, source_uri) 唯一约束保证同库同来源只对应一篇文档；
               内容更新走版本升级（version 递增），重复提交幂等复用。
- ingest_tasks 异步入库任务表：task_id 即 Celery task id，跟踪
               pending -> running -> success/failed 全过程与失败原因，
               供前端轮询任务进度。
- audit_logs   审计表：追加写、不更新不删除，记录"谁在何时对哪个 KB 做了
               什么动作"（上传/删除文档、问答、KB 增删改），detail 是 JSON 字符串。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 文档入库状态
DOC_STATUS_INGESTING = "ingesting"
DOC_STATUS_READY = "ready"
DOC_STATUS_FAILED = "failed"
# 解析质量门禁未通过：保留行与原件，但不写入 Milvus/ES，待人工修复后重传
DOC_STATUS_QUARANTINED = "quarantined"
# 内容与库内已有文档完全一致：只登记血缘，不重复建立索引
DOC_STATUS_DUPLICATE = "duplicate"
# 删除进行中：任何阶段失败都保留该状态，由重试任务继续推进（tombstone）
DOC_STATUS_DELETING = "deleting"

# 入库任务状态
TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_FAILED = "failed"

# 审计动作
AUDIT_ACTION_DOCUMENT_UPLOAD = "document.upload"
AUDIT_ACTION_DOCUMENT_UPLOAD_ASYNC = "document.upload_async"
AUDIT_ACTION_DOCUMENT_DELETE = "document.delete"
AUDIT_ACTION_CHAT_QUERY = "chat.query"
AUDIT_ACTION_KB_CREATE = "kb.create"
AUDIT_ACTION_KB_UPDATE = "kb.update"
AUDIT_ACTION_KB_DELETE = "kb.delete"
AUDIT_ACTION_DEAD_LETTER_REPLAY = "task.dead_letter_replay"


class Base(DeclarativeBase):
    """Alembic 迁移的元数据入口。"""


class DocumentRow(Base):
    """文档元数据：与 Milvus/ES 中的检索数据一一对应（doc_id 稳定）。

    除基础的存储位置与状态外，还承担知识治理职责：
    - 身份：source_id（上游系统原始 ID）、source_type（wiki/ticket/api/...）；
    - 责任：owner（内容责任人）；
    - 一致性：content_hash（归一化正文指纹）用于去重与漂移检测；
    - 时效：source_version（上游版本串）、valid_from/valid_to、collected_at；
    - 权威：authority_level（0-100，参与检索加权）；
    - 血缘：superseded_by（被哪篇替代）、duplicate_of（与哪篇重复）；
    - 可复现：parser_name/parser_version/parse_config_hash + parse_report（质量报告）。
    """

    __tablename__ = "documents"
    __table_args__ = (
        # 同库同来源只对应一篇文档：内容更新走版本升级，不允许同 uri 多 doc_id
        UniqueConstraint("kb_id", "source_uri", name="uq_documents_kb_source"),
        # 同库内容指纹索引：去重查询走索引，避免全表扫描
        Index("ix_documents_kb_content_hash", "kb_id", "content_hash"),
    )

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    source_uri: Mapped[str] = mapped_column(String(1024))
    title: Mapped[str] = mapped_column(String(512), default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    collection: Mapped[str] = mapped_column(String(128), default="")
    es_index: Mapped[str] = mapped_column(String(128), default="")
    raw_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # 文档级 ACL：允许访问的组列表；空数组 = 公开（全员可见），保证存量数据向后兼容
    acl_groups: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default=DOC_STATUS_INGESTING)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ---- 知识治理字段（0004 迁移新增） ----
    # 上游系统原始标识（如工单号 WO-2026-0412、Wiki pageId），非本系统哈希
    source_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # 业务来源类型：file / wiki / ticket / api / table / code / presentation / config
    source_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="file", server_default="file"
    )
    owner: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # 上游版本串（如 v2.3），与内部递增 version 并行存在
    source_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 归一化正文的 sha256：跨来源去重 + 内容漂移检测
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 采集时间（本系统首次/最近一次抓取到该内容的时刻）
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    authority_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 被新版本替代时指向替代者 doc_id；检索默认排除非空者
    superseded_by: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # 与库内已有文档内容重复时指向被复用的 doc_id
    duplicate_of: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 解析链路版本留存：保证"某文档由哪个解析器、哪套参数产出"可追溯
    parser_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parse_config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 解析质量报告（页数/产出率/乱码率/是否通过），JSONB 便于后续扩展指标
    parse_report: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    # 当前版本全部 chunk 的清单（chunk_id/index/content_hash/section_path/role）：
    # 增量入库按它做 Diff 复用未变 chunk 的向量，对账任务按它核对索引完整性。
    # v1.4 起清单写入独立 document_chunks 表，本列仅保留兼容旧数据读取。
    chunk_manifest: Mapped[list[dict]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )


class DocumentChunkRow(Base):
    """chunk 清单行（v1.4 Manifest 拆分）。

    只存定位与校验所需的最小字段（chunk_id / hash / section_path / role），
    不存正文：百万级文档的清单不再压进 documents.chunk_manifest 单个 JSONB，
    避免大文档整行读写放大，也让对账任务可以按 doc_id 流式扫描。
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_doc_index", "doc_id", "chunk_index"),
        # 按知识库 + 版本核对索引 / 流式分页对账走此索引，避免全表扫描
        Index("ix_document_chunks_kb_version", "kb_id", "version"),
    )

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 冗余所属知识库：对账/大规模重建按 kb_id 流式分页时不必回 join documents
    kb_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    # 只有"当前版本"的 chunk 会出现在表里；version 用于排查清单与索引不一致
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    # 章节路径（"第 2 章 / 2.3 报销标准"），与 pipeline 清单字段保持字符串口径
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # 索引状态：indexed=双索引已确认；missing=对账发现索引缺失（修复时整篇重灌后回到 indexed）
    index_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="indexed", server_default="indexed"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class IngestTaskRow(Base):
    """异步入库任务状态：pending → running → success / failed。"""

    __tablename__ = "ingest_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default=TASK_STATUS_PENDING)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 任务创建者（用户 user_id；连接器同步等系统任务为 None）：
    # 任务详情接口据此做属主校验，防止普通用户窥探他人入库任务的细节
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AuditLogRow(Base):
    """审计日志：谁在什么时间对哪个知识库做了什么动作（追加写，不更新不删除）。"""

    __tablename__ = "audit_logs"

    audit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(32), index=True)
    doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # JSON 字符串：动作上下文（如文件名、问题摘要、删除计数）
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DeadLetterEventRow(Base):
    """入库死信：重试耗尽仍失败的事件落此表，人工排查后可按 payload 重放。

    event_id 取任务 id（重试共享同一 task id），重复落信按主键幂等覆盖，
    不会同一次失败堆出多条记录。
    """

    __tablename__ = "dead_letter_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    doc_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 重放所需的原始任务参数（kb_key/filename/source_uri 等）
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# Outbox 事件状态
OUTBOX_STATUS_PENDING = "pending"
# 已被某个 dispatcher 原子领取：其他 dispatcher 不会再拿到同一条事件
OUTBOX_STATUS_DISPATCHING = "dispatching"
OUTBOX_STATUS_DISPATCHED = "dispatched"
OUTBOX_STATUS_FAILED = "failed"

# 事件类型：当前只有异步入库，后续 Connector 同步等复用同一投递通道
OUTBOX_EVENT_INGEST = "ingest.document"

# 投递最大尝试次数：超过后置 failed，由人工/运维重放
OUTBOX_MAX_ATTEMPTS = 5


class OutboxEventRow(Base):
    """Transactional Outbox（v1.4）：与 ingest_tasks 同事务写入，投递失败可重扫。

    解决"PG 已建任务但 Celery 消息未发出"导致的永久 pending：
    - API 在一个事务里同时写任务行与 outbox 事件，提交即代表"事实已发生"；
    - dispatcher 周期扫描 pending 事件补投，投递成功后置 dispatched；
    - 同一 event_id 即 task_id，重复投递由任务侧幂等（doc_id 覆盖写）兜底。
    """

    __tablename__ = "outbox_events"
    __table_args__ = (
        # dispatcher 按 (status, created_at) 扫描，索引让百万级历史事件不拖慢扫描
        Index("ix_outbox_status_created", "status", "created_at"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    # 投递所需的任务参数（JSON 可序列化：只放对象存储 key 与业务字段）
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=OUTBOX_STATUS_PENDING, server_default="pending"
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# 会话消息角色
CONVERSATION_ROLE_USER = "user"
CONVERSATION_ROLE_ASSISTANT = "assistant"


class ConversationRow(Base):
    """多轮会话（v2.0）：一次问答窗口的容器，按 用户 + 知识库 绑定。

    会话是隔离边界：conversation_id 只能被创建它的同租户用户访问，
    追问改写只读取本会话历史，不会跨用户/跨库串上下文。
    """

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_kb", "user_id", "kb_id"),)

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ConversationMessageRow(Base):
    """会话消息：只存问答文本，引用详情仍在 answer 响应/审计里，避免重复存储。"""

    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index("ix_conversation_messages_conv_created", "conversation_id", "created_at"),
    )

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
