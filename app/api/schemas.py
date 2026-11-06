"""API 层出入参模型：只暴露必要字段，避免把内部 metadata 直接透给前端。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.repo import AuditLogInfo, DocumentInfo
from app.domain.evidence import Evidence
from app.domain.kb_config import (
    ChunkingConfig,
    EmbeddingProfile,
    KBConfig,
    RetrievalConfig,
)
from app.generation.citations import Citation


class RetrievalSummary(BaseModel):
    """检索策略摘要（对外只读视图）：混合检索权重、RRF 融合参数、改写/精排开关。"""

    top_k: int
    rerank_top_n: int
    rrf_k: int
    vector_weight: float
    bm25_weight: float
    use_query_rewrite: bool
    use_rerank: bool

    @classmethod
    def from_config(cls, config: RetrievalConfig) -> RetrievalSummary:
        """从内部 RetrievalConfig 抽取对外可暴露的字段。"""
        return cls(
            top_k=config.top_k,
            rerank_top_n=config.rerank_top_n,
            rrf_k=config.rrf_k,
            vector_weight=config.vector_weight,
            bm25_weight=config.bm25_weight,
            use_query_rewrite=config.use_query_rewrite,
            use_rerank=config.use_rerank,
        )


class KBSummary(BaseModel):
    """知识库对外摘要：把内部 KBConfig 拍平成前端友好的结构
    （如 milvus_collection → collection、chunking.size → chunk_size）。"""

    kb_id: str
    kb_code: str
    name: str
    description: str
    sample_questions: list[str] = Field(default_factory=list)
    tenant_id: str
    enabled: bool
    collection: str
    es_index: str
    es_alias: str
    chunk_size: int
    chunk_overlap: int
    retrieval: RetrievalSummary

    @classmethod
    def from_config(cls, kb: KBConfig) -> KBSummary:
        """从内部 KBConfig 构造对外摘要。"""
        return cls(
            kb_id=kb.kb_id,
            kb_code=kb.kb_code,
            name=kb.name,
            description=kb.description,
            sample_questions=list(kb.sample_questions),
            tenant_id=kb.tenant_id,
            enabled=kb.enabled,
            collection=kb.milvus_collection,
            es_index=kb.es_index,
            es_alias=kb.es_alias,
            chunk_size=kb.chunking.size,
            chunk_overlap=kb.chunking.overlap,
            retrieval=RetrievalSummary.from_config(kb.retrieval),
        )


class KBWriteRequest(BaseModel):
    """知识库可写字段；kb_code / kb_id 由标识决定，不允许运行时修改。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    # 前端欢迎页示例问题（纯界面引导，最多 8 条）
    sample_questions: list[str] = Field(default_factory=list, max_length=8)
    # 留空表示当前用户所在租户；显式传其他租户会被拒绝（防跨租户建库）
    tenant_id: str | None = Field(default=None, max_length=64)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingProfile = Field(default_factory=EmbeddingProfile)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    enabled: bool = True


class KBCreateRequest(KBWriteRequest):
    """创建知识库请求：在可写字段之外增加标识字段（创建后不可变）。"""

    kb_code: str = Field(
        description="集合/索引命名依据，只允许小写字母开头，后接小写字母/数字/_/-，长度 2-32"
    )
    kb_id: str | None = Field(
        default=None, max_length=64, description="留空自动生成 kb-{kb_code}"
    )


class KBUpdateRequest(KBWriteRequest):
    """更新知识库请求：与创建共用可写字段，标识字段由路径上的 kb_key 决定。"""


class KBDeleteResponse(BaseModel):
    """删除知识库的响应：回显被删标识便于前端确认。"""

    kb_id: str
    kb_code: str


class ConnectorSummary(BaseModel):
    """连接器配置摘要（v2.0）。"""

    connector_id: str
    type: str
    enabled: bool
    path: str
    owner: str | None = None
    authority_level: int = 0


class ConnectorSyncResponse(BaseModel):
    """连接器同步触发响应：返回任务 id，前端可按 /tasks/{id} 轮询。"""

    task_id: str | None = None
    connector_id: str
    status: str


class ConnectorWebhookPayload(BaseModel):
    """连接器 Webhook 推送（v2.0）：源系统主动上报变更/删除的 source_uri 列表。

    source_uris 里的变更文档通过触发一次全量幂等同步收口（连接器协议本身就是
    全量扫描 + 内容指纹幂等，Webhook 只负责"提前唤醒"）；deleted_uris 直接走
    统一删除状态机，无需等待下一次全量扫描。
    """

    source_uris: list[str] = Field(default_factory=list)
    deleted_uris: list[str] = Field(default_factory=list)


class ConnectorWebhookResponse(BaseModel):
    """Webhook 受理结果：同步任务 id（有变更时）+ 删除文档数。"""

    connector_id: str
    task_id: str | None = None
    deleted: int = 0
    status: str


class DocumentIngestResponse(BaseModel):
    """同步入库响应：返回稳定 doc_id、切块数、存储位置与治理裁决结果。

    reused=True 表示命中"内容未变 / 跨来源重复"，本次没有重新建索引；
    duplicate_of 非空时指向库内已有同内容文档的 doc_id。
    """

    doc_id: str
    source_uri: str
    title: str
    chunk_count: int
    collection: str
    index: str
    raw_object_key: str | None = None
    version: int = 1
    content_hash: str = ""
    reused: bool = False
    duplicate_of: str | None = None
    # 解析质量报告：页数、页产出率、乱码率、是否通过门禁
    quality: dict[str, Any] = Field(default_factory=dict)


class DocumentDeleteResponse(BaseModel):
    """删除文档响应：回显向量库与稀疏索引各自删掉的片段数。"""

    doc_id: str
    deleted_vectors: int
    deleted_keywords: int


class DocumentSummary(BaseModel):
    """文档元数据摘要（来自 PG documents 表），含治理字段。"""

    doc_id: str
    source_uri: str
    title: str
    status: str
    chunk_count: int
    version: int
    # 文档级 ACL：空列表 = 公开（全员可见）
    acl_groups: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    # ---- 知识治理字段 ----
    source_id: str | None = None
    source_type: str = "file"
    owner: str | None = None
    source_version: str | None = None
    content_hash: str | None = None
    authority_level: int = 0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    superseded_by: str | None = None
    duplicate_of: str | None = None
    collected_at: datetime | None = None
    parse_report: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_info(cls, info: DocumentInfo) -> DocumentSummary:
        """从仓储层的 DocumentInfo 记录构造对外模型。"""
        return cls(
            doc_id=info.doc_id,
            source_uri=info.source_uri,
            title=info.title,
            status=info.status,
            chunk_count=info.chunk_count,
            version=info.version,
            acl_groups=info.acl_groups,
            created_at=info.created_at,
            updated_at=info.updated_at,
            source_id=info.source_id,
            source_type=info.source_type,
            owner=info.owner,
            source_version=info.source_version,
            content_hash=info.content_hash,
            authority_level=info.authority_level,
            valid_from=info.valid_from,
            valid_to=info.valid_to,
            superseded_by=info.superseded_by,
            duplicate_of=info.duplicate_of,
            collected_at=info.collected_at,
            parse_report=info.parse_report,
        )


class DocumentListResponse(BaseModel):
    """文档分页响应：total 便于前端渲染分页器。"""

    total: int
    page: int
    size: int
    items: list[DocumentSummary]


class AuditLogEntry(BaseModel):
    """单条审计日志（来自 PG audit_logs 表）。"""

    audit_id: str
    action: str
    user_id: str
    doc_id: str | None
    detail: dict[str, Any] | None
    created_at: datetime

    @classmethod
    def from_info(cls, info: AuditLogInfo) -> AuditLogEntry:
        """从仓储层的 AuditLogInfo 记录构造对外模型。"""
        return cls(
            audit_id=info.audit_id,
            action=info.action,
            user_id=info.user_id,
            doc_id=info.doc_id,
            detail=info.detail,
            created_at=info.created_at,
        )


class AuditLogListResponse(BaseModel):
    """审计日志分页响应。"""

    total: int
    page: int
    size: int
    items: list[AuditLogEntry]


class AsyncIngestResponse(BaseModel):
    """异步入库受理结果；任务实时状态走 GET /tasks/{task_id}。"""

    task_id: str
    status: str


class OwnerPurgeResponse(BaseModel):
    """按责任人批量删除的汇总结果（合规场景：离职、授权撤回、数据主体删除请求）。"""

    owner: str
    deleted_documents: int
    deleted_vectors: int
    deleted_keywords: int


class TaskStatusResponse(BaseModel):
    """异步入库任务状态：doc_id 在成功后才回填，error 在失败时给出原因。"""

    task_id: str
    status: str
    kb_id: str
    doc_id: str | None = None
    error: str | None = None


class DeadLetterSummary(BaseModel):
    """入库死信摘要：payload 保留重放所需的原始任务参数。"""

    event_id: str
    kb_id: str
    doc_id: str | None = None
    error: str | None = None
    created_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class DeadLetterListResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[DeadLetterSummary]


class DeadLetterReplayResponse(BaseModel):
    """死信重放结果：重放生成新任务（task_id 与 event_id 不同）。"""

    event_id: str
    task_id: str
    status: str


class ChatRequest(BaseModel):
    """问答请求：top_k / rewrite / rerank 传 None 表示沿用知识库配置默认值。"""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    rewrite: bool | None = None
    rerank: bool | None = None
    # 多轮会话 id：传入则加载历史做追问改写，缺省/为空表示单轮问答
    conversation_id: str | None = Field(default=None, max_length=64)
    # v2.0 查询路由：auto=按问题语义自动分流；document/structured=强制指定链路
    mode: Literal["auto", "document", "structured"] = "auto"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """拒绝纯空白问题，并统一去掉首尾空白再进入下游。"""
        text = value.strip()
        if not text:
            raise ValueError("question 不能为空")
        return text


class CitationModel(BaseModel):
    """引用来源（答案中 [n] 标记对应的片段）：含原文片段与相关性得分。"""

    index: int
    chunk_id: str
    doc_id: str
    source_uri: str
    title: str
    snippet: str
    score: float

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationModel:
        """从生成层的 Citation 领域对象构造对外模型。"""
        return cls(
            index=citation.index,
            chunk_id=citation.chunk_id,
            doc_id=citation.doc_id,
            source_uri=citation.source_uri,
            title=citation.title,
            snippet=citation.snippet,
            score=citation.score,
        )


class EvidenceModel(BaseModel):
    """统一证据（对外视图）：这条结论来自文档 / 数据库 / 实时接口中的哪一个。

    locator 是证据定位（chunk_id / 可执行 SQL / endpoint 路径），snippet 是进入
    答案的最小片段，observed_at 是观测时间——前端据此渲染"来源可追溯"面板。
    """

    source_type: Literal["document", "database", "api"]
    source_id: str
    locator: str
    snippet: str
    observed_at: datetime
    version: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> EvidenceModel:
        """从领域层 Evidence 构造对外模型。"""
        return cls(
            source_type=evidence.source_type,
            source_id=evidence.source_id,
            locator=evidence.locator,
            snippet=evidence.snippet,
            observed_at=evidence.observed_at,
            version=evidence.version,
            provenance=dict(evidence.provenance),
        )


class AnswerVerificationModel(BaseModel):
    """独立答案验证状态：结论是否有实际送入模型的证据支持。"""

    status: str
    claims_total: int
    cited_claims: int
    coverage: float
    reason: str = ""


class ChatResponse(BaseModel):
    """非流式问答响应：完整答案 + 引用列表 + 统一证据（流式形态见 chat.py 的 SSE 事件）。"""

    kb_id: str
    answer: str
    model: str
    retrieved_count: int
    citations: list[CitationModel]
    # 单轮问答也会分配会话 id，前端把它带回下一轮即可续上上下文
    conversation_id: str | None = None
    # v2.0 查询路由：document / sql / api；sql 路由时 sql 字段返回实际执行的只读查询
    route: str = "document"
    sql: str | None = None
    # v2.1 统一证据：document / database / api 来源可追溯（含定位与观测时间）
    evidence: list[EvidenceModel] = Field(default_factory=list)
    # 降级标记：结构化/实时查询失败时若明确告知（而非静默换文档答案），置 True
    degraded: bool = False
    degrade_reason: str = ""
    # v2.2 答案验证：degraded=0 只说明未触发降级路径，不代表答案结论有证据支持；
    # 前端应据 verification.status 标注"未验证 / 已拒答"，而不是只看 degraded
    verification: AnswerVerificationModel | None = None


class ConversationMessageModel(BaseModel):
    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    """会话历史（v2.0）：按时间正序返回，仅本人 / 同知识库可读。"""

    conversation_id: str
    kb_id: str
    title: str | None = None
    messages: list[ConversationMessageModel]


class HealthResponse(BaseModel):
    """健康检查快照：整体状态 + 各基础设施依赖的细粒度状态。"""

    status: str  # ok / degraded / down
    app_env: str
    auth_mode: str
    kb_count: int
    # 基础设施细粒度状态；skipped 表示该服务未配置（如禁用缓存 / 未挂载 S3）
    services: dict[str, str] = Field(default_factory=dict)
