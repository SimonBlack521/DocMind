"""入库与问答编排：API 层只调用这里，不直接接触存储和模型细节。

入库链路（ingest_bytes → _ingest）：
    解析（DocumentLoader 按文件后缀选解析器，产出纯文本）
    → 切块（TextChunker 按知识库 chunking 配置切分，文档级 ACL 冗余进每个 chunk 的 metadata）
    → Manifest Diff（与上一版 chunk 清单对比：未变 chunk 从 Milvus 取回旧向量复用，
      只对新增/修改的 chunk 调 Embedding，省算力；复用要求 embedding_version 一致）
    → 先写新版（双写 Milvus + ES）→ 再按旧清单精确删除失效 chunk_id
      （不是先删后写：新旧交替窗口内检索始终有完整版本可用，消除半更新窗口）
    → 失效本知识库的检索缓存 → 原始文件归档 S3（失败仅告警，不影响检索）
    → 产出新版 chunk_manifest，由调用方（API/Celery）随 ready 状态落 PG
    兜底：切块为空抛 DocumentParseError；Embedding/存储异常向上传播并回滚本次
    新写入的 chunk（旧版本不受影响），由 Celery 任务层重试并做状态落库。

问答链路（query / query_stream → _retrieve → HybridRetriever → RAGChain）：
    鉴权在 API 层完成，用户组以 allowed_groups 传入
    → 检索缓存（命中直接返回 chunks；key 含用户组，杜绝跨组越权）
    → 查询改写（可选，失败回退原查询）
    → 向量 + BM25 双路并行召回（单路失败不拖垮另一路）
    → RRF 加权融合 → 精排（可选，服务不可用回退 RRF 顺序）
    → LLM 生成答案（无候选片段时不调用 LLM，直接返回固定文案）
    → 引用列表与提示词中的编号 [n] 一一对应
    审计：问答审计日志由 API 层（app/api/routers/chat.py）写入，不在本模块。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

from app.core.errors import AppError, DocumentParseError, DocumentQualityError, ObjectStoreError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.settings import Settings
from app.db.repo import MetadataRepo, acl_visible
from app.db.session import build_engine, build_session_factory
from app.domain.dto import RetrievedChunk
from app.domain.evidence import Evidence
from app.domain.kb_config import KBConfig
from app.generation.chain import AnswerResult, RAGChain
from app.generation.citation_validator import AnswerVerification
from app.generation.citations import Citation
from app.generation.compression import compress_chunks
from app.generation.prompts import LOW_CONFIDENCE_ANSWER, STRUCTURED_UNAVAILABLE_ANSWER
from app.ingestion.chunker import Chunk, TextChunker
from app.ingestion.loader import DocumentLoader, LoadedDocument
from app.ingestion.ocr import build_ocr_engine
from app.ingestion.quality import assess_quality, with_chunk_count
from app.orchestration.agentic import AgenticOutcome, AgenticRAG, StructuredTerminal
from app.orchestration.registry import KBRuntime, RuntimeRegistry
from app.retrieval.cache import RetrievalCache
from app.retrieval.conversation import ConversationQueryRewriter
from app.retrieval.embedding_cache import QueryEmbeddingCache
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_classifier import QueryClassifier
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.query_router import QueryRouter, RequestedMode
from app.storage.keyword_base import KeywordRecord
from app.storage.object_store import ObjectStore, build_document_key
from app.storage.vector_base import VectorRecord, merge_lineage_acl, normalize_acl_groups
from app.structured.api_retriever import ApiAnswer, ApiRetriever
from app.structured.retriever import StructuredAnswer, StructuredRetriever

logger = get_logger(__name__)

_T = TypeVar("_T")


def _run_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    """在同步上下文（API 线程池 / Celery worker）里桥接 async 元数据查询。

    调用线程必须没有运行中的事件循环（asyncio.run 语义）；若在 async 上下文
    被误用会抛 RuntimeError，由调用方捕获降级，不阻断主链路。
    """
    return asyncio.run(coro)


def _query_lineage(
    settings: Settings, kb: KBConfig, content_hashes: list[str]
) -> dict[str, list[Any]]:
    """在独立事件循环内查询血缘文档。

    asyncpg 连接池绑定创建它的事件循环；API 在线程池（run_in_threadpool）调用
    同步 pipeline 时，复用主循环的 engine 会抛 "Future attached to a different loop"
    （真实集成测试中每次查询都失败）。这里每次新建 engine 并在同一 loop 内释放：
    血缘归属是可选增强、每次检索至多查一次，连接建立开销远小于跨循环错误的代价。
    """

    async def _run() -> dict[str, list[Any]]:
        engine = build_engine(settings.database_url)
        try:
            repo = MetadataRepo(build_session_factory(engine))
            return await repo.list_lineage_documents(
                kb_id=kb.kb_id, tenant_id=kb.tenant_id, content_hashes=content_hashes
            )
        finally:
            await engine.dispose()

    return _run_coro(_run())


@dataclass(frozen=True, slots=True)
class IngestResult:
    """入库结果：返回给任务层做状态落库（chunk 数、集合/索引名、归档对象的 key）。

    reused=True 表示内容未变化、跳过了重新索引；duplicate_of 非空表示同库已有
    相同内容的文档，本次只登记血缘、不重复建索引。
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
    # 解析质量报告（页数/产出率/乱码率），随文档落库便于事后审计
    quality: dict[str, object] = field(default_factory=dict)
    # 采集时间：与写入 chunk metadata 的时间戳保持一致，供时效性加权使用
    collected_at: datetime | None = None
    reused: bool = False
    duplicate_of: str | None = None
    # 本次版本的 chunk 清单（含复用与新算的）：调用方随 ready 状态落库，
    # 下次入库按它做增量 Diff，对账任务按它核对索引完整性
    chunk_manifest: list[dict[str, object]] = field(default_factory=list)


# 治理字段 → chunk metadata 的映射：检索阶段的权威度/时效加权、来源筛选都读这里。
# 时间字段统一转 ISO 字符串，保证 Milvus JSON 与 ES metadata 可直接序列化。
_GOVERNANCE_CHUNK_KEYS = (
    "source_id",
    "source_type",
    "owner",
    "source_version",
    "authority_level",
    "valid_from",
    "valid_to",
)


def _chunk_governance_metadata(governance: Mapping[str, object] | None) -> dict[str, object]:
    """把文档级治理字段整理成可写入 chunk metadata 的 JSON 友好结构。"""
    if not governance:
        return {}
    payload: dict[str, object] = {}
    for key in _GOVERNANCE_CHUNK_KEYS:
        value = governance.get(key)
        if value is None:
            continue
        payload[key] = value.isoformat() if isinstance(value, datetime) else value
    return payload


@dataclass(frozen=True, slots=True)
class DeleteResult:
    """删除结果：分别记录向量库与关键词库实际删掉的条数，便于审计与排查。"""

    doc_id: str
    deleted_vectors: int
    deleted_keywords: int


@dataclass(slots=True)
class QueryResult:
    """一次性问答结果：答案文本 + 引用列表；chunks 保留给上层做审计与调试。"""

    answer: str
    citations: list[Citation]
    model: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    # v2.0 查询路由：document=文档检索，sql=结构化查询；sql 字段仅在 sql 路由时非空
    route: str = "document"
    sql: str | None = None
    # v2.1 统一证据：文档/SQL/API 共用同一结构，便于审计与前端追溯来源
    evidence: list[Evidence] = field(default_factory=list)
    # v2.1 降级标记：结构化/实时查询失败时不再静默换文档答案，而是明确告知
    # "实时数据暂不可用"（degraded=True + degrade_reason），由数据新鲜度要求
    # 决定降级策略——用过期文档拼一个"当前库存"比承认失败更危险
    degraded: bool = False
    degrade_reason: str = ""
    # v2.2 独立答案验证状态：引用列表非空不代表答案有证据支持
    verification: AnswerVerification | None = None


@dataclass(slots=True)
class QueryStream:
    """流式问答结果：citations 先行确定，tokens 惰性产出（迭代时才真正请求 LLM）。

    SSE 推送顺序由 API 层控制：先发 citations 事件，再逐段发 token，最后发 done。
    """

    citations: list[Citation]
    model: str
    tokens: Iterator[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
    # 流式响应同样带回路由信息，便于前端标注"本答案来自数据库查询"
    route: str = "document"
    sql: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    # 与 QueryResult.degraded 同语义：流式形态下也要把降级事实带给前端
    degraded: bool = False
    degrade_reason: str = ""
    # 流式答案的验证状态由 API 层在 token 推完后计算并回填（SSE validation 事件）
    verification: AnswerVerification | None = None


class RAGPipeline:
    """编排入口：把知识库配置、存储连接、缓存、模型客户端串成完整链路。

    本身不持有任何存储/模型连接，全部通过 RuntimeRegistry 按知识库懒加载获取；
    object_store（原始文件归档）与 cache（检索缓存）是可选项，传 None 时对应能力自动关闭。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        registry: RuntimeRegistry,
        object_store: ObjectStore | None = None,
        cache: RetrievalCache | None = None,
        repo: MetadataRepo | None = None,
    ) -> None:
        """仅保存依赖引用，不做任何网络连接（连接在首次使用时由 registry 懒建立）。

        repo 是可选的元数据仓库：注入后启用重复内容血缘归属（citation 回溯到
        用户实际授权的逻辑文档）；脚本/测试不注入时该能力自动关闭。
        """
        self._settings = settings
        self._registry = registry
        self._object_store = object_store
        self._cache = cache
        self._repo = repo
        # Query Embedding 缓存（9.7 第一层缓存）：进程级复用，避免同一问题重复向量化。
        # 构造时不建连（redis 客户端懒连接），TTL<=0 时整体禁用
        self._embedding_cache: QueryEmbeddingCache | None = (
            QueryEmbeddingCache(
                settings.redis_url,
                ttl_s=settings.query_embedding_cache_ttl_s,
                max_entries=settings.query_embedding_cache_max_entries,
            )
            if settings.query_embedding_cache_ttl_s > 0
            else None
        )

    def _invalidate_cache(self, kb: KBConfig) -> None:
        """入库 / 删除后失效本知识库的检索缓存；缓存异常不影响主链路。

        先 bump 版本号再 SCAN+DELETE：SCAN 可能因 Redis 抖动超时，
        但版本号 bump 成功就足够让旧 key 后续自然失效（写入/读取都会用新版本）。
        """
        if self._cache is not None:
            self._cache.bump_version(kb.tenant_id, kb.kb_id)
            self._cache.invalidate_kb(kb.tenant_id, kb.kb_id)

    def invalidate_retrieval_cache(self, kb: KBConfig) -> None:
        """公开的缓存失效入口：删除状态变更（DELETING 落库）时立即调用。

        检索缓存 key 含 excluded_doc_ids，DELETING 集合变化后新查询会用新 key；
        但旧 key 的缓存条目在 TTL 内仍可能命中——状态变更瞬间主动失效一次，
        保证"开始删除即对检索不可见"不被缓存窗口破坏。
        """
        self._invalidate_cache(kb)

    def _attribute_lineage(
        self,
        kb: KBConfig,
        chunks: list[RetrievedChunk],
        allowed_groups: frozenset[str] | None,
    ) -> list[RetrievedChunk]:
        """重复内容血缘归属（清单第二优先级第 9 项）：citation 回溯到用户实际授权的来源。

        同一物理内容可能被多个逻辑文档共享（duplicate_of 血缘）：检索期按物理
        chunk 的并集 ACL 过滤保证"任一来源可见即可见"，而引用必须回溯到用户
        实际可见的逻辑文档（可见候选按 repo 排序取第一个：权威度降序、创建时间
        升序）。不改动缓存中的原始 chunks（replace 产新对象）；repo 未注入
        （脚本/测试）或查询失败时原样返回，不阻断主链路。
        """
        if self._repo is None or not chunks:
            return chunks
        hashes = {
            str(chunk.metadata.get("content_hash", ""))
            for chunk in chunks
            if chunk.metadata.get("content_hash")
        }
        if not hashes:
            return chunks
        try:
            lineage = _query_lineage(self._settings, kb, sorted(hashes))
        except Exception as exc:
            logger.warning("血缘归属查询失败，保持物理承载引用：%s", exc)
            return chunks
        if not lineage:
            return chunks
        attributed: list[RetrievedChunk] = []
        for chunk in chunks:
            docs = lineage.get(str(chunk.metadata.get("content_hash", "")), [])
            visible = [doc for doc in docs if acl_visible(doc.acl_groups, allowed_groups)]
            best = visible[0] if visible else None
            if best is None or best.doc_id == chunk.doc_id:
                attributed.append(chunk)
                continue
            attributed.append(
                replace(
                    chunk,
                    doc_id=best.doc_id,
                    title=best.title,
                    source_uri=best.source_uri,
                )
            )
        return attributed

    def reconcile_lineage_acl(
        self, kb: KBConfig, *, content_hash: str, carrier_doc_id: str
    ) -> None:
        """登记 duplicate 血缘后调和承载者索引的可见性并集（清单第二优先级第 9 项）。

        共享同一物理内容的各来源 ACL 不同时，承载 chunk 的可见性取并集
        （"任一来源可见即可见"）：任一到期/新登记来源更宽松时其用户不应漏招。
        必须在同步上下文调用（worker / 线程池；API 的 async 端点用
        run_in_threadpool 包装）；失败只记告警，并集可在下次登记时再调和。
        """
        if self._repo is None:
            return
        try:
            lineage = _query_lineage(self._settings, kb, [content_hash])
        except Exception as exc:
            logger.warning("血缘并集查询失败 content_hash=%s：%s", content_hash, exc)
            return
        docs = lineage.get(content_hash, [])
        if len(docs) <= 1:
            return
        merged = merge_lineage_acl([doc.acl_groups for doc in docs])
        self.refresh_lineage_acl(kb, carrier_doc_id=carrier_doc_id, merged_acl=merged)

    def refresh_lineage_acl(
        self, kb: KBConfig, *, carrier_doc_id: str, merged_acl: list[str]
    ) -> bool:
        """把重复内容血缘的并集 ACL 刷新到承载文档的物理索引（向量库 + 关键词库）。

        重复内容只保留承载者一份物理索引：共享内容的各来源 ACL 不同时，
        承载 chunk 的可见性必须取并集（"任一来源可见即可见"），否则后到来源
        更宽松时其用户会漏招。已是最新并集的 chunk 跳过（幂等）；
        任一侧失败记告警返回 False，不阻断入库主链路。
        """
        runtime = self._registry.runtime(kb)
        normalized = normalize_acl_groups(merged_acl)
        ok = True
        # 向量库：整行回取 → 改 metadata → 覆盖回写（upsert 按主键覆盖）
        try:
            records = runtime.vector_store.get_records_by_doc(carrier_doc_id)
            if records:
                stale = [
                    record
                    for record in records
                    if normalize_acl_groups(record.metadata.get("acl_groups") or [])
                    != normalized
                ]
                if stale:
                    refreshed = [
                        replace(
                            record,
                            metadata={**record.metadata, "acl_groups": normalized},
                        )
                        for record in stale
                    ]
                    runtime.vector_store.delete_by_ids(
                        [record.chunk_id for record in refreshed]
                    )
                    runtime.vector_store.upsert(refreshed, dim=len(refreshed[0].vector))
        except Exception as exc:
            ok = False
            logger.warning("血缘 ACL 向量侧刷新失败 doc_id=%s：%s", carrier_doc_id, exc)
        # 关键词库：index 语义即同 id 覆盖写，无需先删
        try:
            kw_records = runtime.keyword_store.get_records_by_doc(carrier_doc_id)
            if kw_records:
                stale = [
                    record
                    for record in kw_records
                    if normalize_acl_groups(record.metadata.get("acl_groups") or [])
                    != normalized
                ]
                if stale:
                    runtime.keyword_store.upsert(
                        [
                            replace(
                                record,
                                metadata={**record.metadata, "acl_groups": normalized},
                            )
                            for record in stale
                        ]
                    )
        except Exception as exc:
            ok = False
            logger.warning("血缘 ACL 关键词侧刷新失败 doc_id=%s：%s", carrier_doc_id, exc)
        return ok

    def _is_low_confidence(self, chunks: list[RetrievedChunk]) -> bool:
        """低置信拒答判定：有结果但最高分低于 retrieval_min_score（0=关闭）。

        阈值语义见 Settings 注释——作用于精排分数（0~1）；未启用精排时
        RRF 融合分是名次数，阈值应保持关闭，否则会误伤正常结果。
        """
        threshold = self._settings.retrieval_min_score
        if threshold <= 0 or not chunks:
            return False
        top_score = max(chunk.score for chunk in chunks)
        if top_score >= threshold:
            return False
        logger.info("检索最高分 %.4f 低于阈值 %.4f，低置信拒答", top_score, threshold)
        return True

    def _compressed_context(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk] | None:
        """按问题做抽取式上下文压缩；关闭或未产生压缩时返回 None（沿用原文）。

        压缩只作用于送 LLM 的正文，不影响 QueryResult.chunks（审计与引用仍指向原文）。
        """
        if not self._settings.context_compression_enabled or not chunks:
            return None
        return compress_chunks(
            chunks,
            question,
            max_sentences=self._settings.context_compression_max_sentences,
            min_chars=self._settings.context_compression_min_chars,
        )

    def _document_evidence(self, chunks: list[RetrievedChunk]) -> list[Evidence]:
        """把检索片段转成统一证据：定位到 chunk，并带上来源/版本/观测时间。"""
        observed_at = datetime.now(UTC)
        return [
            Evidence(
                source_type="document",
                source_id=chunk.doc_id,
                locator=chunk.chunk_id,
                snippet=chunk.content[:200],
                observed_at=observed_at,
                version=str(chunk.metadata.get("version") or ""),
                provenance={
                    "source_uri": chunk.source_uri,
                    "title": chunk.title,
                },
            )
            for chunk in chunks
        ]

    def _terminal_evidence(
        self, route: str, answer: str, *, sql: str | None = None
    ) -> list[Evidence]:
        """SQL/API 终态证据：来源类型、可复现 SQL 或接口路径、观测时间。"""
        is_sql = route == "sql"
        return [
            Evidence(
                source_type="database" if is_sql else "api",
                source_id="structured" if is_sql else "realtime_api",
                locator=(sql or route),
                snippet=answer[:200],
                observed_at=datetime.now(UTC),
                provenance={"sql": sql} if sql else {"route": route},
            )
        ]

    # ---------------------------------------------------------------- 入库
    def ingest_bytes(
        self,
        kb: KBConfig,
        *,
        data: bytes,
        filename: str,
        source_uri: str | None = None,
        title: str = "",
        version: int = 1,
        acl_groups: list[str] | None = None,
        governance: Mapping[str, object] | None = None,
        old_manifest: list[dict[str, object]] | None = None,
    ) -> IngestResult:
        """文件字节入库：先按后缀解析成文本文档，再走统一的 _ingest 主链路。

        source_uri / title 缺省时用文件名兜底；解析失败抛 DocumentParseError。
        需要先拿到内容指纹做去重/版本裁决的调用方，应先自行 load 再调 ingest_document。
        """
        loader = DocumentLoader(
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            redact_pii=self._settings.privacy_redaction_enabled,
            # OCR 可选：关闭或依赖缺失时 build_ocr_engine 返回 None，解析行为不变
            ocr_engine=build_ocr_engine(enabled=self._settings.ocr_enabled),
            ocr_language=self._settings.ocr_language,
            ocr_min_chars=self._settings.ocr_min_chars,
        )
        document = loader.load_bytes(
            data,
            source_uri=source_uri or f"kb://{kb.kb_code}/{filename}",
            title=title or Path(filename).stem,
            suffix=Path(filename).suffix,
        )
        return self.ingest_document(
            kb,
            document,
            raw_bytes=data,
            version=version,
            acl_groups=acl_groups,
            governance=governance,
            old_manifest=old_manifest,
        )

    def ingest_document(
        self,
        kb: KBConfig,
        document: LoadedDocument,
        *,
        raw_bytes: bytes | None,
        version: int = 1,
        acl_groups: list[str] | None = None,
        governance: Mapping[str, object] | None = None,
        old_manifest: list[dict[str, object]] | None = None,
    ) -> IngestResult:
        """对已解析的文档执行入库主链路。

        单独暴露这一层是为了让 API/Celery 先拿到 document.content_hash，
        完成"同内容跳过 / 内容变化升版本 / 跨来源重复"三项裁决后再写索引。

        governance 是文档级治理字段（来源类型/责任人/权威度/有效期等），
        会随每个 chunk 一起写入检索索引，供检索阶段的加权与筛选使用。
        old_manifest 是该文档上一版的 chunk 清单（来自 PG），用于增量 Diff：
        未变 chunk 的旧向量直接从向量库取回复用，不重算 Embedding。
        """
        return self._ingest(
            kb,
            document,
            raw_bytes=raw_bytes,
            version=version,
            acl_groups=acl_groups,
            governance=governance,
            old_manifest=old_manifest,
        )

    def _ingest(
        self,
        kb: KBConfig,
        document: LoadedDocument,
        *,
        raw_bytes: bytes | None,
        version: int,
        acl_groups: list[str] | None = None,
        governance: Mapping[str, object] | None = None,
        old_manifest: list[dict[str, object]] | None = None,
    ) -> IngestResult:
        """入库主链路：切块 → 增量 Diff → 先写新版 → 清失效旧版 → 失效缓存 → 归档。

        失败兜底：
        - 切块为空抛 DocumentParseError（解析类错误，任务层直接落 failed，不重试）；
        - Embedding / Milvus / ES 异常向上抛，并回滚本次已写入的新 chunk
          （按 id 精确删，旧版本片段不受影响），由任务层落库 failed；
        - 清理失效旧版失败只记告警：新版已可用，残留由检索侧版本去重屏蔽、
          对账任务（reconciler）兜底清除；
        - S3 归档失败只告警：检索数据已可用，归档可之后补偿。
        """
        # 按知识库取运行时（懒加载 + 进程内缓存）：embedding、Milvus、ES 连接
        runtime = self._registry.runtime(kb)
        # 采集时间与治理字段随 chunk 落索引：检索阶段据此做权威度/时效加权
        collected_at = datetime.now(UTC)
        chunk_metadata: dict[str, object] = {
            "title": document.title,
            "source_uri": document.source_uri,
            "mime_type": document.mime_type,
            # 文档级 ACL 冗余进每个 chunk：检索时直接按 metadata 过滤（公开文档为 ["*"]）
            "acl_groups": normalize_acl_groups(acl_groups),
            "collected_at": collected_at.isoformat(),
            # 向量语义版本：检索缓存 key 与增量复用都按它判定新旧，
            # 换 embedding 模型并递增版本后，旧向量不会被误复用
            "embedding_version": self._settings.embedding_version,
        }
        chunk_metadata.update(_chunk_governance_metadata(governance))
        # 按知识库的切块配置（块大小/重叠等）切分正文
        chunker = TextChunker(kb.chunking)
        chunks = chunker.split(
            document.content,
            doc_id=document.doc_id,
            version=version,
            metadata=chunk_metadata,
        )
        if not chunks:
            raise DocumentParseError(f"文档切块结果为空：{document.source_uri}")

        # 父子切片：父块只进 ES（用于命中子块后按 parent_chunk_id 回取整节内容），
        # 不进 Milvus、也不计入 chunk_count——检索只召回子块，父块不占召回名额。
        children = [chunk for chunk in chunks if chunk.metadata.get("chunk_role") != "parent"]
        parents = [chunk for chunk in chunks if chunk.metadata.get("chunk_role") == "parent"]
        if not children:
            raise DocumentParseError(f"文档切块结果为空：{document.source_uri}")

        # ---- 解析质量门禁：不达标的文档直接隔离，不写任何索引 ----
        settings = self._settings
        report = assess_quality(
            document.content,
            min_chars=settings.parse_min_chars,
            min_page_yield=settings.parse_min_page_yield,
            max_garbled_ratio=settings.parse_max_garbled_ratio,
            min_page_chars=settings.parse_min_page_chars,
        )
        report = with_chunk_count(report, len(children))
        quality = report.to_dict()
        if document.pii_findings:
            # 脱敏命中情况随质量报告落库，便于合规审计"哪些文档含哪类 PII"
            quality["pii_findings"] = document.pii_findings
        if not report.passed:
            logger.warning(
                "解析质量未达标，文档进入隔离 kb=%s doc_id=%s reasons=%s",
                kb.kb_code,
                document.doc_id,
                "; ".join(report.reasons),
            )
            raise DocumentQualityError(
                "解析质量未达入库标准：" + "；".join(report.reasons),
                details={"doc_id": document.doc_id, **quality},
            )

        # ---- 增量 Diff：与上一版 chunk 清单对比，未变 chunk 复用旧向量 ----
        # 清单条目同时覆盖子块（Milvus+ES）与父块（仅 ES），对账任务按它核对两侧索引；
        # 复用判定要求 embedding_version 一致：换模型升版本后旧向量一律重算。
        embedding_version = self._settings.embedding_version
        manifest: list[dict[str, object]] = [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.index,
                "content_hash": str(chunk.metadata.get("content_hash", "")),
                "section_path": str(chunk.metadata.get("section_path", "")),
                "role": str(chunk.metadata.get("chunk_role", "child")),
                "embedding_version": embedding_version,
            }
            for chunk in [*children, *parents]
        ]

        # 旧版可复用映射：新版 chunk_id → 旧版 chunk_id（chunk_id 含版本号，版本变 id 变）
        old_reusable: dict[tuple[object, object, object], str] = {}
        for entry in old_manifest or []:
            if entry.get("role", "child") != "child":
                continue
            if entry.get("embedding_version") != embedding_version:
                continue
            old_id = str(entry.get("chunk_id", ""))
            if old_id:
                key = (
                    entry.get("chunk_index"),
                    entry.get("section_path"),
                    entry.get("content_hash"),
                )
                old_reusable[key] = old_id

        reuse_old_ids: dict[str, str] = {}  # 新 chunk_id → 旧 chunk_id
        fresh_chunks: list[Chunk] = []
        for chunk in children:
            key = (
                chunk.index,
                str(chunk.metadata.get("section_path", "")),
                str(chunk.metadata.get("content_hash", "")),
            )
            old_id = old_reusable.get(key)
            if old_id:
                reuse_old_ids[chunk.chunk_id] = old_id
            else:
                fresh_chunks.append(chunk)

        # 从向量库取回未变 chunk 的旧向量；索引里查不到的（数据丢失）自动转为重算
        vector_by_chunk: dict[str, list[float]] = {}
        if reuse_old_ids:
            chunk_by_id = {chunk.chunk_id: chunk for chunk in children}
            old_vectors = runtime.vector_store.get_vectors_by_ids(
                sorted(set(reuse_old_ids.values()))
            )
            for new_id, old_id in reuse_old_ids.items():
                vector = old_vectors.get(old_id)
                if vector is not None:
                    vector_by_chunk[new_id] = vector
                else:
                    fresh_chunks.append(chunk_by_id[new_id])

        # 只对新增/修改的 chunk 调 Embedding：内容小改时绝大部分 chunk 复用，省算力
        if fresh_chunks:
            fresh_vectors = runtime.embedding.embed_documents(
                [chunk.content for chunk in fresh_chunks]
            )
            for chunk, vector in zip(fresh_chunks, fresh_vectors, strict=True):
                vector_by_chunk[chunk.chunk_id] = vector
        dim = runtime.embedding.dim

        # 首次使用自动建 Milvus collection / ES 索引（已存在则是空操作）
        runtime.vector_store.ensure_collection(dim)
        runtime.keyword_store.ensure_index()

        vector_records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                vector=vector_by_chunk[chunk.chunk_id],
                content=chunk.content,
                doc_id=document.doc_id,
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                source_uri=document.source_uri,
                version=chunk.version,
                chunk_index=chunk.index,
                metadata=chunk.metadata,
            )
            for chunk in children
        ]
        # ES 同时收子块与父块：父块供 parent 回取，检索查询用 chunk_role!=parent 把它排除
        keyword_records = [
            KeywordRecord(
                chunk_id=chunk.chunk_id,
                doc_id=document.doc_id,
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                content=chunk.content,
                source_uri=document.source_uri,
                version=chunk.version,
                chunk_index=chunk.index,
                metadata=chunk.metadata,
            )
            for chunk in [*children, *parents]
        ]

        # 先写新版、后清旧版（消除半更新窗口）：
        # 新旧 chunk 在窗口期并存，检索侧按 doc 分组取最高版本去重，始终读到完整版本；
        # 写失败则回滚本次写入——但必须先快照"写入前已存在"的记录：内容未变、仅改
        # ACL/有效期 的同一版本重建会复用相同 chunk_id，upsert 是覆盖而非新建，
        # 直接按 id 全删会把原有索引一起删光（用户故障注入：3 向量 3 关键词 → 0）。
        prev_vectors: dict[str, VectorRecord] = {}
        prev_keywords: dict[str, KeywordRecord] = {}
        if old_manifest:
            # 抽象存储以 None 表示"不支持整篇快照"，此时退化为"只回滚新建 id"
            try:
                old_vector_records = runtime.vector_store.get_records_by_doc(document.doc_id) or []
                old_keyword_records = (
                    runtime.keyword_store.get_records_by_doc(document.doc_id) or []
                )
            except AttributeError:
                old_vector_records = []
                old_keyword_records = []
            prev_vectors = {record.chunk_id: record for record in old_vector_records}
            prev_keywords = {record.chunk_id: record for record in old_keyword_records}
        new_ids = {entry["chunk_id"] for entry in manifest}
        try:
            runtime.vector_store.upsert(vector_records, dim=dim)
            runtime.keyword_store.upsert(keyword_records)
        except Exception:
            self._compensate_partial_write(
                runtime,
                vector_records=vector_records,
                keyword_records=keyword_records,
                prev_vectors=prev_vectors,
                prev_keywords=prev_keywords,
                dim=dim,
            )
            raise

        # 旧版失效 chunk（旧清单有、新清单没有）按 id 精确删除。
        # 失败只告警：残留旧版由检索侧版本去重屏蔽，对账任务兜底清除。
        stale_ids = [
            str(entry.get("chunk_id", ""))
            for entry in old_manifest or []
            if entry.get("chunk_id") and str(entry["chunk_id"]) not in new_ids
        ]
        if stale_ids:
            try:
                runtime.vector_store.delete_by_ids(stale_ids)
                runtime.keyword_store.delete_by_ids(stale_ids)
            except Exception as exc:
                logger.warning(
                    "清理失效旧版 chunk 失败（残留由对账兜底）doc_id=%s err=%s",
                    document.doc_id,
                    exc,
                )
        # 内容已变更：先失效本库检索缓存，再走归档等收尾
        self._invalidate_cache(kb)

        raw_key = self._archive_raw(kb, document, raw_bytes, version)
        logger.info(
            "文档入库完成 kb=%s doc_id=%s chunks=%s parents=%s 复用向量=%s 新算向量=%s",
            kb.kb_code,
            document.doc_id,
            len(children),
            len(parents),
            len(children) - len(fresh_chunks),
            len(fresh_chunks),
        )
        return IngestResult(
            doc_id=document.doc_id,
            source_uri=document.source_uri,
            title=document.title,
            chunk_count=len(children),
            collection=kb.milvus_collection,
            index=kb.es_index,
            raw_object_key=raw_key,
            version=version,
            content_hash=document.content_hash,
            quality=quality,
            collected_at=collected_at,
            chunk_manifest=manifest,
        )

    def _compensate_partial_write(
        self,
        runtime: KBRuntime,
        *,
        vector_records: list[VectorRecord],
        keyword_records: list[KeywordRecord],
        prev_vectors: dict[str, VectorRecord],
        prev_keywords: dict[str, KeywordRecord],
        dim: int,
    ) -> None:
        """双写失败后的补偿：只回滚本次新建的 chunk，被覆盖的旧记录原样恢复。

        区分"新建 vs 覆盖"是同版本重建的关键：仅修改 ACL/有效期时 chunk_id 不变，
        此时 upsert 覆盖了原有记录，若按 id 全删就会把旧索引一起删掉。因此：
        - 写入前不存在的 id → 删除（本次新建的垃圾数据）；
        - 写入前已存在的 id → 用快照回写原记录（恢复旧内容/旧 ACL）。
        补偿本身失败只记日志——原始异常更重要，不能被补偿异常覆盖，
        残留数据由任务重试时的增量重建与对账任务再次清理。
        """
        removed = 0
        restored = 0
        try:
            new_vector_ids = [
                record.chunk_id
                for record in vector_records
                if record.chunk_id not in prev_vectors
            ]
            if new_vector_ids:
                runtime.vector_store.delete_by_ids(new_vector_ids)
                removed += len(new_vector_ids)
            restore_vectors = [
                prev_vectors[record.chunk_id]
                for record in vector_records
                if record.chunk_id in prev_vectors
            ]
            if restore_vectors:
                runtime.vector_store.upsert(restore_vectors, dim=dim)
                restored += len(restore_vectors)
            new_keyword_ids = [
                record.chunk_id
                for record in keyword_records
                if record.chunk_id not in prev_keywords
            ]
            if new_keyword_ids:
                runtime.keyword_store.delete_by_ids(new_keyword_ids)
                removed += len(new_keyword_ids)
            restore_keywords = [
                prev_keywords[record.chunk_id]
                for record in keyword_records
                if record.chunk_id in prev_keywords
            ]
            if restore_keywords:
                runtime.keyword_store.upsert(restore_keywords)
                restored += len(restore_keywords)
        except Exception as exc:  # 补偿失败不能覆盖原始异常
            logger.warning(
                "双写补偿失败 removed=%s restored=%s err=%s", removed, restored, exc
            )

    def _archive_raw(
        self,
        kb: KBConfig,
        document: LoadedDocument,
        raw_bytes: bytes | None,
        version: int,
    ) -> str | None:
        """原始文件归档到 MinIO/S3；失败只告警，不影响已经完成的检索入库。

        归档 payload 优先用原始字节（保留原始格式）；没有原始字节时（如纯文本来源）
        退化为把解析后的文本按 UTF-8 写入。key 按 租户/知识库/文档/版本 分层，
        同一文档新版本不覆盖旧归档。
        """
        if self._object_store is None:
            return None

        filename = Path(urlsplit(document.source_uri).path).name or f"{document.doc_id}.txt"
        key = build_document_key(
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            doc_id=document.doc_id,
            version=version,
            filename=filename,
        )
        payload = raw_bytes if raw_bytes is not None else document.content.encode("utf-8")
        try:
            self._object_store.put_bytes(key, payload, content_type=document.mime_type)
        except ObjectStoreError as exc:
            logger.warning("原始文件归档失败（检索数据已写入）：%s", exc.message)
            return None
        return key

    # ---------------------------------------------------------------- 删除
    def delete_document(self, kb: KBConfig, doc_id: str) -> DeleteResult:
        """删除文档：向量库与关键词库都按 doc_id 删除，随后失效本库检索缓存。

        两库删除相互独立、各自计数；元数据状态由调用方（API/任务层）负责更新。
        """
        runtime = self._registry.runtime(kb)
        result = DeleteResult(
            doc_id=doc_id,
            deleted_vectors=runtime.vector_store.delete_by_doc(doc_id),
            deleted_keywords=runtime.keyword_store.delete_by_doc(doc_id),
        )
        self._invalidate_cache(kb)
        return result

    # ---------------------------------------------------------------- 问答
    def _structured_available(self, kb: KBConfig) -> bool:
        """结构化查询可用性：必须显式启用且配置了只读连接串。"""
        return kb.structured.enabled and bool(kb.structured.sql_url.strip())

    def _api_available(self, kb: KBConfig) -> bool:
        """实时 API 可用性：必须显式启用且至少登记了一个端点。"""
        return kb.api.enabled and bool(kb.api.endpoints)

    def _api_answer(self, kb: KBConfig, question: str) -> ApiAnswer:
        """执行实时 API 查询；失败向上抛 AppError，由调用方决定降级策略。

        v2.1 起禁止静默降级：实时问题（库存/订单状态）静默改用历史文档作答，
        比"明确告知实时数据暂不可用"更危险。Agentic 工具回调自行捕获本异常。
        """
        return ApiRetriever(kb.api).answer(question)

    def _structured_answer(self, kb: KBConfig, question: str) -> StructuredAnswer:
        """执行结构化查询；失败向上抛 AppError，由调用方决定降级策略。

        v2.1 起显式与 auto 模式行为统一：不再静默回退文档检索（调用方在 auto
        模式下转为"明确失败"结果）。Agentic 工具回调自行捕获本异常。
        """
        llm = self._registry.llm() if kb.structured.use_llm_planner else None
        return StructuredRetriever(llm=llm, config=kb.structured).answer(question)

    def _unavailable_answer(self, route: str, exc: AppError) -> str:
        """构造"明确失败"答案：固定文案（具体原因经 degrade_reason 透出）。

        v2.1 降级策略：路由到 sql/api 的查询失败时不回退文档检索——实时库存、
        统计口径这类问题的答案对数据新鲜度敏感，用历史文档拼答案比承认失败
        更危险；这里记日志并返回明确的"数据不可用"文案。
        """
        reason = getattr(exc, "message", None) or str(exc)
        logger.warning(
            "%s 查询失败，明确降级（不回退文档检索）：%s",
            "结构化" if route == "sql" else "实时 API",
            reason,
        )
        return STRUCTURED_UNAVAILABLE_ANSWER

    def _agentic_outcome(
        self,
        kb: KBConfig,
        runtime: KBRuntime,
        question: str,
        *,
        top_k: int | None,
        use_rewrite: bool | None,
        use_rerank: bool | None,
        allowed_groups: frozenset[str] | None,
        excluded_doc_ids: frozenset[str] | None,
        history: list[tuple[str, str]] | None,
    ) -> AgenticOutcome:
        """执行 Agentic 循环：把检索/结构化/API 三种工具以回调形式注入执行器。

        回调复用既有链路，因此 ACL 过滤、检索缓存、SQL 白名单、API 端点约束
        全部保持不变——Agentic 只是多轮调用它们，不绕过任何安全边界。
        """

        def search(sub_question: str) -> list[RetrievedChunk]:
            return self._retrieve(
                kb,
                runtime,
                sub_question,
                top_k=top_k,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
            )

        def structured(sub_question: str) -> StructuredTerminal | None:
            try:
                answer = self._structured_answer(kb, sub_question)
            except AppError:
                # Agentic 循环内工具失败不算终局：返回 None 让反思/换工具继续
                return None
            return StructuredTerminal(route="sql", answer=answer.answer, sql=answer.sql)

        def api(sub_question: str) -> StructuredTerminal | None:
            try:
                answer = self._api_answer(kb, sub_question)
            except AppError:
                return None
            return StructuredTerminal(route="api", answer=answer.answer)

        try:
            llm = self._registry.llm()
        except AppError as exc:
            # 未配置 LLM 时 Agentic 退化为 QueryRouter 单步检索（不反思、不规划）
            logger.warning("Agentic 规划器不可用，退化为单步检索：%s", exc)
            llm = None
        return AgenticRAG(
            llm=llm,
            router=QueryRouter(),
            search=search,
            structured=structured,
            api=api,
            config=kb.agentic,
            structured_available=self._structured_available(kb),
            api_available=self._api_available(kb),
        ).run(question, history or [])

    def query(
        self,
        kb: KBConfig,
        question: str,
        *,
        top_k: int | None = None,
        use_rewrite: bool | None = None,
        use_rerank: bool | None = None,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: frozenset[str] | None = None,
        history: list[tuple[str, str]] | None = None,
        mode: RequestedMode = "auto",
    ) -> QueryResult:
        """非流式问答：检索 → 生成一次完成，等待 LLM 全部输出后返回。

        鉴权发生在 API 层（解析用户身份得到 allowed_groups），这里只负责把
        用户组透传给检索层做“检索时过滤”；审计日志同样由 API 层在返回前写入。
        excluded_doc_ids 是已被新版本替代的文档，检索时直接排除，避免旧版本被引用。
        history 是多轮会话上下文：仅用于追问改写检索 query，生成仍用原始问题。
        mode 是 v2.0 查询路由：auto 按问题语义分流，document/structured 强制指定。
        """
        decision = QueryRouter().route(
            question,
            structured_available=self._structured_available(kb),
            api_available=self._api_available(kb),
            requested=mode,
        )
        if decision.route == "sql":
            try:
                structured = self._structured_answer(kb, question)
            except AppError as exc:
                if mode == "structured":
                    raise
                # auto 模式明确降级：不用历史文档拼统计答案（degraded 标记给前端）
                return QueryResult(
                    answer=self._unavailable_answer("sql", exc),
                    citations=[],
                    route="sql",
                    degraded=True,
                    degrade_reason=str(getattr(exc, "message", exc))[:300],
                )
            return QueryResult(
                answer=structured.answer,
                citations=[],
                model="structured-sql",
                route="sql",
                sql=structured.sql,
                evidence=self._terminal_evidence(
                    "sql", structured.answer, sql=structured.sql
                ),
            )
        if decision.route == "api":
            try:
                api = self._api_answer(kb, question)
            except AppError as exc:
                # 实时问题失败绝不静默换文档答案（"当前库存"答成旧说明书更危险）
                return QueryResult(
                    answer=self._unavailable_answer("api", exc),
                    citations=[],
                    route="api",
                    degraded=True,
                    degrade_reason=str(getattr(exc, "message", exc))[:300],
                )
            return QueryResult(
                answer=api.answer,
                citations=[],
                model="api-retrieval",
                route="api",
                evidence=self._terminal_evidence("api", api.answer),
            )
        runtime = self._registry.runtime(kb)
        if kb.agentic.enabled and mode == "auto":
            # Agentic 多轮检索：子问题本身已自包含，不再做会话追问改写
            outcome = self._agentic_outcome(
                kb,
                runtime,
                question,
                top_k=top_k,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
                history=history,
            )
            if outcome.terminal is not None:
                terminal = outcome.terminal
                return QueryResult(
                    answer=terminal.answer,
                    citations=[],
                    model=f"{terminal.route}-agentic",
                    route=terminal.route,
                    sql=terminal.sql,
                    evidence=self._terminal_evidence(
                        terminal.route, terminal.answer, sql=terminal.sql
                    ),
                )
            chunks = outcome.chunks
            retrieval_question = question
        else:
            retrieval_question = ConversationQueryRewriter(
                self._registry.llm()
            ).rewrite(question, history)
            chunks = self._retrieve(
                kb,
                runtime,
                retrieval_question,
                top_k=top_k,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
            )
        # 重复内容血缘归属：citation 回溯到用户实际授权的逻辑文档（不动缓存原件）
        chunks = self._attribute_lineage(kb, chunks, allowed_groups)
        # 低置信拒答：有检索结果但最高分低于阈值时，不调用 LLM，避免弱证据幻觉
        if self._is_low_confidence(chunks):
            return QueryResult(
                answer=LOW_CONFIDENCE_ANSWER,
                citations=[],
                chunks=chunks,
                evidence=self._document_evidence(chunks),
            )
        answer: AnswerResult = RAGChain(
            self._registry.llm(),
            max_context_chars=self._settings.max_context_chars,
            verify_answers=self._settings.answer_verification_enabled,
            regen_max=self._settings.answer_regen_max,
            min_citation_coverage=self._settings.answer_min_citation_coverage,
        ).answer(
            question, chunks, context_chunks=self._compressed_context(retrieval_question, chunks)
        )
        return QueryResult(
            answer=answer.answer,
            citations=answer.citations,
            model=answer.model,
            chunks=chunks,
            evidence=self._document_evidence(chunks),
            verification=answer.verification,
        )

    def query_stream(
        self,
        kb: KBConfig,
        question: str,
        *,
        top_k: int | None = None,
        use_rewrite: bool | None = None,
        use_rerank: bool | None = None,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: frozenset[str] | None = None,
        history: list[tuple[str, str]] | None = None,
        mode: RequestedMode = "auto",
    ) -> QueryStream:
        """流式问答：检索同步完成（缓存命中同样走这里），生成以惰性迭代器返回，供 SSE 逐段推送。

        注意：tokens 首次被迭代时才真正请求 LLM，LLM 异常不会在调用本方法时抛出，
        而是在推送阶段由 API 层捕获并转成 SSE error 事件。
        结构化路由的答案已完整算出，tokens 退化为单段产出（仍走同一 SSE 协议）。
        """
        decision = QueryRouter().route(
            question,
            structured_available=self._structured_available(kb),
            api_available=self._api_available(kb),
            requested=mode,
        )
        if decision.route == "sql":
            try:
                structured = self._structured_answer(kb, question)
            except AppError as exc:
                if mode == "structured":
                    raise
                # auto 模式明确降级：不用历史文档拼统计答案（degraded 标记给前端）
                return QueryStream(
                    citations=[],
                    model="",
                    tokens=iter([self._unavailable_answer("sql", exc)]),
                    route="sql",
                    degraded=True,
                    degrade_reason=str(getattr(exc, "message", exc))[:300],
                )
            return QueryStream(
                citations=[],
                model="structured-sql",
                tokens=iter([structured.answer]),
                route="sql",
                sql=structured.sql,
                evidence=self._terminal_evidence(
                    "sql", structured.answer, sql=structured.sql
                ),
            )
        if decision.route == "api":
            try:
                api = self._api_answer(kb, question)
            except AppError as exc:
                # 实时问题失败绝不静默换文档答案
                return QueryStream(
                    citations=[],
                    model="",
                    tokens=iter([self._unavailable_answer("api", exc)]),
                    route="api",
                    degraded=True,
                    degrade_reason=str(getattr(exc, "message", exc))[:300],
                )
            return QueryStream(
                citations=[],
                model="api-retrieval",
                tokens=iter([api.answer]),
                route="api",
                evidence=self._terminal_evidence("api", api.answer),
            )
        runtime = self._registry.runtime(kb)
        if kb.agentic.enabled and mode == "auto":
            # Agentic 检索在流式生成前同步完成，之后仍走原 RAGChain 流式通道
            outcome = self._agentic_outcome(
                kb,
                runtime,
                question,
                top_k=top_k,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
                history=history,
            )
            if outcome.terminal is not None:
                terminal = outcome.terminal
                return QueryStream(
                    citations=[],
                    model=f"{terminal.route}-agentic",
                    tokens=iter([terminal.answer]),
                    route=terminal.route,
                    sql=terminal.sql,
                    evidence=self._terminal_evidence(
                        terminal.route, terminal.answer, sql=terminal.sql
                    ),
                )
            chunks = outcome.chunks
            retrieval_question = question
        else:
            retrieval_question = ConversationQueryRewriter(
                self._registry.llm()
            ).rewrite(question, history)
            chunks = self._retrieve(
                kb,
                runtime,
                retrieval_question,
                top_k=top_k,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
            )
        # 重复内容血缘归属：citation 回溯到用户实际授权的逻辑文档（不动缓存原件）
        chunks = self._attribute_lineage(kb, chunks, allowed_groups)
        # 低置信拒答：与一次性问答同一判定，流式产出固定文案（不请求 LLM）
        if self._is_low_confidence(chunks):
            return QueryStream(
                citations=[],
                model="",
                tokens=iter([LOW_CONFIDENCE_ANSWER]),
                chunks=chunks,
                evidence=self._document_evidence(chunks),
            )
        stream = RAGChain(
            self._registry.llm(), max_context_chars=self._settings.max_context_chars
        ).stream_answer(
            question, chunks, context_chunks=self._compressed_context(retrieval_question, chunks)
        )
        return QueryStream(
            citations=stream.citations,
            model=stream.model,
            tokens=stream.tokens,
            chunks=chunks,
            evidence=self._document_evidence(chunks),
        )

    def _retrieve(
        self,
        kb: KBConfig,
        runtime: KBRuntime,
        question: str,
        *,
        top_k: int | None,
        use_rewrite: bool | None,
        use_rerank: bool | None,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]:
        """优先查检索缓存；未命中走混合检索并回填。缓存只存 chunks，答案每次新生成。

        无论成败都记录 retrieval_duration_seconds 指标（finally 保证）。
        """
        started = time.perf_counter()
        try:
            return self._retrieve_inner(
                kb,
                runtime,
                question,
                top_k=top_k,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
            )
        finally:
            metrics.observe(
                "retrieval_duration_seconds", time.perf_counter() - started, kb=kb.kb_code
            )

    def _retrieve_inner(
        self,
        kb: KBConfig,
        runtime: KBRuntime,
        question: str,
        *,
        top_k: int | None,
        use_rewrite: bool | None,
        use_rerank: bool | None,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: frozenset[str] | None = None,
    ) -> list[RetrievedChunk]:
        """先解析“生效参数”再查缓存：调用方传 None 的参数要落成知识库默认值。

        这里的默认值解析规则与 HybridRetriever.retrieve 内部完全一致——否则同一次
        请求会拿一套参数查缓存、另一套参数做检索，导致缓存永远错位。
        """
        # 与 HybridRetriever.retrieve 内部的默认值解析保持一致，保证缓存 key 稳定
        config = kb.retrieval
        eff_top_k = top_k or config.top_k
        eff_rewrite = config.use_query_rewrite if use_rewrite is None else use_rewrite
        eff_rerank = config.use_rerank if use_rerank is None else use_rerank

        cache = self._cache
        cache_key: str | None = None
        if cache is not None and cache.enabled and question.strip():
            cache_key = cache.build_key(
                tenant_id=kb.tenant_id,
                kb_id=kb.kb_id,
                question=question,
                top_k=eff_top_k,
                use_rewrite=eff_rewrite,
                use_rerank=eff_rerank,
                config=config,
                allowed_groups=allowed_groups,
                excluded_doc_ids=excluded_doc_ids,
                embedding_version=self._settings.embedding_version,
                # 版本号进 key：入库/删除/配置变更 bump 后旧 key 自然失效，
                # 即使 invalidate 的 SCAN+DELETE 失败也不影响一致性
                cache_version=cache.get_version(kb.tenant_id, kb.kb_id),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                logger.info("检索缓存命中 kb=%s chunks=%s", kb.kb_code, len(cached))
                return cached

        # 缓存未命中：走完整混合检索（改写 → 双路并行召回 → RRF 融合 → 可选精排）
        retriever = HybridRetriever(
            vector_store=runtime.vector_store,
            keyword_store=runtime.keyword_store,
            embedding=runtime.embedding,
            reranker=self._registry.reranker(),
            rewriter=QueryRewriter(self._registry.llm()),
            classifier=QueryClassifier(self._registry.llm()),
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            config=config,
            # 治理加权参数来自全局配置：权威度与时效性权重，可按环境统一调整
            authority_weight=self._settings.retrieval_authority_weight,
            recency_weight=self._settings.retrieval_recency_weight,
            recency_half_life_days=self._settings.retrieval_recency_half_life_days,
            # 9.7：Query Embedding 缓存 + 精排候选上限 / 高置信跳过阈值
            embedding_cache=self._embedding_cache,
            embedding_version=self._settings.embedding_version,
            rerank_max_candidates=self._settings.rerank_max_candidates,
            rerank_skip_confidence=self._settings.rerank_skip_confidence,
        )
        chunks = retriever.retrieve(
            question,
            top_k=top_k,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            allowed_groups=allowed_groups,
            excluded_doc_ids=excluded_doc_ids,
        )
        # 回填缓存：后续相同问题（含相同用户组）直接命中，免走 Embedding/Milvus/ES
        if cache is not None and cache_key is not None:
            cache.set(cache_key, chunks)
        return chunks
