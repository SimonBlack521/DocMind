"""Milvus 向量库实现（每个知识库一个 collection：rag_{kb_code}）。

为什么每个 KB 独立 collection（而非全租户共享一个）：
- 向量维度由 KB 配置的 embedding 模型决定，不同 KB 维度可能不同，而 Milvus
  的向量维度是 collection 级 schema 属性，同 collection 无法混存不同维度；
- 独立 collection 让 KB 删除/重建的爆炸半径最小，检索时也天然隔离。

schema（ensure_collection 中创建，enable_dynamic_field=False 严格模式）：
    chunk_id    VARCHAR(64)  主键，sha256(doc_id|version|index)
    vector      FLOAT_VECTOR 维度 = KB 的 embedding_dim，COSINE + HNSW（显式 M/ef 参数）
    doc_id/kb_id/tenant_id  VARCHAR(64)  过滤与回溯用
    content     VARCHAR(8192) 片段原文（检索结果直接带回，省去二次查询）
    source_uri  VARCHAR(2048) 来源地址
    version/chunk_index INT32
    metadata    JSON          透传业务元数据；acl_groups 存这里，ACL 过滤用
                              json_contains_any(metadata["acl_groups"], ...) 实现，
                              表达式由 vector_base.scope_expression 统一构造。

安全约定：集合已存在时只校验维度并复用，禁止 drop_old / 自动重建，避免误删数据。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.errors import VectorStoreError
from app.core.logging import get_logger
from app.storage.vector_base import VectorHit, VectorRecord, VectorStore

logger = get_logger(__name__)

_OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "content",
    "source_uri",
    "version",
    "chunk_index",
    "metadata",
]

# 高频过滤标量字段：tenant/kb/doc 过滤几乎出现在每个检索与删除请求里，
# 建 INVERTED 索引避免 Milvus 全量扫描过滤（百万级 chunk 下延迟差异显著）。
_SCALAR_INDEX_FIELDS = ("tenant_id", "kb_id", "doc_id")


def _as_int(value: Any) -> int | None:
    """Milvus describe_index 返回的参数值可能是字符串，统一转 int 后比较。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class MilvusVectorStore(VectorStore):
    """Milvus 向量库实现，索引参数显式注入（默认 HNSW）。

    索引参数（index_type / M / ef_construction / ef_search）由全局配置经
    RuntimeRegistry 注入；集合已存在时会检测索引参数漂移：
    - 入库热路径（ensure_collection）默认只告警不重建（auto_reconcile_index=False），
      避免 drop_index 期间在线查询降级；
    - 重建集中在维护窗口：rag.reconcile_index 任务调 reconcile_index_params()
      执行 drop_index + create_index 原地重建，数据本身不动；
    - 标量过滤字段（tenant_id/kb_id/doc_id）的 INVERTED 索引缺失时随时补建，
      该操作非破坏性，两条路径都允许。
    """

    def __init__(
        self,
        *,
        uri: str,
        collection: str,
        token: str = "",
        db_name: str = "default",
        timeout: float = 10.0,
        index_type: str = "HNSW",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        auto_reconcile_index: bool = False,
    ) -> None:
        self._uri = uri
        self._collection = collection
        self._token = token
        self._db_name = db_name
        self._timeout = timeout
        self._index_type = index_type.upper()
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._auto_reconcile_index = auto_reconcile_index
        self._client: MilvusClient | None = None

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"uri": self._uri, "db_name": self._db_name}
            if self._token:
                kwargs["token"] = self._token
            try:
                self._client = MilvusClient(**kwargs)
            except Exception as exc:
                raise VectorStoreError(
                    f"Milvus 连接失败：{self._uri}", details={"collection": self._collection}
                ) from exc
        return self._client

    def ensure_collection(self, dim: int) -> None:
        """不存在则按固定 schema 创建；存在只校验维度并检查索引参数漂移，绝不重建数据。"""
        client = self._get_client()
        try:
            if client.has_collection(self._collection):
                existing_dim = self._collection_dim(client)
                if existing_dim is not None and existing_dim != dim:
                    raise VectorStoreError(
                        f"向量维度不一致：collection={self._collection} "
                        f"已存在 {existing_dim} 维，当前模型输出 {dim} 维"
                    )
                self._reconcile_index(client, allow_rebuild=self._auto_reconcile_index)
                return

            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                "chunk_id", DataType.VARCHAR, is_primary=True, max_length=64
            )
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
            schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
            schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
            schema.add_field("content", DataType.VARCHAR, max_length=8192)
            schema.add_field("source_uri", DataType.VARCHAR, max_length=2048)
            schema.add_field("version", DataType.INT32)
            schema.add_field("chunk_index", DataType.INT32)
            schema.add_field("metadata", DataType.JSON)

            client.create_collection(
                collection_name=self._collection,
                schema=schema,
                index_params=self._build_index_params(client),
            )
            logger.info(
                "创建 Milvus collection=%s dim=%s index=%s",
                self._collection,
                dim,
                self._index_type,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 集合初始化失败：{self._collection}", details={"error": str(exc)}
            ) from exc

    def _build_index_params(self, client: MilvusClient, *, include_scalar: bool = True) -> Any:
        """按配置生成索引参数：向量 ANN 索引 + 高频过滤标量字段的 INVERTED 索引。

        include_scalar=False 用于"字段上可能已有标量索引"的补建/重建场景：
        Milvus 不允许对已建索引的字段重复 create_index，标量索引统一走
        _ensure_scalar_indexes 幂等补齐。
        """
        index_params = client.prepare_index_params()
        kwargs: dict[str, Any] = {
            "field_name": "vector",
            "index_type": self._index_type,
            "metric_type": "COSINE",
        }
        if self._index_type.startswith("HNSW"):
            kwargs["params"] = {
                "M": self._hnsw_m,
                "efConstruction": self._hnsw_ef_construction,
            }
        index_params.add_index(**kwargs)
        if include_scalar:
            for field_name in _SCALAR_INDEX_FIELDS:
                index_params.add_index(field_name=field_name, index_type="INVERTED")
        return index_params

    def reconcile_index_params(self) -> bool:
        """维护任务入口：集合已存在时按当前配置对账索引（允许原地重建），返回是否变更。"""
        client = self._get_client()
        try:
            if not client.has_collection(self._collection):
                return False
            return self._reconcile_index(client, allow_rebuild=True)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 索引对账失败：{self._collection}", details={"error": str(exc)}
            ) from exc

    def _reconcile_index(self, client: MilvusClient, *, allow_rebuild: bool) -> bool:
        """集合已存在时的索引对账：缺索引补建、漂移按需重建、标量索引补齐。

        只比较"我们关心的"参数：index_type 全家族都比，M/efConstruction 仅 HNSW
        家族比；AUTOINDEX 的实际参数由服务端决定，视为不可比较，配置要求显式
        索引时直接判为漂移。

        allow_rebuild=False（入库热路径默认）：漂移仅告警不重建——drop_index 期间
        查询会降级，重建集中到维护窗口（rag.reconcile_index → reconcile_index_params）。
        返回是否对集合做过变更。
        """
        try:
            existing = self._describe_vector_index(client)
        except Exception as exc:
            logger.warning(
                "读取 Milvus 索引信息失败，跳过漂移检查 collection=%s error=%s",
                self._collection,
                exc,
            )
            return False

        changed = False
        if existing is None:
            # 集合存在但没有向量索引：查询会直接报错，补建是非破坏性操作，
            # 入库路径也允许（不受 allow_rebuild 限制）。
            client.create_index(
                collection_name=self._collection,
                index_params=self._build_index_params(client, include_scalar=False),
            )
            logger.info(
                "补建 Milvus 向量索引 collection=%s index=%s",
                self._collection,
                self._index_type,
            )
            changed = True
        else:
            drifted = existing.get("index_type", "").upper() != self._index_type
            if not drifted and self._index_type.startswith("HNSW"):
                current_m = _as_int(existing.get("M"))
                current_efc = _as_int(existing.get("efConstruction"))
                drifted = (
                    current_m != self._hnsw_m or current_efc != self._hnsw_ef_construction
                )

            if drifted and not allow_rebuild:
                logger.warning(
                    "检测到 Milvus 索引参数漂移 collection=%s existing=%s target=%s/%s；"
                    "入库路径跳过重建，请在维护窗口执行 rag.reconcile_index",
                    self._collection,
                    existing,
                    self._index_type,
                    {"M": self._hnsw_m, "efConstruction": self._hnsw_ef_construction},
                )
            elif drifted:
                index_name = str(existing.get("index_name") or "vector")
                logger.warning(
                    "检测到 Milvus 索引参数漂移 collection=%s existing=%s target=%s/%s，"
                    "原地重建索引",
                    self._collection,
                    existing,
                    self._index_type,
                    {"M": self._hnsw_m, "efConstruction": self._hnsw_ef_construction},
                )
                # 已加载（loaded）的集合不允许直接 drop index：先卸载、重建后再加载恢复查询。
                # 集合也可能未 load（服务刚重启 / Milvus Lite 从不 load），此时 release 会报错，
                # 直接忽略即可——未 load 状态下本来就允许 drop_index。
                try:
                    client.release_collection(collection_name=self._collection)
                except Exception as exc:
                    logger.info(
                        "集合未 load 或 release 失败，跳过 release 直接 drop_index "
                        "collection=%s err=%s",
                        self._collection,
                        exc,
                    )
                client.drop_index(collection_name=self._collection, index_name=index_name)
                client.create_index(
                    collection_name=self._collection,
                    index_params=self._build_index_params(client, include_scalar=False),
                )
                # 重建后 load 回内存：Milvus 会自动加载已加载过的集合（resume_load），
                # 未 load 的集合显式 load 一次确保后续查询不退化
                client.load_collection(collection_name=self._collection)
                changed = True

        return self._ensure_scalar_indexes(client) or changed

    def _ensure_scalar_indexes(self, client: MilvusClient) -> bool:
        """为高频过滤标量字段补建 INVERTED 索引（幂等：已存在则跳过）。

        存量集合（本特性上线前创建）没有标量索引，靠这里在线补齐；
        Milvus 不允许对已建索引字段重复 create_index，因此先按字段名比对。
        """
        try:
            indexed_fields: set[str] = set()
            for name in client.list_indexes(self._collection) or []:
                info: Any = client.describe_index(self._collection, name)
                if isinstance(info, dict) and info.get("field_name"):
                    indexed_fields.add(str(info["field_name"]))
        except Exception as exc:
            logger.warning(
                "读取 Milvus 标量索引信息失败，跳过补建 collection=%s error=%s",
                self._collection,
                exc,
            )
            return False

        changed = False
        for field_name in _SCALAR_INDEX_FIELDS:
            if field_name in indexed_fields:
                continue
            index_params = client.prepare_index_params()
            index_params.add_index(field_name=field_name, index_type="INVERTED")
            client.create_index(collection_name=self._collection, index_params=index_params)
            changed = True
            logger.info(
                "补建 Milvus 标量索引 collection=%s field=%s", self._collection, field_name
            )
        return changed

    def _describe_vector_index(self, client: MilvusClient) -> dict[str, Any] | None:
        """读取 vector 字段上的索引描述；无索引时返回 None。"""
        names: Any = client.list_indexes(self._collection)
        for name in names or []:
            info: Any = client.describe_index(self._collection, name)
            if not isinstance(info, dict):
                continue
            if info.get("field_name") != "vector":
                continue
            return {**info, "index_name": str(name)}
        return None

    def _collection_dim(self, client: MilvusClient) -> int | None:
        try:
            info: Any = client.describe_collection(self._collection)
        except Exception:
            return None

        fields = info.get("fields", []) if isinstance(info, dict) else getattr(info, "fields", [])
        for field in fields:
            name = field.get("name") if isinstance(field, dict) else getattr(field, "name", None)
            if name != "vector":
                continue
            params = field.get("params", {}) if isinstance(field, dict) else {}
            dim = params.get("dim") if isinstance(params, dict) else None
            return int(dim) if dim is not None else None
        return None

    def upsert(self, records: Sequence[VectorRecord], *, dim: int) -> int:
        if not records:
            return 0
        self.ensure_collection(dim)

        for record in records:
            if len(record.vector) != dim:
                raise VectorStoreError(
                    f"chunk {record.chunk_id} 向量维度 {len(record.vector)} 与集合维度 {dim} 不一致"
                )

        rows = [
            {
                "chunk_id": record.chunk_id,
                "vector": list(record.vector),
                "doc_id": record.doc_id,
                "kb_id": record.kb_id,
                "tenant_id": record.tenant_id,
                "content": record.content,
                "source_uri": record.source_uri,
                "version": record.version,
                "chunk_index": record.chunk_index,
                "metadata": record.metadata,
            }
            for record in records
        ]

        client = self._get_client()
        try:
            for start in range(0, len(rows), 500):
                client.upsert(collection_name=self._collection, data=rows[start : start + 500])
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 写入失败：{self._collection}", details={"error": str(exc)}
            ) from exc
        return len(rows)

    def delete_by_doc(self, doc_id: str) -> int:
        client = self._get_client()
        try:
            result: Any = client.delete(
                collection_name=self._collection, filter=f'doc_id == "{doc_id}"'
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 删除失败：{self._collection}",
                details={"doc_id": doc_id, "error": str(exc)},
            ) from exc
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return 0

    def get_vectors_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        """按 chunk_id 批量取向量（client.get 按主键点查，不走 ANN 索引）。"""
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return {}
        client = self._get_client()
        vectors: dict[str, list[float]] = {}
        try:
            for start in range(0, len(ids), 256):
                rows: Any = client.get(
                    collection_name=self._collection,
                    ids=ids[start : start + 256],
                    output_fields=["chunk_id", "vector"],
                )
                for row in rows or []:
                    chunk_id = str(row.get("chunk_id", ""))
                    vector = row.get("vector")
                    if chunk_id and vector is not None:
                        vectors[chunk_id] = list(vector)
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 按 id 取向量失败：{self._collection}", details={"error": str(exc)}
            ) from exc
        return vectors

    def get_records_by_doc(self, doc_id: str) -> list[VectorRecord] | None:
        """按文档取回完整行（含向量与 metadata）：血缘 ACL 调和时整行刷新用。

        单文档 chunk 量有限（切块配置上限内），一次性 query 取回即可；
        记录字段与 upsert 行结构一一对应，可修改 metadata 后原样回写。
        """
        client = self._get_client()
        records: list[VectorRecord] = []
        try:
            rows: Any = client.query(
                collection_name=self._collection,
                filter=f'doc_id == "{doc_id}"',
                output_fields=[
                    "chunk_id",
                    "vector",
                    "doc_id",
                    "kb_id",
                    "tenant_id",
                    "content",
                    "source_uri",
                    "version",
                    "chunk_index",
                    "metadata",
                ],
                limit=16384,
            )
            for row in rows or []:
                records.append(
                    VectorRecord(
                        chunk_id=str(row.get("chunk_id", "")),
                        vector=[float(v) for v in (row.get("vector") or [])],
                        content=str(row.get("content", "")),
                        doc_id=str(row.get("doc_id", "")),
                        kb_id=str(row.get("kb_id", "")),
                        tenant_id=str(row.get("tenant_id", "")),
                        source_uri=str(row.get("source_uri", "")),
                        version=_as_int(row.get("version")) or 0,
                        chunk_index=_as_int(row.get("chunk_index")) or 0,
                        metadata=dict(row.get("metadata") or {}),
                    )
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 按文档取记录失败：{self._collection}",
                details={"doc_id": doc_id, "error": str(exc)},
            ) from exc
        return records

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """按主键精确删除（幂等）：增量重建时清理失效 chunk，不动同文档新版本。"""
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return 0
        client = self._get_client()
        deleted = 0
        try:
            for start in range(0, len(ids), 500):
                result: Any = client.delete(
                    collection_name=self._collection, ids=ids[start : start + 500]
                )
                if isinstance(result, dict):
                    deleted += int(result.get("delete_count", 0))
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 按 id 删除失败：{self._collection}", details={"error": str(exc)}
            ) from exc
        return deleted

    def list_chunk_ids(self, *, batch_size: int = 1000) -> Iterator[list[str]]:
        """分批遍历集合内全部 chunk_id：优先 query_iterator（服务端游标），
        不支持时（如 Milvus Lite）退化为 offset 分页（窗口上限 16384，超限记告警）。"""
        client = self._get_client()
        try:
            iterator = client.query_iterator(
                collection_name=self._collection,
                filter="",
                output_fields=["chunk_id"],
                batch_size=batch_size,
            )
        except (AttributeError, TypeError):
            iterator = None
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 遍历 chunk_id 失败：{self._collection}", details={"error": str(exc)}
            ) from exc

        if iterator is not None:
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    yield [str(row.get("chunk_id", "")) for row in batch if row.get("chunk_id")]
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            return

        offset = 0
        while offset < 16384:
            rows: Any = client.query(
                collection_name=self._collection,
                filter="",
                output_fields=["chunk_id"],
                limit=batch_size,
                offset=offset,
            )
            if not rows:
                break
            yield [str(row.get("chunk_id", "")) for row in rows if row.get("chunk_id")]
            if len(rows) < batch_size:
                break
            offset += len(rows)
        else:
            logger.warning(
                "offset 分页达到 16384 窗口上限 collection=%s，对账可能不完整；"
                "请升级 pymilvus 以启用 query_iterator",
                self._collection,
            )

    def search(
        self, vector: Sequence[float], *, top_k: int, expr: str | None = None
    ) -> list[VectorHit]:
        """COSINE 相似度检索；expr 由 scope_expression 构造（租户/KB/ACL 三重过滤）。

        HNSW 家族索引显式下发 ef（候选队列宽度），召回率/延迟由配置决定而非
        服务端默认值；ef 低于 top_k 时自动抬到 top_k，否则召回上限被 ef 截断。
        """
        client = self._get_client()
        search_params: dict[str, Any] = {"metric_type": "COSINE"}
        if self._index_type.startswith("HNSW"):
            search_params["params"] = {"ef": max(self._hnsw_ef_search, top_k)}
        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "data": [list(vector)],
            "anns_field": "vector",
            "limit": top_k,
            "output_fields": _OUTPUT_FIELDS,
            "search_params": search_params,
            # Strong 一致性：入库后立即可检索（read-your-writes），
            # 默认 Bounded 存在秒级 staleness 窗口，新文档上传后立刻提问会查不到
            "consistency_level": "Strong",
        }
        if expr:
            kwargs["filter"] = expr
        try:
            raw_hits: Any = client.search(**kwargs)
        except Exception as exc:
            raise VectorStoreError(
                f"Milvus 检索失败：{self._collection}", details={"error": str(exc)}
            ) from exc

        hits: list[VectorHit] = []
        for item in raw_hits or []:
            for raw in item:
                hit = _parse_hit(raw)
                if hit is not None:
                    hits.append(hit)
        return hits

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None


def _parse_hit(raw: Any) -> VectorHit | None:
    if isinstance(raw, dict):
        chunk_id = str(raw.get("id") or raw.get("chunk_id") or "")
        score = float(raw.get("distance", 0.0))
        entity = raw.get("entity") or {}
    else:
        chunk_id = str(getattr(raw, "id", ""))
        score = float(getattr(raw, "distance", 0.0))
        entity = getattr(raw, "entity", {}) or {}

    if not chunk_id:
        return None
    if not isinstance(entity, dict):
        entity = {}

    metadata = entity.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    return VectorHit(
        chunk_id=chunk_id,
        score=score,
        content=str(entity.get("content", "")),
        doc_id=str(entity.get("doc_id", "")),
        source_uri=str(entity.get("source_uri", "")),
        metadata=metadata if isinstance(metadata, dict) else {},
    )
