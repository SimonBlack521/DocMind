"""Elasticsearch BM25 稀疏索引，与 Milvus 稠密向量检索互补（混合检索的稀疏腿）。

每个知识库一个索引 rag_{kb_code}_chunks，并挂别名 rag_{kb_code}_search：
读写都走别名，未来需要零停机重建（换 analyzer、改 mapping）时新建索引再切别名即可。

mapping 要点：id 类字段全部 keyword（精确过滤）；content/title/section_path 是 text 走
分词打分（检索按 title^3 / section_path^2 / content^1 多字段加权，另带 content 短语
子句与 source_uri.text 文件名匹配——错误码、文件名、制度名称类精确查询更准）；
acl_groups 是 keyword 数组，检索时用 terms 查询做文档级 ACL 过滤——查询组列表
由用户组 ∪ 公开哨兵 "*" 组成（与 Milvus 侧 scope_expression 同一约定），
命中任一即可见；metadata object(enabled=false) 只存不索引，透传回业务层。

注意：title/section_path/source_uri.text 是后加字段，存量文档在下次写入前不参与
这些字段的打分（ES 不会自动重建存量文档的倒排），功能降级为仅 content 检索，不报错。

生产环境建议把 content 的 analyzer 换成已安装的 IK（ik_smart），中文召回效果更好；
这里默认 standard，保证不装额外插件也能跑通。
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from typing import Any

from elasticsearch import Elasticsearch, helpers

from app.core.errors import KeywordStoreError
from app.core.logging import get_logger
from app.storage.keyword_base import KeywordHit, KeywordRecord, KeywordStore
from app.storage.vector_base import ACL_PUBLIC_GROUP

logger = get_logger(__name__)

_SOURCE_FIELDS = [
    "chunk_id",
    "doc_id",
    "content",
    "title",
    "section_path",
    "source_uri",
    "version",
    "chunk_index",
    "metadata",
]


class ElasticsearchKeywordStore(KeywordStore):
    """ES 实现：bulk 覆盖写（_id=chunk_id 保证幂等）、bool must+filter 检索。"""

    def __init__(
        self,
        *,
        url: str,
        index: str,
        alias: str,
        username: str | None = None,
        password: str | None = None,
        analyzer: str = "standard",
        timeout: float = 30.0,
        number_of_shards: int = 1,
        number_of_replicas: int = 0,
    ) -> None:
        self._index = index
        self._alias = alias
        self._analyzer = analyzer
        self._shards = max(1, number_of_shards)
        self._replicas = max(0, number_of_replicas)
        auth: tuple[str, str] | None = (
            (username, password) if username and password else None
        )
        self._client = Elasticsearch(hosts=[url], basic_auth=auth, request_timeout=timeout)

    def _index_settings(self) -> dict[str, Any]:
        """v1.4-3: ES 索引 settings——按规模动态 shards/replicas，批量写入时禁用 refresh。"""
        return {
            "number_of_shards": self._shards,
            "number_of_replicas": self._replicas,
            # 批量写入阶段（bulk upsert）不走实时 refresh，查询时再触发
            "refresh_interval": "1s",
            # BM25 TF/IDF 归一化：忽略停用词、限制字段长度（百万级文档防膨胀）
            "analysis": {
                "analyzer": {
                    "default": {"type": self._analyzer},
                }
            },
        }

    def ensure_index(self) -> None:
        try:
            if self._client.indices.exists_alias(name=self._alias):
                return

            if not self._client.indices.exists(index=self._index):
                self._client.indices.create(
                    index=self._index,
                    settings=self._index_settings(),
                    mappings={
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "kb_id": {"type": "keyword"},
                            "tenant_id": {"type": "keyword"},
                            "content": {"type": "text", "analyzer": self._analyzer},
                            # 多字段加权检索：标题/章节路径独立成字段（清单第 23 项）
                            "title": {"type": "text", "analyzer": self._analyzer},
                            "section_path": {"type": "text", "analyzer": self._analyzer},
                            # keyword 精确过滤不变；text 子字段支持文件名类查询分词匹配
                            "source_uri": {
                                "type": "keyword",
                                "ignore_above": 1024,
                                "fields": {
                                    "text": {"type": "text", "analyzer": self._analyzer}
                                },
                            },
                            "version": {"type": "integer"},
                            "chunk_index": {"type": "integer"},
                            # 父子切片：parent 子块只用于回取，检索时用 must_not 排除
                            "chunk_role": {"type": "keyword"},
                            # 文档级 ACL：与 Milvus 同一约定，公开文档为 ["*"]
                            "acl_groups": {"type": "keyword"},
                            "metadata": {"type": "object", "enabled": False},
                        }
                    },
                )
                logger.info("创建 ES 索引=%s", self._index)
            else:
                # 存量索引补字段映射：新增字段/子字段是幂等操作，不影响既有文档
                # （既有文档在新字段上无倒排，下次写入时才参与打分）
                self._client.indices.put_mapping(
                    index=self._index,
                    properties={
                        "acl_groups": {"type": "keyword"},
                        "chunk_role": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": self._analyzer},
                        "section_path": {"type": "text", "analyzer": self._analyzer},
                        "source_uri": {
                            "type": "keyword",
                            "ignore_above": 1024,
                            "fields": {"text": {"type": "text", "analyzer": self._analyzer}},
                        },
                    },
                )

            self._client.indices.put_alias(index=self._index, name=self._alias)
            logger.info("创建 ES 别名=%s -> %s", self._alias, self._index)
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 索引初始化失败：{self._index}", details={"error": str(exc)}
            ) from exc

    def upsert(self, records: Sequence[KeywordRecord]) -> int:
        if not records:
            return 0
        self.ensure_index()

        actions = [
            {
                "_op_type": "index",
                "_index": self._index,
                "_id": record.chunk_id,
                "_source": {
                    "chunk_id": record.chunk_id,
                    "doc_id": record.doc_id,
                    "kb_id": record.kb_id,
                    "tenant_id": record.tenant_id,
                    "content": record.content,
                    # 提升为一级字段：多字段加权检索直接打分，不再依赖 metadata 回读
                    "title": str(record.metadata.get("title") or ""),
                    "section_path": str(record.metadata.get("section_path") or ""),
                    "source_uri": record.source_uri,
                    "version": record.version,
                    "chunk_index": record.chunk_index,
                    # pipeline 入库时已把 acl_groups 归一化进 chunk metadata；缺省按公开兜底
                    "acl_groups": record.metadata.get("acl_groups") or [ACL_PUBLIC_GROUP],
                    "chunk_role": record.metadata.get("chunk_role") or "child",
                    "metadata": record.metadata,
                },
            }
            for record in records
        ]

        try:
            result: Any = helpers.bulk(
                self._client, actions, refresh=False, raise_on_error=False, stats_only=False
            )
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 写入失败：{self._index}", details={"error": str(exc)}
            ) from exc

        success = int(result[0]) if isinstance(result, tuple) else 0
        errors = list(result[1]) if isinstance(result, tuple) else []
        if errors:
            raise KeywordStoreError(
                "ES 部分写入失败", details={"failed": len(errors), "first_error": str(errors[0])}
            )
        return success

    def delete_by_doc(self, doc_id: str) -> int:
        try:
            response: Any = self._client.delete_by_query(
                index=self._alias, query={"term": {"doc_id": doc_id}}, refresh=False
            )
            self._client.indices.refresh(index=self._alias)
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 删除失败：{self._alias}", details={"doc_id": doc_id, "error": str(exc)}
            ) from exc

        body = getattr(response, "body", response)
        return int(body.get("deleted", 0)) if isinstance(body, dict) else 0

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """按 chunk_id 精确删除（幂等）：增量重建时清理失效 chunk，不动同文档新版本。

        v1.4-3 批量优化：每 1000 条一批 delete_by_query，全部删完后统一 refresh 一次，
        避免每批 refresh 带来的 I/O 放大（百万级增量重建时可省数分钟刷新开销）。
        """
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return 0
        deleted = 0
        try:
            for start in range(0, len(ids), 1000):
                response: Any = self._client.delete_by_query(
                    index=self._alias,
                    query={"terms": {"chunk_id": ids[start : start + 1000]}},
                    refresh=False,
                )
                body = getattr(response, "body", response)
                if isinstance(body, dict):
                    deleted += int(body.get("deleted", 0))
            # 批量删完后统一 refresh 一次（查询时能立刻看到效果）
            self._client.indices.refresh(index=self._alias)
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 按 id 删除失败：{self._alias}", details={"error": str(exc)}
            ) from exc
        return deleted

    def list_chunk_ids(self, *, batch_size: int = 1000) -> Iterator[list[str]]:
        """scroll 遍历全部 chunk_id（_source=False 只带 _id），供对账任务使用。"""
        try:
            batch: list[str] = []
            for hit in helpers.scan(
                self._client,
                index=self._alias,
                query={"query": {"match_all": {}}, "sort": "_doc"},
                _source=False,
                size=batch_size,
            ):
                chunk_id = str(hit.get("_id", ""))
                if chunk_id:
                    batch.append(chunk_id)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 遍历 chunk_id 失败：{self._alias}", details={"error": str(exc)}
            ) from exc

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[KeywordHit]:
        """按 chunk_id 批量回取（mget）：父子切片回取父块时只发一次请求。"""
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return []
        try:
            response: Any = self._client.mget(index=self._alias, ids=ids)
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 批量回取失败：{self._alias}", details={"error": str(exc)}
            ) from exc

        body = getattr(response, "body", response)
        docs = body.get("docs", []) if isinstance(body, dict) else []
        hits: list[KeywordHit] = []
        for doc in docs:
            if not doc.get("found"):
                continue
            hits.append(_to_keyword_hit(doc.get("_source", {}), fallback_id=doc.get("_id", "")))
        return hits

    def get_records_by_doc(self, doc_id: str) -> list[KeywordRecord] | None:
        """按文档取回完整记录（含 kb_id/tenant_id/version）：血缘 ACL 调和刷新用。

        单文档 chunk 量有限（含父块也在切块上限内），一次 terms 查询取回即可。
        """
        try:
            response: Any = self._client.search(
                index=self._alias,
                query={"term": {"doc_id": doc_id}},
                size=10000,
                # 等价于 REST 的 _source 字段列表（8.x 客户端 source 形参不接受序列）
                source_includes=[
                    "chunk_id",
                    "doc_id",
                    "kb_id",
                    "tenant_id",
                    "content",
                    "source_uri",
                    "version",
                    "chunk_index",
                    "metadata",
                ],
            )
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 按文档取记录失败：{self._alias}",
                details={"doc_id": doc_id, "error": str(exc)},
            ) from exc
        body = getattr(response, "body", response)
        raw_hits = body.get("hits", {}).get("hits", []) if isinstance(body, dict) else []
        records: list[KeywordRecord] = []
        for hit in raw_hits:
            src = hit.get("_source") or {}
            records.append(
                KeywordRecord(
                    chunk_id=str(src.get("chunk_id", "")),
                    doc_id=str(src.get("doc_id", "")),
                    kb_id=str(src.get("kb_id", "")),
                    tenant_id=str(src.get("tenant_id", "")),
                    content=str(src.get("content", "")),
                    source_uri=str(src.get("source_uri", "")),
                    version=int(src.get("version") or 0),
                    chunk_index=int(src.get("chunk_index") or 0),
                    metadata=dict(src.get("metadata") or {}),
                )
            )
        return records

    def search(
        self,
        query: str,
        *,
        top_k: int,
        kb_id: str | None = None,
        allowed_groups: Collection[str] | None = None,
        excluded_doc_ids: Collection[str] | None = None,
    ) -> list[KeywordHit]:
        filters: list[dict[str, Any]] = []
        if kb_id:
            filters.append({"term": {"kb_id": kb_id}})
        if allowed_groups is not None:
            # 与 Milvus 侧同一约定：公开文档的 acl_groups 含哨兵 "*"
            visible = sorted(
                {group.strip() for group in allowed_groups if group.strip()}
                | {ACL_PUBLIC_GROUP}
            )
            filters.append({"terms": {"acl_groups": visible}})
        bool_query: dict[str, Any] = {
            # 多字段加权（清单第 23 项）：标题 ^3 / 章节路径 ^2 / 正文 ^1，
            # best_fields 取单字段最高分——错误码、文件名、制度名称类精确查询
            # 会被标题/章节命中显著提前，不再被正文长文本稀释
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "section_path^2", "content"],
                        "type": "best_fields",
                    }
                },
                # 短语子句：查询整体作为短语命中正文时额外加权（提升精确问句的区分度）
                {"match_phrase": {"content": {"query": query, "boost": 2.0}}},
                # 文件名/路径类查询：source_uri 的 text 子字段分词匹配
                {"match": {"source_uri.text": {"query": query, "boost": 2.0}}},
            ],
            "minimum_should_match": 1,
            "filter": filters,
            # 父块只用于回取，绝不作为检索候选返回（否则会挤占 top_k 名额）
            "must_not": [{"term": {"chunk_role": "parent"}}],
        }
        if excluded_doc_ids:
            # 排除被新版本替代的文档：过滤在 ES 内执行，不占用 top_k 返回名额
            bool_query["must_not"].append(
                {"terms": {"doc_id": sorted({doc_id for doc_id in excluded_doc_ids if doc_id})}}
            )
        request = {
            "query": {"bool": bool_query},
            "size": top_k,
            "_source": _SOURCE_FIELDS,
        }
        try:
            response: Any = self._client.search(index=self._alias, **request)
        except Exception as exc:
            raise KeywordStoreError(
                f"ES 检索失败：{self._alias}", details={"error": str(exc)}
            ) from exc

        body = getattr(response, "body", response)
        raw_hits = body.get("hits", {}).get("hits", []) if isinstance(body, dict) else []

        hits: list[KeywordHit] = []
        for hit in raw_hits:
            hits.append(
                _to_keyword_hit(
                    hit.get("_source", {}),
                    fallback_id=hit.get("_id", ""),
                    score=float(hit.get("_score") or 0.0),
                )
            )
        return hits

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.warning("关闭 ES 客户端失败", exc_info=True)


def _to_keyword_hit(
    source: dict[str, Any], *, fallback_id: str = "", score: float = 0.0
) -> KeywordHit:
    """把 ES `_source` 统一映射成 KeywordHit（search / mget 两条路径共用）。"""
    metadata = source.get("metadata", {})
    return KeywordHit(
        chunk_id=str(source.get("chunk_id") or fallback_id),
        score=score,
        content=str(source.get("content", "")),
        doc_id=str(source.get("doc_id", "")),
        source_uri=str(source.get("source_uri", "")),
        metadata=metadata if isinstance(metadata, dict) else {},
    )
