"""检索缓存单元测试：key 规则、命中/序列化往返、按库失效、TTL 与降级（fakeredis 内存实现）。"""

from __future__ import annotations

import fakeredis
import pytest
import redis

from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig
from app.retrieval.cache import KEY_PREFIX, RetrievalCache


def _chunk(chunk_id: str, content: str = "片段内容") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=0.88,
        doc_id="doc-1",
        source_uri="kb://finance/a.md",
        title="文档 A",
        metadata={"title": "文档 A", "page": 1},
    )


@pytest.fixture()
def fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture()
def cache(fake_redis: fakeredis.FakeRedis) -> RetrievalCache:
    return RetrievalCache("redis://unused", ttl_s=300, client=fake_redis)


def _key(cache: RetrievalCache, **overrides: object) -> str:
    kwargs = {
        "tenant_id": "default",
        "kb_id": "kb-finance",
        "question": "报销比例是多少",
        "top_k": 20,
        "use_rewrite": True,
        "use_rerank": False,
        "config": RetrievalConfig(),
    }
    kwargs.update(overrides)
    return cache.build_key(**kwargs)  # type: ignore[arg-type]


def test_disabled_cache_is_noop() -> None:
    cache = RetrievalCache("redis://unused", ttl_s=0)
    assert not cache.enabled
    assert cache.get("any") is None
    cache.set("any", [_chunk("c1")])  # 不抛错
    assert cache.invalidate_kb("default", "kb-finance") == 0


def test_set_then_get_roundtrip(cache: RetrievalCache) -> None:
    key = _key(cache)
    assert cache.get(key) is None

    chunks = [_chunk("c1"), _chunk("c2", "另一段")]
    cache.set(key, chunks)
    hit = cache.get(key)
    assert hit is not None
    assert [c.chunk_id for c in hit] == ["c1", "c2"]
    assert hit[0] == chunks[0]  # dataclass 全字段相等（含 metadata）


def test_set_writes_ttl(cache: RetrievalCache, fake_redis: fakeredis.FakeRedis) -> None:
    key = _key(cache)
    cache.set(key, [_chunk("c1")])
    ttl = fake_redis.ttl(key)
    assert 0 < ttl <= 300


def test_key_changes_with_params_and_config(cache: RetrievalCache) -> None:
    base = _key(cache)
    assert base.startswith(f"{KEY_PREFIX}:default:kb-finance:")
    assert _key(cache, question="另一个问题") != base
    assert _key(cache, top_k=10) != base
    assert _key(cache, use_rerank=True) != base
    assert _key(cache, kb_id="kb-hr") != base
    assert _key(cache, tenant_id="other") != base
    # 检索配置变更（如调权重）→ key 变化，旧缓存自然失效
    assert _key(cache, config=RetrievalConfig(vector_weight=0.7, bm25_weight=0.3)) != base
    # ACL 用户组参与 key：不同组的用户不能共享召回缓存
    assert _key(cache, allowed_groups=frozenset({"finance"})) != base
    assert _key(cache, allowed_groups=frozenset({"hr"})) != _key(
        cache, allowed_groups=frozenset({"finance"})
    )
    # 组顺序不影响 key
    assert _key(cache, allowed_groups=frozenset({"hr", "finance"})) == _key(
        cache, allowed_groups=frozenset({"finance", "hr"})
    )
    # embedding 版本进 key：换模型升版本后旧缓存自动失效，无需手动清理
    assert _key(cache, embedding_version="v2") != base
    assert _key(cache, embedding_version="") == base  # 缺省保持既有行为
    # 与检索无关的字段不在配置模型内，无需考虑；同参数 key 必须稳定
    assert _key(cache) == base


def test_acl_key_is_type_sensitive() -> None:
    """ACL 三态与集合内容必须互不碰撞（用户复现：不同权限命中同一缓存）。"""
    cache = RetrievalCache("redis://unused", ttl_s=300, client=fakeredis.FakeRedis(
        decode_responses=True
    ))
    base = _key(cache)
    # 不限权（None）不能与"显式空组"或"仅 * 组"共用缓存
    unrestricted = _key(cache, allowed_groups=None)
    empty_groups = _key(cache, allowed_groups=frozenset())
    star_group = _key(cache, allowed_groups=frozenset({"*"}))
    assert len({unrestricted, empty_groups, star_group}) == 3
    assert unrestricted == base  # 缺省即"不限权"，语义不变
    # 逗号拼接碰撞：{"finance","hr"} vs {"finance,hr"} 必须是不同 key
    two_groups = _key(cache, allowed_groups=frozenset({"finance", "hr"}))
    comma_group = _key(cache, allowed_groups=frozenset({"finance,hr"}))
    assert two_groups != comma_group
    # 排除集合同样使用规范 JSON：空集合与含空串的集合不碰撞
    assert _key(cache, excluded_doc_ids=frozenset()) == base
    assert _key(cache, excluded_doc_ids=frozenset({"a,b"})) != _key(
        cache, excluded_doc_ids=frozenset({"a", "b"})
    )


def test_private_cache_not_shared_across_permission_sets(
    cache: RetrievalCache,
) -> None:
    """复现用户场景：高权限用户先查，低权限用户不能命中其私有缓存。"""
    private = _key(cache, allowed_groups=frozenset({"finance"}))
    cache.set(private, [_chunk("c-private", "财务私有内容")])
    other = _key(cache, allowed_groups=frozenset({"hr"}))
    assert cache.get(other) is None
    assert cache.get(_key(cache, allowed_groups=frozenset({"finance,hr"}))) is None


def test_invalidate_kb_only_clears_own_prefix(cache: RetrievalCache) -> None:
    key_finance = _key(cache, kb_id="kb-finance")
    key_hr = _key(cache, kb_id="kb-hr")
    cache.set(key_finance, [_chunk("c1")])
    cache.set(key_hr, [_chunk("c2")])

    removed = cache.invalidate_kb("default", "kb-finance")
    assert removed == 1
    assert cache.get(key_finance) is None
    assert cache.get(key_hr) is not None  # 其他知识库不受影响


def test_corrupted_payload_treated_as_miss(
    cache: RetrievalCache, fake_redis: fakeredis.FakeRedis
) -> None:
    key = _key(cache)
    fake_redis.set(key, "not-a-json")
    assert cache.get(key) is None
    assert fake_redis.exists(key) == 0  # 脏 key 被清理


class _BrokenRedis:
    """所有操作都抛 RedisError，验证降级路径。"""

    def get(self, key: str) -> None:
        raise redis.RedisError("connection refused")

    def setex(self, *args: object) -> None:
        raise redis.RedisError("connection refused")

    def scan_iter(self, **kwargs: object) -> None:
        raise redis.RedisError("connection refused")

    def delete(self, *args: object) -> None:
        raise redis.RedisError("connection refused")

    def close(self) -> None:
        return None


def test_redis_failure_degrades_gracefully() -> None:
    cache = RetrievalCache("redis://unused", ttl_s=300, client=_BrokenRedis())  # type: ignore[arg-type]
    assert cache.get("k") is None  # 读失败按未命中
    cache.set("k", [_chunk("c1")])  # 写失败不抛错
    assert cache.invalidate_kb("default", "kb-finance") == 0  # 失效失败返回 0
    cache.close()
