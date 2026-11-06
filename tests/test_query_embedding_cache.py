"""Query Embedding 缓存（9.7 第一层缓存）。"""

from __future__ import annotations

from app.retrieval.embedding_cache import QueryEmbeddingCache


def _cache(ttl_s: int = 3600, max_entries: int = 4) -> QueryEmbeddingCache:
    # 不传 redis_url：只用进程内 LRU，测试不依赖外部服务
    return QueryEmbeddingCache(ttl_s=ttl_s, max_entries=max_entries)


def test_local_lru_hit_after_set() -> None:
    cache = _cache()
    kwargs = {
        "tenant_id": "default",
        "kb_id": "kb-1",
        "model": "fake",
        "embedding_version": "v1",
        "text": "报销流程",
    }
    assert cache.get(**kwargs) is None
    cache.set(**kwargs, vector=[0.1, 0.2])
    assert cache.get(**kwargs) == [0.1, 0.2]


def test_key_isolated_by_model_version_and_tenant() -> None:
    """换模型 / 升 embedding_version 后旧向量不可复用，缓存 key 必须变化。"""
    cache = _cache()
    base = {
        "tenant_id": "default",
        "kb_id": "kb-1",
        "model": "m1",
        "embedding_version": "v1",
        "text": "问题",
    }
    cache.set(**base, vector=[1.0])
    assert cache.get(**{**base, "model": "m2"}) is None
    assert cache.get(**{**base, "embedding_version": "v2"}) is None
    assert cache.get(**{**base, "tenant_id": "other"}) is None


def test_lru_evicts_oldest_entry() -> None:
    cache = _cache(max_entries=2)
    for text in ("问题一", "问题二", "问题三"):
        cache.set(
            tenant_id="default",
            kb_id="kb-1",
            model="m",
            embedding_version="v1",
            text=text,
            vector=[float(len(text))],
        )
    assert (
        cache.get(
            tenant_id="default",
            kb_id="kb-1",
            model="m",
            embedding_version="v1",
            text="问题一",
        )
        is None
    )
    assert (
        cache.get(
            tenant_id="default",
            kb_id="kb-1",
            model="m",
            embedding_version="v1",
            text="问题三",
        )
        is not None
    )


def test_disabled_when_ttl_non_positive() -> None:
    cache = _cache(ttl_s=0)
    assert cache.enabled is False
    assert (
        cache.get(
            tenant_id="default",
            kb_id="kb-1",
            model="m",
            embedding_version="v1",
            text="问题",
        )
        is None
    )
