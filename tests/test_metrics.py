"""监控指标测试：Metrics 聚合逻辑、HTTP 中间件埋点、缓存命中/未命中计数。"""

from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient

from app.core.metrics import Metrics, metrics
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig
from app.main import create_app
from app.retrieval.cache import RetrievalCache


def test_metrics_inc_observe_snapshot() -> None:
    m = Metrics()
    m.inc("requests_total", route="/health", status="200")
    m.inc("requests_total", route="/health", status="200")
    m.observe("latency_seconds", 0.5, route="/health")
    m.observe("latency_seconds", 1.5, route="/health")

    snap = m.snapshot()
    key = 'requests_total{route="/health",status="200"}'
    assert snap["counters"][key] == 2.0
    summary = snap["summaries"]['latency_seconds{route="/health"}']
    assert summary == {"count": 2, "sum": 2.0, "max": 1.5, "avg": 1.0}
    assert snap["uptime_seconds"] >= 0
    # 无缓存计数时命中率为 None
    assert snap["retrieval_cache_hit_rate"] is None


def test_metrics_cache_hit_rate() -> None:
    m = Metrics()
    m.inc("retrieval_cache_hits_total")
    m.inc("retrieval_cache_misses_total", 3)
    assert m.snapshot()["retrieval_cache_hit_rate"] == 0.25


def test_metrics_endpoint_collects_http_requests() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    counter_key = 'http_requests_total{method="GET",route="/health",status="200"}'
    assert payload["counters"][counter_key] >= 1
    summary = payload["summaries"]['http_request_duration_seconds{route="/health"}']
    assert summary["count"] >= 1
    assert "retrieval_cache_hit_rate" in payload


def test_cache_hit_and_miss_are_counted() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    cache = RetrievalCache("redis://unused", ttl_s=300, client=client)
    key = cache.build_key(
        tenant_id="default",
        kb_id="kb-finance",
        question="指标测试问题",
        top_k=20,
        use_rewrite=True,
        use_rerank=False,
        config=RetrievalConfig(),
    )
    before = metrics.snapshot()["counters"]
    hits_before = before.get("retrieval_cache_hits_total", 0.0)
    misses_before = before.get("retrieval_cache_misses_total", 0.0)

    assert cache.get(key) is None  # 未命中
    chunk = RetrievedChunk(chunk_id="c1", content="内容", score=0.9, doc_id="d1", source_uri="u")
    cache.set(key, [chunk])
    assert cache.get(key) is not None  # 命中

    after = metrics.snapshot()["counters"]
    assert after["retrieval_cache_hits_total"] == hits_before + 1
    assert after["retrieval_cache_misses_total"] == misses_before + 1
