"""实时 API 检索测试（v2.0）：端点选择、请求参数、响应摘要与失败降级。

用 httpx.MockTransport 模拟外部接口，不发起真实网络请求。
"""

from __future__ import annotations

import httpx
import pytest

from app.core.errors import StructuredQueryError
from app.domain.kb_config import ApiEndpointConfig, ApiRetrievalConfig
from app.retrieval.query_router import QueryRouter
from app.structured.api_retriever import ApiRetriever


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_api_retriever_returns_json_summary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "当前库存是多少"
        assert request.headers["X-Token"] == "secret"
        return httpx.Response(200, json={"sku": "A-1", "stock": 42})

    config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[
            ApiEndpointConfig(
                endpoint_id="inventory",
                url="https://example.test/inventory",
                description="库存查询",
                headers={"X-Token": "secret"},
            )
        ],
    )

    answer = ApiRetriever(config, client=_client(handler)).answer("当前库存是多少")

    assert answer.endpoint_id == "inventory"
    assert answer.status_code == 200
    assert '"stock": 42' in answer.answer
    assert "inventory" in answer.answer


def test_api_retriever_selects_endpoint_by_description_overlap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[
            ApiEndpointConfig(endpoint_id="orders", url="https://example.test/orders"),
            ApiEndpointConfig(
                endpoint_id="inventory",
                url="https://example.test/inventory",
                description="库存数量查询",
            ),
        ],
    )

    answer = ApiRetriever(config, client=_client(handler)).answer("现在库存还有多少")

    assert answer.endpoint_id == "inventory"


def test_api_retriever_maps_http_error_to_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[ApiEndpointConfig(endpoint_id="orders", url="https://example.test/orders")],
    )

    with pytest.raises(StructuredQueryError) as exc_info:
        ApiRetriever(config, client=_client(handler)).answer("订单状态")
    assert exc_info.value.details["status_code"] == 500


def test_api_retriever_maps_network_error_to_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[ApiEndpointConfig(endpoint_id="orders", url="https://example.test/orders")],
    )

    with pytest.raises(StructuredQueryError):
        ApiRetriever(config, client=_client(handler)).answer("订单状态")


def test_api_retriever_truncates_plain_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 5000)

    config = ApiRetrievalConfig(
        enabled=True,
        max_chars=1000,
        endpoints=[ApiEndpointConfig(endpoint_id="status", url="https://example.test/status")],
    )

    answer = ApiRetriever(config, client=_client(handler)).answer("实时状态")

    assert len(answer.answer) <= 1000


def test_api_retriever_blocks_private_and_metadata_hosts() -> None:
    """SSRF 防护：内网/回环/云元数据地址默认拒绝，不能仅依赖"管理员配置过 URL"。"""
    config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[ApiEndpointConfig(endpoint_id="internal", url="http://127.0.0.1:8080/x")],
    )

    with pytest.raises(StructuredQueryError) as exc_info:
        ApiRetriever(config, client=_client(lambda request: httpx.Response(200))).answer(
            "内部状态"
        )
    assert "内网" in exc_info.value.message or "本机" in exc_info.value.message

    metadata_config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[
            ApiEndpointConfig(
                endpoint_id="metadata",
                url="http://169.254.169.254/latest/meta-data/",
            )
        ],
    )
    with pytest.raises(StructuredQueryError):
        ApiRetriever(
            metadata_config, client=_client(lambda request: httpx.Response(200))
        ).answer("元数据")


def test_api_retriever_rejects_oversized_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 500)

    config = ApiRetrievalConfig(
        enabled=True,
        max_response_bytes=100,
        endpoints=[ApiEndpointConfig(endpoint_id="status", url="https://example.test/status")],
    )

    with pytest.raises(StructuredQueryError) as exc_info:
        ApiRetriever(config, client=_client(handler)).answer("实时状态")
    assert "大小上限" in exc_info.value.message


def test_api_retriever_requires_endpoint_match_when_multiple() -> None:
    """多端点且完全无匹配时返回路由失败，而不是盲调第一个端点。"""
    config = ApiRetrievalConfig(
        enabled=True,
        endpoints=[
            ApiEndpointConfig(endpoint_id="orders", url="https://example.test/orders"),
            ApiEndpointConfig(
                endpoint_id="inventory",
                url="https://example.test/inventory",
                description="库存数量查询",
            ),
        ],
    )

    with pytest.raises(StructuredQueryError) as exc_info:
        ApiRetriever(config, client=_client(lambda request: httpx.Response(200))).answer(
            "今天天气怎么样"
        )
    assert "没有匹配" in exc_info.value.message


def test_api_config_requires_endpoint_when_enabled() -> None:
    with pytest.raises(ValueError):
        ApiRetrievalConfig(enabled=True)

    with pytest.raises(ValueError):
        ApiEndpointConfig(endpoint_id="bad", url="ftp://example.test")


def test_router_routes_realtime_question_to_api() -> None:
    decision = QueryRouter().route(
        "当前库存还有多少", structured_available=False, api_available=True
    )

    assert decision.route == "api"


def test_router_prefers_sql_when_both_available() -> None:
    decision = QueryRouter().route(
        "统计各部门报销总额", structured_available=True, api_available=True
    )

    assert decision.route == "sql"


def test_router_falls_back_to_document_without_realtime_marker() -> None:
    decision = QueryRouter().route(
        "年假有多少天", structured_available=True, api_available=True
    )

    assert decision.route == "document"
