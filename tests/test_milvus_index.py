"""S1 索引层：显式 HNSW 参数、索引漂移对账、检索 ef 下发的单元测试。

用假 MilvusClient 替换 pymilvus.MilvusClient（monkeypatch），不依赖真实 Milvus。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.core.settings import Settings
from app.storage import milvus_store
from app.storage.milvus_store import MilvusVectorStore


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, Any]] = []

    def add_field(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.fields.append((name, kwargs))


class _FakeMilvusClient:
    """最小可用的 MilvusClient 替身：记录调用，索引描述可由用例注入。"""

    # 类属性：各用例通过类属性注入"已存在集合的索引描述"
    existing_index: dict[str, Any] | None = None
    has_existing_collection = True

    def __init__(self, **kwargs: Any) -> None:
        self.created_collections: list[dict[str, Any]] = []
        self.created_indexes: list[Any] = []
        self.dropped_indexes: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.index_params_created: list[_FakeIndexParams] = []

    # ---- 元数据 ----
    def has_collection(self, name: str) -> bool:
        return type(self).has_existing_collection

    def describe_collection(self, name: str) -> dict[str, Any]:
        return {"fields": [{"name": "vector", "params": {"dim": 8}}]}

    def list_indexes(self, collection_name: str) -> list[str]:
        if type(self).existing_index is None:
            return []
        return ["vector"]

    def describe_index(self, collection_name: str, index_name: str) -> dict[str, Any]:
        info = type(self).existing_index or {}
        return {**info, "field_name": "vector"}

    # ---- 建集合 / 索引 ----
    def create_schema(self, **kwargs: Any) -> _FakeSchema:
        return _FakeSchema()

    def prepare_index_params(self) -> _FakeIndexParams:
        params = _FakeIndexParams()
        self.index_params_created.append(params)
        return params

    def create_collection(self, **kwargs: Any) -> None:
        self.created_collections.append(kwargs)

    def create_index(self, collection_name: str, index_params: Any) -> None:
        self.created_indexes.append(index_params)

    def drop_index(self, collection_name: str, index_name: str) -> None:
        self.dropped_indexes.append(index_name)

    def release_collection(self, collection_name: str) -> None:
        pass

    def load_collection(self, collection_name: str) -> None:
        pass

    # ---- 检索 ----
    def search(self, **kwargs: Any) -> list[Any]:
        self.search_calls.append(kwargs)
        return [[]]

    def close(self) -> None:  # pragma: no cover - 替身无需释放
        pass


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMilvusClient]:
    _FakeMilvusClient.existing_index = None
    _FakeMilvusClient.has_existing_collection = True
    monkeypatch.setattr(milvus_store, "MilvusClient", _FakeMilvusClient)
    return _FakeMilvusClient


def _store(**kwargs: Any) -> MilvusVectorStore:
    defaults: dict[str, Any] = {"uri": "http://x", "collection": "rag_test"}
    defaults.update(kwargs)
    return MilvusVectorStore(**defaults)


def _last_client(store: MilvusVectorStore) -> _FakeMilvusClient:
    client = store._get_client()
    assert isinstance(client, _FakeMilvusClient)
    return client


# ---------------------------------------------------------------- 建集合
def test_create_collection_with_explicit_hnsw(fake_client: type[_FakeMilvusClient]) -> None:
    fake_client.has_existing_collection = False
    store = _store(index_type="HNSW", hnsw_m=24, hnsw_ef_construction=300)
    store.ensure_collection(8)

    client = _last_client(store)
    assert len(client.created_collections) == 1
    index_params = client.created_collections[0]["index_params"]
    assert index_params.indexes[0]["index_type"] == "HNSW"
    assert index_params.indexes[0]["metric_type"] == "COSINE"
    assert index_params.indexes[0]["params"] == {"M": 24, "efConstruction": 300}


def test_create_collection_flat_without_hnsw_params(
    fake_client: type[_FakeMilvusClient],
) -> None:
    fake_client.has_existing_collection = False
    store = _store(index_type="FLAT")
    store.ensure_collection(8)

    index_params = _last_client(store).created_collections[0]["index_params"]
    assert index_params.indexes[0]["index_type"] == "FLAT"
    assert "params" not in index_params.indexes[0]


# ---------------------------------------------------------------- 索引漂移对账
def test_index_drift_autoindex_to_hnsw_rebuilds(
    fake_client: type[_FakeMilvusClient],
) -> None:
    """索引漂移（AUTOINDEX→HNSW）：入库路径只告警不重建；维护入口 reconcile_index_params 才重建。

    项22 契约：drop_index 期间查询会降级，重建集中到维护窗口（rag.reconcile_index），
    入库热路径（ensure_collection，auto_reconcile 默认关）绝不重建索引。
    """
    fake_client.existing_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
    store = _store(index_type="HNSW")
    store.ensure_collection(8)

    client = _last_client(store)
    # 入库路径：漂移不重建，仅补齐缺失的标量索引
    assert client.dropped_indexes == []
    assert all(
        idx.indexes[0]["field_name"] != "vector" for idx in client.created_indexes
    )

    # 维护入口：允许原地重建
    assert store.reconcile_index_params() is True
    assert client.dropped_indexes == ["vector"]
    vector_rebuilds = [
        idx for idx in client.created_indexes if idx.indexes[0]["field_name"] == "vector"
    ]
    assert len(vector_rebuilds) == 1
    assert vector_rebuilds[0].indexes[0]["index_type"] == "HNSW"


def test_index_drift_hnsw_params_rebuilds(fake_client: type[_FakeMilvusClient]) -> None:
    fake_client.existing_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "M": "8",  # describe_index 参数值可能是字符串
        "efConstruction": "100",
    }
    store = _store(index_type="HNSW", hnsw_m=16, hnsw_ef_construction=200)
    store.ensure_collection(8)

    client = _last_client(store)
    # 入库路径：参数漂移不重建
    assert client.dropped_indexes == []
    # 维护入口：按当前配置重建
    assert store.reconcile_index_params() is True
    assert client.dropped_indexes == ["vector"]
    vector_rebuilds = [
        idx for idx in client.created_indexes if idx.indexes[0]["field_name"] == "vector"
    ]
    assert len(vector_rebuilds) == 1


def test_index_no_drift_keeps_index(fake_client: type[_FakeMilvusClient]) -> None:
    fake_client.existing_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "M": "16",
        "efConstruction": "200",
    }
    store = _store(index_type="HNSW", hnsw_m=16, hnsw_ef_construction=200)
    store.ensure_collection(8)

    client = _last_client(store)
    assert client.dropped_indexes == []
    # 向量索引无漂移不动；存量集合缺标量索引（项22），幂等补齐 tenant_id/kb_id/doc_id
    scalar_fields = {
        idx.indexes[0]["field_name"]
        for idx in client.created_indexes
        if idx.indexes[0].get("index_type") == "INVERTED"
    }
    assert scalar_fields == {"tenant_id", "kb_id", "doc_id"}
    assert all(
        idx.indexes[0]["field_name"] != "vector" for idx in client.created_indexes
    )


def test_missing_index_is_created(fake_client: type[_FakeMilvusClient]) -> None:
    fake_client.existing_index = None  # 集合存在但没有索引
    store = _store(index_type="HNSW")
    store.ensure_collection(8)

    client = _last_client(store)
    assert client.dropped_indexes == []
    # 缺向量索引必须补建（非破坏性，入库路径允许）+ 3 个标量索引补齐
    assert len(client.created_indexes) == 4
    fields = [idx.indexes[0]["field_name"] for idx in client.created_indexes]
    assert fields.count("vector") == 1
    assert sorted(f for f in fields if f != "vector") == ["doc_id", "kb_id", "tenant_id"]


# ---------------------------------------------------------------- 检索 ef 下发
def test_search_passes_ef_for_hnsw(fake_client: type[_FakeMilvusClient]) -> None:
    store = _store(index_type="HNSW", hnsw_ef_search=64)
    store.search([0.1] * 8, top_k=10)

    params = _last_client(store).search_calls[0]["search_params"]
    assert params == {"metric_type": "COSINE", "params": {"ef": 64}}


def test_search_ef_not_below_top_k(fake_client: type[_FakeMilvusClient]) -> None:
    store = _store(index_type="HNSW", hnsw_ef_search=16)
    store.search([0.1] * 8, top_k=32)

    params = _last_client(store).search_calls[0]["search_params"]
    assert params["params"]["ef"] == 32  # ef 被抬到 top_k，避免召回上限被截断


def test_search_no_ef_for_flat(fake_client: type[_FakeMilvusClient]) -> None:
    store = _store(index_type="FLAT")
    store.search([0.1] * 8, top_k=10)

    params = _last_client(store).search_calls[0]["search_params"]
    assert params == {"metric_type": "COSINE"}


# ---------------------------------------------------------------- 配置校验
def test_settings_hnsw_defaults() -> None:
    settings = Settings()
    assert settings.milvus_index_type == "HNSW"
    assert settings.milvus_hnsw_m == 16
    assert settings.milvus_hnsw_ef_construction == 200
    assert settings.milvus_hnsw_ef_search == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("milvus_hnsw_m", 3),
        ("milvus_hnsw_m", 65),
        ("milvus_hnsw_ef_construction", 7),
        ("milvus_hnsw_ef_construction", 2048),
        ("milvus_hnsw_ef_search", 0),
    ],
)
def test_settings_hnsw_range_validation(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})
