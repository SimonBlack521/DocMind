"""稳定 ID 规则测试：doc_id / chunk_id 必须可重复计算且互不冲突。"""

from __future__ import annotations

from app.ingestion.chunker import compute_chunk_id
from app.ingestion.loader import compute_doc_id, normalize_source_uri


def test_normalize_url_removes_fragment_and_lowercases_host() -> None:
    uri = normalize_source_uri("HTTPS://Example.COM/docs/a.pdf#page=2")
    assert uri == "https://example.com/docs/a.pdf"


def test_normalize_local_path_unifies_separators() -> None:
    assert normalize_source_uri(r"C:\docs\a.txt") == "C:/docs/a.txt"
    assert normalize_source_uri("C:/docs/a.txt/") == "C:/docs/a.txt"


def test_doc_id_stable_for_equivalent_source_uris() -> None:
    assert compute_doc_id("t1", "kb-finance", "https://Example.com/docs/a.pdf") == compute_doc_id(
        "t1", "kb-finance", "https://example.com/docs/a.pdf#page=2"
    )
    assert compute_doc_id("t1", "kb-finance", r"C:\docs\a.txt") == compute_doc_id(
        "t1", "kb-finance", "C:/docs/a.txt"
    )


def test_doc_id_isolated_by_tenant_and_kb() -> None:
    base = compute_doc_id("t1", "kb-1", "https://example.com/a.pdf")
    assert base != compute_doc_id("t2", "kb-1", "https://example.com/a.pdf")
    assert base != compute_doc_id("t1", "kb-2", "https://example.com/a.pdf")


def test_chunk_id_deterministic_and_unique() -> None:
    doc_id = compute_doc_id("t1", "kb-1", "https://example.com/a.pdf")
    assert compute_chunk_id(doc_id, 1, 0) == compute_chunk_id(doc_id, 1, 0)
    ids = {compute_chunk_id(doc_id, 1, index) for index in range(10)}
    assert len(ids) == 10
    assert compute_chunk_id(doc_id, 2, 0) not in ids
    # 结构路径与内容指纹同样参与哈希：同一位置的不同小节/不同内容必须得到不同 id
    assert compute_chunk_id(doc_id, 1, 0, section_path="制度 > 报销") != compute_chunk_id(
        doc_id, 1, 0, section_path="制度 > 住宿"
    )
    assert compute_chunk_id(doc_id, 1, 0, content_hash="a") != compute_chunk_id(
        doc_id, 1, 0, content_hash="b"
    )
