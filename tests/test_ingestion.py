"""接入层测试：切块边界、chunk_id 稳定性、HTML 文本抽取。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import UnsupportedFileTypeError
from app.domain.kb_config import ChunkingConfig
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader


def test_chunker_splits_within_size_and_keeps_ids_unique() -> None:
    text = "第一段内容。第二段内容。第三段内容。" * 40
    chunker = TextChunker(ChunkingConfig(size=100, overlap=20))
    chunks = chunker.split(text, doc_id="doc-1")

    assert chunks
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunker_is_deterministic() -> None:
    text = "甲乙丙丁。" * 60
    chunker = TextChunker(ChunkingConfig(size=80, overlap=10))
    first = chunker.split(text, doc_id="doc-1", version=1)
    second = chunker.split(text, doc_id="doc-1", version=1)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.content for c in first] == [c.content for c in second]


def test_loader_extracts_html_text(tmp_path: Path) -> None:
    html = tmp_path / "policy.html"
    html.write_text(
        "<html><body><script>alert('x')</script><h1>报销制度</h1><p>差旅需提前审批</p></body></html>",
        encoding="utf-8",
    )

    loader = DocumentLoader(tenant_id="t1", kb_id="kb-finance")
    document = loader.load_path(html)

    assert "报销制度" in document.content
    assert "差旅需提前审批" in document.content
    assert "alert" not in document.content
    assert document.mime_type == "text/html"
    assert len(document.doc_id) == 64


def test_loader_rejects_unsupported_suffix(tmp_path: Path) -> None:
    unsupported = tmp_path / "manual.doc"
    unsupported.write_bytes(b"\xd0\xcf\x11\xe0")

    with pytest.raises(UnsupportedFileTypeError):
        DocumentLoader(tenant_id="t1", kb_id="kb-hr").load_path(unsupported)
