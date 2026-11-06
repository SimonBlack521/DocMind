"""Document Tree 与父子切片测试：结构还原、摘要、父块生成与回取标识。"""

from __future__ import annotations

from app.domain.kb_config import ChunkingConfig
from app.ingestion.chunker import TextChunker, split_sections
from app.ingestion.tree import build_document_tree, iter_tree, summarize_section

DOC = """# 差旅制度

本制度适用于全体员工。

## 住宿标准

一线城市住宿每晚不超过500元，二线城市不超过400元。

### 特殊情形

超标需提前审批并说明原因，否则不予报销。

## 报销时限

出差结束后10个工作日内提交报销单，逾期需部门负责人签字。
"""


def test_build_tree_restores_heading_hierarchy() -> None:
    """标题层级被还原成树：一级标题包含两个二级小节，其中一个带三级子节。"""
    nodes = build_document_tree(split_sections(DOC))
    assert len(nodes) == 1
    root = nodes[0]
    assert root.path == "差旅制度"
    assert root.level == 1
    assert [child.path for child in root.children] == ["差旅制度 > 住宿标准", "差旅制度 > 报销时限"]
    assert root.children[0].children[0].path == "差旅制度 > 住宿标准 > 特殊情形"
    assert len(iter_tree(nodes)) == 4


def test_summaries_skip_headings_and_are_extractive() -> None:
    """摘要跳过标题行，取正文首段，可复现（同一文本两次得到同一结果）。"""
    summary = summarize_section("## 住宿标准\n一线城市住宿每晚不超过500元。")
    assert summary.startswith("一线城市住宿每晚不超过500元")
    assert summarize_section("## 只有标题") == ""
    assert summary == summarize_section("## 住宿标准\n一线城市住宿每晚不超过500元。")


def test_enabled_parent_child_emits_parent_and_child_links() -> None:
    """开启父子切片：多块小节生成一个父块，子块带 parent_chunk_id；同一父块只生成一次。"""
    chunker = TextChunker(
        ChunkingConfig(size=30, overlap=5, min_chunk_len=1, parent_child=True)
    )
    chunks = chunker.split(DOC, doc_id="doc-tree", version=2, metadata={"title": "差旅制度"})

    parents = [c for c in chunks if c.metadata["chunk_role"] == "parent"]
    children = [c for c in chunks if c.metadata["chunk_role"] == "child"]
    assert parents, "长小节应生成父块"
    assert all(c.index == -1 and c.metadata["children_count"] >= 2 for c in parents)
    assert any(c.metadata.get("parent_chunk_id") for c in children)

    parent_ids = {c.chunk_id for c in parents}
    linked = {c.metadata["parent_chunk_id"] for c in children if c.metadata.get("parent_chunk_id")}
    assert linked <= parent_ids

    # 父块内容就是整节正文，且父块 id 由"文档+版本+小节路径+整节指纹"决定
    for parent in parents:
        assert parent.content in DOC
        assert len(parent.chunk_id) == 64


def test_parent_child_disabled_keeps_flat_chunks() -> None:
    """默认关闭时保持原有扁平切块，不出现 parent/child 角色字段。"""
    chunker = TextChunker(ChunkingConfig(size=60, overlap=10))
    chunks = chunker.split(DOC, doc_id="doc-tree")
    assert all("parent_chunk_id" not in chunk.metadata for chunk in chunks)
    assert all("chunk_role" not in chunk.metadata for chunk in chunks)


def test_oversized_section_skips_parent_generation() -> None:
    """整节超过 parent_max_chars 时不再生成父块，避免上下文 token 失控。"""
    long_section = "## 长小节\n" + ("制度条款内容。" * 400)
    chunker = TextChunker(
        ChunkingConfig(size=100, overlap=10, parent_child=True, parent_max_chars=200)
    )
    chunks = chunker.split(long_section, doc_id="doc-long")
    assert not [c for c in chunks if c.metadata.get("chunk_role") == "parent"]
    assert chunks
