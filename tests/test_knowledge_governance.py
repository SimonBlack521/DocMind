"""知识治理审计测试：用 scripts/data/governance 下的真实样例锁定"当前能力边界"。

这些测试有双重作用：
1. 正向验证——现有的解析 / 切块 / 元数据能力确实可用（回归保护）；
2. 边界声明——把"已实现"与"尚未实现"的点固化成断言，避免能力被误判。

覆盖审计项：
    1. 知识源格式覆盖与解析方式（含代码仓库 .py/.zip、产品配置 .yaml）
    2. 文档元数据字段清单
    6/7. chunk 策略与结构感知差异
    8. chunk_id 生成规则（document_id + 结构路径 + 序号 + 内容指纹）
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.core.errors import DocumentParseError
from app.domain.kb_config import ChunkingConfig
from app.ingestion.chunker import TextChunker, compute_chunk_id
from app.ingestion.governance import decide
from app.ingestion.loader import DocumentLoader, LoadedDocument
from app.storage.vector_base import ACL_PUBLIC_GROUP, normalize_acl_groups

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data" / "governance"


def _loader() -> DocumentLoader:
    return DocumentLoader(tenant_id="tenant-gov", kb_id="kb-governance")


# ---------------------------------------------------------------- 1. 格式覆盖
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("wiki_hr_policy.md", "差旅与报销制度"),
        ("service_ops.txt", "payment-gateway"),
        ("wiki_product_config.html", "会员等级产品配置"),
        ("tickets.csv", "WO-2026-0412"),
        ("api_reference.json", "order-query-api"),
        ("contract.docx", "采购合同摘要"),
        ("product_intro.pptx", "经营回顾"),
        ("ledger.xlsx", "EXP-2026-0001"),
        ("manual.pdf", "Payment Gateway Operations Manual"),
    ],
)
def test_supported_source_types_are_parseable(filename: str, expected: str) -> None:
    """九类样例（Markdown/文本/HTML/CSV/JSON/Word/PPT/Excel/PDF）都能抽出正文。"""
    document = _loader().load_path(FIXTURE_DIR / filename)
    assert expected in document.content
    assert document.mime_type
    assert len(document.doc_id) == 64


@pytest.mark.parametrize("filename", ["code_sample.py", "wiki_product_config.yaml"])
def test_code_repo_and_config_sources_are_parsed(filename: str) -> None:
    """代码仓库（.py）与产品配置（.yaml）已纳入支持列表，按文本解析。

    .py 额外走 AST 结构化：顶层类/函数渲染成 `## 类：X` / `## 函数：Y` 小节标记，
    供结构感知切块识别边界，避免把两个函数混进同一个块。
    """
    document = _loader().load_path(FIXTURE_DIR / filename)
    assert document.content
    assert len(document.content_hash) == 64

    if filename.endswith(".py"):
        assert "## 类：OrderStatus" in document.content
        assert "## 函数：assert_transition" in document.content
        assert document.parser_name == "python-ast"
    else:
        assert "points_multiplier: 1.5" in document.content


def test_zip_archive_is_loaded_with_file_boundaries(tmp_path: Path) -> None:
    """代码仓库压缩包：逐文件抽取并加 `## 文件：路径` 标记，结构边界可被切块识别。"""
    import zipfile

    archive_path = tmp_path / "repo.zip"
    code_source = (FIXTURE_DIR / "code_sample.py").read_text("utf-8")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("orders/state_machine.py", code_source)
        archive.writestr("docs/policy.md", "# 订单状态\n\n已支付订单可发货。\n")
        archive.writestr("assets/logo.bin", b"\x00\x01")  # 不支持的后缀被跳过

    document = _loader().load_path(archive_path)
    assert "## 文件：docs/policy.md" in document.content
    assert "## 文件：orders/state_machine.py" in document.content
    assert "logo.bin" not in document.content
    assert document.parser_name == "zipfile"


def test_binary_parsers_flatten_structure_into_plain_text() -> None:
    """二进制解析只输出纯文本：PDF 有页码前缀，Excel 有工作表前缀，但都不是结构化块。"""
    pdf = _loader().load_path(FIXTURE_DIR / "manual.pdf")
    assert "## 第 1 页" in pdf.content

    xlsx = _loader().load_path(FIXTURE_DIR / "ledger.xlsx")
    assert "## 工作表：费用台账" in xlsx.content

    docx = _loader().load_path(FIXTURE_DIR / "contract.docx")
    # 表格被拍平成 "单元格 | 单元格"，行列关系丢失
    assert "第一批 | 2026-11-15 | 120000" in docx.content


def test_empty_document_is_rejected_before_indexing() -> None:
    """空内容在入库前就抛错——保证不会把空文档静默写入索引。"""
    with pytest.raises(DocumentParseError):
        _loader().load_bytes(b"   \n  ", source_uri="kb://x/empty.txt", suffix=".txt")


# ---------------------------------------------------------------- 2. 元数据字段
def test_loaded_document_metadata_carries_fingerprint_and_parser_identity() -> None:
    """LoadedDocument 已带内容指纹与解析器身份；业务治理字段由 DB 层承担。"""
    names = {item.name for item in fields(LoadedDocument)}
    assert names == {
        "doc_id",
        "source_uri",
        "title",
        "content",
        "mime_type",
        "content_hash",
        "parser_name",
        "parser_version",
        "pii_findings",
    }

    document = _loader().load_path(FIXTURE_DIR / "wiki_hr_policy.md")
    assert len(document.content_hash) == 64
    assert document.parser_name == "builtin-text"

    pdf = _loader().load_path(FIXTURE_DIR / "manual.pdf")
    assert pdf.parser_name == "pypdf"
    assert pdf.parser_version

    # owner / source_type / 有效期 / 权威度等属于 DB 治理字段，不放在解析结果上
    assert {"owner", "source_type", "authority_level", "valid_from", "valid_to"}.isdisjoint(names)


def test_chunk_metadata_carries_structure_path_and_offsets() -> None:
    """chunk 已带结构路径、标题锚点与原文字符区间，引用可回指到具体小节。"""
    document = _loader().load_path(FIXTURE_DIR / "wiki_hr_policy.md")
    chunks = TextChunker(ChunkingConfig()).split(
        document.content,
        doc_id=document.doc_id,
        metadata={
            "title": document.title,
            "source_uri": document.source_uri,
            "mime_type": document.mime_type,
            "acl_groups": normalize_acl_groups(None),
        },
    )
    assert chunks
    metadata = chunks[0].metadata

    assert {"doc_id", "chunk_index", "version", "title", "source_uri", "mime_type"} <= set(metadata)
    assert len(metadata["content_hash"]) == 64
    # ACL 公开文档落哨兵 "*"，与 Milvus/ES 过滤约定一致
    assert metadata["acl_groups"] == [ACL_PUBLIC_GROUP]

    # 结构化定位信息：结构路径 + 锚点 + 原文区间
    assert {"section_path", "section_anchor", "char_start", "char_end"} <= set(metadata)
    assert metadata["section_path"].startswith("差旅与报销制度")
    assert metadata["section_anchor"]
    assert isinstance(metadata["char_start"], int)
    assert metadata["char_end"] > metadata["char_start"]
    # 父子切片（parent_chunk_id）属于 P2 的 Document Tree，当前不生成
    assert "parent_chunk_id" not in metadata


# ---------------------------------------------------------------- 6/7. 切块策略
def test_chunking_is_recursive_then_packed_with_overlap() -> None:
    """两阶段策略：段内优先在段落/句子边界切，再装箱；同一小节内相邻块尾部有 overlap。

    注意：结构层先分段，overlap 只在"同一小节内部"成立；跨小节不做重叠，
    避免把上一节末尾的句子带进下一节，污染证据边界。
    """
    document = _loader().load_path(FIXTURE_DIR / "wiki_hr_policy.md")
    config = ChunkingConfig(size=80, overlap=20)
    chunks = TextChunker(config).split(document.content, doc_id=document.doc_id)

    assert all(len(chunk.content) <= config.size for chunk in chunks)
    assert len(chunks) > 1

    same_section = [
        (left, right)
        for left, right in itertools.pairwise(chunks)
        if left.metadata["section_path"] == right.metadata["section_path"]
    ]
    assert same_section, "样例正文应在小节内部产生多块，才能验证段内 overlap"
    left, right = same_section[0]
    assert left.content[-10:] in right.content[: config.overlap + 10]


def test_chunking_is_structure_aware_per_file_type() -> None:
    """同一 TextChunker，但结构标记来自各自解析器：Markdown 标题、PDF 页码、Excel 工作表。

    切块粒度参数仍由知识库统一配置，差异来自"结构边界"而不是为每种格式写一套切块器。
    """
    chunker = TextChunker(ChunkingConfig(size=120, overlap=20))
    paths: dict[str, str] = {}
    for filename in ("wiki_hr_policy.md", "manual.pdf", "ledger.xlsx"):
        document = _loader().load_path(FIXTURE_DIR / filename)
        chunks = chunker.split(document.content, doc_id=document.doc_id)
        assert chunks and all(len(c.content) <= 120 for c in chunks)
        # 结构路径的第一跳即来源结构：标题 / 页码 / 工作表
        paths[filename] = chunks[0].metadata["section_path"]
    assert paths["manual.pdf"].startswith("第 ")
    assert paths["ledger.xlsx"].startswith("工作表")
    assert paths["wiki_hr_policy.md"].startswith("差旅与报销制度")


def test_chunk_size_and_overlap_are_kb_level_configurable() -> None:
    """size/overlap 是知识库级参数（默认 512/60），当前没有自动调优。"""
    default = ChunkingConfig()
    assert default.size == 512
    assert default.overlap == 60
    assert ChunkingConfig(size=256, overlap=32).size == 256


# ---------------------------------------------------------------- 8. chunk_id
def test_chunk_id_is_hash_of_doc_version_path_index_and_content() -> None:
    """chunk_id = sha256(doc_id|version|section_path|index|content_hash)：五元组构成稳定锚点。"""
    document = _loader().load_path(FIXTURE_DIR / "wiki_hr_policy.md")
    chunks = TextChunker(ChunkingConfig()).split(document.content, doc_id=document.doc_id)

    for chunk in chunks:
        expected = hashlib.sha256(
            "|".join(
                [
                    document.doc_id,
                    str(chunk.version),
                    str(chunk.metadata["section_path"]),
                    str(chunk.index),
                    str(chunk.metadata["content_hash"]),
                ]
            ).encode()
        ).hexdigest()
        assert chunk.chunk_id == expected == compute_chunk_id(
            document.doc_id,
            chunk.version,
            chunk.index,
            section_path=str(chunk.metadata["section_path"]),
            content_hash=str(chunk.metadata["content_hash"]),
        )


def test_content_change_with_same_version_yields_new_chunk_ids() -> None:
    """同一版本下改内容，chunk_id 随内容指纹变化——id 本身即可检测内容漂移。"""
    chunker = TextChunker(ChunkingConfig(size=100, overlap=10))
    first = chunker.split("甲" * 300, doc_id="doc-x", version=1)
    second = chunker.split("乙" * 300, doc_id="doc-x", version=1)
    assert [c.chunk_id for c in first] != [c.chunk_id for c in second]


def test_doc_id_is_source_based_but_content_hash_detects_duplicates() -> None:
    """doc_id 由 tenant|kb|source_uri 决定（同内容不同来源是两篇文档）；
    跨来源去重由 content_hash + 元数据层 duplicate_of 承担，不再依赖 doc_id。"""
    loader = _loader()
    same_content = "完全相同的正文内容"
    first = loader.load_bytes(None, text=same_content, source_uri="kb://gov/a.txt", suffix=".txt")
    second = loader.load_bytes(None, text=same_content, source_uri="kb://gov/b.txt", suffix=".txt")
    assert first.doc_id != second.doc_id
    assert first.content == second.content
    assert first.content_hash == second.content_hash


def _decide_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "new_content_hash": "h1",
        "new_acl": ["finance"],
        "new_source_id": "src-1",
        "new_source_type": "file",
        "new_owner": "alice",
        "new_source_version": "1",
        "new_authority_level": 5,
        "existing_status": "ready",
        "existing_content_hash": "h1",
        "existing_acl": ["finance"],
        "existing_source_id": "src-1",
        "existing_source_type": "file",
        "existing_owner": "alice",
        "existing_source_version": "1",
        "existing_authority_level": 5,
        "new_valid_from": None,
        "new_valid_to": None,
        "existing_valid_from": None,
        "existing_valid_to": None,
        "new_config_hash": "cfg-1",
        "existing_config_hash": "cfg-1",
    }
    base.update(overrides)
    return base


def test_decide_reuses_when_all_governance_inputs_unchanged() -> None:
    assert decide(**_decide_kwargs()).decision == "reuse"  # type: ignore[arg-type]


def test_decide_rebuilds_when_validity_became_expired() -> None:
    """回归（用户复现）：有效期从未来改成已过期，不能再返回 reused=true。"""
    future = datetime.now(UTC) + timedelta(days=30)
    past = datetime.now(UTC) - timedelta(days=1)
    decision = decide(
        **_decide_kwargs(existing_valid_to=future, new_valid_to=past)  # type: ignore[arg-type]
    )
    assert decision.decision == "ingest_new_version"


def test_decide_rebuilds_when_chunk_or_embedding_fingerprint_changed() -> None:
    """回归：切块参数 / Embedding 版本变化后，旧索引不能被直接复用。"""
    assert (
        decide(**_decide_kwargs(new_config_hash="cfg-2")).decision  # type: ignore[arg-type]
        == "ingest_new_version"
    )
    # 历史文档未记录指纹：按"已变化"处理，宁重建一次
    assert (
        decide(**_decide_kwargs(existing_config_hash=None)).decision  # type: ignore[arg-type]
        == "ingest_new_version"
    )
