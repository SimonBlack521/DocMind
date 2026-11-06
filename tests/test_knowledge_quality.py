"""知识治理规则与解析质量门禁的单元测试（P0 能力）。

覆盖：
- 内容指纹的稳定性（空白归一化）与敏感性（正文变化）
- 版本裁决：内容变化自动升版本、内容不变沿用
- source_type 校验与后缀推断
- 质量门禁：短文本、乱码、页面产出率、空切块
"""

from __future__ import annotations

import pytest

from app.domain.governance import (
    compute_config_hash,
    compute_content_hash,
    infer_source_type,
    normalize_source_type,
    resolve_version,
)
from app.ingestion.quality import assess_quality, split_pages, with_chunk_count


def test_content_hash_ignores_whitespace_but_detects_text_change() -> None:
    """同一内容不同换行/缩进得到同一指纹；正文变化则指纹变化。"""
    compact = "差旅报销制度\n申请单编号必须填写"
    pretty = "  差旅报销制度\r\n\r\n   申请单编号必须填写\n"
    assert compute_content_hash(compact) == compute_content_hash(pretty)
    assert compute_content_hash(compact) != compute_content_hash("差旅报销制度\n不需要申请单")


def test_resolve_version_bumps_only_when_content_changes() -> None:
    """内容变化 → 版本 +1；内容不变 → 沿用原版本；首次入库 → 至少 1。"""
    assert resolve_version(None, 1, "h1") == 1
    assert resolve_version("h1", 3, "h1") == 3
    assert resolve_version("h1", 3, "h2") == 4


def test_source_type_inference_and_validation() -> None:
    """后缀推断覆盖 wiki/ticket/api/table 等，非法显式值被拒绝。"""
    assert infer_source_type("kb://gov/policy.md") == "wiki"
    assert infer_source_type("kb://gov/tickets.csv") == "table"
    assert infer_source_type("kb://gov/api.json") == "api"
    assert infer_source_type("kb://gov/contract.docx") == "file"

    assert normalize_source_type(None, source_uri="kb://gov/a.md", suffix=".md") == "wiki"
    assert normalize_source_type("ticket", source_uri="kb://gov/a.csv", suffix=".csv") == "ticket"
    with pytest.raises(ValueError):
        normalize_source_type("unknown", source_uri="kb://gov/a.md", suffix=".md")


def test_config_hash_is_order_insensitive() -> None:
    """切块参数指纹只反映取值，不受字典序影响，便于跨进程比对。"""
    assert compute_config_hash({"size": 512, "overlap": 60}) == compute_config_hash(
        {"overlap": 60, "size": 512}
    )
    assert compute_config_hash({"size": 512}) != compute_config_hash({"size": 256})


# ---------------------------------------------------------------- 质量门禁
def test_quality_gate_accepts_normal_document() -> None:
    report = assess_quality("第一段内容。" * 40)
    assert report.passed
    assert report.char_count > 0
    assert report.page_yield is None  # 无页码标记的文档不做页产出率判定


def test_quality_gate_rejects_too_short_text() -> None:
    report = assess_quality("只有几个字", min_chars=20)
    assert not report.passed
    assert "低于下限" in "；".join(report.reasons)


def test_quality_gate_rejects_garbled_text() -> None:
    """大量控制字符/替换字符视为编码错乱。"""
    garbled = "\ufffd" * 100 + "正常文字"
    report = assess_quality(garbled, max_garbled_ratio=0.05)
    assert not report.passed
    assert "乱码率" in "；".join(report.reasons)


def test_quality_gate_rejects_low_page_yield() -> None:
    """4 页里只有 1 页有正文：疑似扫描件或解析异常，应被隔离。"""
    content = "\n".join(
        [
            "## 第 1 页",
            "封面标题与版本号",
            "## 第 2 页",
            "（空）",
            "## 第 3 页",
            "（空）",
            "## 第 4 页",
            "（空）",
        ]
    )
    pages = split_pages(content)
    assert len(pages) == 4
    report = assess_quality(content, min_page_yield=0.3, min_page_chars=4)
    assert not report.passed
    assert "页面产出率" in "；".join(report.reasons)
    assert report.page_yield == pytest.approx(0.25)


def test_empty_chunk_result_fails_gate() -> None:
    base = assess_quality("内容足够长" * 20)
    assert base.passed
    failed = with_chunk_count(base, 0)
    assert not failed.passed
    assert "切块结果为空" in "；".join(failed.reasons)


def test_report_is_json_serializable() -> None:
    """质量报告要能直接写进 JSONB 列。"""
    payload = assess_quality("正文内容" * 50).to_dict()
    assert payload["passed"] is True
    assert isinstance(payload["reasons"], list)
