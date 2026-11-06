"""解析质量门禁：把"解析是否可用"从主观判断变成可量化、可拦截的规则。

三把尺子（均在入库前执行，不通过就不写任何索引）：
- 正文长度：过短通常意味着解析失败或扫描件无 OCR；
- 页面产出率：带页码的文档（PDF/PPT）逐页统计有文字的页占比，避免"只解析出封面"；
- 乱码率：不可打印控制字符与替换字符 U+FFFD 的占比，识别编码错乱。

注意：这里只做"结构是否可用"的判定，不做内容对错的语义判断。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# 解析器输出的页码标记（见 parsers._parse_pdf 与 pptx 的页分隔约定）
_PAGE_MARKER = re.compile(r"^## 第 \d+ 页\s*$", re.MULTILINE)

# 视为乱码的 Unicode 类别：控制符/格式符/私用区/代理区/未分配
_GARBLED_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Cn"})
_ALLOWED_CONTROL = frozenset({"\n", "\r", "\t"})


@dataclass(frozen=True, slots=True)
class QualityReport:
    """解析质量报告：既用于门禁判定，也原样落库（parse_report）供事后审计。"""

    char_count: int
    page_count: int
    pages_with_text: int
    page_yield: float | None
    garbled_ratio: float
    passed: bool
    reasons: tuple[str, ...] = ()
    chunk_count: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """转成 JSON 可序列化的结构，直接写进 documents.parse_report（JSONB）。"""
        payload: dict[str, object] = {
            "char_count": self.char_count,
            "page_count": self.page_count,
            "pages_with_text": self.pages_with_text,
            "page_yield": self.page_yield,
            "garbled_ratio": round(self.garbled_ratio, 4),
            "chunk_count": self.chunk_count,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


def split_pages(content: str) -> list[str]:
    """按页码标记切页；无页码标记的文档返回空列表（不参与页产出率判定）。"""
    parts = _PAGE_MARKER.split(content)
    if len(parts) <= 1:
        return []
    return [part.strip() for part in parts if part.strip()]


def _garbled_ratio(content: str) -> float:
    if not content:
        return 0.0
    garbled = sum(
        1
        for char in content
        if (char != "\n" and char not in _ALLOWED_CONTROL
            and (char == "\ufffd" or unicodedata.category(char) in _GARBLED_CATEGORIES))
    )
    return garbled / len(content)


def assess_quality(
    content: str,
    *,
    min_chars: int = 20,
    min_page_yield: float = 0.3,
    max_garbled_ratio: float = 0.05,
    min_page_chars: int = 10,
) -> QualityReport:
    """执行三项门禁并返回报告；调用方据 passed 决定是否隔离文档。"""
    text = content.strip()
    char_count = len(text)
    pages = split_pages(text)
    pages_with_text = sum(1 for page in pages if len(page) >= min_page_chars)
    page_count = len(pages)
    page_yield = (pages_with_text / page_count) if page_count else None
    garbled_ratio = _garbled_ratio(text)

    reasons: list[str] = []
    if char_count < min_chars:
        reasons.append(f"正文字符数 {char_count} 低于下限 {min_chars}")
    if page_yield is not None and page_yield < min_page_yield:
        reasons.append(
            f"页面产出率 {page_yield:.2%} 低于下限 {min_page_yield:.0%}"
            f"（{pages_with_text}/{page_count} 页有文本，疑似扫描件或解析异常）"
        )
    if garbled_ratio > max_garbled_ratio:
        reasons.append(
            f"乱码率 {garbled_ratio:.2%} 超过上限 {max_garbled_ratio:.0%}（疑似编码错乱）"
        )

    return QualityReport(
        char_count=char_count,
        page_count=page_count,
        pages_with_text=pages_with_text,
        page_yield=page_yield,
        garbled_ratio=garbled_ratio,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def with_chunk_count(report: QualityReport, chunk_count: int) -> QualityReport:
    """切块后补记块数；块数为 0 视为门禁不通过。"""
    if chunk_count <= 0:
        return QualityReport(
            char_count=report.char_count,
            page_count=report.page_count,
            pages_with_text=report.pages_with_text,
            page_yield=report.page_yield,
            garbled_ratio=report.garbled_ratio,
            passed=False,
            reasons=(*report.reasons, "切块结果为空"),
            chunk_count=0,
            extra=report.extra,
        )
    return QualityReport(
        char_count=report.char_count,
        page_count=report.page_count,
        pages_with_text=report.pages_with_text,
        page_yield=report.page_yield,
        garbled_ratio=report.garbled_ratio,
        passed=report.passed,
        reasons=report.reasons,
        chunk_count=chunk_count,
        extra=report.extra,
    )
