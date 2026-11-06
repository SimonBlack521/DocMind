"""引用生成：把检索片段转成可回溯的引用列表（编号与提示词中的 [n] 对齐）。

编号对齐机制（本模块最核心的约定）：
    prompts.build_context_block 按 chunks 顺序从 1 开始给片段编号 [1]、[2]…，
    build_citations 用同样的顺序生成 index = start_index + offset——
    两者遍历的是同一份 chunks、同样的顺序，因此天然一致：
    LLM 在答案里写 [2]，前端就能按 index=2 在引用列表里找到对应文档，
    无需对 LLM 输出做额外的编号提取或映射校验。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.domain.dto import RetrievedChunk

_SNIPPET_LEN = 240


@dataclass(frozen=True, slots=True)
class Citation:
    """单条引用：index 对应答案中的 [n]；snippet 是截断摘要，仅供前端展示。"""

    index: int
    chunk_id: str
    doc_id: str
    source_uri: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any]
    # 父块摘要（Parent-Child 双重定位，清单第三优先级第 17 项）：
    # snippet 是命中的子块（精确定位），parent_snippet 是送入 LLM 的父块节选
    # （语义上下文）；非父子切片模式或父块缺失时为空串
    parent_snippet: str = ""


def _snippet(content: str) -> str:
    """压缩所有连续空白为单个空格后截断到 _SNIPPET_LEN：摘要只用于展示，不参与生成。"""
    text = " ".join(content.split())
    return text if len(text) <= _SNIPPET_LEN else f"{text[:_SNIPPET_LEN]}..."


def build_citations(
    chunks: Sequence[RetrievedChunk], *, start_index: int = 1
) -> list[Citation]:
    """按片段顺序生成引用，index 从 1 开始，和提示词里的 [1]、[2] 一一对应。"""
    return [
        Citation(
            index=start_index + offset,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source_uri=chunk.source_uri,
            title=chunk.title or chunk.source_uri,
            snippet=_snippet(chunk.content),
            score=chunk.score,
            metadata=dict(chunk.metadata),
            # 父子双重定位：父块摘要供用户核对"该结论出自哪一节"
            parent_snippet=(
                _snippet(str(chunk.metadata["parent_content"]))
                if chunk.metadata.get("parent_content")
                else ""
            ),
        )
        for offset, chunk in enumerate(chunks)
    ]
