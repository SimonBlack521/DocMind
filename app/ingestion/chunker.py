"""文本切块：结构感知的递归切分 + 稳定 chunk_id，保证入库幂等、混合检索可去重。

chunk_id = sha256(doc_id | version | section_path | index | content_hash)
- doc_id + section_path 定位“这段内容属于哪篇文档的哪一节”；
- content_hash 保证内容变化即可被识别（同版本改内容不会产生同一个 id）；
- index 处理同一节内重复内容（例如两段完全相同的占位文本）。

切块分两层：
1. 结构层 split_sections：优先按文档结构切段——Markdown/HTML 标题层级、
   PDF/PPT 的页码标记、代码仓库压缩包的“文件”标记。每段带结构路径 path。
2. 粒度层 递归切分 + 贪心装箱：每段内部按 separators 优先级递归劈到 ≤size，
   再装箱合并，块间保留 overlap 字符尾部重叠。
结构层保证“不会把两节内容混进同一个块”，粒度层保证单块不超长。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.kb_config import ChunkingConfig
from app.ingestion.tree import Section, summarize_section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_PAGE_RE = re.compile(r"^## 第\s+(\d+)\s+页\s*$")
_FILE_RE = re.compile(r"^## 文件：(.+)\s*$")
_ANCHOR_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    content: str
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_chunk_id(
    doc_id: str, version: int, index: int, *, section_path: str = "", content_hash: str = ""
) -> str:
    """块级稳定标识：文档 + 版本 + 结构路径 + 序号 + 内容指纹。"""
    payload = f"{doc_id}|{version}|{section_path}|{index}|{content_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_parent_chunk_id(
    doc_id: str, version: int, *, section_path: str = "", content_hash: str = ""
) -> str:
    """父块标识：只由"文档 + 版本 + 小节路径 + 整节内容指纹"决定。

    与子块 id 用不同的命名空间（前缀 parent），避免父子 id 碰撞；
    同一小节重复出现时内容指纹不同，父块不会互相覆盖。
    """
    payload = f"parent|{doc_id}|{version}|{section_path}|{content_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _anchor(path: str) -> str:
    """标题锚点：把结构路径转成 URL 友好的 slug，便于引用定位回原文小节。"""
    return _ANCHOR_RE.sub("-", path).strip("-").lower()


def split_sections(text: str, *, structure_aware: bool = True) -> list[Section]:
    """按标题/页码/文件标记切段；没有结构标记时退化为“全文一段”。

    实现方式是逐行扫描并记录绝对偏移，避免用正则切分后丢失原文位置。
    """
    if not structure_aware:
        return [Section(path="", text=text, start=0, level=0)]

    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, int, int, list[str]]] = []
    breadcrumbs: list[str] = []
    current_path = ""
    current_level = 0
    current_start = 0
    current_lines: list[str] = []
    offset = 0

    def flush(end: int) -> None:
        nonlocal current_lines, current_start
        body = "".join(current_lines).strip()
        if body:
            # 段首偏移需要回退到第一行非空内容的位置，这里用段起点近似即可
            sections.append((current_path, current_start, current_level, [body]))
        current_lines = []
        current_start = end

    for line in lines:
        stripped = line.strip()
        page_match = _PAGE_RE.match(stripped)
        file_match = _FILE_RE.match(stripped)
        heading_match = _HEADING_RE.match(stripped)

        if page_match or file_match or heading_match:
            flush(offset)
            if file_match:
                # 代码仓库/压缩包：文件边界重置面包屑
                breadcrumbs = [f"文件：{file_match.group(1)}"]
                current_level = 1
            elif page_match:
                breadcrumbs = [f"第 {page_match.group(1)} 页"]
                current_level = 1
            else:
                assert heading_match is not None
                current_level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                # 保留同级及以上的父级，丢弃更深的旧层级
                breadcrumbs = [*breadcrumbs[: current_level - 1], title]
            current_path = " > ".join(breadcrumbs)
            current_start = offset
            current_lines = [line]
        else:
            if current_lines:
                current_lines.append(line)
            else:
                # 文档开头没有标题：先归入空路径，遇到第一个标题时再切走
                current_lines.append(line)
                current_start = offset
        offset += len(line)

    flush(offset)
    return [
        Section(path=path, text=body, start=start, level=level)
        for path, start, level, [body] in sections
    ]


class TextChunker:
    """结构感知切分：先按结构分段，再在段内递归切分并按 size 装箱。"""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def split(
        self,
        text: str,
        *,
        doc_id: str,
        version: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """切出 Chunk 列表；空文本返回 []，过短碎片（<min_chunk_len）在多块时丢弃。"""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        sections = split_sections(normalized, structure_aware=self._config.structure_aware)
        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            contents = self._pack_section(section.text)
            section_hash = hashlib.sha256(section.text.encode("utf-8")).hexdigest()
            section_summary = summarize_section(section.text)
            # 父子切片：只有"一个小节被切成多块"时才值得生成父块；
            # 整节本身很短（一块装得下）或超过 parent_max_chars（会撑爆上下文）都不生成
            parent_id = (
                compute_parent_chunk_id(
                    doc_id, version, section_path=section.path, content_hash=section_hash
                )
                if self._config.parent_child
                and len(contents) > 1
                and len(section.text) <= self._config.parent_max_chars
                else ""
            )
            if parent_id:
                parent_metadata: dict[str, Any] = dict(metadata or {})
                parent_metadata.update(
                    {
                        "doc_id": doc_id,
                        "chunk_index": -1,
                        "version": version,
                        "content_hash": section_hash,
                        "chunk_role": "parent",
                        "parent_chunk_id": parent_id,
                        "section_path": section.path,
                        "section_anchor": _anchor(section.path),
                        "section_summary": section_summary,
                        "char_start": section.start,
                        "char_end": section.start + len(section.text),
                        "children_count": len(contents),
                    }
                )
                chunks.append(
                    Chunk(
                        chunk_id=parent_id,
                        doc_id=doc_id,
                        index=-1,
                        content=section.text,
                        version=version,
                        metadata=parent_metadata,
                    )
                )

            cursor = 0
            for content in contents:
                stripped = content.strip()
                if not stripped:
                    continue
                # 过滤合并后残留的过短碎片，但全文只有一段时保留
                if len(stripped) < self._config.min_chunk_len and len(contents) > 1:
                    continue

                # 在段内定位该块的位置：从上次位置附近找，兼容 overlap 回退
                probe_from = max(0, cursor - self._config.overlap)
                relative = section.text.find(stripped, probe_from)
                if relative < 0:
                    relative = section.text.find(stripped)
                if relative < 0:
                    relative = cursor
                if relative > cursor:
                    cursor = relative
                char_start = section.start + relative
                char_end = char_start + len(stripped)

                content_hash = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
                chunk_metadata: dict[str, Any] = dict(metadata or {})
                chunk_metadata.update(
                    {
                        "doc_id": doc_id,
                        "chunk_index": index,
                        "version": version,
                        "content_hash": content_hash,
                        # 结构化定位信息：引用可回指到原文小节与字符区间
                        "section_path": section.path,
                        "section_anchor": _anchor(section.path),
                        "section_summary": section_summary,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )
                if self._config.parent_child:
                    chunk_metadata["chunk_role"] = "child"
                if parent_id:
                    # 命中的子块可凭此回取整节内容（父块）送生成
                    chunk_metadata["parent_chunk_id"] = parent_id
                chunks.append(
                    Chunk(
                        chunk_id=compute_chunk_id(
                            doc_id,
                            version,
                            index,
                            section_path=section.path,
                            content_hash=content_hash,
                        ),
                        doc_id=doc_id,
                        index=index,
                        content=stripped,
                        version=version,
                        metadata=chunk_metadata,
                    )
                )
                index += 1
                cursor = max(cursor, relative + len(stripped) - self._config.overlap)
        return chunks

    def _pack_section(self, section_text: str) -> list[str]:
        pieces = self._split_recursive(
            section_text, self._config.size, list(self._config.separators)
        )
        return self._pack(pieces, self._config.size, self._config.overlap)

    def _split_recursive(self, text: str, size: int, separators: list[str]) -> list[str]:
        """按分隔符优先级递归切到 ≤size；分隔符拼回片段末尾，保留换行/句号。"""
        if len(text) <= size:
            return [text]
        if not separators:
            return [text[i : i + size] for i in range(0, len(text), size)]

        separator = separators[0]
        rest = separators[1:]
        if not separator or separator not in text:
            return self._split_recursive(text, size, rest)

        parts = [part for part in text.split(separator) if part]
        if len(parts) == 1:
            return self._split_recursive(parts[0], size, rest)

        # 把分隔符拼回片段，避免换行/句号丢失
        segments = [part + separator for part in parts[:-1]] + [parts[-1]]
        pieces: list[str] = []
        for segment in segments:
            if len(segment) <= size:
                pieces.append(segment)
            else:
                pieces.extend(self._split_recursive(segment, size, rest))
        return pieces

    def _pack(self, pieces: list[str], size: int, overlap: int) -> list[str]:
        """贪心装箱：碎片顺序填入当前块，溢出时开新块并带上前块尾部 overlap 字符。"""
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece.strip():
                continue
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= size:
                current += piece
                continue

            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            merged = f"{tail}{piece}" if tail else piece
            current = merged if len(merged) <= size else piece

        if current.strip():
            chunks.append(current)
        return chunks
