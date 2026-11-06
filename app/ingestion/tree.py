"""Document Tree：把结构段组织成层级节点，并为每个节点生成抽取式小节摘要。

为什么用"抽取式"摘要而不是 LLM 摘要：
- 入库阶段对每个小节调用 LLM 会让解析成本与耗时随文档长度线性上涨，
  且摘要不可复现（同文档两次入库得到不同摘要，破坏幂等）；
- 本项目的 Document Tree 只承担两个职责：给父子切片提供稳定的层级锚点，
  以及给检索结果补充"这一段讲什么"的导航信息。首段/首句足以胜任；
- 若后续需要生成式摘要，可在离线任务里回填 summary，不影响树结构。

节点 id 不在这里生成：同一小节可能重复出现（同名标题），
父块 id 必须同时绑定内容指纹，因此由 chunker 的 compute_parent_chunk_id 负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 摘要长度上限：约 1-2 句，够做导航又不占用生成上下文
SUMMARY_LIMIT = 200


@dataclass(frozen=True, slots=True)
class Section:
    """结构段：path 是面包屑（如“差旅制度 > 报销时限”），start 是原文起始偏移。

    level 对应标题层级（# 为 1）；页码/文件标记按 1 级处理；文档开头无标题的自由文本为 0。
    """

    path: str
    text: str
    start: int
    level: int = 0


def summarize_section(text: str, *, limit: int = SUMMARY_LIMIT) -> str:
    """抽取小节摘要：跳过标题行，取首个正文段落的开头若干字符。"""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return ""

    summary = ""
    for line in lines[:3]:
        candidate = f"{summary} {line}".strip() if summary else line
        if len(candidate) >= limit:
            summary = candidate[:limit].rstrip()
            break
        summary = candidate
    if len(summary) > limit:
        summary = summary[:limit].rstrip()
    # 截断后补省略号，明确告知下游"这不是完整内容"
    return summary + "…" if len(summary) >= limit else summary


@dataclass(frozen=True, slots=True)
class TreeNode:
    """文档树节点：path 是面包屑，start/end 是原文绝对偏移，children 是下级小节。"""

    path: str
    level: int
    text: str
    start: int
    end: int
    summary: str
    children: tuple[TreeNode, ...] = ()


@dataclass(slots=True)
class _MutableNode:
    path: str
    level: int
    text: str
    start: int
    end: int
    summary: str
    children: list[_MutableNode] = field(default_factory=list)


def _freeze(node: _MutableNode) -> TreeNode:
    return TreeNode(
        path=node.path,
        level=node.level,
        text=node.text,
        start=node.start,
        end=node.end,
        summary=node.summary,
        children=tuple(_freeze(child) for child in node.children),
    )


def build_document_tree(sections: list[Section]) -> tuple[TreeNode, ...]:
    """按标题层级把线性 section 列表还原成树。

    规则：level 更深的节点挂到最近的上级节点下；同级节点共享父级；
    level 0（文档开头无标题的自由文本）作为根节点。
    """
    roots: list[_MutableNode] = []
    stack: list[_MutableNode] = []

    for section in sections:
        node = _MutableNode(
            path=section.path,
            level=section.level,
            text=section.text,
            start=section.start,
            end=section.start + len(section.text),
            summary=summarize_section(section.text),
        )
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return tuple(_freeze(root) for root in roots)


def iter_tree(nodes: tuple[TreeNode, ...]) -> list[TreeNode]:
    """先序遍历，返回扁平节点列表（便于统计/导出结构）。"""
    flat: list[TreeNode] = []
    for node in nodes:
        flat.append(node)
        flat.extend(iter_tree(node.children))
    return flat
