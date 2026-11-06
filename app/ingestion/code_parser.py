"""代码类知识源的结构化解析：把 Python 源文件切成"模块 + 类/函数"小节。

为什么不用纯文本切：函数/类天然是代码的语义边界，按 AST 切出的块不会把两个
函数拼在一起，检索命中时也更容易把"具体实现"与"调用约定"区分开。
解析失败（语法不完整/非 UTF-8）时退化为原文，绝不让代码文件直接入库失败。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeSection:
    path: str
    source: str


def _node_label(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"函数：{node.name}"
    if isinstance(node, ast.ClassDef):
        return f"类：{node.name}"
    return ""


def python_sections(source: str) -> list[CodeSection]:
    """按顶层定义切分；每个定义保留原始缩进与完整源码（含装饰器与 docstring）。"""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return [CodeSection(path="", source=source)]

    lines = source.splitlines(keepends=True)
    sections: list[CodeSection] = []

    # 模块头：第一个顶层定义之前的 import/常量，作为独立小节
    first_start = min(
        (
            (node.lineno - 1) for node in tree.body if _node_label(node)
        ),
        default=len(lines),
    )
    header = "".join(lines[:first_start])
    if header.strip():
        sections.append(CodeSection(path="模块", source=header))

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno - 1
        # 装饰器的行号早于 def/class 行，需要往前扩展到装饰器起点
        if node.decorator_list:
            start = min(dec.lineno - 1 for dec in node.decorator_list)
        end = node.end_lineno or len(lines)
        body = "".join(lines[start:end])
        # 顶层定义下挂的子类/方法不再单独切分：保持实现完整性，交由粒度层处理
        sections.append(CodeSection(path=_node_label(node), source=body))
    return sections


def render_python_structure(source: str) -> str:
    """渲染成带结构标记的文本，交给结构感知切块识别小节边界。"""
    sections = python_sections(source)
    if len(sections) <= 1 and not sections[0].path:
        return source
    blocks = [f"## {section.path}\n{section.source.strip()}" for section in sections]
    return "\n\n".join(blocks)
