"""跨层传输对象：检索结果在检索/重排/生成之间流转时使用。

在整体链路中的角色：向量检索（Milvus）与稀疏检索（ES）各自返回不同结构，
进入融合（RRF）与重排（reranker）之前先统一成 RetrievedChunk，之后的所有
环节只面向这一个数据结构编程，新增检索通道时无需改动下游。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    """一路检索（向量 / BM25）返回的候选片段。

    字段语义：
    - score：该路检索的原生分数（向量距离 / BM25 分），只在单路内部可比；
      跨路融合用 RRF 按排名折算，不要直接相加两路 score；
    - source_uri：对象存储中的原文位置（s3://...），供答案引用与前端下载；
    - metadata：透传文档侧扩展信息（页码、acl_groups 等），rerank/生成层不解析，
      只原样带回给调用方。
    """

    chunk_id: str
    content: str
    score: float
    doc_id: str
    source_uri: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
