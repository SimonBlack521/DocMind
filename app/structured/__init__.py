"""结构化数据检索（v2.0）：只读 SQL 执行 + LLM 生成 SQL + 结果格式化。

与文档检索（app/retrieval）并行的一条能力：统计类问题需要精确数字，
走"查询业务库"而不是从文档片段里猜。安全边界见 sql_executor 的 SQLGuard。
"""

from app.structured.retriever import StructuredAnswer, StructuredRetriever
from app.structured.sql_executor import ReadOnlySQLExecutor, SQLGuard, SQLResult

__all__ = [
    "ReadOnlySQLExecutor",
    "SQLGuard",
    "SQLResult",
    "StructuredAnswer",
    "StructuredRetriever",
]
