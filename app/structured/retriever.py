"""结构化检索器：问题 → LLM 生成 SQL → 安全校验执行 → 确定性格式化答案。

为什么答案不再交给 LLM 组织：
    SQL 的结果已经是精确数字，再让 LLM"转述"一遍只会引入新的幻觉风险
    （改数、加解释、编造单位）。这里直接把行列拼成结构化文本，答案与
    查询结果一一对应、可复核；SQL 本身也随响应返回，便于审计。

规划失败（模型不可用 / 生成的不是 SELECT / 表不在白名单）时抛
StructuredQueryError，由编排层决定降级回文档检索还是向上报错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.errors import StructuredQueryError
from app.core.logging import get_logger
from app.domain.kb_config import StructuredConfig
from app.providers.llm import ChatMessage, LLMClient
from app.structured.sql_executor import ReadOnlySQLExecutor, SQLResult

logger = get_logger(__name__)

# 单条 SQL 生成用低温度：统计口径类任务需要稳定输出，不需要创造性
_PLANNER_TEMPERATURE = 0.0
_PLANNER_MAX_TOKENS = 512
# 答案里最多展开多少行：完整结果仍在 rows 字段里返回，文本只是可读摘要
_ANSWER_MAX_ROWS = 20
_SQL_START_PATTERN = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# 从"说明文字 + SQL"的混合输出里定位语句起点（不能锚定行首，前面常有寒暄）
_SQL_SEARCH_PATTERN = re.compile(r"\b(select|with)\b", re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_SQL_PLANNER_SYSTEM = (
    "你是企业数据查询助手。请根据用户问题生成一条只读 SQL 查询。\n"
    "硬性要求：\n"
    "1. 只能输出一条 SELECT 或 WITH 语句，禁止任何写操作；\n"
    "2. 只能访问下面给出的表，字段名必须来自列清单，不要臆造；\n"
    "3. 不要输出解释、注释或 Markdown 代码块，只输出 SQL 本体；\n"
    "4. 结果行数不要超过 {max_rows} 行。\n"
    "可用表结构：\n{schema}"
)


class StructuredRetriever:
    """面向单个知识库的结构化检索器；每次请求构造一次（Engine 进程级复用）。"""

    def __init__(
        self,
        *,
        llm: LLMClient | None,
        config: StructuredConfig,
        executor: ReadOnlySQLExecutor | None = None,
    ) -> None:
        if executor is not None:
            self._executor = executor
        else:
            self._executor = ReadOnlySQLExecutor(
                url=config.sql_url,
                allowed_tables=config.allowed_tables,
                max_rows=config.max_rows,
                timeout_s=config.timeout_s,
                default_schema=config.default_schema,
            )
        self._llm = llm
        self._config = config

    def answer(self, question: str) -> StructuredAnswer:
        """完整链路：规划 SQL → 校验执行 → 拼接答案。"""
        sql = self._plan_sql(question)
        result = self._executor.execute(sql)
        return StructuredAnswer(
            answer=format_rows(result),
            # 返回实际执行的 SQL（已补 LIMIT/已校验），而不是规划器的原始输出
            sql=result.sql,
            columns=result.columns,
            rows=result.rows,
            truncated=result.truncated,
        )

    def _plan_sql(self, question: str) -> str:
        """生成 SQL：优先 LLM 规划；关闭规划时只接受问题本身就是 SELECT 的调试场景。"""
        if self._config.use_llm_planner:
            if self._llm is None:
                raise StructuredQueryError("SQL 规划器不可用（LLM 未初始化）")
            schema = self._describe_schema()
            response = self._llm.chat(
                [
                    ChatMessage(
                        role="system",
                        content=_SQL_PLANNER_SYSTEM.format(
                            max_rows=self._config.max_rows, schema=schema
                        ),
                    ),
                    ChatMessage(role="user", content=question),
                ],
                temperature=_PLANNER_TEMPERATURE,
                max_tokens=_PLANNER_MAX_TOKENS,
            )
            sql = extract_sql(response.text)
            if not sql:
                raise StructuredQueryError(
                    "SQL 规划失败：模型未返回可识别查询",
                    details={"raw": response.text[:200]},
                )
            return sql

        if _SQL_START_PATTERN.match(question.strip()):
            return question.strip()
        raise StructuredQueryError("SQL 规划已关闭，且问题不是一条 SELECT 语句")

    def _describe_schema(self) -> str:
        """把白名单表结构压成提示词文本，并注入业务语义层（字段/指标口径）。

        字段名本身不携带业务含义（amount 可能含税也可能不含税，status=2 可能
        是"已完成"也可能是"处理中"），因此把运维配置的语义说明与指标定义
        一起给模型，避免"SQL 语法正确但业务口径错误"。
        """
        description = self._executor.describe_tables()
        lines: list[str] = []
        for table, columns in description.items():
            if columns:
                lines.append(f"- {table}({', '.join(columns)})")
            else:
                lines.append(f"- {table}")
        schema = "\n".join(lines) if lines else "（未读取到表结构，只能使用下列表名）"
        semantics = self._semantic_text()
        return f"{schema}\n业务语义（必须按此口径生成 SQL）：\n{semantics}" if semantics else schema

    def _semantic_text(self) -> str:
        """渲染 field_descriptions / metric_definitions 为规划器可读的说明文本。"""
        sections: list[str] = []
        if self._config.field_descriptions:
            fields = "\n".join(
                f"- {name}：{desc}"
                for name, desc in self._config.field_descriptions.items()
            )
            sections.append(f"字段含义：\n{fields}")
        if self._config.metric_definitions:
            metrics = "\n".join(
                f"- {name}：{definition}"
                for name, definition in self._config.metric_definitions.items()
            )
            sections.append(f"指标口径：\n{metrics}")
        return "\n".join(sections)


def extract_sql(text: str) -> str:
    """从 LLM 输出中抽出 SQL：去代码围栏、从第一个 SELECT/WITH 起截断。"""
    cleaned = text.strip()
    fenced = _CODE_FENCE_PATTERN.findall(cleaned)
    if fenced:
        cleaned = fenced[0].strip()
    match = _SQL_SEARCH_PATTERN.search(cleaned)
    if match is None:
        return ""
    return cleaned[match.start() :].strip().rstrip(";").strip()


def format_rows(result: SQLResult, *, max_rows: int = _ANSWER_MAX_ROWS) -> str:
    """把查询结果转成可读文本；不调用 LLM，保证数字与查询结果严格一致。"""
    if not result.rows:
        return "数据库查询未返回匹配记录。"
    lines = [f"根据数据库查询结果（共 {len(result.rows)} 行）："]
    for index, row in enumerate(result.rows[:max_rows], start=1):
        cells = "，".join(f"{column}: {_format_value(value)}" for column, value in row.items())
        lines.append(f"{index}. {cells}")
    if len(result.rows) > max_rows:
        lines.append(f"（仅展示前 {max_rows} 行）")
    if result.truncated:
        lines.append("（结果已达到行数上限，可能存在未展示的数据）")
    return "\n".join(lines)


def _format_value(value: object) -> str:
    """单元格值转字符串；None 统一显示"空"，避免出现"None"这种程序味输出。"""
    if value is None:
        return "空"
    text = str(value)
    return text if len(text) <= 200 else f"{text[:200]}…"


@dataclass(frozen=True, slots=True)
class StructuredAnswer:
    """结构化检索结果：答案文本 + 实际执行的 SQL + 原始行列（供审计/调试）。"""

    answer: str
    sql: str
    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool
