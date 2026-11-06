"""结构化查询测试（v2.0）：路由规则 + SQL 安全护栏 + 只读执行 + 答案格式化。

全部用例使用临时 SQLite 文件库，不依赖外部服务；LLM 用假实现，
只验证"生成 SQL → 校验 → 执行 → 拼答案"的链路与失败分支。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.core.errors import StructuredQueryError
from app.domain.kb_config import StructuredConfig
from app.providers.llm import ChatMessage, LLMClient, LLMResponse
from app.retrieval.query_router import QueryRouter
from app.structured.retriever import StructuredRetriever, extract_sql, format_rows
from app.structured.sql_executor import ReadOnlySQLExecutor, SQLGuard, SQLResult


class FakeLLM(LLMClient):
    """返回预置文本的假 LLM；记录调用次数用于断言是否触发了规划。"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.chat_calls = 0

    @property
    def model(self) -> str:
        return "fake-planner"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(text=self._text, model=self.model)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        return iter([self._text])


def _prepare_db(tmp_path: Path, rows: int = 5) -> str:
    """建一个临时 SQLite 库并写入测试数据，返回连接串。"""
    path = tmp_path / "stats.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE expenses (department TEXT, amount INTEGER)"))
        for index in range(rows):
            connection.execute(
                text("INSERT INTO expenses (department, amount) VALUES (:d, :a)"),
                {"d": f"dept-{index}", "a": 100 + index},
            )
    engine.dispose()
    return url


# ------------------------------------------------------------------ SQL 安全护栏
def test_guard_rejects_write_and_multi_statement() -> None:
    guard = SQLGuard(["expenses"], max_rows=10)

    with pytest.raises(StructuredQueryError):
        guard.validate("DELETE FROM expenses")
    with pytest.raises(StructuredQueryError):
        guard.validate("UPDATE expenses SET amount = 0")
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM expenses; DROP TABLE expenses")
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM information_schema.tables")


def test_guard_rejects_table_outside_allowlist() -> None:
    guard = SQLGuard(["expenses"])

    with pytest.raises(StructuredQueryError) as exc_info:
        guard.validate("SELECT * FROM users")
    assert exc_info.value.details["table"] == "users"


def test_guard_accepts_select_and_ignores_literals() -> None:
    guard = SQLGuard(["expenses"])

    assert guard.validate("SELECT department FROM expenses") == (
        "SELECT department FROM expenses"
    )
    # 关键字检查不应被字符串字面量误伤（'delete' 只是数据值）
    assert guard.validate("SELECT * FROM expenses WHERE department = 'delete'") == (
        "SELECT * FROM expenses WHERE department = 'delete'"
    )
    # 调用方已写 LIMIT 时原样保留（执行层按行数上限截断结果）
    assert guard.validate("SELECT * FROM expenses LIMIT 3") == (
        "SELECT * FROM expenses LIMIT 3"
    )


def test_guard_rejects_nested_and_system_objects() -> None:
    guard = SQLGuard(["expenses"])

    # 子查询/CTE 中引用非白名单表也必须被拒绝（正则方案的典型绕过点）
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM (SELECT * FROM users) AS sub")
    with pytest.raises(StructuredQueryError):
        guard.validate("WITH x AS (SELECT * FROM users) SELECT * FROM x")
    # 系统表即使带 Schema 前缀也拒绝
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM pg_catalog.pg_tables")


def test_guard_accepts_cte_over_allowlisted_tables() -> None:
    guard = SQLGuard(["expenses"])

    assert guard.validate(
        "WITH x AS (SELECT * FROM expenses) SELECT * FROM x"
    ) == "WITH x AS (SELECT * FROM expenses) SELECT * FROM x"


def test_guard_rejects_nested_cte_scope_bypass() -> None:
    """回归（用户复现）：内层 CTE 不得把外层同名真实表"洗白"。

    旧实现把整棵树的 CTE 名称汇总，外层真实表 users 因与内层 CTE 同名被跳过；
    正确语义是按作用域解析：内层 CTE 只在内层可见，外层 users 仍是真实表。
    """
    guard = SQLGuard(["expenses"])
    # 内层声明 users CTE，外层同名的真实表 users 必须被白名单拒绝
    sql = (
        "WITH outer_q AS (SELECT * FROM expenses) "
        "SELECT * FROM users WHERE id IN ("
        "WITH users AS (SELECT id FROM expenses) SELECT id FROM users)"
    )
    with pytest.raises(StructuredQueryError) as exc_info:
        guard.validate(sql)
    assert "users" in str(exc_info.value.details.get("table", ""))


def test_guard_rejects_dangerous_functions() -> None:
    guard = SQLGuard(["expenses"])

    with pytest.raises(StructuredQueryError) as exc_info:
        guard.validate("SELECT pg_sleep(10) FROM expenses")
    assert exc_info.value.details["function"] == "pg_sleep"


def test_executor_caps_user_supplied_limit(tmp_path: Path) -> None:
    """模型写出超大 LIMIT 也不能绕过系统硬上限。"""
    url = _prepare_db(tmp_path, rows=5)
    executor = ReadOnlySQLExecutor(
        url=url, allowed_tables=["expenses"], max_rows=2, timeout_s=5.0
    )

    result = executor.execute("SELECT department FROM expenses LIMIT 10000000")

    assert len(result.rows) == 2
    assert result.truncated is True
    # 审计保留的是模型生成、经校验的原始 SQL
    assert result.sql == "SELECT department FROM expenses LIMIT 10000000"


# ------------------------------------------------------------------ 只读执行
def test_executor_returns_rows_and_caps_output(tmp_path: Path) -> None:
    url = _prepare_db(tmp_path, rows=5)
    executor = ReadOnlySQLExecutor(
        url=url, allowed_tables=["expenses"], max_rows=3, timeout_s=5.0
    )

    result = executor.execute(
        "SELECT department, amount FROM expenses ORDER BY amount DESC"
    )

    assert result.columns == ["department", "amount"]
    assert len(result.rows) == 3
    assert result.truncated is True
    assert result.rows[0] == {"department": "dept-4", "amount": 104}


def test_executor_maps_db_error_to_domain_error(tmp_path: Path) -> None:
    url = _prepare_db(tmp_path, rows=1)
    executor = ReadOnlySQLExecutor(url=url, allowed_tables=["expenses"])

    with pytest.raises(StructuredQueryError):
        executor.execute("SELECT missing_column FROM expenses")


def test_executor_describes_allowed_tables(tmp_path: Path) -> None:
    url = _prepare_db(tmp_path, rows=1)
    executor = ReadOnlySQLExecutor(url=url, allowed_tables=["expenses"])

    assert executor.describe_tables() == {"expenses": ["department", "amount"]}


# ------------------------------------------------------------------ 答案格式化
def test_format_rows_matches_query_result() -> None:
    result = SQLResult(
        columns=["department", "amount"],
        rows=[{"department": "dept-1", "amount": None}, {"department": "dept-2", "amount": 20}],
        truncated=False,
    )

    answer = format_rows(result)

    assert "共 2 行" in answer
    assert "department: dept-1，amount: 空" in answer


def test_format_rows_empty_result() -> None:
    assert format_rows(SQLResult(columns=[], rows=[], truncated=False)) == (
        "数据库查询未返回匹配记录。"
    )


# ------------------------------------------------------------------ SQL 抽取
def test_extract_sql_handles_code_fence_and_prose() -> None:
    assert extract_sql("```sql\nSELECT * FROM expenses;\n```") == "SELECT * FROM expenses"
    assert extract_sql("好的，查询如下：\nSELECT count(*) FROM expenses") == (
        "SELECT count(*) FROM expenses"
    )
    assert extract_sql("我无法回答这个问题") == ""


# ------------------------------------------------------------------ 检索器
def test_retriever_end_to_end_with_llm_planner(tmp_path: Path) -> None:
    url = _prepare_db(tmp_path, rows=5)
    config = StructuredConfig(
        enabled=True, sql_url=url, allowed_tables=["expenses"], max_rows=10
    )
    llm = FakeLLM("SELECT count(*) AS total FROM expenses")

    answer = StructuredRetriever(llm=llm, config=config).answer("报销记录一共有多少条？")

    assert llm.chat_calls == 1
    # 返回的是实际执行的 SQL：无 LIMIT 时执行层会补 max_rows+1（用于判断截断）
    assert answer.sql.startswith("SELECT count(*) AS total FROM expenses")
    assert "total: 5" in answer.answer


def test_retriever_rejects_unsafe_planner_output(tmp_path: Path) -> None:
    url = _prepare_db(tmp_path, rows=1)
    config = StructuredConfig(enabled=True, sql_url=url, allowed_tables=["expenses"])

    with pytest.raises(StructuredQueryError):
        StructuredRetriever(llm=FakeLLM("DROP TABLE expenses"), config=config).answer("清理数据")


def test_retriever_without_planner_requires_select(tmp_path: Path) -> None:
    url = _prepare_db(tmp_path, rows=1)
    config = StructuredConfig(
        enabled=True,
        sql_url=url,
        allowed_tables=["expenses"],
        use_llm_planner=False,
    )

    answer = StructuredRetriever(llm=None, config=config).answer(
        "SELECT department FROM expenses"
    )
    assert len(answer.rows) == 1

    with pytest.raises(StructuredQueryError):
        StructuredRetriever(llm=None, config=config).answer("今天天气怎么样")


# ------------------------------------------------------------------ 查询路由
def test_router_defaults_to_document_when_structured_disabled() -> None:
    decision = QueryRouter().route("统计各部门报销总额", structured_available=False)

    assert decision.route == "document"
    assert decision.reason == "structured_disabled"


def test_router_routes_statistics_question_to_sql() -> None:
    decision = QueryRouter().route(
        "统计各部门报销总额并按金额排名", structured_available=True
    )

    assert decision.route == "sql"


def test_router_keeps_fact_question_in_document() -> None:
    """"年假多少天"这类弱特征不能误判为统计问题。"""
    decision = QueryRouter().route("年假有多少天？", structured_available=True)

    assert decision.route == "document"


def test_router_honours_explicit_mode() -> None:
    router = QueryRouter()

    assert router.route("统计总额", structured_available=True, requested="document").route == (
        "document"
    )
    assert router.route("随便问一句", structured_available=True, requested="structured").route == (
        "sql"
    )
    with pytest.raises(StructuredQueryError):
        router.route("统计总额", structured_available=False, requested="structured")


def test_router_detects_raw_sql_question() -> None:
    decision = QueryRouter().route(
        "select count(*) from expenses", structured_available=True
    )

    assert decision.route == "sql"
    assert decision.reason == "sql_statement"


# ------------------------------------------------------------------ 配置校验
def test_structured_config_requires_target_when_enabled() -> None:
    with pytest.raises(ValueError):
        StructuredConfig(enabled=True)
    with pytest.raises(ValueError):
        StructuredConfig(enabled=True, sql_url="sqlite:///tmp.db")

    config = StructuredConfig(
        enabled=True, sql_url="sqlite:///tmp.db", allowed_tables=["expenses"]
    )
    assert config.max_rows == 100


# ------------------------------------------------ 跨 Schema 隔离（v2.1 清单 P0-1）
def test_sql_guard_rejects_cross_schema_same_name_table() -> None:
    """同名表位于不同 Schema：仅凭表名匹配不得通过白名单。

    allowed_tables=["expenses"] 只授权"expenses"这一个对象；
    confidential.expenses 是另一个对象，不能因为表名相同就放行。
    """
    guard = SQLGuard(["expenses"], max_rows=10)
    guard.validate("SELECT * FROM expenses")  # 裸表名：白名单命中
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM confidential.expenses")
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM public.expenses")


def test_sql_guard_default_schema_pins_bare_names() -> None:
    """配置 default_schema 后，裸表名被钉死到该 Schema 解析。"""
    guard = SQLGuard(["expenses"], default_schema="public", max_rows=10)
    guard.validate("SELECT * FROM expenses")
    guard.validate("SELECT * FROM PUBLIC.expenses")  # 大小写不敏感
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM confidential.expenses")


def test_sql_guard_qualified_whitelist_rejects_bare_reference() -> None:
    """白名单只有 schema.table 时，裸表名因解析歧义被拒绝（防止 search_path 借道）。"""
    guard = SQLGuard(["public.expenses"], max_rows=10)
    guard.validate("SELECT * FROM public.expenses")
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM expenses")
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM main.expenses")


def test_sql_guard_rejects_three_part_name() -> None:
    """三段式（catalog.schema.table）属于跨库访问，一律拒绝。"""
    guard = SQLGuard(["expenses"], max_rows=10)
    with pytest.raises(StructuredQueryError):
        guard.validate("SELECT * FROM otherdb.public.expenses")


def test_executor_releases_slot_after_completion(tmp_path: Path) -> None:
    """查询正常结束后并发名额必须归还（v2.1 清单 P0-2 的回归锚点）。"""
    from app.structured import sql_executor as module

    url = _prepare_db(tmp_path)
    executor = ReadOnlySQLExecutor(url=url, allowed_tables=["expenses"], max_rows=10)
    before = module._QUERY_SLOTS._value
    executor.execute("SELECT COUNT(*) FROM expenses")
    assert module._QUERY_SLOTS._value == before
