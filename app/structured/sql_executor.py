"""只读 SQL 执行器：把"LLM 生成的 SQL"约束在可审计、不可越界的安全边界内。

为什么不能直接执行 LLM 生成的 SQL：
    LLM 可能生成 DROP/UPDATE 等破坏性语句，也可能借子查询读取白名单之外的表
    （如 information_schema、其他业务库）。因此执行前必须经过 SQLGuard 的
    AST 静态检查，执行时再叠加行数上限、超时与并发上限三道运行时护栏。

四层护栏（缺一不可）：
    1. 语法层：sqlglot 解析 AST，只允许单条 SELECT / WITH（含 UNION）查询，
       禁止任何 DML/DDL/命令节点与危险函数；
    2. 对象层：遍历 AST 中全部表引用（含 CTE、子查询、JOIN），逐一校验
       Schema 与 Table 白名单，拒绝系统表 / 跨库访问；
    3. 行数层：执行时统一包裹 `LIMIT max_rows+1`，原始 SQL 里的 LIMIT 不能
       绕过系统硬上限（多取一行仅用于判断是否截断）；
    4. 运行时：数据库会话级 statement_timeout（PostgreSQL / MySQL）、
       并发信号量与线程池超时。

已知限制：Python 无法强杀线程，超时后仅停止等待并尝试取消 future；
生产环境必须同时使用独立只读账号，并在数据库侧配置只读事务、查询超时
与资源限制。AST 校验只是第一层防线，不能替代数据库账号权限。
"""

# pyright: reportPrivateImportUsage=false
# sqlglot 的类型桩把 Expression 等公共基类标记为私有导出，但它们是官方文档
# 推荐的公共 API（exp.Expression / exp.Table 等），此处关闭该误报。
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

import sqlglot
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import exp
from sqlglot.errors import ParseError as SqlglotParseError
from sqlglot.optimizer.scope import traverse_scope

from app.core.errors import StructuredQueryError
from app.core.logging import get_logger

logger = get_logger(__name__)

# 危险函数：文件读写、命令执行、长睡眠、跨库连接。命中即拒绝。
_DANGEROUS_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "lo_import",
        "lo_export",
        "dblink",
        "load_file",
        "xp_cmdshell",
        "benchmark",
        "sleep",
    }
)
# 系统 Schema：即使被误加进白名单也不允许访问
_SYSTEM_SCHEMAS = frozenset(
    {"information_schema", "pg_catalog", "performance_schema", "mysql", "sys"}
)
# 禁止的 AST 节点：写操作、DDL、数据库命令、SELECT INTO
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Drop",
    "Alter",
    "Create",
    "TruncateTable",
    "Grant",
    "Revoke",
    "Command",
    "Copy",
    "Into",
    "Pragma",
)

# 不同知识库可能指向不同业务库；Engine 按 (URL, 超时) 复用（内部自带连接池）
_ENGINES: dict[tuple[str, float], Engine] = {}
_ENGINE_LOCK = threading.Lock()
# 超时依赖"在另一个线程里执行查询"，用共享小池避免每次查询新建线程池
_QUERY_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="structured-sql")
# 并发上限：超时查询仍可能在数据库侧执行，必须限制同时发送的语句数量
_QUERY_SLOTS = threading.BoundedSemaphore(4)


def _dialect_for(url: str) -> str | None:
    """按连接串推断 sqlglot 方言；未知方言交给默认解析器（标准 SQL）。"""
    lowered = url.lower()
    if lowered.startswith(("postgresql", "postgres")):
        return "postgres"
    if lowered.startswith("mysql"):
        return "mysql"
    if lowered.startswith("sqlite"):
        return "sqlite"
    return None


def _build_engine(url: str, timeout_s: float) -> Engine:
    """按连接串构建同步 Engine，并把查询超时下沉到数据库会话。"""
    kwargs: dict[str, object] = {"pool_pre_ping": True, "future": True}
    connect_args: dict[str, object] = {}
    timeout_ms = max(1, int(timeout_s * 1000))
    if url.startswith("sqlite"):
        # check_same_thread=False：超时在线程池中执行；timeout 为 SQLite busy timeout
        connect_args.update({"check_same_thread": False, "timeout": timeout_s})
    elif url.lower().startswith(("postgresql", "postgres")):
        # psycopg2 支持 options；服务端超时后语句会被真正取消
        connect_args["options"] = f"-c statement_timeout={timeout_ms}"
    elif url.lower().startswith("mysql"):
        connect_args["read_timeout"] = timeout_s
    if connect_args:
        kwargs["connect_args"] = connect_args
    return create_engine(url, **kwargs)


def engine_for(url: str, timeout_s: float = 5.0) -> Engine:
    """进程级缓存 Engine，避免每次查询都重建连接池。"""
    key = (url, timeout_s)
    with _ENGINE_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = _build_engine(url, timeout_s)
            _ENGINES[key] = engine
        return engine


@dataclass(frozen=True, slots=True)
class SQLResult:
    """一次只读查询的结果：实际执行的 SQL + 列名 + 行数据 + 是否被截断。"""

    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool
    sql: str = ""


class SQLGuard:
    """SQL 静态校验：AST 级只读校验 + 表/Schema 白名单 + 危险函数拦截。"""

    def __init__(
        self,
        allowed_tables: list[str],
        *,
        max_rows: int = 100,
        dialect: str | None = None,
        default_schema: str = "",
    ) -> None:
        """白名单解析（跨 Schema 隔离的关键）：

        - 裸表名（如 "expenses"）只授权"不带 Schema 的引用"；配置了 default_schema
          时它等价于 default_schema.expenses 这一个具体对象；
        - 带 Schema 的白名单项（如 "public.expenses"）只授权完全相同的 schema.table；
        - 绝不允许"仅凭表名匹配"通过跨 Schema 引用：confidential.expenses 与
          expenses 是两个不同的对象，同名不等于同权。
        """
        self._allowed_bare: set[str] = set()
        self._allowed_qualified: set[str] = set()
        for entry in allowed_tables:
            name = entry.strip().lower()
            if not name:
                continue
            if "." in name:
                self._allowed_qualified.add(name)
            else:
                self._allowed_bare.add(name)
                if default_schema.strip():
                    self._allowed_qualified.add(f"{default_schema.strip().lower()}.{name}")
        self._max_rows = max_rows
        self._dialect = dialect
        # 裸表名的解析目标 Schema；空串表示"未配置"（此时裸表名只按裸白名单匹配）
        self._default_schema = default_schema.strip().lower()

    def validate(self, sql: str) -> str:
        """校验并返回原始 SQL；不合法抛 StructuredQueryError（含具体原因）。

        返回原 SQL 而不是重写后的 AST：执行层会统一包裹 LIMIT，调用方与审计
        看到的仍是模型生成、经过校验的语句，便于复核。
        """
        cleaned = sql.strip()
        if not cleaned:
            raise StructuredQueryError("SQL 为空")
        cleaned = cleaned.rstrip(";").strip()
        if not cleaned:
            raise StructuredQueryError("SQL 为空")

        root = self._parse_single(cleaned)
        if not isinstance(root, exp.Query):
            raise StructuredQueryError("只允许 SELECT / WITH 查询")
        self._reject_forbidden_nodes(root)
        self._check_tables(root)
        self._check_functions(root)
        return cleaned

    def _parse_single(self, sql: str) -> exp.Expr:
        """解析为单条语句；语法错误、多语句、空语句一律拒绝。"""
        try:
            statements = sqlglot.parse(sql, read=self._dialect)
        except SqlglotParseError as exc:
            raise StructuredQueryError(
                "SQL 语法解析失败", details={"error": str(exc)[:200]}
            ) from exc
        statements = [statement for statement in statements if statement is not None]
        if len(statements) != 1:
            raise StructuredQueryError("只允许单条 SQL 语句，禁止一次提交多条")
        return statements[0]

    def _reject_forbidden_nodes(self, root: exp.Expr) -> None:
        """遍历整棵 AST（含 CTE 与子查询），命中写操作/DLL/命令节点即拒绝。"""
        for node_name in _FORBIDDEN_NODE_NAMES:
            node_type = getattr(exp, node_name, None)
            if node_type is None:
                continue
            if isinstance(root, node_type):
                raise StructuredQueryError(f"SQL 包含禁止的语句类型：{node_name}")
            for _ in root.find_all(node_type):
                raise StructuredQueryError(f"SQL 包含禁止的语句类型：{node_name}")

    def _check_tables(self, root: exp.Expr) -> None:
        """按 SQL 作用域校验全部真实表引用；CTE 别名只在本作用域内视为派生表。

        之前的实现把整棵语法树的 CTE 名称汇总成全局集合，再统一跳过同名 Table，
        存在嵌套 CTE 作用域绕过：内层声明一个 CTE 后，外层同名的真实表也被当作
        CTE 跳过，从而绕过白名单。这里改用 sqlglot 的作用域解析
        （traverse_scope）：每个作用域独立解析 WITH 定义，sources 中只有真正
        落到外部对象的 exp.Table 才需要做白名单校验，派生表/CTE 一律是 Scope。
        解析失败时 fail-closed（拒绝执行），避免校验器与执行器语义不一致。

        匹配规则（严格模式，防跨 Schema 绕过）：
        - 带 Schema 的引用必须与白名单完全一致（schema.table 精确匹配），
          绝不回退到"仅表名匹配"；
        - 不带 Schema 的引用按配置的 default_schema 解析后精确匹配；
          未配置 default_schema 时只接受白名单中的裸表名；
        - 三段式（catalog.schema.table）属于跨库访问，一律拒绝。
        """
        tables = self._scoped_real_tables(root)
        if not tables:
            raise StructuredQueryError("无法从 SQL 中解析出要访问的表")
        for table in tables:
            name = table.name.lower()
            schema = (table.db or "").lower()
            if not name:
                continue
            if schema in _SYSTEM_SCHEMAS or name.startswith("pg_"):
                raise StructuredQueryError(
                    f"禁止访问系统表：{table.sql(dialect=self._dialect)}",
                    details={"table": table.sql(dialect=self._dialect)},
                )
            if (table.catalog or "").strip():
                raise StructuredQueryError(
                    f"禁止跨库访问（三段式表名）：{table.sql(dialect=self._dialect)}",
                    details={"table": table.sql(dialect=self._dialect)},
                )
            if schema:
                # 显式指定 Schema：必须精确命中 schema.table，不做表名兜底匹配
                if f"{schema}.{name}" not in self._allowed_qualified:
                    raise StructuredQueryError(
                        f"表不在白名单内（跨 Schema 引用需精确配置）："
                        f"{table.sql(dialect=self._dialect)}",
                        details={"table": table.sql(dialect=self._dialect)},
                    )
                continue
            # 未指定 Schema：命中裸白名单直接通过；否则按 default_schema 解析后精确匹配
            if name in self._allowed_bare:
                continue
            if self._default_schema and f"{self._default_schema}.{name}" in self._allowed_qualified:
                continue
            raise StructuredQueryError(
                f"表不在白名单内：{table.sql(dialect=self._dialect)}",
                details={"table": table.sql(dialect=self._dialect)},
            )

    def _scoped_real_tables(self, root: exp.Expr) -> list[exp.Table]:
        """遍历作用域并收集真实表（CTE / 子查询派生表不在其中）。"""
        tables: list[exp.Table] = []
        try:
            for scope in traverse_scope(root):
                for source in scope.sources.values():
                    if isinstance(source, exp.Table):
                        tables.append(source)
        except Exception as exc:
            raise StructuredQueryError(
                "SQL 作用域解析失败，拒绝执行", details={"error": str(exc)[:200]}
            ) from exc
        return tables

    def _check_functions(self, root: exp.Expr) -> None:
        """危险函数拦截（文件/命令/睡眠/跨库）。"""
        for function in root.find_all(exp.Func):
            # Anonymous 函数（自定义/方言函数）的 sql_name() 恒为 ANONYMOUS，
            # 真实函数名在 .name；两者都检查，避免 pg_sleep 这类调用漏网
            names = {function.sql_name().lower(), str(function.name or "").lower()}
            name = next((item for item in names if item in _DANGEROUS_FUNCTIONS), "")
            if name:
                raise StructuredQueryError(
                    f"SQL 包含禁止的函数：{name}", details={"function": name}
                )


class ReadOnlySQLExecutor:
    """白名单只读执行器：校验 → 并发/超时控制 → 硬 LIMIT → 截断结果行数。"""

    def __init__(
        self,
        *,
        url: str,
        allowed_tables: list[str],
        max_rows: int = 100,
        timeout_s: float = 5.0,
        default_schema: str = "",
    ) -> None:
        if not url.strip():
            raise StructuredQueryError("结构化查询未配置数据库连接串")
        self._url = url
        self._max_rows = max_rows
        self._timeout_s = timeout_s
        self._guard = SQLGuard(
            allowed_tables,
            max_rows=max_rows,
            dialect=_dialect_for(url),
            default_schema=default_schema,
        )
        self._allowed_tables = list(allowed_tables)

    def describe_tables(self) -> dict[str, list[str]]:
        """读取白名单表的列名，供 SQL 规划提示词使用；读取失败仅返回空列。

        提示词里带列名能显著提升 LLM 生成 SQL 的准确率（避免臆造字段名），
        但这是尽力而为的增强，不影响后续校验与执行。
        """
        engine = engine_for(self._url, self._timeout_s)
        description: dict[str, list[str]] = {}
        for table in self._allowed_tables:
            if "." in table:
                schema, name = table.split(".", 1)
            else:
                schema, name = None, table
            try:
                columns = inspect(engine).get_columns(name, schema=schema)
            except Exception as exc:  # 元数据读取失败不影响主链路
                logger.warning("读取表结构失败 table=%s：%s", table, exc)
                description[table] = []
                continue
            description[table] = [str(column["name"]) for column in columns]
        return description

    def execute(self, sql: str) -> SQLResult:
        """校验并执行 SQL；超时/执行失败统一转 StructuredQueryError。

        并发名额的释放时机与查询的"真实生命周期"绑定（add_done_callback）：
        等待超时只代表调用方不再等，后台线程与数据库侧的语句仍在执行——
        若此时释放名额，BoundedSemaphore(4) 将无法约束实际运行中的语句数
        （4 条慢查询超时后又会放 4 条新查询进来）。超时路径尝试 cancel；
        只有尚未开工的任务能被取消（回调仍会触发归还名额），已开跑的任务
        由数据库端 statement_timeout 兜底结束后才归还名额。
        """
        validated = self._guard.validate(sql)
        # 原始 SQL 只作为审计/展示信息；真正执行的是加了硬 LIMIT 的包裹语句
        safe_sql = self._effective_sql(validated)
        if not _QUERY_SLOTS.acquire(timeout=self._timeout_s):
            raise StructuredQueryError(
                "结构化查询并发已达上限", details={"timeout_s": self._timeout_s}
            )
        try:
            future = _QUERY_POOL.submit(self._run, safe_sql)
        except Exception:
            # 提交失败（如解释器关停）时名额尚未绑定 future 回调，必须手动归还
            _QUERY_SLOTS.release()
            raise
        # 名额归还挂在 future 完成回调上：成功/异常/取消任一终态都恰好归还一次
        future.add_done_callback(lambda _done: _QUERY_SLOTS.release())
        try:
            columns, rows, truncated = future.result(timeout=self._timeout_s)
        except FutureTimeoutError as exc:
            # 线程无法强杀；尝试取消（未开始时可取消），并依赖数据库端 statement_timeout
            future.cancel()
            logger.warning("结构化查询超时，已尝试取消等待 sql=%s", safe_sql[:200])
            raise StructuredQueryError(
                "结构化查询超时", details={"timeout_s": self._timeout_s}
            ) from exc
        except SQLAlchemyError as exc:
            raise StructuredQueryError(
                "结构化查询执行失败", details={"error": str(exc)[:300]}
            ) from exc
        return SQLResult(
            columns=columns, rows=rows, truncated=truncated, sql=validated
        )

    def _effective_sql(self, sql: str) -> str:
        """用派生表包裹原查询并强制 LIMIT max_rows+1。

        不能只判断"原 SQL 是否已有 LIMIT"：模型可以写 `LIMIT 10000000` 绕过上限。
        包裹后原始 LIMIT 再大也受外层硬上限约束；多取一行用于判断是否截断。
        """
        return f"SELECT * FROM ({sql}) AS _rag_limited LIMIT {self._max_rows + 1}"

    def _run(self, sql: str) -> tuple[list[str], list[dict[str, object]], bool]:
        """实际执行：多取一行用于判断是否被截断，最终最多返回 max_rows 行。

        PostgreSQL 安全护栏：如果配置了 default_schema，执行前先 SET search_path
        把裸表名的解析范围钉死到应用层白名单的那个 schema，防止数据库连接
        的 search_path 中出现其他同名表时被静默命中——应用层 SQLGuard 已经
        做了钉死（裸表名等价 default_schema.table），执行层再钉一次 search_path
        是双重冗余保护。
        """
        engine = engine_for(self._url, self._timeout_s)
        is_postgres = self._url.lower().startswith(("postgresql", "postgres"))
        with engine.connect() as connection:
            if is_postgres and self._guard._default_schema:
                # $user 与 public 保留在 search_path 尾部，允许 SQL 函数/扩展对象正常解析
                connection.execute(
                    text(
                        f'SET search_path TO "{self._guard._default_schema}", "$user", public'
                    )
                )
            result = connection.execute(text(sql))
            # Row 迭代产出的是数据行，列名必须显式取 keys()（SIM118 是误报）
            columns = [str(key) for key in result.keys()]  # noqa: SIM118
            fetched = result.fetchmany(self._max_rows + 1)
        truncated = len(fetched) > self._max_rows
        rows = [
            {column: value for column, value in zip(columns, row, strict=False)}
            for row in fetched[: self._max_rows]
        ]
        return columns, rows, truncated
