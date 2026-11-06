"""知识库配置：一个业务线对应一个知识库，配置以 YAML 落在 configs/kbs/ 下。

命名约定（强约束，避免不同业务库共用集合导致数据串库）：
    Milvus collection = rag_{kb_code}
    ES index          = rag_{kb_code}_chunks
    ES alias          = rag_{kb_code}_search

kb_code 为什么不可变：
    kb_code 是上述所有物理资源的命名根。数据一旦入库，Milvus collection、ES
    索引、别名里都"烙"着创建时的 kb_code；如果允许改 kb_code，新名字指向的是
    一批不存在/为空的资源，旧资源则变成无人引用的孤儿数据——既查不到历史
    文档又泄漏存储。因此创建后禁止修改 kb_code（要改只能新建库并迁移数据）。
    kb_id 同样唯一，但它是逻辑标识（权限、外键引用用），与物理命名解耦。

配置文件即数据库：没有单独的"知识库表"，YAML 文件是知识库的权威存储，
KBRegistry 是其在内存中的只读镜像；API 的增删改最终都落到本模块的
save/delete/find 函数上。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.errors import KBConfigError, KBNotFoundError

KB_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


class ChunkingConfig(BaseModel):
    """文档切块参数。extra="forbid"：YAML 里多写的键直接报错，防止拼错键名
    （如 overlapp）被静默忽略后按默认值切块，排查困难。"""

    model_config = ConfigDict(extra="forbid")

    size: int = 512
    overlap: int = 60
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "。", "；", "！", "？", " ", ""]
    )
    min_chunk_len: int = 20
    # 结构感知切分：按 Markdown/HTML 标题、PDF/PPT 页码、压缩包文件边界分段，
    # 避免把不同小节混进同一个块；关闭后全部按纯文本递归切
    structure_aware: bool = True
    # 父子切片：长小节额外生成一个"父块"（整节内容）并写入检索索引的旁路，
    # 检索命中子块后按 parent_chunk_id 回取父块送生成，兼顾精确定位与上下文完整
    parent_child: bool = False
    # 父块最大字符数：超过该长度的小节不再生成父块（避免上下文 token 失控）
    parent_max_chars: int = 1500

    @field_validator("size")
    @classmethod
    def _size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunking.size 必须大于 0")
        return value

    @field_validator("overlap")
    @classmethod
    def _overlap_lt_size(cls, value: int, info: Any) -> int:
        size = info.data.get("size", 512)
        if value < 0 or value >= size:
            raise ValueError("chunking.overlap 必须在 [0, size) 区间内")
        return value

    @field_validator("parent_max_chars")
    @classmethod
    def _parent_size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunking.parent_max_chars 必须大于 0")
        return value


class EmbeddingProfile(BaseModel):
    """知识库级 Embedding 覆盖项；为空则回落到全局 Settings。"""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    dim: int | None = None

    @field_validator("dim")
    @classmethod
    def _dim_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("embedding.dim 必须大于 0")
        return value


class RetrievalConfig(BaseModel):
    """检索/融合/重排参数。vector_weight 与 bm25_weight 参与两路归一化加权，
    use_query_rewrite / use_rerank 是管线开关，可按业务库粒度关闭以省延迟。"""

    model_config = ConfigDict(extra="forbid")

    top_k: int = 20
    rerank_top_n: int = 6
    rrf_k: int = 60
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    use_query_rewrite: bool = True
    use_rerank: bool = True
    # 自适应召回（9.3）：不写死 K。简单问题取 base_k/2（不低于 top_k_min），
    # 多跳/复合问题放大到 base_k×2（不超过 top_k_max）；调用方显式传 top_k 时不生效
    adaptive_top_k: bool = True
    top_k_min: int = 5
    top_k_max: int = 40
    # 召回阶段分数断层早停（9.7）：相邻名次相对跌幅 >= 该值时丢弃后续候选；0=关闭。
    # 只作用于单路内部（同一打分尺度），不跨向量/BM25 比较。
    # 默认 0=关闭（清单第三优先级第 15 项：未用 Golden Dataset 验证不损失召回率之前
    # 默认关闭）；按知识库显式配置 >0 即开启，作为实验开关
    score_cliff_ratio: float = 0.0
    # MMR 多样性（9.5）：1.0=关闭（纯相关性）；越小越强调同一文档/相邻片段之间的差异
    mmr_lambda: float = 1.0
    # 图增强重排（v2.0 GraphRAG 最小版）：按候选片段共享词元图调整顺序；默认关闭
    graph_rerank_enabled: bool = False
    # 图中心度在融合分中的权重：0=完全忽略图结构，1=与相关性同权
    graph_rerank_boost: float = 0.5
    # 父块上下文预算：单个父块最多注入多少字符（0=不截断），以及一次问答中
    # 所有父块扩展内容的总字符上限（超出后后续命中退回子块，避免上下文膨胀）
    parent_max_chars: int = 2000
    parent_context_max_chars: int = 8000

    @field_validator("parent_max_chars", "parent_context_max_chars")
    @classmethod
    def _parent_budget_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("父子检索上下文预算不能为负数（0 表示不限制）")
        return value

    @field_validator("graph_rerank_boost")
    @classmethod
    def _graph_boost_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("retrieval.graph_rerank_boost 必须在 [0, 1] 区间内")
        return value

    @field_validator("top_k", "rerank_top_n", "rrf_k")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("检索参数必须大于 0")
        return value

    @field_validator("top_k_min", "top_k_max")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("自适应 top_k 边界不能为负数（0 表示不启用该边界）")
        return value

    @field_validator("vector_weight", "bm25_weight")
    @classmethod
    def _weight_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("检索权重必须在 [0, 1] 区间内")
        return value

    @field_validator("score_cliff_ratio")
    @classmethod
    def _cliff_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("score_cliff_ratio 必须在 [0, 1] 区间内（0 表示关闭）")
        return value

    @field_validator("mmr_lambda")
    @classmethod
    def _mmr_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("mmr_lambda 必须在 [0, 1] 区间内（1 表示关闭多样性）")
        return value

    @model_validator(mode="after")
    def _adaptive_bounds(self) -> RetrievalConfig:
        if self.top_k_min > 0 and self.top_k_max > 0 and self.top_k_max < self.top_k_min:
            raise ValueError("retrieval.top_k_max 不能小于 top_k_min")
        return self


class ConnectorConfig(BaseModel):
    """知识源连接器配置（v2.0 Connector Framework）。

    连接器负责"从外部系统拉取文档"，入库仍走统一的 governance / 质量门禁 /
    增量 Diff 链路——连接器只解决"数据从哪来"，不绕过任何安全与质量约束。
    """

    model_config = ConfigDict(extra="forbid")

    connector_id: str
    # 目前支持本地目录；后续 sql/api/对象存储按同一 protocol 扩展
    type: str = "local_directory"
    enabled: bool = True
    # local_directory：扫描根目录（相对路径按服务进程工作目录解析）
    path: str = ""
    recursive: bool = True
    # 只采集这些后缀；留空表示不限（仍受 MAX_FILE_BYTES 与解析器能力限制）
    include_suffixes: list[str] = Field(
        default_factory=lambda: [".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv"]
    )
    # 治理字段默认值，可由文件级元数据覆盖
    owner: str | None = None
    # 与文档治理字段同一口径：0=最低，数值越大越权威
    authority_level: int = 0
    source_type: str = "file"
    # 自动同步间隔（分钟）：0=不自动调度，仅手动 /sync 或 Webhook 触发；
    # >0 时由 rag.sync_due_connectors 周期任务按间隔派发全量幂等同步
    sync_interval_minutes: int = 0

    @field_validator("connector_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("connector_id 不能为空")
        return cleaned


class StructuredConfig(BaseModel):
    """结构化数据检索（v2.0）：让知识库在"文档问答"之外支持"统计类问题"。

    安全边界（缺一不可，全部在此显式配置，不做隐式放宽）：
    - sql_url 必须是**只读账号**的连接串（由运维配置，系统不提供写权限）；
    - allowed_tables 是表级白名单：LLM 生成的 SQL 只能引用白名单内的表；
    - max_rows / timeout_s 限制单次查询的行数与耗时，避免拖垮业务库。

    enabled=false（默认）时整条结构化链路完全关闭，不影响既有知识库。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # SQLAlchemy 同步连接串，如 sqlite:///./data/stats.db 或 postgresql+psycopg://...
    sql_url: str = ""
    # 表名白名单（大小写不敏感，可写 schema.table）
    allowed_tables: list[str] = Field(default_factory=list)
    # 裸表名的解析 Schema（如 "public"）：配置后白名单里的裸表名被钉死到该 Schema，
    # 跨 Schema 同名表（confidential.expenses）无法借"仅表名匹配"绕过白名单
    default_schema: str = ""
    max_rows: int = 100
    timeout_s: float = 5.0
    # 是否允许 LLM 生成 SQL（关闭时仅支持显式传入 select 的问题）
    use_llm_planner: bool = True
    # 业务语义层：字段含义（如 "expenses.amount": "含税报销金额"），避免模型
    # 把 amount 猜成不含税、把 status=2 猜成已完成
    field_descriptions: dict[str, str] = Field(default_factory=dict)
    # 指标口径字典（如 "报销总额": "符合统计条件的报销金额之和；本季度按业务时区"），
    # 让 SQL 语法正确之外，业务口径也正确
    metric_definitions: dict[str, str] = Field(default_factory=dict)

    @field_validator("max_rows")
    @classmethod
    def _max_rows_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("structured.max_rows 必须大于 0")
        return value

    @field_validator("timeout_s")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("structured.timeout_s 必须大于 0")
        return value

    @model_validator(mode="after")
    def _enabled_requires_target(self) -> StructuredConfig:
        if self.enabled:
            if not self.sql_url.strip():
                raise ValueError("structured.enabled=true 时必须配置 structured.sql_url")
            if not self.allowed_tables:
                raise ValueError(
                    "structured.enabled=true 时必须配置至少一个 structured.allowed_tables"
                )
        return self


class AgenticConfig(BaseModel):
    """Agentic RAG（v2.0）：多轮"规划 → 工具执行 → 反思补检索"，默认关闭。

    开启后每次问答会多消耗 1~max_rounds 次规划/反思 LLM 调用与多轮检索，
    适合复杂多跳问题；简单问答建议保持关闭以控制延迟与成本。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # 是否用 LLM 规划/反思；关闭时退化为 QueryRouter 单步检索（仅保留循环框架）
    use_llm_planner: bool = True
    # 总轮次上限（含初始计划）与每轮子问题上限
    max_rounds: int = 3
    max_sub_queries: int = 3
    # 证据合并后的最大片段数，防止多轮检索撑爆生成上下文
    max_evidence_chunks: int = 20
    # 少于该片段数视为证据不足，触发反思补检索
    min_evidence_chunks: int = 3
    # 执行预算：单次请求最多发起多少次工具调用（含 SQL/API），以及墙钟截止时间
    # 预算耗尽后即使还有计划步骤也不再执行，防止复杂问题成本失控
    max_tool_calls: int = 8
    deadline_ms: int = 30_000

    @field_validator(
        "max_rounds", "max_sub_queries", "max_evidence_chunks", "max_tool_calls", "deadline_ms"
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("agentic 的轮次/子问题/证据上限必须大于 0")
        return value

    @field_validator("min_evidence_chunks")
    @classmethod
    def _min_evidence_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("agentic.min_evidence_chunks 不能为负数")
        return value

    @model_validator(mode="after")
    def _evidence_bounds(self) -> AgenticConfig:
        if self.min_evidence_chunks > self.max_evidence_chunks:
            raise ValueError("agentic.min_evidence_chunks 不能大于 max_evidence_chunks")
        return self


class ApiEndpointConfig(BaseModel):
    """一个实时查询 API（v2.0）：库存/订单状态等"必须现查"的问题走这里。

    安全边界：URL 必须由运维在配置中显式登记（不做任意 URL 请求），
    请求参数固定为单一 query_param，超时与返回长度都有上限。
    """

    model_config = ConfigDict(extra="forbid")

    endpoint_id: str
    url: str
    description: str = ""
    timeout_s: float = 3.0
    query_param: str = "q"
    # 静态认证头（token 等由运维注入配置文件，不要写进代码）
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("endpoint_id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("api.endpoints[].endpoint_id 不能为空")
        return cleaned

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("api.endpoints[].url 必须是 http(s) 地址")
        return cleaned

    @field_validator("timeout_s")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("api.endpoints[].timeout_s 必须大于 0")
        return value


class ApiRetrievalConfig(BaseModel):
    """实时 API 检索配置；enabled=false（默认）时整条链路关闭。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    endpoints: list[ApiEndpointConfig] = Field(default_factory=list)
    # 返回给 LLM/用户的响应最多保留多少字符，防止超大 JSON 撑爆上下文
    max_chars: int = 4000
    # 响应体字节上限：超过直接拒绝，防止恶意/异常接口返回巨型响应拖垮进程
    max_response_bytes: int = 1_000_000
    # 允许的 Content-Type 前缀；命中任一前缀才接受响应（缺省头按兼容处理）
    allowed_content_types: list[str] = Field(
        default_factory=lambda: ["application/json", "text/"]
    )
    # 多端点时最低词元重叠分：低于该值视为"没有匹配的接口"，而不是盲调第一个
    min_match_score: int = 1
    # 企业内网接口可显式开启；默认拒绝私有/回环/链路本地/云元数据地址（SSRF 防护）
    allow_private_hosts: bool = False

    @field_validator("max_chars", "max_response_bytes", "min_match_score")
    @classmethod
    def _max_chars_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("api 的响应上限/匹配阈值必须大于 0")
        return value

    @model_validator(mode="after")
    def _enabled_requires_endpoint(self) -> ApiRetrievalConfig:
        if self.enabled and not self.endpoints:
            raise ValueError("api.enabled=true 时必须配置至少一个 api.endpoints")
        return self


class KBConfig(BaseModel):
    """一个知识库的完整配置（与单个 YAML 文件一一对应）。

    kb_id 是逻辑主键（对外 API、权限判断用），kb_code 是物理命名根
    （不可变，原因见模块 docstring）。三个派生属性 milvus_collection /
    es_index / es_alias 是命名约定的唯一实现处——全项目禁止在别处手拼
    这些名字，否则约定就失去约束力。
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    kb_code: str
    name: str
    description: str = ""
    # 前端欢迎页的示例问题（纯界面引导，不参与检索）；为空时前端回退到通用提示
    sample_questions: list[str] = Field(default_factory=list)
    tenant_id: str = "default"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingProfile = Field(default_factory=EmbeddingProfile)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    # 第一阶段只做知识库级 ACL；文档级 ACL 字段预留，二期启用
    doc_acl_enabled: bool = False
    enabled: bool = True
    # v2.0：外部知识源连接器列表（默认空，不影响既有知识库配置）
    connectors: list[ConnectorConfig] = Field(default_factory=list)
    # v2.0：结构化查询（统计类问题走只读 SQL，默认关闭）
    structured: StructuredConfig = Field(default_factory=StructuredConfig)
    # v2.0：实时 API 查询（库存/状态类问题，默认关闭）
    api: ApiRetrievalConfig = Field(default_factory=ApiRetrievalConfig)
    # v2.0：Agentic RAG（多轮规划与反思，默认关闭）
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)

    @field_validator("kb_id", "name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不能为空")
        return value.strip()

    @field_validator("sample_questions")
    @classmethod
    def _clean_questions(cls, value: list[str]) -> list[str]:
        """示例问题只用于界面引导：去空白、丢空串，并限制条数与单条长度。"""
        items: list[str] = []
        for item in value:
            text = item.strip()
            if not text:
                continue
            if len(text) > 200:
                raise ValueError("sample_questions 单条不能超过 200 字")
            items.append(text)
        if len(items) > 8:
            raise ValueError("sample_questions 最多 8 条")
        return items

    @field_validator("kb_code")
    @classmethod
    def _valid_kb_code(cls, value: str) -> str:
        if not KB_CODE_PATTERN.match(value):
            raise ValueError("kb_code 只允许小写字母开头，后接小写字母/数字/_/-，长度 2-32")
        return value

    @property
    def milvus_collection(self) -> str:
        return f"rag_{self.kb_code}"

    @property
    def es_index(self) -> str:
        return f"rag_{self.kb_code}_chunks"

    @property
    def es_alias(self) -> str:
        return f"rag_{self.kb_code}_search"


def validation_details(error: ValidationError) -> list[dict[str, Any]]:
    """把 pydantic 校验错误压成 [{location, message}]，供 API 响应 details 字段
    原样透传给前端，定位到具体哪个字段不合法。"""
    return [
        {"location": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
        for item in error.errors()
    ]


def load_kb_config(path: Path) -> KBConfig:
    """从单个 YAML 文件加载知识库配置。"""
    if not path.is_file():
        raise KBConfigError(f"知识库配置文件不存在：{path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise KBConfigError(
            f"知识库配置 YAML 解析失败：{path.name}", details={"error": str(exc)}
        ) from exc

    if not isinstance(raw, dict):
        raise KBConfigError(f"知识库配置必须是 YAML 字典：{path.name}")

    try:
        return KBConfig.model_validate(raw)
    except ValidationError as exc:
        raise KBConfigError(
            f"知识库配置校验失败：{path.name}", details={"errors": validation_details(exc)}
        ) from exc


def discover_kb_configs(directory: Path) -> list[KBConfig]:
    """扫描目录下所有 YAML，按 kb_code 排序返回。"""
    if not directory.is_dir():
        raise KBConfigError(f"知识库配置目录不存在：{directory}")

    paths = sorted({*directory.glob("*.yaml"), *directory.glob("*.yml")})
    return [load_kb_config(path) for path in paths]


def find_kb_config_path(directory: Path, kb_code: str) -> Path | None:
    """按 kb_code 定位配置文件；文件名不强制等于 kb_code，兼容人工命名的存量 YAML。"""
    for path in sorted({*directory.glob("*.yaml"), *directory.glob("*.yml")}):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(raw, dict) and raw.get("kb_code") == kb_code:
            return path
    return None


def save_kb_config(
    directory: Path, config: KBConfig, *, path: Path | None = None
) -> Path:
    """写 YAML 配置（先写隐藏临时文件再原子替换，避免进程崩溃留下半截文件）。

    原子写为什么重要：知识库配置在请求热路径上被反复读取（KBRegistry 重建、
    API 直接 load）。若直接覆盖写目标文件，进程在写一半时崩溃、或读者撞上
    写到一半的文件，都会拿到语法残缺的 YAML 导致整个知识库不可用。先写
    .{name}.tmp 再 os.replace：同一文件系统内 replace 是原子操作，读者要么
    看到旧版本、要么看到完整新版本，不存在中间态；崩溃残留的只是隐藏 tmp
    文件，不影响目标。

    新建时文件名固定为 {kb_code}.yaml；更新时回写原文件，保持人工命名不变。
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = path or directory / f"{config.kb_code}.yaml"
    data = config.model_dump(mode="json")
    # 默认分隔符是一串带换行的字符串，写进 YAML 可读性很差；与默认值一致时省略
    if data["chunking"].get("separators") == ChunkingConfig().separators:
        data["chunking"].pop("separators", None)
    payload = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)
    return target


def delete_kb_config(path: Path) -> None:
    """删除配置文件；文件已不存在时视为成功（幂等）。"""
    path.unlink(missing_ok=True)


class KBRegistry:
    """内存中的知识库注册表，支持按 kb_id 或 kb_code 查询。

    构造期即校验 kb_id / kb_code 全局唯一（重复直接抛 KBConfigError 拒绝启动），
    因此运行期 resolve() 不需要再处理歧义。注册表本身不可变：配置变更后由
    调用方整体重建一个新 Registry 替换引用，避免读写并发下的加锁复杂度。
    """

    def __init__(self, configs: list[KBConfig]) -> None:
        self._by_id: dict[str, KBConfig] = {}
        self._by_code: dict[str, KBConfig] = {}
        for config in configs:
            if config.kb_id in self._by_id:
                raise KBConfigError(f"kb_id 重复：{config.kb_id}")
            if config.kb_code in self._by_code:
                raise KBConfigError(f"kb_code 重复：{config.kb_code}")
            self._by_id[config.kb_id] = config
            self._by_code[config.kb_code] = config

    @classmethod
    def from_directory(cls, directory: Path) -> KBRegistry:
        """扫描配置目录并构建注册表；目录不存在或任一文件非法都会抛错。"""
        return cls(discover_kb_configs(directory))

    def resolve(self, key: str) -> KBConfig:
        """kb_id 和 kb_code 都能命中。"""
        config = self._by_id.get(key) or self._by_code.get(key)
        if config is None:
            raise KBNotFoundError(f"知识库不存在：{key}", details={"kb": key})
        return config

    def list_enabled(self) -> list[KBConfig]:
        """仅返回 enabled=true 的知识库；停用库对检索/问答不可见但配置保留。"""
        return [c for c in self._by_code.values() if c.enabled]

    def __len__(self) -> int:
        return len(self._by_id)
