"""全局配置：环境变量 / .env 统一收敛到 Settings，业务代码不直接读 os.environ。

在整体链路中的角色：这是全项目的"配置单一事实来源"。FastAPI 启动、Celery worker、
检索管线、鉴权模块等都通过 get_settings() 拿到同一份 Settings 实例。
统一收敛的好处：
- 类型与默认值集中声明，拼错环境变量名会在启动期暴露（pydantic 校验）；
- 生产护栏（见 _validate_production_safety）在进程启动时统一拦截危险误配，
  而不是等请求进来才发现"线上裸奔无鉴权"。

优先级：环境变量 > .env 文件 > 字段默认值（pydantic-settings 默认行为）；
字段名与环境变量名大小写不敏感（case_sensitive=False），如 app_env ↔ APP_ENV。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全量运行配置，按"应用 / 鉴权 / 存储 / 限流 / 检索 / 模型"分组声明。

    设计要点：
    - extra="ignore"：.env 里允许存在本类未声明的键（如第三方工具共用一个 .env），
      不会导致启动失败；
    - 所有跨进程组件（API 与 Celery worker）读同一份配置，因此任何需要两边一致的
      开关（如 celery_eager、embedding_dim）都必须放在这里而不是各进程各读各的。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "Enterprise RAG"
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"
    # 允许跨域的前端地址，逗号分隔；留空表示不启用 CORS
    cors_allow_origins: str = ""

    # ---------- 鉴权 ----------
    # dev：本地占位身份（禁止生产）；oidc：企业 SSO（OIDC + JWT）；ldap：目录服务（预留）
    auth_mode: Literal["dev", "oidc", "ldap"] = "dev"
    dev_tenant_id: str = "default"
    dev_user_id: str = "dev-user"
    # AUTH_MODE=oidc 时必填前三项；claim 名按企业 IdP 实际下发字段调整
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_tenant_claim: str = "tenant_id"
    oidc_kb_claim: str = "allowed_kb_ids"
    # 组 claim：映射到 CurrentUser.allowed_groups，用于文档级 ACL；Keycloak 默认下发 groups
    oidc_group_claim: str = "groups"
    oidc_allow_missing_groups: bool = False
    # 同样 fail-closed：多知识库环境下未下发 KB 授权 claim 默认按"无库可访问"处理
    oidc_kb_claim: str = "allowed_kb_ids"
    oidc_allow_missing_kb_ids: bool = False
    # AUTH_MODE=ldap 时必填前两项；HTTP Basic 凭证 → 服务账号搜用户 → 用户 rebind 验密码
    ldap_server_url: str = ""
    ldap_user_search_base: str = ""
    # 服务账号 bind（留空则匿名 bind）；用于搜索用户与用户组
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""
    # 用户搜索过滤器，{username} 为占位符（自动做 filter 转义）
    ldap_user_search_filter: str = "(uid={username})"
    # 组搜索 base，留空回落 user_search_base；过滤器占位符 {user_dn} / {username}
    ldap_group_search_base: str = ""
    ldap_group_search_filter: str = "(member={user_dn})"
    ldap_group_name_attribute: str = "cn"
    # 组 → 知识库映射，JSON 字符串，如 {"finance":["kb-1"],"hr":["kb-2"]}；留空表示不限制
    ldap_group_kb_map: str = ""
    ldap_timeout_s: float = 10.0

    # ---------- 关系库 / 缓存 ----------
    # 默认 PostgreSQL；企业已有 MySQL 时换成 mysql+asyncmy://...
    database_url: str = "postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag"
    redis_url: str = "redis://127.0.0.1:6379/0"
    # 测试时置 true：Celery 任务在当前进程内同步执行，无需启动 worker（禁止用于生产）
    celery_eager: bool = False
    # Celery 任务时限（秒）：软时限先到（SoftTimeLimitExceeded，给任务留清理机会），
    # 硬时限兜底强杀，防止单个解析/入库任务卡死 worker
    task_soft_time_limit_seconds: int = 540
    task_time_limit_seconds: int = 600
    # 检索热点缓存 TTL（秒）；<= 0 表示禁用缓存。只缓存召回 chunks，不缓存答案
    retrieval_cache_ttl_s: int = 300
    # 多副本配置同步：每 N 秒最多检查一次 Redis 配置版本号；<= 0 表示禁用
    kb_config_sync_interval_s: int = 30

    # ---------- 频率限制 ----------
    # rate_limit_enabled=false 时所有 check() 直接放行（测试环境默认关闭）
    rate_limit_enabled: bool = False
    # 每用户每分钟 chat 请求上限（问答/流式问答共享 action="chat"）
    rate_limit_chat_rpm: int = 30
    # 每用户每分钟 upload 请求上限（同步/异步上传共享 action="upload"）
    rate_limit_upload_rpm: int = 5

    # ---------- Milvus ----------
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"
    # ---------- Milvus 索引（向量 ANN） ----------
    # 百万级向量必须用显式图索引：AUTOINDEX 会把参数决策交给服务端，无法做
    # 召回率/延迟的确定性调优，也无法在评测中复现结果。默认 HNSW（召回与 QPS
    # 综合最优）；数据量小或追求 100% 召回时可切 FLAT（暴力精确检索）。
    milvus_index_type: Literal["HNSW", "FLAT", "IVF_FLAT", "AUTOINDEX"] = "HNSW"
    # HNSW 三个关键参数（仅 milvus_index_type=HNSW 时生效）：
    #   M               每个节点的最大出边数，越大图越稠密：召回↑ 内存↑ 建索引↑，常用 8~64
    #   ef_construction 建索引时的候选队列宽度：越大图质量越好（召回↑），构建越慢，常用 100~500
    #   ef_search       查询时的候选队列宽度：召回率与延迟的开关，常用 top_k~512，
    #                   必须 >= 检索 top_k，否则召回上限被 ef 截断
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_hnsw_ef_search: int = 64
    # 入库热路径遇到索引参数漂移（配置与线上集合不一致）时是否原地重建索引：
    # 默认 False——只告警，重建集中到维护任务 rag.reconcile_index（维护窗口执行），
    # 避免 drop_index 期间在线查询降级。
    milvus_auto_reconcile_index: bool = False

    # ---------- Elasticsearch（BM25 稀疏检索） ----------
    es_url: str = "http://127.0.0.1:9200"
    es_username: str | None = None
    es_password: str | None = None
    # v1.4-3: ES 索引规模动态调参（百万级 chunk 场景建议 shards=3~5）
    es_analyzer: str = "standard"
    es_number_of_shards: int = 1
    es_number_of_replicas: int = 0

    # ---------- 对象存储（MinIO / S3 兼容） ----------
    s3_endpoint_url: str = "http://127.0.0.1:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "rag-raw"
    s3_region: str = "us-east-1"
    s3_use_ssl: bool = False

    # ---------- 知识库 ----------
    kb_config_dir: Path = BASE_DIR / "configs" / "kbs"
    default_tenant_id: str = "default"
    # 知识库写操作（创建/更新/删除）的管理员组；留空表示任何未受限用户都可管理。
    # 受限用户（IdP 下发 allowed_kb_ids）始终不具备管理权限
    kb_admin_group: str = ""
    # RBAC 分层（v1.2）：贡献者（上传/删除文档）与审计员（查看审计日志）的授权组
    kb_contributor_group: str = ""
    kb_auditor_group: str = ""
    # 平台管理员组：跨知识库的运维动作（预留）
    platform_admin_group: str = ""
    # 严格模式：开启后所有角色必须显式配置授权组，未配置即"无人拥有该角色"（Fail-Closed）。
    # 生产环境建议开启；默认关闭以兼容开发/单租户部署
    rbac_strict: bool = False

    # ---------- Embedding ----------
    embedding_backend: Literal["remote", "local"] = "remote"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_timeout_s: float = 30.0
    embedding_batch_size: int = 32
    # 远程 Embedding 并发上限（进程内信号量）：高并发检索/入库时防止把模型网关打挂，
    # 也避免线程池里堆积过多阻塞调用；0 = 不限制（清单第 24 项）
    embedding_max_concurrency: int = 16
    # 向量语义版本：换模型/换维度/换归一化策略时必须递增（如 v1→v2）。
    # 增量入库只复用同版本向量；检索缓存 key 也携带它，切换后旧缓存自动失效
    embedding_version: str = "v1"

    # ---------- Reranker ----------
    rerank_backend: Literal["remote", "local", "ollama", "none"] = "remote"
    rerank_model: str = "BAAI/bce-reranker-base-zh"
    rerank_api_base: str = ""
    rerank_api_key: str = ""
    rerank_timeout_s: float = 30.0
    rerank_ollama_base: str = "http://127.0.0.1:11434"
    # 本地 cross-encoder 的批大小：同 batch 内按文本长度排序分桶，减少 padding 浪费
    rerank_batch_size: int = 16
    # 送精排的候选池上限：候选越多推理越慢，必须给全局上限，避免个别复杂查询拖垮 P99
    rerank_max_candidates: int = 50
    # 按需精排：向量首名相似度 >= 该值且问题简单时跳过重型 Reranker（0=关闭，始终精排）。
    # 精排是交叉编码器，代价高；FAQ 类高置信问题交给 RRF 融合结果即可
    rerank_skip_confidence: float = 0.0
    # 远程 Reranker 并发上限（进程内信号量）；0 = 不限制
    rerank_max_concurrency: int = 8

    # ---------- Query Embedding 缓存（9.7 第一层缓存） ----------
    # 同一问题在 TTL 内复用向量，避免重复调用 Embedding 模型；<=0 表示禁用
    query_embedding_cache_ttl_s: int = 3600
    query_embedding_cache_max_entries: int = 2048

    # ---------- LLM ----------
    llm_backend: Literal["remote", "ollama"] = "remote"
    llm_model: str = "qwen2.5-72b-instruct"
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_timeout_s: float = 120.0
    # 远程 LLM 并发上限（进程内信号量）：流式与非流式共用同一额度，
    # 占满时新请求排队等待而不是同时打到模型网关；0 = 不限制
    llm_max_concurrency: int = 8

    # ---------- 模型降级与检索默认值 ----------
    # 顺序含义：优先远程 API，其次本地 CUDA，最后 CPU
    model_device_preference: str = "remote,cuda,cpu"
    chunk_size: int = 512
    chunk_overlap: int = 60
    retrieve_top_k: int = 20
    rerank_top_n: int = 6
    rrf_k: int = 60

    # ---------- 检索结果治理加权 ----------
    # 在 RRF 融合之后按来源权威度与采集新鲜度做乘法加权：
    #   weight = (1 + 权威权重 × level/100) × (1 + 新鲜度权重 × 0.5^(age/half_life))
    # 任意一项置 0 即关闭该维度；两项都为 0 时完全保持 RRF 顺序
    retrieval_authority_weight: float = 0.2
    retrieval_recency_weight: float = 0.1
    retrieval_recency_half_life_days: float = 180.0

    # ---------- 低置信拒答 ----------
    # 精排后最高分低于该阈值时不调用 LLM，直接返回低置信拒答文案。
    # 阈值作用于精排分数（0~1），建议仅在 use_rerank=true 时启用（如 0.3）；
    # 未启用精排时 RRF 融合分是名次数、不是相似度，阈值无意义，应保持 0（关闭）
    retrieval_min_score: float = 0.0

    # ---------- 生成上下文预算 ----------
    # 送 LLM 的参考资料总字符预算：超出的片段截断并在末尾标注，防止长上下文
    # 撑爆 token 上限 / 稀释关键证据；0 表示不限制
    max_context_chars: int = 12000

    # ---------- 答案独立验证（v2.2） ----------
    # 引用校验只保证"引用列表与正文标注一致"，不能证明正文结论有证据支持：
    # 实测 20 条真实回答中 16 条没有可识别的 [n]，而引用列表平均 7.9 条。
    # 因此增加独立的答案验证状态（按送入模型的实际证据逐句校验）：
    # - 关键结论无支持或与证据冲突 → 优先重生成一次，仍失败则标注"未验证"；
    # - answer_regen_max=0 表示不重生成，只标注状态。
    answer_verification_enabled: bool = True
    answer_regen_max: int = 1
    # 有引用结论的比例下限：低于该比例且存在实质结论时标记 unverified
    answer_min_citation_coverage: float = 0.5

    # ---------- 上下文压缩（9.6） ----------
    # 抽取式压缩：按当前问题只保留相关句子，但强制保留条件/否定/时间/数值/权限等
    # 关键约束；只抽句不改写，定位 metadata 不变。短于 min_chars 的片段不压缩
    context_compression_enabled: bool = True
    context_compression_max_sentences: int = 8
    context_compression_min_chars: int = 200

    # ---------- 隐私与合规 ----------
    # 入库前对邮箱/手机号/身份证/银行卡做占位符脱敏（原文保留在对象存储，按 ACL 回查）
    privacy_redaction_enabled: bool = True
    # ---------- 多模态 OCR（v2.0，可选） ----------
    # 扫描版 PDF 抽不出文字时，对页面内嵌图片做 OCR；依赖缺失时自动跳过
    ocr_enabled: bool = False
    # tesseract 语言包，中文需安装 chi_sim
    ocr_language: str = "chi_sim+eng"
    # 页面文本少于该字符数才触发 OCR，避免文本型 PDF 被无谓识别
    ocr_min_chars: int = 20
    # 文档保留期（天）：0 表示不自动清理；>0 时由维护任务删除 collected_at 超期的文档
    document_retention_days: int = 0
    # 被新版本替代的旧文档保留期（天）：Active Version 切换后旧版只标记 superseded，
    # 保留观察窗供审计/回滚，超窗由 rag.purge_superseded_documents 走统一删除状态机清理；
    # 0 表示不自动清理
    superseded_retention_days: int = 30
    # ---------- 后台维护（Celery beat 周期） ----------
    # 索引对账（reconcile_index）周期（分钟）：0 表示不自动调度（手动或运维按需触发）；
    # 对账用于发现孤儿 chunk / 缺失 chunk，按清单自动修复；建议 30~240 分钟
    reconcile_interval_minutes: int = 60
    # Outbox 补投周期（秒）：0 表示关闭自动补投（仍可手动触发 rag.dispatch_outbox）；
    # 建议 10~60 秒，故障窗口内最多延迟一个周期
    outbox_dispatch_interval_seconds: int = 30
    # 多轮会话：追问改写读取的历史消息条数（user+assistant 合计）
    conversation_history_limit: int = 8
    # 会话保留期（天）：updated_at 超过该窗口未活跃的会话由 rag.purge_expired_conversations
    # 周期任务删除（会话含问答内容，长期保留有合规风险）；0 表示不过期
    conversation_retention_days: int = 30
    # 保留期清理批次大小：每次删除这么多条，避免一次性长时间占用 worker
    retention_purge_batch: int = 500
    # 删除失败重试周期（秒）：0 表示关闭；删除任何阶段失败都会留 DELETING，
    # 由 rag.resume_pending_deletions 周期推进，建议 60~600 秒
    deletion_retry_interval_seconds: int = 300
    # 连接器定时同步扫描周期（秒）：beat 按该周期检查各连接器
    # sync_interval_minutes 是否到期并派发同步；0 表示关闭定时同步
    connector_sync_scan_seconds: int = 300

    # ---------- 解析质量门禁 ----------
    # 不达标的文档进入 quarantined 状态，不写索引，避免污染检索结果
    parse_min_chars: int = 20
    parse_min_page_yield: float = 0.3
    parse_max_garbled_ratio: float = 0.05
    # 单页至少这么多字符才算"该页解析出了有效文本"
    parse_min_page_chars: int = 10

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """切块/向量维度的数值合法性校验：启动期 fail-fast，避免脏配置写入向量库。"""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE 必须大于 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须在 [0, CHUNK_SIZE) 区间内")
        if self.embedding_dim <= 0:
            raise ValueError("EMBEDDING_DIM 必须大于 0")
        if not 4 <= self.milvus_hnsw_m <= 64:
            raise ValueError("MILVUS_HNSW_M 必须在 [4, 64] 区间内")
        if not 8 <= self.milvus_hnsw_ef_construction <= 1024:
            raise ValueError("MILVUS_HNSW_EF_CONSTRUCTION 必须在 [8, 1024] 区间内")
        if not 1 <= self.milvus_hnsw_ef_search <= 32768:
            raise ValueError("MILVUS_HNSW_EF_SEARCH 必须在 [1, 32768] 区间内")
        if self.max_context_chars < 0:
            raise ValueError("MAX_CONTEXT_CHARS 不能为负数（0 表示不限制）")
        if not 0.0 <= self.retrieval_min_score <= 1.0:
            raise ValueError("RETRIEVAL_MIN_SCORE 必须在 [0, 1] 区间内（0 表示关闭）")
        if not 0.0 <= self.rerank_skip_confidence <= 1.0:
            raise ValueError("RERANK_SKIP_CONFIDENCE 必须在 [0, 1] 区间内（0 表示关闭）")
        if self.rerank_batch_size <= 0:
            raise ValueError("RERANK_BATCH_SIZE 必须大于 0")
        if self.rerank_max_candidates <= 0:
            raise ValueError("RERANK_MAX_CANDIDATES 必须大于 0")
        if self.context_compression_max_sentences < 0:
            raise ValueError("CONTEXT_COMPRESSION_MAX_SENTENCES 不能为负数")
        if self.context_compression_min_chars < 0:
            raise ValueError("CONTEXT_COMPRESSION_MIN_CHARS 不能为负数")
        if self.query_embedding_cache_max_entries <= 0:
            raise ValueError("QUERY_EMBEDDING_CACHE_MAX_ENTRIES 必须大于 0")
        return self

    @model_validator(mode="after")
    def _validate_production_safety(self) -> Settings:
        """生产护栏：两类"开发便利"配置在 APP_ENV=prod 下直接拒绝启动。

        - AUTH_MODE=dev：所有请求都会被当成同一个占位用户放行，等于线上无鉴权，
          任何人都能读写全部租户数据，必须杜绝；
        - CELERY_EAGER=true：入库任务退化为请求进程内同步执行，大文件解析会
          长时间阻塞事件循环，且绕过了 worker 的并发隔离与重试机制。

        之所以在配置加载期（而不是请求期）拦截：误配属于部署事故，越早炸越好，
        进程起不来会立刻被运维发现；若放行到请求期才报错，往往已经造成数据泄露。
        """
        if self.app_env == "prod" and self.auth_mode == "dev":
            raise ValueError("生产环境（APP_ENV=prod）禁止 AUTH_MODE=dev，请配置 oidc 或 ldap")
        if self.app_env == "prod" and self.celery_eager:
            raise ValueError("生产环境禁止 CELERY_EAGER=true")
        if self.app_env == "prod":
            # RBAC Fail-Closed：生产环境强制严格模式，空授权组不能自动升级成管理员；
            # 管理员组必须显式配置，避免"没配组=人人可管理知识库"的默认放权。
            if not self.rbac_strict:
                self.rbac_strict = True
            if not self.kb_admin_group.strip():
                raise ValueError("生产环境必须配置 KB_ADMIN_GROUP（RBAC 严格模式）")
        return self

    @property
    def model_device_order(self) -> list[str]:
        """把 MODEL_DEVICE_PREFERENCE 解析成有序列表，例如 ['remote', 'cuda', 'cpu']。"""
        return [
            item.strip().lower()
            for item in self.model_device_preference.split(",")
            if item.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级单例 Settings。

    lru_cache 保证整个进程只解析一次环境变量/.env（校验器也只跑一次），
    所有模块拿到的都是同一个对象。测试需要覆盖配置时，应清缓存
    （get_settings.cache_clear()）或 monkeypatch 环境变量后再取。
    """
    return Settings()
