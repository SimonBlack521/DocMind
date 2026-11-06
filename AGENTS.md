# AGENTS.md — 项目交接与协作约定

> 本文件随仓库迁移，是代码之外的上下文载体。新会话（包括换机、服务器上重新打开）
> 请先读本文件 + `README.md` + `configs/kbs/*.yaml`，再动手改代码。

## 1. 项目定位

企业级 RAG 服务：FastAPI + Celery + Milvus + Elasticsearch + PostgreSQL + Redis + S3 兼容对象存储。
能力范围：

- 混合检索：向量召回 + BM25 召回，RRF 融合，可选 Reranker（remote / local / ollama）
- 文档级 ACL：`tenant_id + kb_id` 隔离 + `allowed_groups` 检索期过滤（公开文档哨兵 `*`）
- 鉴权：`dev` / `oidc`（JWT）/ `ldap`（骨架已接，绑定与组搜索待完善）
- 生成：非流式与 SSE 流式，引用编号与提示词片段严格对齐，L1 文本匹配 + L2 数字事实校验
- 结构化：Query Router 分流 → sqlglot AST 校验 → 只读 SQL（硬 LIMIT + 超时 + 并发信号量）
- 实时 API 检索：端点白名单 + SSRF 防护 + 响应大小/类型限制
- Agentic RAG：规划 → 工具执行 → 证据覆盖反思，带工具调用次数与 deadline 预算
- 统一证据：`document` / `database` / `api` 共用 `Evidence`，挂到 `QueryResult` / `QueryStream`

## 2. 关键文件地图

| 关注点 | 文件 |
|---|---|
| API 路由 | `app/api/routers/`（chat / documents / kbs / tasks / connectors / health / metrics / audit） |
| 主编排 | `app/orchestration/pipeline.py`（query / query_stream / ingest_bytes / evidence） |
| 混合检索 | `app/retrieval/hybrid.py`（改写 → 多变体召回 → RRF → 精排 → MMR → 父块扩展） |
| 查询路由 | `app/retrieval/query_router.py`（规则 + confidence + intent） |
| Agentic 循环 | `app/orchestration/agentic.py`（Planner / Reflect / Coverage / 预算） |
| 生成与引用 | `app/generation/`（prompts / citations / citation_validator / answer） |
| 结构化查询 | `app/structured/`（retriever / sql_executor / api_retriever） |
| 入库与切块 | `app/ingestion/`（parsers / chunker / governance） |
| 异步任务 | `app/tasks/`（ingest / connectors / maintenance / celery_app） |
| 删除状态机 | `app/orchestration/deletion.py` + `app/db/repo.py` |
| 配置模型 | `app/domain/kb_config.py`、`app/core/settings.py`、`configs/kbs/*.yaml` |
| 统一证据 | `app/domain/evidence.py` |

## 3. 启动与运行

```bash
# 依赖栈（Linux/Windows 均可；Celery 不在 compose 内）
docker compose -f docker-compose.dev.yml up -d

# Linux 必须（Elasticsearch）
sudo sysctl -w vm.max_map_count=262144

python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                      # 改连接地址与密钥

alembic upgrade head
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# worker + beat（Linux 默认 prefork；Windows 必须 --pool=solo）
python -m celery -A app.tasks.celery_app.celery_app worker --loglevel=info
python -m celery -A app.tasks.celery_app.celery_app beat --loglevel=info
```

GPU 机器（如 V100S 32G）：`pip install -e ".[dev,local-models]"` 可跑本地
`sentence-transformers`（bge-m3 embedding / bge-reranker）；仍可保持 `*_BACKEND=remote`。

## 4. 质量门槛（每次改动必跑）

```bash
python -m ruff check app tests scripts
python -m pyright app scripts
python -m pytest tests -q
```

- 基线（2026-11-05）：**325 passed**（含 2 条真实依赖集成测试，栈未就绪时自动 skip）；
  ruff All checks passed；pyright **0 errors**（89 条可选依赖缺失 warning 属正常，
  如 torch/sqlglot/redis/boto3）。
- pyright 若因 Node 堆内存崩溃：`NODE_OPTIONS=--max-old-space-size=3072`。
- **禁止删除测试断言**；修实现而不是改断言（除非断言本身写错，需说明原因）。
- 保持既有外部 API 契约：新增字段必须有默认值或可空；不重命名既有响应字段。

## 5. 硬性约定

- 中文注释与 docstring；公开函数/方法写完整类型注解。
- 不搞一次性大重构；`pipeline.py` / `hybrid.py` / `agentic.py` / `repo.py` 等热点文件改动前先 Read。
- 同步/异步裁决统一：请求内走 async，Celery 任务内用 `resources.run()` 桥接，
  不要在 async 上下文里直接 `asyncio.run`。
- SQL 安全：只允许 sqlglot AST 校验通过的只读查询；禁止绕过 `SQLGuard`、
  禁止放宽硬 LIMIT / statement_timeout / 信号量上限。
- 引用编号 `[n]` 必须与 `build_context_block` 的 chunks 顺序一致；
  修改检索排序逻辑时同步检查 `citations.build_citations`。
- 生产护栏（`APP_ENV=prod`）：禁止 `AUTH_MODE=dev`、禁止 `CELERY_EAGER=true`；
  强制 `rbac_strict=true` 且必须配置 `KB_ADMIN_GROUP`，否则拒绝启动。
- 删除必须走 `DocumentDeletionService`（状态机 + tombstone + 定时重试），
  不要新增"直接删 PG 行"的路径。

## 6. 已完成的加固（v2.0.x，勿回退）

**P0（v2.0.1）**

- SQL：AST 校验 + 危险函数拦截 + 子查询包壳硬 `LIMIT` + 会话超时 + 并发信号量
- RBAC：生产强制 strict；任务详情校验 `allowed_kb_ids`
- 删除：统一状态机 DELETING→tombstone，对象删除失败保留元数据并由
  `rag.resume_pending_deletions` 定时重试
- Outbox：`FOR UPDATE SKIP LOCKED` 原子 claim + lease 回收 + `dispatching` 状态
- API 检索：SSRF（scheme/userinfo/私有 IP/元数据 host/DNS）、响应大小与 Content-Type 限制
- 知识库写操作鉴权、审计

**P1（v2.0.2）**

- Query Router：`confidence` / `intent`；制度语境消歧；多意图与弱实时词降级
- SQL 业务语义层：`field_descriptions` / `metric_definitions` 注入规划提示词
- Agentic：`CoverageReport` 证据覆盖反思 + `max_tool_calls` / `deadline_ms` 预算
- Graph 模块正名：`graph_enhanced_rerank`（Graph-enhanced Reranking，非完整 GraphRAG）
- 引用 L2：数字归一化（千分位/小数）后校验答案数字必须出现在片段中
- MMR 复用已入库 chunk 向量（`get_vectors_by_ids`），仅补缺失 Embedding
- 父块预算：`retrieval.parent_max_chars`（句末优先截断）+ `parent_context_max_chars`（全局）
- 连接器删除同步：全量扫描 + `source_uri` 稳定映射 + 内容指纹幂等 + PG 已入库集合差集
- 统一 Evidence Schema，覆盖文档 / SQL / API / 拒答全部返回点

## 7. 配置要点

- 知识库用 YAML（`KB_CONFIG_DIR=./configs/kbs`），模型为 `extra="forbid"`，字段名写错会直接报错。
- `structured.enabled=true` 必须同时配置 `sql_url` 与至少一个 `allowed_tables`。
- `api.enabled=true` 必须配置至少一个 `endpoints`，端点有 `min_match_score` 阈值。
- `agentic.enabled=true` 会额外消耗 1~`max_rounds` 次 LLM 调用；默认关闭。
- 检索默认：`top_k=20`、`rerank_top_n=6`、`rrf_k=60`；自适应 K 由 `adaptive_top_k` 控制。
- 模型后端按 `MODEL_DEVICE_PREFERENCE=remote,cuda,cpu` 降级；remote 不可用时才用本地。

## 8. 迁移到 Linux 服务器

- **不要复制 `.venv`**（Windows 二进制，Linux 不可用），服务器上重建虚拟环境。
- 不要复制 `__pycache__`、`.pytest_cache`、`.ruff_cache`、`logs`、`.idea`。
- `.env` 需要改：`DATABASE_URL` / `REDIS_URL` / `MILVUS_URI` / `ES_URL` / `S3_ENDPOINT_URL`
  以及各模型 API 地址与 Key。**不要提交 `.env`**。
- 首次启动顺序：`docker compose up -d` → 等 ES/Milvus healthy → `alembic upgrade head`
  → 启动 API → 启动 worker 与 beat。
- 端口与防火墙：19530（Milvus）、9200（ES）、5432（PG）、6379（Redis）、8333（S3 网关）。
- 若服务器只用远程模型 API，不需要 GPU；需要本地模型时注意 V100S 32G 的显存预算
  （embedding 与 reranker 不要同时按大 batch 常驻）。

## 9. 已知限制

- `auth_mode=ldap` 仅骨架接入，实际用户绑定/组搜索逻辑待实现。
- HPA 只对 Worker 生效；API 实例数建议手动管理。
- Redis 不可用时限流降级到进程内 dict，多副本下总速率 = 单实例 limit × 实例数。
- Ollama reranker 不可用时静默回退 RRF 顺序，不报错。
- 百万级 benchmark 已完成（2026-11-05，10K–100K–500K–1M 四档），原始结果在 `benchmark_report.json`，汇总解读在 `BENCHMARK_REPORT.md`。方法学边界：① 向量为归一化随机正态（非真实 embedding，Recall 数字是随机分布下界）；② 500K/1M 精确 Recall 未测（全量遍历 + 点积在大集合上太慢）；③ 测试脚本 `/tmp/benchmark_rag.py` 未纳入源码包。

## 10. 工作方式

- 先读代码再改；改完必须跑第 4 节三件套并贴出结果。
- 交付物（报告/文档）放工作区根目录；中间脚本/草稿放临时目录，不要污染仓库。
- 与用户沟通使用中文；代码引用用 `file:///` 可点击链接。
