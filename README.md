# Enterprise RAG

企业级 RAG（Retrieval-Augmented Generation）服务：混合检索（向量 BM25 融合 RRF）+ LLM 生成 + 文档级 ACL 权限控制。

```
                    ┌─────────────────┐
  ┌───────┐   HTTP  │   FastAPI API   │
  │ 客户端 │ ──────▶ │  /api/v1/kbs/*  │ ──┐
  └───────┘   SSE   │  /health /readyz│   │
                    └────────┬────────┘   │ 依赖注入
                             │            │  CurrentUser
                    ┌────────▼────────┐   │  Pipeline
                    │   RAG Pipeline  │ ◀─┘  RateLimiter
                    │ ┌─────────────┐ │
                    │ │  Hybrid     │ │     ┌──────────┐
                    │ │  Retriever  │ ├────▶│ Milvus    │ 向量库
                    │ │  RRF 融合   │ │     └──────────┘
                    │ └──────┬──────┘ │     ┌──────────┐
                    │        │        ├────▶│ ES / BM25 │ 稀疏索引
                    │   Reranker       │     └──────────┘
                    │   (可选)         │
                    │        │        │     ┌──────────┐
                    │   Generation    ├────▶│ LLM API   │ 远程 / Ollama
                    └────────┬────────┘     └──────────┘
                             │
                    ┌────────▼────────┐
                    │  Celery Worker  │ ──▶ 文档解析 / 切块 / 入库
                    │  (异步任务)      │     ┌──────────┐
                    └────────┬────────┘     │ SeaweedFS │ 对象存储
                             │              └──────────┘
                    ┌────────▼────────┐     ┌──────────┐
                    │ PostgreSQL      │     │ Redis     │ 缓存 + 限流
                    │ (元数据 + 审计)   │     └──────────┘
                    └─────────────────┘
```

## 快速启动

### 模式 A：Docker Compose（推荐）

```bash
# 开发模式（依赖容器化，API 热重载）
docker compose -f docker-compose.dev.yml up -d

# 生产模式（9 服务完整栈：api + worker + milvus + es + pg + redis + seaweedfs + etcd）
docker compose -f docker-compose.prod.yml up -d --build

# 应用启动后跑 DB 迁移
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### 模式 B：本地 venv（不需要完整栈）

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

pip install -e ".[dev]"
cp .env.example .env            # 按需改配置

# 启动 API（不需要完整 Milvus/ES/Redis 也能跑基础端点）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 启动 worker（异步入库用）
python -m celery -A app.tasks.celery_app.celery_app worker --concurrency 4
```

## 鉴权模式

| 模式 | 适用场景 | 必需配置 |
|------|---------|---------|
| `auth_mode=dev` | 本地开发、冒烟测试 | 无 —— 用占位用户 `dev-user` |
| `auth_mode=oidc` | 企业 SSO（Keycloak / Auth0） | `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL` |
| `auth_mode=ldap` | 目录服务（预留） | `LDAP_SERVER_URL`, `LDAP_USER_SEARCH_BASE` |

**生产安全护栏**：`APP_ENV=prod` 强制拒绝 `AUTH_MODE=dev` 和 `CELERY_EAGER=true`。

## 核心端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/kbs/{kb_key}/chat` | 知识库问答（非流式） |
| POST | `/api/v1/kbs/{kb_key}/chat:stream` | SSE 流式问答 |
| POST | `/api/v1/kbs/{kb_key}/documents` | 文档同步入库（≤32MB） |
| POST | `/api/v1/kbs/{kb_key}/documents:async` | 文档异步入库（Celery） |
| DELETE | `/api/v1/kbs/{kb_key}/documents/{doc_id}` | 删除文档 |
| GET | `/api/v1/kbs/{kb_key}/documents` | 分页列出文档 |
| POST/PUT/DELETE | `/api/v1/kbs` | KB 管理 API（需 KB_ADMIN_GROUP） |
| GET | `/health` | 完整健康状态 + 依赖探活 |
| GET | `/healthz` | liveness（永远 200） |
| GET | `/readyz` | readiness（prod 任一依赖 down → 503） |
| GET | `/metrics` | JSON 快照 |
| GET | `/metrics/prometheus` | Prometheus exposition format |

## 频率限制

生产默认关闭。设置 `RATE_LIMIT_ENABLED=true` 开启：
- `RATE_LIMIT_CHAT_RPM=30` — 每用户每分钟 chat 请求上限
- `RATE_LIMIT_UPLOAD_RPM=5` — 每用户每分钟 upload 请求上限
- Redis 不可用时自动降级到进程内 dict 实现，不阻断主链路

被拒返回 HTTP 429 + `Retry-After` header。

## 环境变量速查

| 变量 | 默认 | 说明 |
|------|------|------|
| `APP_ENV` | `dev` | `dev` / `test` / `prod` |
| `AUTH_MODE` | `dev` | `dev` / `oidc` / `ldap` |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL 连接串 |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis 连接串 |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 连接 |
| `ES_URL` | `http://127.0.0.1:9200` | Elasticsearch |
| `S3_ENDPOINT_URL` | `http://127.0.0.1:9000` | 对象存储（MinIO / SeaweedFS） |
| `LLM_BACKEND` | `remote` | `remote` / `ollama` |
| `EMBEDDING_BACKEND` | `remote` | `remote` / `local` |
| `RERANK_BACKEND` | `none` | `none` / `remote` / `ollama` / `local` |
| `RATE_LIMIT_ENABLED` | `false` | 生产防刷开关 |
| `CELERY_EAGER` | `false` | 测试同步模式（prod 禁止） |

完整配置见 `app/core/settings.py`。

## 测试

```bash
# 三件套（每次提交前必跑）
python -m ruff check app tests scripts
python -m pyright app
python -m pytest tests/ -q          # 113+ passed

# 检索评测（本地知识库）
python scripts/eval_retrieval.py --kb finance --top-n 10

# 生成评测（含忠实度、引用、数值一致性）
python scripts/eval_generation.py --kb finance --show-answers
```

## 部署

- **Docker Compose**：`docker compose -f docker-compose.prod.yml up -d --build`
- **Kubernetes**：`kubectl apply -f k8s/`（含 ConfigMap / Secret / API / Worker / HPA）
- **GitHub Actions CI**：push / PR 自动跑 ruff + pyright + pytest（Python 3.11 + 3.12 matrix）

## 已知限制

- Reranker 走 `ollama` 时要求 Ollama 实例可访问；未启动会静默回退原顺序，不影响主链路
- `auth_mode=ldap` 仅做了骨架接入，实际绑定 / 组搜索逻辑待实现
- HPA 只对 Worker 生效；API 实例数建议手动管理（CPU 敏感）
- Redis 不可用时限流自动降级，多副本部署下总速率会是单实例 limit × 实例数
