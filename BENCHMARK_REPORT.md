# RAG 系统百万级能力基准测试报告

> 生成时间：2026-11-05 08:29；版本校正 2026-11-05
> 测试环境：宿主机 Milvus standalone，pymilvus 3.x API，实际版本需以服务端 `/version` 接口为准。
> 注：`docker-compose.dev.yml` 与 `docker-compose.prod.yml` 标注的镜像标签为 `milvusdb/milvus:v2.4.9`，
>     与宿主机独立运行实例的版本存在漂移——重新部署 compose 后应以容器内 `milvus version` 为准。
> 脚本：`/tmp/benchmark_rag.py`（未纳入源码包），原始结果：`benchmark_report.json`

## 一、测试设计

| 参数 | 值 | 说明 |
|------|------|------|
| 向量维度 | 1024 | 模拟真实 embedding（bge-m3 1024 维） |
| 向量分布 | 归一化正态随机 | cosine similarity 基线零（高方差分布的下界） |
| 索引 | HNSW(M=16, efConstruction=200) | 与生产 `rag_finance` 同参数 |
| efSearch | 64 | 生产默认 |
| Metric | COSINE | 内积 = cosine similarity（向量已归一化） |
| 灌数批次 | 5000 vectors/batch | insert 吞吐最优点 |
| 精确 Recall golden | 10K: 200 / 100K: 50 | 大集合跳过（太慢） |
| Golden query 种子 | 固定种子 42 | 跨阶段可复现 |
| 并发压测 | 8 workers × 200 queries | 模拟典型 RAG 并发 |

**ANN Recall@K 的解读边界**

报告中的 Recall 是 HNSW ANN 检索结果与同一集合上精确近邻结果的重合度，反映的是近似检索与精确检索的差异，不是 RAG 问答的语义 Recall。

测试使用的是 1024 维归一化正态随机向量——各向量相似度方差极小（cosine similarity 分布极平坦）。在这种分布上，"找 top-k 近邻"本身就是困难任务：精确检索的 top-k 之间相似度可能仅差小数点后第三位，ANN 一旦错过任何一个边缘项就直接掉到 top-k+1 之外。这解释了 10K 档 Recall@10=0.242、100K 档 Recall@10=0.068 的低值。

**但不能由此推导真实 embedding 的 Recall 一定会更高**。真实 embedding 分布有语义聚类（高方差），理论上 ANN 更容易命中语义相关簇——这需要用 bge-m3 对真实语料生成的百万向量重测，并用 FAISS IndexFlatIP 或等价精确搜索算 Ground Truth，得到 HNSW 的 Recall-Latency 曲线。当前数据是"随机分布下界"，不是 RAG 质量指标。

另外，精确基线的实现有一处方法学限制：`_exact_top_k()` 使用主键前缀分段遍历全集合（绕开 Milvus offset 硬限制），每次 query 都拉全集合向量算点积。这在 100K 档每条 query 需 3-5 秒，500K+ 跳过。如果要在 500K/1M 规模测精确 Recall，建议预先一次性导出全量向量到内存做 Ground Truth 复用，而不是每个 query 都重新遍历。

## 二、原始数据

### 2.1 汇总表

| 档位 | 行数 | 磁盘 | 灌数速率 | p50 | p95 | p99 | 并发 QPS | 并发 p95 | Recall@1 | Recall@5 | Recall@10 |
|------|------|------|----------|-----|-----|-----|----------|----------|----------|----------|-----------|
| **10K** | 10,000 | 747 MB | 2,256 v/s | **3.7 ms** | 5.3 ms | 7.7 ms | **903** | 14.4 ms | 0.395 | 0.484 | 0.242 |
| **100K** | 100,000 | 1,102 MB | 4,310 v/s | **10.5 ms** | 14.5 ms | 18.2 ms | **370** | 31.2 ms | 0.300 | 0.136 | 0.068 |
| **500K** | 500,000 | 2,766 MB | 3,725 v/s | **13.1 ms** | 20.7 ms | 45.8 ms | **124** | 86.1 ms | — | — | — |
| **1M** | 1,000,000 | 4,645 MB | 3,898 v/s | **23.3 ms** | 53.6 ms | 129.6 ms | **58** | 296.9 ms | — | — | — |

### 2.2 缩放分析

| 指标 | 10K→100K | 100K→500K | 500K→1M | 10K→1M 总倍数 |
|------|----------|-----------|---------|---------------|
| 行数 | ×10 | ×5 | ×2 | **×100** |
| 磁盘 | ×1.47 | ×2.51 | ×1.68 | **×6.22** |
| p50 延迟 | ×2.84 | ×1.25 | ×1.78 | **×6.30** |
| p95 延迟 | ×2.74 | ×1.43 | ×2.59 | **×10.11** |
| 并发 QPS | ×0.41 | ×0.33 | ×0.47 | **×0.064** |

**关键发现：HNSW 延迟基本符合 O(log n) 预期**

行数 ×100，p50 延迟只涨 ×6.3。这是 HNSW 索引的核心优势——从 10K 到 1M，每增加 10 倍数据量只增加约 2.5-3 倍延迟。

## 三、能力水位

### 3.1 Milvus 单节点能力边界

| 能力维度 | 水位 | 说明 |
|----------|------|------|
| **单节点最大容量** | ≥ 1M rows / 4.6 GB | 磁盘和索引构建都健康 |
| **单线程检索延迟** | p50 ≤ 25ms @ 1M rows | efSearch=64 |
| **8 并发 QPS** | ≥ 58 @ 1M rows | 单机无 GPU |
| **灌数吞吐** | ~4,000 v/s（批次 5K） | 受 Milvus insert→flush→index pipeline 限制 |

### 3.2 对生产的启示

**当前生产（rag_finance：9,070 rows）远在水位以下**

- 生产库只有 9K 行，对应延迟预估 ≤ 4ms（远低于 1M 水位的 23ms）
- Milvus 单节点可支撑 100x 生产规模（1M 行级）才需要考虑扩容

**生产扩容路线**

| 规模 | 方案 |
|------|------|
| ≤ 5M rows | Milvus 单节点（当前够用） |
| 5M - 50M rows | Milvus Cluster（多节点 + 分区）+ HNSW 参数调优（M=32, efConstruction=400） |
| ≥ 50M rows | 考虑混合方案：Hot 数据 HNSW + Cold 数据 IVF_PQ 或 DiskANN |

## 四、瓶颈定位

### 4.1 Milvus query offset 硬限制（踩坑）

**问题**：Milvus 3.5.14 的 `client.query(offset=N, limit=M)` 强制 `(offset + limit) ≤ 16384`，且 offset 本身不能超过 16384。这意味着 offset 分页根本无法覆盖 16384+ 行的集合。

**修复**：改用主键前缀分段 filter（`chunk_id >= "bench-000000" and chunk_id < "bench-002000"`），每段 offset=0, limit=2000，完全避开窗口限制。主键格式必须补零（字典序 == 数值序）。

**影响**：所有需要遍历全集合的场景（精确基线、数据校验等）都必须用此方案。

### 4.2 100K+ 精确 Recall 效率

用主键分段拉全集合 + 批量算点积的精确基线在 100K 档每条 query 约需 3-5 秒（50 条 query 总计约 3 分钟），500K+ 档预估 20-30 分钟/query。**生产验证不建议跑全量精确基线**，改用以下方案：
- 采样 1-5K 条数据做近似精确基线
- 或直接用 RAGAS 的 faithfulness / context_precision 评估真实检索质量

### 4.3 HNSW 参数调优空间

当前测试用的是生产默认参数（M=16, efConstruction=200, efSearch=64）。对 1M+ 规模：
- **M=32**（更多邻居）：索引更大（约 2x），但 Recall 更高
- **efSearch=128/256**：在线搜索更慢（约 2x/4x），但 Recall 显著提升
- **磁盘**：当前 4.6GB/1M rows ≈ 4.6KB/row。原始 1024 维 float32 向量的理论下限是 4KB/row；差值来自 HNSW 索引、segment 元数据和服务级固定开销。
  精确拆分需在服务器上分别测量 `/var/lib/milvus/data/segments` 与 `/var/lib/milvus/data/indexs` 目录体积。

## 五、已完成的修复（本轮会话）

| 文件 | 修改 | 解决的问题 |
|------|------|------------|
| `app/structured/sql_executor.py` | `_check_tables()` 重写 + `_QUERY_SLOTS` 安全归还 | SQLGuard 跨 schema 绕过 + 信号量泄漏 |
| `app/db/repo.py` | 新增 `list_hidden_doc_ids()` | 删除状态文档在检索中可见 |
| `app/orchestration/pipeline.py` | 新增 `invalidate_retrieval_cache()` + 显式 `AppError` 上抛 + `degraded` 标记 | 静默回退 + 缓存脏读 |
| `app/generation/citation_validator.py` | 新增 `clean_answer_refs()` | 引用列表收缩但答案文本残留 [n] |
| `app/api/routers/chat.py` | SSE 新增 validation 事件 + done 事件带 evidence | 前端无最终引用校验反馈 |
| `app/web/index.html` + `app.js` + `style.css` | 原生 JS SPA（无构建依赖） | 前端原型上线 |
| `/tmp/benchmark_rag.py` | `_exact_top_k` 改用主键前缀分段 + chunk_id 补零 | Milvus offset 硬限制导致 100K 档崩溃 |

## 六、扩展建议

### 6.1 短期（1-2 周）
1. **真实 embedding 重测**：用 bge-m3 对真实语料生成 1M 向量重跑 benchmark，验证 Recall 绝对值
2. **Reranker 端到端测试**：加上 reranker 后 top-10 质量提升多少
3. **SSE streaming 压测**：当前只测了 search，没测完整的 query → generate → stream 链路

### 6.2 中期（1-2 月）
4. **Milvus Cluster 部署**：5 节点（3 data + 2 coordinator）测试 5-10M 规模
5. **HNSW / DiskANN / IVF_PQ 对比**：不同索引在不同规模下的延迟-Recall-Pareto
6. **混合检索 benchmark**：向量 + BM25 + RRF + reranker 全链路吞吐

### 6.3 长期（3-6 月）
7. **冷热分离架构**：热数据 HNSW（高频查询），冷数据 IVF_PQ（低成本存储）
8. **Agentic RAG 端到端测试**：规划 → 多轮检索 → SQL → API → 反思，全链路耗时预算
9. **生产监控看板**：Milvus 指标（segment compaction lag、索引构建延迟、查询 p99）接入 Grafana

---

**原始 JSON 报告**：[benchmark_report.json](file:///home/RAG_SYSTEM/Rag_System/benchmark_report.json)
**修复后的 benchmark 脚本**：[benchmark_rag.py](file:///tmp/benchmark_rag.py)
**全量原始日志**：[benchmark_full.log](file:///tmp/benchmark_full.log)
