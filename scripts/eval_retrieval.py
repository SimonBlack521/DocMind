"""检索质量离线评测：对 golden 问题集跑真实混合检索（不经 LLM 生成）。

指标覆盖审计要求的检索/证据/冗余/成本四组：
    - 检索：Recall@1/5/10、MRR、nDCG@K；
    - 证据：证据完整率、边界截断率、相邻块依赖率、引用定位成功率
      （golden 可选 `expect_evidence`，填写原文中的连续证据句）；
    - 冗余：重复 Chunk 比例、同文档候选占比、重复 Token 比例；
    - 成本：返回 Chunk 数、单条向量存储估算、上下文 Token、检索 P95。
无答案问题（golden 标 `expect_no_answer: true`）不参与检索指标，只用于生成阶段的拒答评测。

设计：
    - 直接装配 HybridRetriever（向量 + BM25 双路召回 + RRF），不经过 API / 检索缓存；
    - 默认关闭查询改写与精排，测纯检索链路；--rewrite / --rerank 可按 KB 配置打开；
    - --seed 会把评测语料重灌进目标知识库（doc_id 由 source_uri 哈希稳定生成，
      覆盖式写入，幂等可重复）；golden 的 expect_titles 任一命中即算该题命中；
    - --exact-baseline 会从 Milvus 全量取出候选池向量，用纯 Python 精确余弦算出
      "理论最优 top-N"（相当于 FLAT 暴力检索），再与 ANN（HNSW）实际召回对比，
      输出近似索引召回保持率（ANN 命中了精确 top-N 中多少比例），用于量化
      HNSW 参数（M/efConstruction/ef_search）造成的召回衰减。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

BASE_DIR = Path(__file__).resolve().parents[1]
# 脚本直接运行时无 pytest 的 pythonpath 注入，需要把项目根加入 sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.evaluation.metrics import (  # noqa: E402 - 必须在 sys.path 注入之后导入
    cost_metrics,
    evidence_metrics,
    ndcg_at_k,
    p95,
    recall_at_k,
    reciprocal_rank,
    redundancy_metrics,
)

if TYPE_CHECKING:  # 仅类型标注用，运行时这两个模块在 evaluate() 内延迟导入
    from app.core.settings import Settings
    from app.domain.kb_config import KBConfig

DEFAULT_CORPUS = BASE_DIR / "scripts" / "data" / "eval_finance_corpus.jsonl"
DEFAULT_GOLDEN = BASE_DIR / "scripts" / "data" / "eval_finance_golden.jsonl"
RECALL_KS = (1, 5, 10)


def _cosine(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度：评测语料规模小，无需 numpy 依赖。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_exact_pool(settings: Settings, kb: KBConfig) -> list[dict]:
    """从 Milvus 全量取出候选池（chunk_id/vector/title），作为精确检索基线。"""
    from pymilvus import MilvusClient

    kwargs: dict = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    client = MilvusClient(**kwargs)
    try:
        rows = client.query(
            collection_name=kb.milvus_collection,
            filter=f'kb_id == "{kb.kb_id}"',
            output_fields=["chunk_id", "vector", "metadata"],
            limit=65536,
        )
    finally:
        client.close()
    pool: list[dict] = []
    for row in rows or []:
        metadata = row.get("metadata") or {}
        pool.append(
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "vector": list(row.get("vector") or []),
                "title": str(metadata.get("title", "")),
            }
        )
    return pool


def _exact_top_n(
    query_vector: list[float], pool: list[dict], top_n: int
) -> list[tuple[str, str]]:
    """对候选池做暴力精确余弦排序，返回 top-N 的 (chunk_id, title)。"""
    scored = [
        (_cosine(query_vector, row["vector"]), row["chunk_id"], row["title"])
        for row in pool
        if row["vector"]
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk_id, title) for _, chunk_id, title in scored[:top_n]]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                rows.append(json.loads(line))
            else:
                print(f"[warn] {path.name}:{lineno} 空行已跳过")
    return rows


def seed_corpus(kb_code: str, corpus_path: Path) -> int:
    """把语料覆盖式重灌进知识库；返回入库篇数。"""
    from app.core.settings import get_settings
    from app.domain.kb_config import KBRegistry
    from app.orchestration.pipeline import RAGPipeline
    from app.orchestration.registry import RuntimeRegistry

    settings = get_settings()
    kb = KBRegistry.from_directory(settings.kb_config_dir).resolve(kb_code)
    registry = RuntimeRegistry(settings)
    try:
        pipeline = RAGPipeline(settings=settings, registry=registry)
        rows = load_jsonl(corpus_path)
        for row in rows:
            title = row["title"]
            pipeline.ingest_bytes(
                kb,
                data=row["content"].encode("utf-8"),
                filename=f"{title}.txt",
                source_uri=f"kb://{kb.kb_code}/{title}",
                title=title,
            )
            print(f"[seed] {title}")
        return len(rows)
    finally:
        registry.close()


def evaluate(
    kb_code: str,
    golden_path: Path,
    *,
    top_n: int,
    use_rewrite: bool,
    use_rerank: bool,
    exact_baseline: bool = False,
) -> int:
    from app.core.settings import get_settings
    from app.domain.kb_config import KBRegistry
    from app.orchestration.registry import RuntimeRegistry
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.query_rewrite import QueryRewriter

    settings = get_settings()
    kb = KBRegistry.from_directory(settings.kb_config_dir).resolve(kb_code)
    registry = RuntimeRegistry(settings)
    try:
        runtime = registry.runtime(kb)
        retriever = HybridRetriever(
            vector_store=runtime.vector_store,
            keyword_store=runtime.keyword_store,
            embedding=runtime.embedding,
            reranker=registry.reranker(),
            rewriter=QueryRewriter(registry.llm()),
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            config=kb.retrieval,
        )

        # 精确基线：全量候选池向量一次性取出，逐题算精确余弦 top-N（≈FLAT 暴力检索）
        exact_pool: list[dict] | None = None
        if exact_baseline:
            exact_pool = _load_exact_pool(settings, kb)
            if not exact_pool:
                print("[warn] 精确基线候选池为空，请先用 --seed 灌入语料")
                exact_pool = None
            else:
                print(f"[exact] 候选池向量数：{len(exact_pool)}")

        golden = load_jsonl(golden_path)
        answerable = [row for row in golden if not row.get("expect_no_answer")]
        # 两套口径分开统计：
        # - recall_at_k：标准 Recall@K = mean(|前K结果 ∩ 标注全集| / |标注全集|)
        # - hit_rate_at_k：前 K 个结果中"至少命中一个相关项"的查询比例
        recall_sums = {k: 0.0 for k in RECALL_KS}
        hits_at_k = {k: 0 for k in RECALL_KS}
        rr_sum = 0.0
        ndcg_sum = 0.0
        evidence_sums: dict[str, float] = {}
        redundancy_sums: dict[str, float] = {}
        context_tokens = 0
        latencies_ms: list[float] = []
        chunk_total = 0
        # 近似索引召回保持率：ANN 实际召回命中精确 top-N 的比例
        ann_retain_sum = 0.0
        ann_retain_count = 0
        exact_hits_at_k = {k: 0 for k in RECALL_KS}

        print(f"\n逐题明细（top_n={top_n} rewrite={use_rewrite} rerank={use_rerank}）：")
        for idx, row in enumerate(golden, 1):
            question = row["question"]
            expect = set(row.get("expect_titles") or [])
            started = time.perf_counter()
            chunks = retriever.retrieve(
                question,
                top_k=top_n,
                rerank_top_n=top_n,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
            )
            elapsed = time.perf_counter() - started
            latencies_ms.append(elapsed * 1000)
            chunk_total += len(chunks)

            redundancy = redundancy_metrics(chunks)
            for key, value in redundancy.items():
                redundancy_sums[key] = redundancy_sums.get(key, 0.0) + value
            cost = cost_metrics(chunks, vector_dim=retriever.embedding_dim, latencies_ms=[elapsed])
            context_tokens += int(cost["context_token_estimate"])

            if row.get("expect_no_answer"):
                print(f"  [{idx:02d}] 无答案题（仅参与拒答评测） {question}")
                continue

            flags = [chunk.title in expect for chunk in chunks]
            # 同一文档的多个 chunk 只能算一次命中（chunk 级计数会让 recall>1）
            relevance_ids = [chunk.title for chunk in chunks]
            rank = next((i for i, flag in enumerate(flags, 1) if flag), None)
            # golden_total 是 golden 标注全集大小（expect_titles 的条数），
            # 清单 E01 要求分母来自全集而非检索列表中的相关项数
            golden_total = len(expect)
            for k in RECALL_KS:
                recall = recall_at_k(
                    flags, k, golden_total=golden_total, relevance_ids=relevance_ids
                )
                recall_sums[k] += recall
                hits_at_k[k] += int(recall > 0)
            rr_sum += reciprocal_rank(flags)
            # golden 可选 expect_grades（{标题: 等级}）：提供时 nDCG 用全集等级严格口径
            grades_map = row.get("expect_grades") or {}
            if grades_map:
                ndcg_sum += ndcg_at_k(
                    flags,
                    top_n,
                    graded_relevance=[float(grades_map.get(c.title, 0.0)) for c in chunks],
                    golden_grades=[float(g) for g in grades_map.values()],
                    relevance_ids=relevance_ids,
                )
            else:
                ndcg_sum += ndcg_at_k(
                    flags, top_n, golden_total=golden_total, relevance_ids=relevance_ids
                )

            if exact_pool is not None:
                # 纯向量通道对比：ANN（HNSW，走真实 Milvus 检索）vs 精确余弦 top-N
                query_vector = runtime.embedding.embed_query(question)
                exact_top = _exact_top_n(query_vector, exact_pool, top_n)
                exact_ids = {chunk_id for chunk_id, _ in exact_top}
                ann_hits = runtime.vector_store.search(query_vector, top_k=top_n)
                ann_ids = {hit.chunk_id for hit in ann_hits}
                if exact_ids:
                    ann_retain_sum += len(ann_ids & exact_ids) / len(exact_ids)
                    ann_retain_count += 1
                exact_flags = [title in expect for _, title in exact_top]
                for k in RECALL_KS:
                    exact_hits_at_k[k] += int(
                        recall_at_k(exact_flags, k, golden_total=len(expect)) > 0
                    )

            if row.get("expect_evidence"):
                metrics = evidence_metrics(chunks, row["expect_evidence"])
                for key, value in metrics.items():
                    evidence_sums[key] = evidence_sums.get(key, 0.0) + value

            mark = f"命中@{rank}" if rank is not None else "未命中"
            print(f"  [{idx:02d}] {mark:<8} {question}（{elapsed * 1000:.0f}ms）")
            if rank is None:
                print(f"       期望：{sorted(expect)}；实际前5：{[c.title for c in chunks[:5]]}")

        total = max(len(answerable), 1)
        print("\n汇总：")
        for k in RECALL_KS:
            print(
                f"  recall@{k:<2}   = {recall_sums[k] / total:.3f}"
                "  # 标准口径：|前K∩标注全集|/|标注全集| 的均值"
            )
            print(
                f"  hit_rate@{k:<2} = {hits_at_k[k]}/{total} = {hits_at_k[k] / total:.3f}"
                "  # 前K至少命中一个相关项的查询比例"
            )
        print(f"  MRR       = {rr_sum / total:.3f}")
        print(f"  nDCG@{top_n:<3} = {ndcg_sum / total:.3f}")

        if ann_retain_count:
            print(f"\n近似索引对账（{ann_retain_count} 道有答案题，向量通道 top-{top_n}）：")
            print(
                f"  ann_recall_keep_ratio = {ann_retain_sum / ann_retain_count:.3f}"
                "  # ANN 命中精确 top-N 的比例，1.0 表示无召回衰减"
            )
            for k in RECALL_KS:
                print(
                    f"  exact_recall@{k:<2}        = {exact_hits_at_k[k]}/{total}"
                    f" = {exact_hits_at_k[k] / total:.3f}  # 精确检索（FLAT 等效）的理论上限"
                )

        evidence_rows = sum(1 for row in answerable if row.get("expect_evidence"))
        if evidence_rows:
            print(f"\n证据质量（{evidence_rows} 道带证据标注的题）：")
            for key, value in evidence_sums.items():
                print(f"  {key:<26} = {value / evidence_rows:.3f}")

        print("\n冗余与成本：")
        for key, value in redundancy_sums.items():
            print(f"  {key:<26} = {value / len(golden):.3f}")
        print(f"  avg_chunks_per_query     = {chunk_total / len(golden):.1f}")
        print(f"  avg_context_tokens       = {context_tokens / len(golden):.0f}")
        p95_note = (
            ""
            if len(latencies_ms) >= 20
            else f"（样本仅 {len(latencies_ms)} 条，P95 实为最大值，仅供小样本冒烟参考）"
        )
        print(f"  retrieval_p95_ms         = {p95(latencies_ms):.1f}{p95_note}")
        return 0
    finally:
        registry.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量离线评测（recall@k / MRR）")
    parser.add_argument("--kb", default="finance", help="知识库 kb_code，默认 finance")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="golden JSONL 路径")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="评测语料 JSONL 路径")
    parser.add_argument("--no-seed", action="store_true", help="跳过语料重灌（已入库时使用）")
    parser.add_argument("--top-n", type=int, default=10, help="召回截断深度，默认 10")
    parser.add_argument("--rewrite", action="store_true", help="开启查询改写（会调用 LLM）")
    parser.add_argument("--rerank", action="store_true", help="开启精排（按 RERANK_BACKEND）")
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="同一 golden 集分别跑 关闭/开启 精排并各输出一份指标（9.2 要求召回前后对比）",
    )
    parser.add_argument(
        "--exact-baseline",
        action="store_true",
        help="对账近似索引：用精确余弦（FLAT 等效）作基线，输出 ANN 召回保持率",
    )
    args = parser.parse_args()

    if not args.no_seed:
        count = seed_corpus(args.kb, args.corpus)
        print(f"[seed] 语料重灌完成，共 {count} 篇")

    if args.compare_rerank:
        # 精排只能重排已召回候选，找不回召回阶段漏掉的证据：
        # 必须同时看"召回后（未精排）"与"精排后"两套指标，才能区分召回瓶颈与排序瓶颈
        for enabled in (False, True):
            print(f"\n{'=' * 20} rerank={'on' if enabled else 'off'} {'=' * 20}")
            code = evaluate(
                args.kb,
                args.golden,
                top_n=args.top_n,
                use_rewrite=args.rewrite,
                use_rerank=enabled,
                exact_baseline=args.exact_baseline,
            )
            if code != 0:
                return code
        return 0

    return evaluate(
        args.kb,
        args.golden,
        top_n=args.top_n,
        use_rewrite=args.rewrite,
        use_rerank=args.rerank,
        exact_baseline=args.exact_baseline,
    )


if __name__ == "__main__":
    sys.exit(main())
