"""百万级规模压测（v1.4）：在无真实 Milvus/ES 集群时评估管线关键路径的延迟与吞吐。

测什么（component-level mock，不假装测了外部存储）：
1. chunking：用 TextChunker 切分合成文档，测"文档/秒、块/秒、平均块长"
   —— 入库链路里唯一纯 CPU、可稳定复现的环节；
2. fusion：用合成候选集反复跑 RRF 融合，测单次融合延迟分位值与并发吞吐
   —— 检索链路里与存储无关的 CPU 开销，候选规模可参数化到百万级 chunk 池。

不测什么：Milvus/ES 网络往返、Embedding 推理、重排服务——这些必须用真实集群，
mock 出来的数字没有参考价值；本脚本只给出"应用层自身开销"的基线。

用法示例（PowerShell）：
    python scripts/benchmark_scale.py --mode both --chunks 1000000 --queries 500 \
        --concurrency 16 --docs 500 --size 512 --json reports/bench.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
# 脚本直接运行时没有 pytest 的 pythonpath 注入，需要把项目根加入 sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.domain.dto import RetrievedChunk  # noqa: E402 - 必须在 sys.path 注入之后导入
from app.domain.kb_config import ChunkingConfig  # noqa: E402
from app.ingestion.chunker import TextChunker  # noqa: E402
from app.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """延迟统计：均值 + 分位值，单位毫秒。"""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True, slots=True)
class FusionReport:
    """融合压测结果：延迟分布 + 并发吞吐 + 数据规模。"""

    pool_size: int
    queries: int
    concurrency: int
    depth: int
    latency: LatencyStats
    throughput_qps: float


@dataclass(frozen=True, slots=True)
class ChunkingReport:
    """切块压测结果：文档数、块数、耗时与吞吐。"""

    documents: int
    chunks: int
    total_chars: int
    elapsed_s: float
    docs_per_s: float
    chunks_per_s: float
    avg_chunk_chars: float


def _latency_stats(values: list[float]) -> LatencyStats:
    """毫秒延迟统计；样本为空时返回全 0，避免脚本因参数为 0 直接崩。"""
    if not values:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    return LatencyStats(
        count=len(ordered),
        mean_ms=round(statistics.fmean(ordered), 3),
        p50_ms=round(_percentile(ordered, 0.50), 3),
        p95_ms=round(_percentile(ordered, 0.95), 3),
        p99_ms=round(_percentile(ordered, 0.99), 3),
        max_ms=round(ordered[-1], 3),
    )


def _percentile(ordered: list[float], ratio: float) -> float:
    """最近秩分位值：样本量小时比插值法更稳定。"""
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


def _synthetic_channels(
    rng: random.Random, pool_size: int, depth: int
) -> list[list[RetrievedChunk]]:
    """构造两路合成候选（向量路与 BM25 路有部分重叠），模拟真实融合输入。"""
    vector_ids = rng.sample(range(pool_size), depth)
    # 第二路取 60% 的相同 id + 40% 新 id：既有叠加加分，也有单路独有候选
    overlap = vector_ids[: max(1, int(depth * 0.6))]
    extra = rng.sample(range(pool_size), depth - len(overlap))
    bm25_ids = overlap + extra
    rng.shuffle(bm25_ids)

    def make(chunk_id: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=f"c{chunk_id}",
            content="synthetic content",
            score=score,
            doc_id=f"d{chunk_id % 1000}",
            source_uri=f"s3://bench/{chunk_id}",
        )

    vector = [make(chunk_id, 1.0 - rank / depth) for rank, chunk_id in enumerate(vector_ids)]
    bm25 = [make(chunk_id, depth - rank) for rank, chunk_id in enumerate(bm25_ids)]
    return [vector, bm25]


def benchmark_fusion(
    *,
    pool_size: int,
    queries: int,
    concurrency: int,
    depth: int,
    k: int,
    seed: int,
) -> FusionReport:
    """在并发线程里重复执行 RRF 融合，统计单次延迟与整体吞吐。"""

    def one_query(index: int) -> float:
        # 每个任务独立随机源：多线程共享 Random 会串行化并污染结果
        rng = random.Random(seed + index)
        channels = _synthetic_channels(rng, pool_size, depth)
        started = time.perf_counter()
        reciprocal_rank_fusion(channels, weights=[0.5, 0.5], k=k)
        return (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        latencies = list(executor.map(one_query, range(queries)))
    elapsed = time.perf_counter() - started
    return FusionReport(
        pool_size=pool_size,
        queries=queries,
        concurrency=concurrency,
        depth=depth,
        latency=_latency_stats(latencies),
        throughput_qps=round(queries / elapsed, 2) if elapsed > 0 else 0.0,
    )


def _synthetic_document(index: int, sections: int, section_chars: int) -> str:
    """构造多小节文档：标题 + 正文，覆盖结构感知切分路径。"""
    parts: list[str] = [f"# 压测文档 {index}"]
    for section in range(sections):
        body = (f"第{section}节内容。" * (section_chars // 6 + 1))[:section_chars]
        parts.append(f"## 小节 {section}\n{body}")
    return "\n\n".join(parts)


def benchmark_chunking(
    *, documents: int, size: int, sections: int, section_chars: int
) -> ChunkingReport:
    """用真实 TextChunker 切分合成文档，统计吞吐与块长度。"""
    chunker = TextChunker(ChunkingConfig(size=size, overlap=60))
    started = time.perf_counter()
    total_chunks = 0
    total_chars = 0
    for index in range(documents):
        text = _synthetic_document(index, sections, section_chars)
        chunks = chunker.split(text, doc_id=f"doc-{index}")
        total_chunks += len(chunks)
        total_chars += sum(len(chunk.content) for chunk in chunks)
    elapsed = time.perf_counter() - started
    return ChunkingReport(
        documents=documents,
        chunks=total_chunks,
        total_chars=total_chars,
        elapsed_s=round(elapsed, 3),
        docs_per_s=round(documents / elapsed, 2) if elapsed > 0 else 0.0,
        chunks_per_s=round(total_chunks / elapsed, 2) if elapsed > 0 else 0.0,
        avg_chunk_chars=round(total_chars / total_chunks, 1) if total_chunks else 0.0,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 管线百万级 mock 压测")
    parser.add_argument("--mode", choices=["fusion", "chunking", "both"], default="both")
    parser.add_argument("--chunks", type=int, default=1_000_000, help="融合压测的 chunk 池规模")
    parser.add_argument("--queries", type=int, default=200, help="融合压测查询次数")
    parser.add_argument("--concurrency", type=int, default=8, help="并发查询线程数")
    parser.add_argument("--depth", type=int, default=40, help="单路候选数量")
    parser.add_argument("--k", type=int, default=60, help="RRF 平滑参数 k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docs", type=int, default=200, help="切块压测文档数")
    parser.add_argument("--size", type=int, default=512, help="切块大小")
    parser.add_argument("--sections", type=int, default=4, help="每篇文档小节数")
    parser.add_argument("--section-chars", type=int, default=600, help="每小节字符数")
    parser.add_argument("--json", type=str, default="", help="把结果写入 JSON 文件")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report: dict[str, object] = {}
    if args.mode in {"fusion", "both"}:
        fusion = benchmark_fusion(
            pool_size=args.chunks,
            queries=args.queries,
            concurrency=args.concurrency,
            depth=args.depth,
            k=args.k,
            seed=args.seed,
        )
        report["fusion"] = asdict(fusion)
        lat = fusion.latency
        print(
            f"[fusion] pool={fusion.pool_size} queries={fusion.queries} "
            f"concurrency={fusion.concurrency} depth={fusion.depth}"
        )
        print(
            f"  latency mean={lat.mean_ms}ms p50={lat.p50_ms}ms "
            f"p95={lat.p95_ms}ms p99={lat.p99_ms}ms max={lat.max_ms}ms"
        )
        print(f"  throughput={fusion.throughput_qps} qps")

    if args.mode in {"chunking", "both"}:
        chunking = benchmark_chunking(
            documents=args.docs,
            size=args.size,
            sections=args.sections,
            section_chars=args.section_chars,
        )
        report["chunking"] = asdict(chunking)
        print(
            f"[chunking] docs={chunking.documents} chunks={chunking.chunks} "
            f"elapsed={chunking.elapsed_s}s"
        )
        print(
            f"  throughput={chunking.docs_per_s} docs/s {chunking.chunks_per_s} chunks/s "
            f"avg_chunk_chars={chunking.avg_chunk_chars}"
        )

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
