"""Chunk 参数标定：在评测语料上扫描 size/overlap，用证据完整率与成本选参。

为什么要单独标定：
- chunk size 太小 → 证据被切断（边界截断率高），相邻块依赖变强；
- chunk size 太大 → 向量数量下降但单块噪声变多，上下文 token 与精排成本上升；
- overlap 增大能缓解截断，但会造成正文重复、向量与 embedding token 膨胀。
这里用"证据完整率（golden 证据句是否完整落在某一块内）+ 成本"做量化取舍，
结论可复现，且不依赖向量库/LLM（纯本地计算）。

用法：
    .venv\\Scripts\\python scripts\\eval_chunking.py
    .venv\\Scripts\\python scripts\\eval_chunking.py --sizes 128,256,512,1024 --overlaps 0,32,60,128
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eval_retrieval import DEFAULT_CORPUS, DEFAULT_GOLDEN, load_jsonl  # noqa: E402

from app.domain.kb_config import ChunkingConfig  # noqa: E402
from app.evaluation.metrics import estimate_tokens  # noqa: E402
from app.ingestion.chunker import TextChunker  # noqa: E402


def evaluate_combo(
    corpus: list[dict],
    golden: list[dict],
    *,
    size: int,
    overlap: int,
    vector_dim: int,
) -> dict[str, float]:
    """对一组 (size, overlap) 计算证据完整率与成本指标。"""
    chunker = TextChunker(ChunkingConfig(size=size, overlap=overlap))
    chunk_count = 0
    total_chars = 0
    raw_chars = 0
    fingerprints: set[str] = set()
    intact = 0
    evidence_total = 0
    embedding_tokens = 0
    max_len = 0

    for doc in corpus:
        raw = str(doc["content"])
        raw_chars += len(raw)
        chunks = chunker.split(raw, doc_id=str(doc["title"]))
        contents = [chunk.content for chunk in chunks]
        chunk_count += len(chunks)
        total_chars += sum(len(content) for content in contents)
        max_len = max(max_len, *(len(content) for content in contents)) if contents else max_len
        embedding_tokens += sum(estimate_tokens(content) for content in contents)
        fingerprints.update(contents)

        # 期望证据只在该文档的块里找：跨文档命中没有意义
        expected = [
            row
            for row in golden
            if str(doc["title"]) in (row.get("expect_titles") or [])
        ]
        for row in expected:
            for evidence in row.get("expect_evidence") or []:
                evidence_total += 1
                intact += int(any(evidence in content for content in contents))

    return {
        "chunk_count": float(chunk_count),
        "avg_chars": total_chars / chunk_count if chunk_count else 0.0,
        "max_chars": float(max_len),
        "duplicate_ratio": 1.0 - len(fingerprints) / chunk_count if chunk_count else 0.0,
        # 正文膨胀率：overlap 造成的重复字符占比
        "inflation": total_chars / raw_chars if raw_chars else 0.0,
        "evidence_intact_rate": intact / evidence_total if evidence_total else 0.0,
        "embedding_tokens": float(embedding_tokens),
        "vector_mb": chunk_count * vector_dim * 4 / (1024 * 1024),
    }


def _recommend(rows: list[dict[str, float]]) -> dict[str, float]:
    """选参规则：先保住证据完整率（允许比最优低 1 个百分点），再取 embedding token 最少。"""
    best_intact = max(row["evidence_intact_rate"] for row in rows)
    candidates = [row for row in rows if row["evidence_intact_rate"] >= best_intact - 0.01]
    return min(candidates, key=lambda row: (row["embedding_tokens"], row["chunk_count"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk size / overlap 参数标定")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--sizes", default="128,256,512,1024", help="逗号分隔")
    parser.add_argument("--overlaps", default="0,32,60,128", help="逗号分隔")
    parser.add_argument("--vector-dim", type=int, default=1024)
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    overlaps = [int(value) for value in args.overlaps.split(",") if value.strip()]
    corpus = load_jsonl(args.corpus)
    golden = load_jsonl(args.golden)

    rows: list[dict[str, float]] = []
    print(
        f"\n{'size':>6} {'ovlp':>6} {'chunks':>7} {'avg':>6} {'max':>6} "
        f"{'dup%':>6} {'inflate':>8} {'intact':>7} {'emb_tok':>9} {'vec_MB':>8}"
    )
    for size, overlap in itertools.product(sizes, overlaps):
        if overlap >= size:
            continue
        metrics = evaluate_combo(
            corpus, golden, size=size, overlap=overlap, vector_dim=args.vector_dim
        )
        row = {"size": float(size), "overlap": float(overlap), **metrics}
        rows.append(row)
        print(
            f"{size:>6} {overlap:>6} {metrics['chunk_count']:>7.0f} "
            f"{metrics['avg_chars']:>6.0f} {metrics['max_chars']:>6.0f} "
            f"{metrics['duplicate_ratio'] * 100:>6.1f} {metrics['inflation']:>8.2f} "
            f"{metrics['evidence_intact_rate']:>7.3f} {metrics['embedding_tokens']:>9.0f} "
            f"{metrics['vector_mb']:>8.2f}"
        )

    if rows:
        best = _recommend(rows)
        print(
            "\n推荐参数："
            f"size={best['size']:.0f} overlap={best['overlap']:.0f} "
            f"（证据完整率={best['evidence_intact_rate']:.3f}，"
            f"embedding token={best['embedding_tokens']:.0f}，"
            f"chunk 数={best['chunk_count']:.0f}）"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
