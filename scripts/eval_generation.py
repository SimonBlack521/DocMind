"""生成质量离线评测：答案忠实度（faithfulness）+ 引用覆盖率 + 期望文档命中率。

设计：
    - 复用 eval_retrieval 的 golden 集与语料重灌逻辑（expect_titles 作为期望文档）；
    - 每题走完整 RAG 管线（检索 + LLM 生成），记录答案、引用与耗时；
    - 默认再调一次 LLM 当裁判：逐条断言核对参考片段，输出 0-1 支持度；
    - --no-judge 只跑生成不评分，适合快速冒烟或没有第二个模型额度时使用。

用法：
    .venv\\Scripts\\python scripts\\eval_generation.py --kb finance
    .venv\\Scripts\\python scripts\\eval_generation.py --kb finance --no-seed --no-judge
    .venv\\Scripts\\python scripts\\eval_generation.py --kb finance --limit 5 --rewrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 同目录脚本复用：语料重灌与 JSONL 读取保持单一实现
from eval_retrieval import DEFAULT_CORPUS, DEFAULT_GOLDEN, load_jsonl, seed_corpus  # noqa: E402

from app.evaluation.metrics import refusal_quality, summarize_refusal  # noqa: E402

JUDGE_SYSTEM_PROMPT = """你是严格的 RAG 答案审核员。任务：把【模型回答】拆成事实性断言，
逐条判断能否在【参考片段】中找到直接依据。
只输出 JSON，不要输出解释：{"score": <0-1 之间的小数>, "unsupported": [<未获支持的断言>]}
约束：
- unsupported 里每条必须原样摘自【模型回答】，不得改写、不得编造回答里没出现的数字或结论；
- 片段给出相同事实（数字、主体、条件一致）即算支持，不要求措辞一致；
- 纯拒答或没有事实性断言时 score 记 1.0。"""

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:万|亿|%|元|天|日|年|月|小时|分钟|个工作日|工作日)?")
_REFUSAL_MARKERS = ("未找到相关信息", "未检索到", "无法回答", "没有相关", "暂时无法")
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _is_refusal(answer: str) -> bool:
    """模型是否给出"知识库未覆盖"的拒答（规则判定，不依赖裁判模型）。"""
    return any(marker in answer for marker in _REFUSAL_MARKERS)


def _citation_accuracy(answer: str, titles: list[str], expected: set[str]) -> tuple[float, float]:
    """引用准确率与引用有效率。

    - 有效率：引用编号是否落在实际送入模型的片段范围内（防模型编造 [99]）；
    - 准确率：在被引用的片段里，有多少比例来自 golden 期望文档
      （没有期望文档标注时退化为有效率）。
    """
    cited = {int(index) for index in _CITATION_RE.findall(answer)}
    if not cited:
        return 0.0, 0.0
    valid = {index for index in cited if 1 <= index <= len(titles)}
    valid_rate = len(valid) / len(cited)
    if not expected:
        return valid_rate, valid_rate
    accurate = {index for index in valid if titles[index - 1] in expected}
    return valid_rate, len(accurate) / len(valid) if valid else 0.0


def _numeric_support(answer: str, contexts: list[str]) -> float | None:
    """无裁判时的客观代理指标：回答中的数字是否在参考片段里出现过。

    3B 级小模型当裁判容易误判，因此额外给出不依赖 LLM 的数值一致性，便于交叉核对。
    """
    numbers = _NUMBER_RE.findall(answer)
    if not numbers:
        return None
    haystack = re.sub(r"\s+", "", "".join(contexts))
    supported = sum(1 for number in numbers if number in haystack)
    return supported / len(numbers)


def _parse_judge_json(text: str) -> dict:
    """容忍模型在 JSON 前后加 markdown 代码块或说明文字。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"score": None, "unsupported": [], "raw": text}
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return {"score": None, "unsupported": [], "raw": text}
    score = parsed.get("score")
    if not isinstance(score, (int, float)):
        return {"score": None, "unsupported": [], "raw": text}
    return {
        "score": max(0.0, min(1.0, float(score))),
        "unsupported": parsed.get("unsupported") or [],
    }


def _judge(llm: Any, question: str, answer: str, contexts: list[str]) -> dict:
    from app.providers.llm import ChatMessage

    user_prompt = (
        f"【问题】\n{question}\n\n"
        f"【模型回答】\n{answer}\n\n"
        f"【参考片段】\n" + "\n---\n".join(contexts)
    )
    response = llm.chat(
        [
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return _parse_judge_json(response.text)


def run(
    kb_code: str,
    golden_path: Path,
    *,
    top_n: int,
    limit: int | None,
    use_rewrite: bool,
    use_rerank: bool,
    judge: bool,
    show_answers: bool = False,
) -> int:
    from app.core.settings import get_settings
    from app.domain.kb_config import KBRegistry
    from app.orchestration.pipeline import RAGPipeline
    from app.orchestration.registry import RuntimeRegistry

    settings = get_settings()
    kb = KBRegistry.from_directory(settings.kb_config_dir).resolve(kb_code)
    registry = RuntimeRegistry(settings)
    try:
        pipeline = RAGPipeline(settings=settings, registry=registry)
        golden = load_jsonl(golden_path)
        if limit is not None:
            golden = golden[:limit]

        rows: list[dict] = []
        print(
            f"\n逐题明细（top_n={top_n} rewrite={use_rewrite} "
            f"rerank={use_rerank} judge={judge}）："
        )
        for idx, item in enumerate(golden, 1):
            question = item["question"]
            expect = set(item.get("expect_titles") or [])
            expected_no_answer = bool(item.get("expect_no_answer"))

            started = time.perf_counter()
            result = pipeline.query(
                kb,
                question,
                top_k=top_n,
                use_rewrite=use_rewrite,
                use_rerank=use_rerank,
            )
            elapsed = time.perf_counter() - started

            titles = [chunk.title for chunk in result.chunks]
            hit = any(title in expect for title in titles) if expect else True
            cited = bool(_CITATION_RE.search(result.answer))
            # 送模型/裁判的都是"实际上下文"：父子切片时优先父块正文
            contexts = [
                str(chunk.metadata.get("parent_content") or chunk.content)
                for chunk in result.chunks
            ]
            numeric = _numeric_support(result.answer, contexts)
            citation_valid, citation_accurate = _citation_accuracy(
                result.answer, titles, expect
            )
            refusal_label = refusal_quality(
                predicted_refusal=_is_refusal(result.answer),
                expected_no_answer=expected_no_answer,
            )

            verdict = {"score": None, "unsupported": []}
            if judge:
                verdict = _judge(registry.llm(), question, result.answer, contexts)

            rows.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "hit": hit,
                    "cited": cited,
                    "citation_valid": citation_valid,
                    "citation_accurate": citation_accurate,
                    "refusal": refusal_label,
                    "expected_no_answer": expected_no_answer,
                    "score": verdict["score"],
                    "numeric": numeric,
                    "unsupported": verdict["unsupported"],
                    "seconds": round(elapsed, 2),
                }
            )
            score_text = "n/a" if verdict["score"] is None else f"{verdict['score']:.2f}"
            numeric_text = "n/a" if numeric is None else f"{numeric:.2f}"
            print(
                f"  [{idx:02d}] 命中={'Y' if hit else 'N'} 引用={'Y' if cited else 'N'} "
                f"拒答={refusal_label} 忠实度={score_text} 数值一致={numeric_text} "
                f"{elapsed:.1f}s {question}"
            )
            if show_answers:
                print(f"       回答：{result.answer}")
            if verdict["unsupported"]:
                for claim in verdict["unsupported"]:
                    print(f"       未支持：{claim}")

        total = len(rows)
        if not total:
            print("\n[error] golden 集为空，无法评测")
            return 1

        hit_rate = sum(1 for row in rows if row["hit"]) / total
        cited_rate = sum(1 for row in rows if row["cited"]) / total
        citation_valid = sum(row["citation_valid"] for row in rows) / total
        citation_accurate = sum(row["citation_accurate"] for row in rows) / total
        refusal_summary = summarize_refusal([row["refusal"] for row in rows])
        numeric_rows = [row["numeric"] for row in rows if row["numeric"] is not None]
        print("\n汇总：")
        print(f"  题数            = {total}")
        print(f"  期望文档命中率  = {hit_rate:.3f}")
        print(f"  引用覆盖率      = {cited_rate:.3f}")
        print(f"  引用编号有效率  = {citation_valid:.3f}")
        print(f"  引用准确率      = {citation_accurate:.3f}")
        print(
            "  拒答质量        = "
            f"correct_answer={refusal_summary['correct_answer']} "
            f"correct_refusal={refusal_summary['correct_refusal']} "
            f"false_refusal={refusal_summary['false_refusal']} "
            f"hallucination={refusal_summary['hallucination']}"
        )
        print(f"  裁判模型        = {registry.llm().model}")
        if judge:
            scored = [row["score"] for row in rows if row["score"] is not None]
            if scored:
                mean = sum(scored) / len(scored)
                print(f"  答案忠实度均值  = {mean:.3f}（有效评分 {len(scored)}/{total}）")
                print(f"  完全忠实题目数  = {sum(1 for s in scored if s >= 0.99)}/{len(scored)}")
                if mean < 0.7:
                    print("  提示：裁判为小模型时低分常见误判，请结合数值一致性与人工抽检复核")
            else:
                print("  答案忠实度      = 无法解析裁判输出，请检查模型是否返回 JSON")
        if numeric_rows:
            numeric_mean = sum(numeric_rows) / len(numeric_rows)
            print(
                f"  数值一致均值    = {numeric_mean:.3f}"
                f"（{len(numeric_rows)}/{total} 题含数字）"
            )
        print(f"  平均生成耗时    = {sum(row['seconds'] for row in rows) / total:.1f}s")
        return 0
    finally:
        registry.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成质量离线评测（忠实度 / 引用 / 命中率）")
    parser.add_argument("--kb", default="finance", help="知识库 kb_code，默认 finance")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="golden JSONL 路径")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="评测语料 JSONL 路径")
    parser.add_argument("--no-seed", action="store_true", help="跳过语料重灌（已入库时使用）")
    parser.add_argument("--top-n", type=int, default=6, help="送入生成的片段数，默认 6")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题，便于快速验证")
    parser.add_argument("--rewrite", action="store_true", help="开启查询改写（会调用 LLM）")
    parser.add_argument("--rerank", action="store_true", help="开启精排（按 RERANK_BACKEND）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判，只输出生成结果")
    parser.add_argument("--show-answers", action="store_true", help="逐题打印完整回答")
    args = parser.parse_args()

    if not args.no_seed:
        count = seed_corpus(args.kb, args.corpus)
        print(f"[seed] 语料重灌完成，共 {count} 篇")

    return run(
        args.kb,
        args.golden,
        top_n=args.top_n,
        limit=args.limit,
        use_rewrite=args.rewrite,
        use_rerank=args.rerank,
        judge=not args.no_judge,
        show_answers=args.show_answers,
    )


if __name__ == "__main__":
    sys.exit(main())
