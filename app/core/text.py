"""中英文混合词面 Tokenization：英文/数字按词，中文按单字 + 双字 bigram。

为什么不用纯单字切分（清单第三优先级第 13/16 项）：
"企业员工报销标准"按单字拆开互不相干，词面重合度虚高却没有"词"的粒度；
双字 bigram（"报销""销标""标准"）是轻量、确定性的中文词面近似，
在 MMR 多样性惩罚与上下文压缩相关性判断里都比单字更接近真实语义重合。
保留单字是为了让单字词（如"税"）也能命中。

本模块只做纯计算，供 retrieval / generation 两层共用，保证两处口径一致。
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿]+")


def token_set(text: str) -> set[str]:
    """词面 token 集合（小写归一）：英文数字词 + 中文单字与双字 bigram。"""
    lowered = text.lower()
    tokens = set(_WORD_RE.findall(lowered))
    for run in _CJK_RE.findall(lowered):
        tokens.update(run)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens
