"""综合同一事件多帧的 OCR 结果，得出最终原文与置信度。"""
from __future__ import annotations

from collections import Counter

from .recognizer import OCRResult


def _normalize(text: str) -> str:
    return "".join(text.split()).lower()


def resolve_frames(results: list[OCRResult]) -> OCRResult | None:
    """多帧投票：优先选出现次数最多的非空结果；相同则取平均置信度更高者。"""
    valid = [r for r in results if r.text and r.text.strip()]
    if not valid:
        return None
    counter = Counter(_normalize(r.text) for r in valid)
    top = counter.most_common(1)[0][1]
    candidates = [r for r in valid if _normalize(r.text) == counter.most_common(1)[0][0]]
    # 若最高票数为 1，说明各帧不一致，取置信度最高者
    if top == 1:
        def conf_key(r: OCRResult):
            return r.confidence if r.confidence is not None else 0.5
        chosen = max(valid, key=conf_key)
    else:
        chosen = max(candidates, key=lambda r: r.confidence if r.confidence is not None else 0.5)
    conf = chosen.confidence
    if conf is None:
        conf = min(0.85, 0.5 + 0.1 * top)
    return OCRResult(text=chosen.text, confidence=conf, backend=chosen.backend)

