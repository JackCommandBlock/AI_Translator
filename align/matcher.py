"""把 OCR / ASR 结果对齐到 ASS 事件时间轴。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ASRMatch:
    text: str
    confidence: float
    source: str = "asr"


def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = overlap_ms(a_start, a_end, b_start, b_end)
    union = max(1, (a_end - a_start) + (b_end - b_start) - inter)
    return inter / union


def _word_overlaps(word_start: int, word_end: int, ev_start: int, ev_end: int) -> bool:
    return overlap_ms(word_start, word_end, ev_start, ev_end) > 0


def _join_words(words: list[str], source_lang: str) -> str:
    text = "".join(words)
    if source_lang.split("-")[0] in ("ja", "zh", "ko"):
        return text
    # 对空格敏感语言，按词之间加空格（简化的词边界恢复）
    return " ".join(w for w in words)


def align_asr_to_events(events, segments, source_lang: str = "ja") -> dict[int, ASRMatch]:
    """将整段 ASR 结果按时间分配到每个 ASS 事件。"""
    matches: dict[int, ASRMatch] = {}
    if not segments:
        return matches

    has_words = any(seg.words for seg in segments)

    for ev in events:
        ev_words: list[str] = []
        best_conf = 0.0
        overlap_segments = []
        for seg in segments:
            ov = overlap_ms(ev.start_ms, ev.end_ms, seg.start_ms, seg.end_ms)
            if ov <= 0:
                continue
            overlap_segments.append((ov, seg))

        if has_words:
            for _, seg in overlap_segments:
                for w in seg.words:
                    if _word_overlaps(w.start_ms, w.end_ms, ev.start_ms, ev.end_ms):
                        ev_words.append(w.word)
                best_conf = max(best_conf, seg.confidence)
            if ev_words:
                text = _join_words(ev_words, source_lang)
                matches[ev.id] = ASRMatch(text=text, confidence=best_conf)
            continue

        if not overlap_segments:
            continue
        # 无词级时间戳：选择重叠最大的段
        overlap_segments.sort(key=lambda x: x[0], reverse=True)
        _, best = overlap_segments[0]
        matches[ev.id] = ASRMatch(text=best.text, confidence=best.confidence)

    return matches

