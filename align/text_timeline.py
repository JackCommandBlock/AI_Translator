"""文本时间线与事件匹配。

把每个采样时刻识别到的文本按内容聚成文本轨道（TextTrack），记录其出现的时间点
（存在区间）；再用时间覆盖 + ASR 相似度 + 样式相似度把文本轨道一对一分配给各个
ASS 事件。样式是时间完全重叠时的主要判别信号，位置/对齐只作弱信号。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from ocr.style_features import VisualStyle

VALID_TYPES = ("dialogue", "title", "note", "sign", "lyrics", "unknown")


@dataclass
class TextTrack:
    key: int
    text: str
    type: str
    presence: list[int] = field(default_factory=list)  # 该文本出现的采样时刻（ms）
    confidence: float | None = None
    style: object | None = None  # VisualStyle

    @property
    def span(self) -> tuple[int, int]:
        return (min(self.presence), max(self.presence))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "text": self.text,
            "type": self.type,
            "presence": list(self.presence),
            "confidence": self.confidence,
            "style": self.style.to_dict() if self.style else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextTrack":
        return cls(
            key=int(d.get("key")),
            text=str(d.get("text") or ""),
            type=str(d.get("type") or "unknown"),
            presence=[int(x) for x in (d.get("presence") or [])],
            confidence=_to_conf(d.get("confidence")),
            style=VisualStyle.from_dict(d.get("style")),
        )


def _to_conf(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(text: str) -> str:
    return "".join(str(text).split()).lower()


def text_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _touches(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """两个区间是否相交（含端点，单点样本也算命中）。"""
    return a_start <= b_end and a_end >= b_start


def build_text_timeline(
    frame_ocrs,
    gap_ms: int = 1500,
    similarity_threshold: float = 0.8,
) -> list[TextTrack]:
    """把逐帧 OCR 结果按内容聚成文本轨道。

    - 按归一化文本聚类；
    - 同一文本若出现时间间隔超过 gap_ms，则拆成多条轨道；
    - 相近文本（编辑距离很小）做模糊合并。
    """
    observations: list[tuple[int, str, str, float | None, object]] = []
    for fo in frame_ocrs:
        for b in fo.blocks:
            text = (b.text or "").strip()
            if not text:
                continue
            observations.append(
                (
                    int(fo.time_ms),
                    text,
                    getattr(b, "type", "unknown") or "unknown",
                    getattr(b, "confidence", None),
                    getattr(b, "style", None),
                )
            )
    observations.sort(key=lambda o: o[0])

    buckets: list[dict] = []
    for time_ms, text, typ, conf, style in observations:
        best: dict | None = None
        best_sim = 0.0
        for tr in buckets:
            if time_ms - tr["last_time"] > gap_ms:
                continue
            sim = max(text_similarity(text, t) for t in tr["texts"])
            if sim > best_sim:
                best_sim = sim
                best = tr
        if best is not None and best_sim >= similarity_threshold:
            best["texts"].append(text)
            best["types"].append(typ)
            best["presence"].append(time_ms)
            if conf is not None:
                best["confs"].append(conf)
            if style is not None:
                best["styles"].append(style)
            best["last_time"] = time_ms
        else:
            buckets.append(
                {
                    "texts": [text],
                    "types": [typ],
                    "presence": [time_ms],
                    "confs": [conf] if conf is not None else [],
                    "styles": [style] if style is not None else [],
                    "last_time": time_ms,
                }
            )

    tracks: list[TextTrack] = []
    for i, tr in enumerate(buckets):
        tracks.append(
            TextTrack(
                key=i,
                text=_representative_text(tr["texts"]),
                type=_representative_type(tr["types"]),
                presence=sorted(tr["presence"]),
                confidence=(sum(tr["confs"]) / len(tr["confs"])) if tr["confs"] else None,
                style=_representative_style(tr["styles"]),
            )
        )
    return tracks


def _representative_text(texts: list[str]) -> str:
    target = Counter(normalize_text(t) for t in texts).most_common(1)[0][0]
    for t in texts:
        if normalize_text(t) == target:
            return t
    return texts[0]


def _representative_type(types: list[str]) -> str:
    known = [t for t in types if t and t != "unknown"]
    if known:
        return Counter(known).most_common(1)[0][0]
    return "unknown"


def _representative_style(styles):
    # 只有提取到文字主色才算「有样式信号」；空 VisualStyle（例如 LLM 未返回 box）
    # 不参与聚合，避免在 original.json 里留下全 None 的假样式。
    valid = [s for s in styles if s is not None and s.text_color is not None]
    if not valid:
        return None
    tc = [s.text_color for s in valid if s.text_color]
    oc = [s.outline_color for s in valid if s.outline_color]
    bd = [s.bold for s in valid if s.bold is not None]
    rg = [s.region for s in valid if s.region is not None]

    def mean_color(cs):
        if not cs:
            return None
        return tuple(int(round(v)) for v in np.mean(np.array(cs, float), axis=0))

    return VisualStyle(
        text_color=mean_color(tc),
        outline_color=mean_color(oc),
        bold=(sum(bd) / len(bd)) if bd else None,
        region=(
            (sum(r[0] for r in rg) / len(rg), sum(r[1] for r in rg) / len(rg))
            if rg
            else None
        ),
        font_height=max((s.font_height or 0) for s in valid) or None,
    )


def _asr_text(a) -> str:
    if a is None:
        return ""
    if isinstance(a, dict):
        return (a.get("text") or "").strip()
    return (getattr(a, "text", "") or "").strip()


def match_events_to_tracks(events, tracks, asr_by_event, cfg, styles=None) -> dict[int, int]:
    """用「时间覆盖 + ASR 相似度 + 样式相似度」把文本轨道一对一分配给事件。

    把逐事件贪心换成候选边打分 + 二分图最大收益分配，样式是时间完全重叠时的关键
    判别信号；位置/对齐只作弱信号。styles 为 {event_id: AssStyle}，缺省等价于关闭
    样式信号。
    """
    from ocr.style_features import style_similarity

    styles = styles or {}
    threshold = float(getattr(cfg, "asr_similarity_threshold", 0.6))
    style_w = float(getattr(cfg, "style_match_weight", 0.8))
    style_min = float(getattr(cfg, "style_similarity_min", 0.35))
    use_style = bool(getattr(cfg, "use_style_match", True))

    n, m = len(events), len(tracks)
    cost = [[0.0] * m for _ in range(n)]
    for i, ev in enumerate(events):
        window = (ev.start_ms, ev.end_ms)
        asr = _asr_text(asr_by_event.get(ev.id))
        st = styles.get(ev.id)
        for j, t in enumerate(tracks):
            if not _touches(t.span[0], t.span[1], window[0], window[1]):
                continue
            s = overlap(t.span[0], t.span[1], window[0], window[1]) / max(
                1, ev.duration_ms
            )
            if t.span[0] >= ev.start_ms and t.span[1] <= ev.end_ms:
                s += 0.5
            if asr and text_similarity(t.text, asr) >= threshold:
                s += 1.0
            if use_style and st is not None and t.style is not None:
                sim = style_similarity(t.style, st, cfg)
                if sim >= style_min:
                    s += style_w * sim
            cost[i][j] = s

    assign = _assign_max(cost)
    return {events[i].id: tracks[j].key for i, j in assign.items()}


def _assign_max(cost):
    """收益最大化的一对一分配。

    优先用 scipy 的 linear_sum_assignment 求全局最优（无新依赖时不会引入 scipy），
    不可用时退化为按得分贪心。
    """
    n, m = len(cost), len(cost[0]) if cost else 0
    if n == 0 or m == 0:
        return {}
    size = max(n, m)
    M = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            M[i][j] = -(cost[i][j] if i < n and j < m else 0.0)
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(np.array(M, float))
        return {
            int(i): int(j)
            for i, j in zip(ri, ci)
            if i < n and j < m and cost[i][j] > 1e-6
        }
    except Exception:
        edges = sorted(
            (
                (cost[i][j], i, j)
                for i in range(n)
                for j in range(m)
                if cost[i][j] > 1e-6
            ),
            reverse=True,
        )
        used_i, used_j, res = set(), set(), {}
        for _, i, j in edges:
            if i in used_i or j in used_j:
                continue
            res[i] = j
            used_i.add(i)
            used_j.add(j)
        return res


def infer_type(block_type, text, asr_text, cfg, content_classifier=None) -> str:
    """类型推断优先级：语音命中 > 视觉模型判定类型 > 纯文本分类。"""
    threshold = float(getattr(cfg, "asr_similarity_threshold", 0.6))
    if asr_text and text_similarity(text, asr_text) >= threshold:
        return "dialogue"
    if block_type and block_type != "unknown":
        return block_type
    if content_classifier is not None:
        try:
            types = content_classifier.classify([text])
            if types and types[0] and types[0] != "unknown":
                return types[0]
        except Exception:
            pass
    return "unknown"

