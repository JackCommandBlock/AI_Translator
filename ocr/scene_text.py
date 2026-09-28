"""全帧多文本 OCR 与内容类型分类。

位置信息不可信，因此不再裁剪字幕区域，而是把整帧缩到长边 1280 后识别画面中的
【所有】文字块。每条文字返回 text 与内容类型 type；region_hint 仅作弱提示，
不参与任何匹配或类型决策。

本地 OCR 兜底路径：rapidocr/easyocr 本身返回文本框与坐标，可直接做全帧检测；
检测到文本后，再用纯文本 LLM 做类型分类，避免把整图发给大模型以节省 token。
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from translate.client import chat_completion

from .style_features import extract_visual_style


logger = logging.getLogger("pipeline.ocr.scene_text")

VALID_TYPES = ("dialogue", "title", "note", "sign", "lyrics", "unknown")

_TYPE_ALIASES = {
    "dialog": "dialogue",
    "dialogues": "dialogue",
    "subtitle": "dialogue",
    "speech": "dialogue",
    "caption": "dialogue",
    "person": "title",
    "name": "title",
    "place": "title",
    "annotation": "note",
    "parenthetical": "note",
    "poster": "sign",
    "screen": "sign",
    "board": "sign",
    "song": "lyrics",
    "singing": "lyrics",
}


@dataclass
class TextBlock:
    text: str
    type: str = "unknown"      # dialogue|title|note|sign|lyrics|unknown
    region_hint: str = ""      # 仅作弱提示，不参与决策
    confidence: float | None = None
    box: object | None = None  # 绝对像素四点 [[x,y],...]，或 None
    style: object | None = None  # VisualStyle

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
            "region_hint": self.region_hint,
            "confidence": self.confidence,
            "box": self.box,
            "style": self.style.to_dict() if self.style else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextBlock":
        from .style_features import VisualStyle
        return cls(
            text=str(d.get("text") or ""),
            type=normalize_type(d.get("type")),
            region_hint=str(d.get("region_hint") or d.get("region") or ""),
            confidence=_to_confidence(d.get("confidence")),
            box=d.get("box"),
            style=VisualStyle.from_dict(d.get("style")),
        )


@dataclass
class FrameOCR:
    time_ms: int
    blocks: list[TextBlock] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"time_ms": self.time_ms, "blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: dict) -> "FrameOCR":
        return cls(
            time_ms=int(d.get("time_ms") or 0),
            blocks=[TextBlock.from_dict(b) for b in (d.get("blocks") or [])],
        )


def normalize_type(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in VALID_TYPES:
        return v
    return _TYPE_ALIASES.get(v, "unknown")


def _to_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


def resize_long_edge(image: np.ndarray, long_edge: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _to_png_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


_JSON_ESCAPE_CHARS = set('"\\/bfnrtu')


def _repair_json_escapes(text: str) -> str:
    """修复模型原样写出的非法 JSON 转义（典型如 \\N）。

    视觉模型常把多行分隔符 \"\\N\" 写进 JSON 字符串：有的写成单个反斜杠 \"\\N\"
    （非法），有的正确地写成 \"\\\\N\"。旧实现按“单个反斜杠 + 非转义字符”逐处补
    反斜杠，会把已经正确的 \"\\\\N\" 误伤成 \"\\\\\\N\" 导致解析失败。

    这里改成按“连续反斜杠段 + 后继字符”判断：后继是合法转义字符时整段原样保留；
    否则保证反斜杠成对，只给落单的反斜杠补一个，从而把后继字符当作普通字符。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and text[j] == "\\":
            j += 1
        run = j - i
        nxt = text[j] if j < n else ""
        if nxt in _JSON_ESCAPE_CHARS:
            # 后继是合法转义字符，整段（含后继字符）原样保留
            out.append(text[i : j + 1])
        else:
            # 后继不是合法转义字符：奇数个反斜杠会多出一个落单反斜杠，补一个成对
            out.append("\\" * (run + 1) if run % 2 == 1 else "\\" * run)
            out.append(nxt)
        i = j + 1
    return "".join(out)


def _parse_json(text: str):
    raw = (text or "").strip()
    repaired = _repair_json_escapes(raw)
    cleaned = re.sub(r"^```(?:json)?\s*", "", repaired)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 仅在解析失败时尝试轻量修复：去掉尾随逗号，以及从文本中抽取最外层数组/对象。
    candidates = [_strip_trailing_commas(cleaned)]
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        candidates.append(_strip_trailing_commas(m.group(0)))
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        candidates.append(_strip_trailing_commas(m.group(0)))
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法解析为 JSON: {raw[:200]!r}")


def _strip_trailing_commas(text: str) -> str:
    """去掉 ] 或 } 前的尾随逗号；合法 JSON 不会命中，仅用于容错。"""
    return re.sub(r",\s*([\]}])", r"\1", text)


SCENE_TEXT_PROMPT = (
    "识别这张视频画面中的【所有】文字，不要只识别底部字幕。"
    "每条独立文字单独输出一条。对每条文字判断类型："
    "dialogue(对白/旁白)、title(标题/人名/地名)、note(括号或补充说明)、"
    "sign(画面内招牌/海报/屏幕等场景文字)、lyrics(歌词)。"
    "box 为该文字的归一化包围盒（0~1，左上右下 x1,y1,x2,y2）。"
    "只输出一个合法的 JSON 数组，不要输出任何解释、注释或尾随逗号。"
    '格式示例：[{"i":0,"text":"你好","type":"dialogue","box":[0,0,1,0.2]}]。'
    " 多行文字用 \\N 分隔。"
)


def parse_blocks(payload) -> list[TextBlock]:
    data = payload
    if isinstance(data, str):
        data = _parse_json(data)
    if isinstance(data, dict):
        data = data.get("items") or data.get("results") or data.get("blocks") or [data]
    if not isinstance(data, list):
        return []
    blocks: list[TextBlock] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            TextBlock(
                text=text,
                type=normalize_type(item.get("type")),
                region_hint=str(item.get("region") or item.get("region_hint") or ""),
                confidence=_to_confidence(item.get("confidence")),
                box=_parse_box(item.get("box")),
            )
        )
    return blocks


def _parse_box(value):
    """把 LLM 返回的 [x1,y1,x2,y2]（0~1）解析为普通浮点列表，非法时返回 None。"""
    if not value:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(box) != 4:
        return None
    return box


def _norm_box_to_abs(box, w, h):
    """[x1,y1,x2,y2] 归一化盒 -> 绝对像素四点 [[x,y],...]。"""
    if not box:
        return None
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1 * w))))
    x2 = max(0, min(w - 1, int(round(x2 * w))))
    y1 = max(0, min(h - 1, int(round(y1 * h))))
    y2 = max(0, min(h - 1, int(round(y2 * h))))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class SceneTextOCR:
    """全帧多文本 OCR：把整帧交给视觉模型，返回画面中所有文字块及其类型。"""

    def __init__(self, client, model: str, source_lang: str = "ja", temperature: float = 0.2):
        self.client = client
        self.model = model
        self.source_lang = source_lang
        self.temperature = temperature

    def recognize(self, frame: np.ndarray) -> list[TextBlock]:
        img = resize_long_edge(frame, 1280)
        b64 = _to_png_b64(img)
        prompt = SCENE_TEXT_PROMPT
        if self.source_lang:
            prompt += f"\n视频语言：{self.source_lang}。"
        resp = chat_completion(
            self.client,
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        blocks = parse_blocks(resp.choices[0].message.content or "")
        h, w = img.shape[:2]
        for b in blocks:
            b.box = _norm_box_to_abs(b.box, w, h)     # [x1,y1,x2,y2] -> 4 点
            b.style = extract_visual_style(img, b.box, w, h)
        return blocks


CONTENT_TYPE_PROMPT = (
    "判断下面每条字幕文本的内容类型。类型可选："
    "dialogue(对白/旁白)、title(标题/人名/地名)、note(括号或补充说明)、"
    "sign(画面内招牌/海报/屏幕文字)、lyrics(歌词)、unknown(无法判断)。"
    '只输出 JSON 数组 [{"i":0,"type":"dialogue"}]，不要解释。'
)


class ContentTypeClassifier:
    """纯文本 LLM 类型分类，作为视觉模型判定类型的兜底。"""

    def __init__(self, client, model: str, temperature: float = 0.2):
        self.client = client
        self.model = model
        self.temperature = temperature

    def classify(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        items = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        user = CONTENT_TYPE_PROMPT + "\n\n" + items
        resp = chat_completion(
            self.client,
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "你是字幕内容分类助手，只输出要求的 JSON。"},
                {"role": "user", "content": user},
            ],
        )
        data = _parse_json(resp.choices[0].message.content or "")
        return _types_from_payload(data, len(texts))


def _types_from_payload(data, n: int) -> list[str]:
    types: dict[int, str] = {}
    if isinstance(data, dict):
        if any(k in data for k in ("i", "index", "type")):
            data = [data]
        else:
            for k, v in data.items():
                try:
                    i = int(k)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n:
                    types[i] = normalize_type(v)
            return [types.get(i, "unknown") for i in range(n)]
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("i", item.get("index", -1)))
            except (TypeError, ValueError):
                continue
            if 0 <= i < n:
                types[i] = normalize_type(item.get("type"))
    return [types.get(i, "unknown") for i in range(n)]


def _region_hint(box, frame: np.ndarray) -> str:
    if not box:
        return ""
    h, w = frame.shape[:2]
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
    except (TypeError, ValueError, IndexError):
        return ""
    cx = (min(xs) + max(xs)) / 2.0 / max(1, w)
    cy = (min(ys) + max(ys)) / 2.0 / max(1, h)
    v = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
    hz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
    if v == "middle" and hz == "center":
        return "center"
    return f"{v}-{hz}"


class LocalSceneTextOCR:
    """本地 OCR 全帧检测：直接取每个文本框，再用纯文本 LLM 做类型分类。"""

    def __init__(self, backend, classifier: ContentTypeClassifier | None = None):
        self.backend = backend
        self.classifier = classifier

    def recognize(self, frame: np.ndarray) -> list[TextBlock]:
        detected = self.backend.recognize_blocks(frame)
        blocks: list[TextBlock] = []
        h, w = frame.shape[:2]
        for d in detected:
            text = (d.text or "").strip()
            if not text:
                continue
            vis = extract_visual_style(frame, d.box, w, h)
            blocks.append(
                TextBlock(
                    text=text,
                    type="unknown",
                    region_hint=_region_hint(d.box, frame),
                    confidence=d.confidence,
                    box=d.box,
                    style=vis,
                )
            )
        if self.classifier and blocks:
            try:
                types = self.classifier.classify([b.text for b in blocks])
                for b, t in zip(blocks, types):
                    b.type = t
            except Exception:
                pass
        return blocks

