"""解析 [V4+ Styles] 段，把每个事件的 Style 映射为可比较的视觉预期。

ASS 文件里的 Style 定义（主色、描边色、粗体、对齐、字号等）是「事件侧样式指纹」，
配合画面侧 OCR 提取的实测外观一起做样式相似度，能在时间完全重叠时把字幕正确对应
到各自的事件。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_COLOR_RE = re.compile(
    r"&H([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})"
)


def _f(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _b(value) -> bool:
    v = str(value or "").strip().lower()
    return v in ("1", "-1", "true", "yes", "on")


def parse_ass_color(value: str) -> tuple[int, int, int]:
    # &HAABBGGRR -> (R, G, B)，AA 为透明度，这里忽略
    m = _COLOR_RE.match((value or "").strip())
    if not m:
        return (255, 255, 255)
    a, b, g, r = (int(x, 16) for x in m.groups())
    return (r, g, b)


@dataclass
class AssStyle:
    name: str
    fontname: str = ""
    fontsize: float = 20.0
    primary: tuple[int, int, int] = (255, 255, 255)
    secondary: tuple[int, int, int] = (255, 0, 0)
    outline: tuple[int, int, int] = (0, 0, 0)
    back: tuple[int, int, int] = (0, 0, 0)
    bold: bool = False
    italic: bool = False
    alignment: int = 2      # 1-9，仅作弱信号
    outline_w: float = 2.0
    shadow: float = 0.0
    margin_l: int = 10
    margin_r: int = 10
    margin_v: int = 10


def parse_styles(ass) -> dict[str, AssStyle]:
    styles: dict[str, AssStyle] = {}
    in_styles = False
    fmt: dict[str, int] = {}
    for line in ass.header_lines:          # header_lines 已含 [V4+ Styles] 段
        s = line.strip()
        low = s.lower()
        if low.startswith("[v4"):
            in_styles = True
            fmt.clear()
            continue
        if not in_styles:
            continue
        if low.startswith("["):
            break
        if low.startswith("format:"):
            cols = [c.strip() for c in s.split(":", 1)[1].split(",")]
            fmt = {c.lower(): i for i, c in enumerate(cols)}
            continue
        if low.startswith("style:"):
            vals = [c.strip() for c in s.split(":", 1)[1].split(",")]

            def get(name, default=""):
                i = fmt.get(name.lower())
                return vals[i] if i is not None and i < len(vals) else default

            name = get("Name")
            if not name:
                continue
            styles[name] = AssStyle(
                name=name,
                fontname=get("Fontname"),
                fontsize=_f(get("Fontsize"), 20.0),
                primary=parse_ass_color(get("PrimaryColour")),
                secondary=parse_ass_color(get("SecondaryColour")),
                outline=parse_ass_color(get("OutlineColour")),
                back=parse_ass_color(get("BackColour")),
                bold=_b(get("Bold")),
                italic=_b(get("Italic")),
                alignment=_i(get("Alignment"), 2),
                outline_w=_f(get("Outline"), 2.0),
                shadow=_f(get("Shadow"), 0.0),
                margin_l=_i(get("MarginL"), 10),
                margin_r=_i(get("MarginR"), 10),
                margin_v=_i(get("MarginV"), 10),
            )
    return styles


def style_for_events(ass, events) -> dict[int, AssStyle]:
    styles = parse_styles(ass)
    return {ev.id: styles[ev.style] for ev in events if ev.style in styles}
