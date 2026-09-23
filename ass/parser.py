"""读取并解析 ASS 文件。

原则：尽量以原始文本保存，仅在 Dialogue 的 Text 部分做修改。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{2})$")


def parse_time(text: str) -> int:
    """'H:MM:SS.cc' -> 毫秒。"""
    m = TIME_RE.match(text.strip())
    if not m:
        raise ValueError(f"无法解析时间: {text!r}")
    h, mm, ss, cs = (int(x) for x in m.groups())
    return h * 3600_000 + mm * 60_000 + ss * 1000 + cs * 10


def format_time(ms: int) -> str:
    """毫秒 -> 'H:MM:SS.cc'。"""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, rem = divmod(rem, 1000)
    cs = rem // 10
    return f"{h}:{mm:02d}:{ss:02d}.{cs:02d}"


@dataclass
class DialogueEvent:
    """单条 Dialogue 事件。"""
    id: int
    start_ms: int
    end_ms: int
    layer: str
    style: str
    name: str
    margin_l: str
    margin_r: str
    margin_v: str
    effect: str
    text: str
    raw_prefix: str = ""  # "Dialogue: Layer,Start,End,...,Effect,"
    start_raw: str = ""
    end_raw: str = ""

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class AssFile:
    """ASS 文件的内存表示。"""
    header_lines: list[str] = field(default_factory=list)
    events_format_line: str = ""
    events: list[DialogueEvent] = field(default_factory=list)
    source_path: Path | None = None


def parse_ass(path: str | Path) -> AssFile:
    """读取 ASS 文件，返回结构化的 AssFile。"""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    ass = AssFile(source_path=path)
    in_events = False
    event_idx = 0

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("[events]"):
            in_events = True
            ass.header_lines.append(line)
            continue
        if in_events:
            if stripped.startswith("Format:"):
                ass.events_format_line = line
                ass.header_lines.append(line)
                continue
            if stripped.startswith("Dialogue:"):
                ev = _parse_dialogue(line, event_idx)
                ass.events.append(ev)
                event_idx += 1
                continue
            # Events 区内其它行（注释、空行等）保留在 header
            if stripped == "" or stripped.startswith(";"):
                ass.header_lines.append(line)
                continue
            # 未知行也保留，避免丢内容
            ass.header_lines.append(line)
            continue
        ass.header_lines.append(line)

    return ass


def _parse_dialogue(line: str, idx: int) -> DialogueEvent:
    """解析单行 Dialogue。

    Text 中可能含逗号，因此按前 9 个逗号切分；raw_prefix 保留原始字节级前缀，
    保证写回时除 Text 外的部分与原文件完全一致。
    """
    # 定位第 9 个逗号（即 Text 字段前的分隔符）
    comma_pos = -1
    for _ in range(9):
        comma_pos = line.find(",", comma_pos + 1)
        if comma_pos == -1:
            break
    if comma_pos == -1:
        raise ValueError(f"Dialogue 字段不足: {line!r}")
    raw_prefix = line[: comma_pos + 1]
    text = line[comma_pos + 1 :]
    head = line[:comma_pos]
    fields = [p.strip() for p in head.split(",")]
    layer, start_raw, end_raw, style, name, ml, mr, mv, effect = fields
    if layer.lower().startswith("dialogue:"):
        layer = layer[len("dialogue:") :].strip()
    return DialogueEvent(
        id=idx,
        start_ms=parse_time(start_raw),
        end_ms=parse_time(end_raw),
        layer=layer,
        style=style,
        name=name,
        margin_l=ml,
        margin_r=mr,
        margin_v=mv,
        effect=effect,
        text=text,
        raw_prefix=raw_prefix,
        start_raw=start_raw,
        end_raw=end_raw,
    )
