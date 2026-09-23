"""将翻译后的 ASS 结构写回文件，仅替换 Dialogue Text。"""
from __future__ import annotations

from pathlib import Path

from .parser import AssFile


def write_ass(ass: AssFile, output_path: str | Path, texts: dict[int, str] | None = None) -> None:
    """写回 ASS。

    texts 为 {event_id: new_text}。若为 None，则直接使用 events 中的 text。
    """
    output_path = Path(output_path)
    lines: list[str] = []
    events_started = False
    event_iter = iter(ass.events)
    current_event = next(event_iter, None)

    # header_lines 中已包含 [Events]、Format 行以及空行/注释。
    for line in ass.header_lines:
        lines.append(line)
        if line.strip().lower().startswith("[events]"):
            events_started = True

    # 若 header 里没有 Events 段（理论不会），补上
    if not events_started:
        lines.append("[Events]")
        lines.append(ass.events_format_line)

    # 逐条输出 Dialogue；header 中已经不含 Dialogue，这里统一追加。
    for ev in ass.events:
        new_text = ev.text
        if texts is not None and ev.id in texts:
            new_text = texts[ev.id]
        lines.append(f"{ev.raw_prefix}{new_text}")

    # 去掉末尾多余空行，保留单个换行
    content = "\n".join(lines).rstrip("\n") + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8-sig", newline="")

