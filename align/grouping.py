"""字幕上下文分组。

连续、时间相邻且说话人一致的事件合并为一个字幕组，
用于上下文翻译与跨事件的原文纠错。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubtitleGroup:
    event_ids: list[int] = field(default_factory=list)

    @property
    def start_id(self) -> int:
        return self.event_ids[0]

    @property
    def end_id(self) -> int:
        return self.event_ids[-1]


def build_groups(events, gap_threshold_ms: int = 1200) -> list[SubtitleGroup]:
    """根据时间间隔与说话人名称建立字幕组。"""
    groups: list[SubtitleGroup] = []
    current: SubtitleGroup | None = None
    prev_speaker = None
    prev_end = None

    for ev in events:
        speaker = ev.name.strip().lower()
        if current is None:
            current = SubtitleGroup(event_ids=[ev.id])
        else:
            gap = ev.start_ms - prev_end
            continuous = gap <= gap_threshold_ms
            same_speaker = (speaker == prev_speaker) or (prev_speaker in ("", "undefined"))
            if continuous and same_speaker:
                current.event_ids.append(ev.id)
            else:
                groups.append(current)
                current = SubtitleGroup(event_ids=[ev.id])
        prev_speaker = speaker
        prev_end = ev.end_ms

    if current is not None:
        groups.append(current)
    return groups

