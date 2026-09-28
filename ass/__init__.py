"""ASS 字幕解析、标签处理与写回。"""
from .parser import AssFile, DialogueEvent, parse_ass
from .styles import AssStyle, parse_ass_color, parse_styles, style_for_events
from .writer import write_ass

__all__ = [
    "AssFile",
    "DialogueEvent",
    "parse_ass",
    "write_ass",
    "AssStyle",
    "parse_ass_color",
    "parse_styles",
    "style_for_events",
]

