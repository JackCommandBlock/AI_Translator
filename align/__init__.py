"""时间轴对齐、文本时间线与字幕分组。"""
from .matcher import ASRMatch, align_asr_to_events, overlap_ms, iou
from .grouping import SubtitleGroup, build_groups
from .text_timeline import (
    TextTrack,
    build_text_timeline,
    infer_type,
    match_events_to_tracks,
    text_similarity,
)

__all__ = [
    "ASRMatch",
    "align_asr_to_events",
    "overlap_ms",
    "iou",
    "SubtitleGroup",
    "build_groups",
    "TextTrack",
    "build_text_timeline",
    "infer_type",
    "match_events_to_tracks",
    "text_similarity",
]

