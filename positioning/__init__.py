"""字幕定位包。

根据原画面中烧录字幕的包围盒，为译文字幕生成候选位置，用 ffmpeg 把候选
位置渲染到代表帧上，再交给多模态大模型评分，选出最优位置写回 ``\pos``。
"""
from .positioner import position_subtitles, apply_positions

__all__ = ["position_subtitles", "apply_positions"]
