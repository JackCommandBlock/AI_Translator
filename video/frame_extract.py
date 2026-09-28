"""按 ASS 事件时间轴从视频抽取代表性帧。"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg import Ffmpeg


def _sample_times(start_ms: int, end_ms: int, count: int) -> list[int]:
    """在 [start, end] 内取 count 个采样点（含首尾，中间均布）。"""
    if count <= 1:
        return [start_ms]
    return [int(start_ms + (end_ms - start_ms) * i / (count - 1)) for i in range(count)]


def extract_frames_for_events(
    video_path: str | Path,
    events,
    out_dir: str | Path,
    frames_per_event: int = 3,
    ffmpeg: Ffmpeg | None = None,
) -> dict[int, list[Path]]:
    """返回 {event_id: [frame_path, ...]}。"""
    ffmpeg = ffmpeg or Ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[Path]] = {}
    for ev in events:
        times = _sample_times(ev.start_ms, ev.end_ms, frames_per_event)
        paths: list[Path] = []
        for k, ms in enumerate(times):
            sec = ms / 1000.0
            fp = out_dir / f"ev{ev.id:04d}_{k}.png"
            ffmpeg.run_checked(
                ["-y", "-ss", f"{sec:.3f}", "-i", str(video_path), "-frames:v", "1", str(fp)]
            )
            if fp.exists():
                paths.append(fp)
        result[ev.id] = paths
    return result


def extract_frames_at_times(
    video_path: str | Path,
    times_ms: list[int],
    out_dir: str | Path,
    prefix: str = "frame",
    ffmpeg: Ffmpeg | None = None,
) -> list[Path]:
    ffmpeg = ffmpeg or Ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ms in enumerate(times_ms):
        fp = out_dir / f"{prefix}_{i:04d}.png"
        ffmpeg.run_checked(
            ["-y", "-ss", f"{ms / 1000.0:.3f}", "-i", str(video_path), "-frames:v", "1", str(fp)]
        )
        if fp.exists():
            paths.append(fp)
    return paths


def extract_frames_at_times_map(
    video_path: str | Path,
    times_ms: list[int],
    out_dir: str | Path,
    prefix: str = "frame",
    ffmpeg: Ffmpeg | None = None,
) -> list[tuple[int, Path]]:
    """与 extract_frames_at_times 相同，但返回 (time_ms, path) 以便时间对齐。"""
    ffmpeg = ffmpeg or Ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[tuple[int, Path]] = []
    for i, ms in enumerate(times_ms):
        fp = out_dir / f"{prefix}_{i:04d}.png"
        ffmpeg.run_checked(
            ["-y", "-ss", f"{ms / 1000.0:.3f}", "-i", str(video_path), "-frames:v", "1", str(fp)]
        )
        if fp.exists():
            result.append((ms, fp))
    return result

