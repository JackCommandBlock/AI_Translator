"""从视频抽取音频并转换为适合 ASR 的 WAV。"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg import Ffmpeg


def extract_audio_wav(
    video_path: str | Path,
    out_path: str | Path,
    sample_rate: int = 16000,
    ffmpeg: Ffmpeg | None = None,
) -> Path:
    """抽取单声道 16kHz WAV，供 Whisper 系列模型使用。"""
    ffmpeg = ffmpeg or Ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_checked(
        [
            "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out_path),
        ]
    )
    return out_path

