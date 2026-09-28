"""ffmpeg / ffprobe 的轻量封装。"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path


logger = logging.getLogger("pipeline.video.ffmpeg")


class Ffmpeg:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin

    def run(self, args: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.ffmpeg, "-hide_banner"]
        if quiet:
            cmd += ["-loglevel", "error"]
        cmd += args
        logger.debug("执行 ffmpeg: %s", " ".join(cmd))
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def run_checked(self, args: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
        proc = self.run(args, quiet=quiet)
        if proc.returncode != 0:
            logger.error("ffmpeg 执行失败：%s", proc.stderr.strip())
            raise RuntimeError(f"ffmpeg 执行失败: {proc.stderr.strip()}")
        return proc


def probe(path: str | Path) -> dict:
    """用 ffprobe 读取媒体信息。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    logger.debug("执行 ffprobe: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        logger.error("ffprobe 执行失败：%s", proc.stderr.strip())
        raise RuntimeError(f"ffprobe 执行失败: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def get_duration_ms(path: str | Path) -> int:
    info = probe(path)
    duration = float(info["format"]["duration"])
    return int(round(duration * 1000))

