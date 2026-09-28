"""统一日志配置与脱敏工具。

项目内各模块通过 ``logging.getLogger("pipeline.*")`` 记录运行步骤，
网络请求统一由 ``translate.client.chat_completion`` 记录。日志同时输出到
控制台和 ``log_dir/log_file``，文件采用滚动策略，避免无限增长。

为了避免把密钥、token 等敏感信息写进日志，格式化时会统一脱敏。
"""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "passwd",
)

_SENSITIVE_RE = re.compile(
    r"(?i)\b(" + "|".join(SENSITIVE_KEYS) + r")\b\s*[:=]\s*['\"]?[^,\s'\"}]+"
)


def redact(text: str) -> str:
    """把常见密钥形式替换为 ``***``，防止日志泄漏敏感信息。"""
    if not text:
        return text
    return _SENSITIVE_RE.sub(r"\1=***", text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return redact(msg)


def setup_logging(
    log_dir: str | Path = "logs",
    log_file: str = "pipeline.log",
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """初始化 pipeline 日志器（幂等，重复调用不会叠加 handler）。"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(_resolve_level(level))

    # 允许按新配置重新初始化，同时避免重复调用造成 handler 叠加。
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    fmt = RedactingFormatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """返回 ``pipeline`` 或 ``pipeline.<name>`` 命名空间下的 logger。"""
    if not name:
        return logging.getLogger("pipeline")
    return logging.getLogger(f"pipeline.{name}")


def _resolve_level(level: str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)
