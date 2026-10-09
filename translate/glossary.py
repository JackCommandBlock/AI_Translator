"""术语表加载与匹配。"""
from __future__ import annotations

import json
from pathlib import Path


def _read_glossary_file(p: Path) -> dict[str, str]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        out = {}
        for item in data:
            if isinstance(item, dict):
                src = item.get("source") or item.get("src") or item.get("原文")
                dst = item.get("target") or item.get("dst") or item.get("译文")
                if src and dst:
                    out[str(src)] = str(dst)
        return out
    return {str(k): str(v) for k, v in data.items()}


def load_glossary(path: str | Path) -> dict[str, str]:
    """加载术语表。

    ``path`` 既可以是单个 JSON 文件，也可以是包含多个 ``*.json`` 的目录；
    目录模式下会按文件名排序后合并所有表。
    """
    p = Path(path)
    if not p.exists():
        return {}
    if p.is_dir():
        out: dict[str, str] = {}
        for fp in sorted(p.glob("*.json")):
            out.update(_read_glossary_file(fp))
        return out
    return _read_glossary_file(p)


def format_glossary(glossary: dict[str, str], limit: int = 200) -> str:
    if not glossary:
        return ""
    items = list(glossary.items())[:limit]
    return "\n".join(f"{k} → {v}" for k, v in items)


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """把术语表按长词优先替换到文本中（仅对源语文本使用）。"""
    if not text or not glossary:
        return text
    for src in sorted(glossary, key=len, reverse=True):
        if src in text:
            text = text.replace(src, glossary[src])
    return text

