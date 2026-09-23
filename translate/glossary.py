"""术语表加载与匹配。"""
from __future__ import annotations

import json
from pathlib import Path


def load_glossary(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
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

