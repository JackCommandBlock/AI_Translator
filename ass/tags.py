"""ASS 特效标签与文本的分离 / 还原。

Text 中形如 {\\an8}、{\\pos(960,100)} 的 override 标签需与正文分离，
只翻译正文，翻译后原样拼回。换行符 \\N、\\n 也需保留。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


TAG_RE = re.compile(r"\{[^{}]*\}")
NEWLINE_TOKEN = "\u23ce"  # ⏎ 作为 LLM 翻译时的换行占位符


@dataclass
class Token:
    kind: str  # "tag" | "newline" | "text"
    value: str


def tokenize(text: str) -> list[Token]:
    """把 ASS Text 拆成 tag / newline / text 三种 token。"""
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        tag_m = TAG_RE.match(text, pos)
        if tag_m:
            tokens.append(Token("tag", tag_m.group(0)))
            pos = tag_m.end()
            continue
        if text.startswith("\\N", pos):
            tokens.append(Token("newline", "\\N"))
            pos += 2
            continue
        if text.startswith("\\n", pos):
            tokens.append(Token("newline", "\\n"))
            pos += 2
            continue
        if text.startswith("\\h", pos):
            tokens.append(Token("newline", "\\h"))
            pos += 2
            continue
        j = pos
        while j < len(text):
            if text[j] == "{":
                break
            if text.startswith("\\N", j) or text.startswith("\\n", j) or text.startswith("\\h", j):
                break
            j += 1
        tokens.append(Token("text", text[pos:j]))
        pos = j
    return tokens


def extract_translatable(tokens: list[Token]) -> str:
    """把正文 token 合并为待翻译文本，换行用占位符表示。"""
    parts: list[str] = []
    for t in tokens:
        if t.kind == "text":
            parts.append(t.value)
        elif t.kind == "newline":
            parts.append(NEWLINE_TOKEN)
    return "".join(parts)


def recompose(tokens: list[Token], translated: str) -> str:
    """把翻译结果按 token 结构拼回 ASS Text。

    翻译文本按换行占位符拆成若干行，逐行填回该行第一个 text token；
    tag 与换行 token 原样保留。同一行中因内联标签被拆开的多个 text
    token 会合并到第一个 text token（内联位置略有损失，但标签不丢失）。
    """
    lines = translated.split(NEWLINE_TOKEN)
    out: list[str] = []
    line_idx = 0
    text_emitted = False
    for t in tokens:
        if t.kind == "tag":
            out.append(t.value)
        elif t.kind == "newline":
            out.append(t.value)
            line_idx += 1
            text_emitted = False
        else:
            if not text_emitted:
                out.append(lines[line_idx] if line_idx < len(lines) else "")
                text_emitted = True
            # 同一行后续的 text token 已合并到第一个 text token，跳过
    return "".join(out)


def has_tags(text: str) -> bool:
    return bool(TAG_RE.search(text))


def clean_tags(text: str) -> str:
    """去掉所有 override 标签，仅用于检查等场景。"""
    return TAG_RE.sub("", text).replace("\\N", " ").replace("\\n", " ")
