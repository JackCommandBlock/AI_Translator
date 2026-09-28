"""OpenAI 兼容客户端的统一封装。

所有大模型网络请求都应通过 :func:`chat_completion` 发起，这样模型、消息规模、
耗时、token 用量以及异常都能被统一记录到 ``pipeline.network`` 日志中。
"""
from __future__ import annotations

import logging
import time
from typing import Any


logger = logging.getLogger("pipeline.network")


def build_client(config):
    """按 gemini_example.py 的方式创建 OpenAI 兼容客户端。"""
    from openai import OpenAI

    logger.debug("创建 OpenAI 兼容客户端 base_url=%s", config.api_base_url)
    return OpenAI(base_url=config.api_base_url, api_key=config.api_key)


def _describe_message(message: dict) -> dict:
    """把消息压缩成适合日志的摘要，避免输出大段 base64 图像。"""
    role = message.get("role", "?")
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "chars": len(content), "preview": content[:120]}
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(type(item).__name__))
                continue
            kind = item.get("type")
            if kind == "text":
                text = str(item.get("text") or "")
                parts.append(f"text({len(text)}字符)")
            elif kind == "image_url":
                url = str(item.get("image_url", {}).get("url") or "")
                parts.append("image(base64)" if url.startswith("data:") else f"image({url[:80]})")
            else:
                parts.append(str(kind or "?"))
        return {"role": role, "parts": parts}
    return {"role": role}


def _usage_dict(resp: Any) -> dict[str, Any] | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def chat_completion(client, **kwargs) -> Any:
    """带日志的 ``client.chat.completions.create`` 调用。"""
    model = kwargs.get("model", "?")
    messages = kwargs.get("messages") or []
    temperature = kwargs.get("temperature")

    logger.info(
        "LLM 请求 model=%s messages=%d temperature=%s",
        model,
        len(messages),
        temperature,
    )
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            logger.debug("  [message %d] %s", i, _describe_message(msg))

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        logger.exception(
            "LLM 请求失败 model=%s messages=%d",
            model,
            len(messages),
        )
        raise

    elapsed = time.perf_counter() - start
    content = ""
    choices = getattr(resp, "choices", None) or []
    if choices and getattr(choices[0], "message", None) is not None:
        content = choices[0].message.content or ""
    usage = _usage_dict(resp)
    logger.info(
        "LLM 响应 model=%s 耗时=%.2fs 返回字符=%d 用量=%s",
        model,
        elapsed,
        len(content),
        usage,
    )
    if content:
        logger.debug("LLM 响应内容: %s", content[:500])
    return resp
