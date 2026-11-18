"""Minimal Ollama client: JSON-mode chat, model fallback, CPU fallback on CUDA OOM."""
from __future__ import annotations

import json
import logging
import re

import httpx

from ..config import get_settings
from ..gpu import gpu_lock, is_oom

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction (models sometimes wrap it in prose or ```json fences)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON in: {text[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"bad JSON: {e}: {text[:200]}") from e


async def _call(client: httpx.AsyncClient, model: str, system: str, user: str, cpu: bool) -> str:
    cfg = get_settings().llm
    opts = {"temperature": 0.4, "num_ctx": 4096}
    if cpu:
        opts["num_gpu"] = 0
    r = await client.post(
        f"{cfg.base_url}/api/chat",
        json={"model": model, "stream": False, "format": "json", "keep_alive": cfg.keep_alive,
              "options": opts,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
    )
    if r.status_code != 200:
        raise LLMError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"]


async def chat_json(system: str, user: str) -> dict:
    cfg = get_settings().llm
    async with gpu_lock():
        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            last: Exception | None = None
            for model in filter(None, [cfg.model, cfg.fallback_model]):
                for cpu in (False, True):
                    try:
                        return parse_json(await _call(client, model, system, user, cpu))
                    except (LLMError, httpx.HTTPError, KeyError) as e:
                        last = e
                        if not cpu and is_oom(str(e)):
                            log.warning("LLM out of GPU memory, retrying on CPU")
                            continue
                        if "not found" in str(e).lower():
                            log.warning("model %s not pulled; run: ollama pull %s", model, model)
                        break
            raise LLMError(str(last))


async def is_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{get_settings().llm.base_url}/api/tags")
            return r.status_code == 200
    except httpx.HTTPError:
        return False
