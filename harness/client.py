"""Lanes: every one remote, metered, BYOK and OpenAI-compatible.

A lane swap is a base_url and a key. Claude Pro is never a lane (v3 §3).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]
Transport = Callable[[str, Mapping[str, str], bytes, float], JsonDict]


@dataclass(frozen=True)
class Lane:
    name: str
    provider: str
    base_url: str
    model_id: str
    api_key_env: str
    # Set thinking levels explicitly: GLM defaults to maximum when absent (v3 §5).
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    verified: bool = False  # base_url/model_id checked against live provider docs


LANES: dict[str, Lane] = {
    "deepseek-flash": Lane(
        "deepseek-flash",
        "deepseek",
        "https://api.deepseek.com",
        "deepseek-flash",
        "DEEPSEEK_API_KEY",
        verified=True,
    ),
    "deepseek-v4-pro": Lane(
        "deepseek-v4-pro",
        "deepseek",
        "https://api.deepseek.com",
        "deepseek-v4-pro",
        "DEEPSEEK_API_KEY",
        verified=True,
    ),
    # Unverified: confirm base_url, model_id and thinking parameter before funding.
    "glm-5.3-flash": Lane(
        "glm-5.3-flash",
        "zai",
        "https://api.z.ai/api/paas/v4",
        "glm-5.3-flash",
        "ZAI_API_KEY",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "grok-build-0.1": Lane(
        "grok-build-0.1",
        "xai",
        "https://api.x.ai/v1",
        "grok-build-0.1",
        "XAI_API_KEY",
    ),
    "nemotron-3-ultra": Lane(
        "nemotron-3-ultra",
        "nvidia",
        "https://integrate.api.nvidia.com/v1",
        "nvidia/nemotron-3-ultra",
        "NVIDIA_API_KEY",
    ),
}


class MissingKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    cache_hit_tokens: int
    cache_miss_tokens: int
    output_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    served_model: str
    usage: Usage
    wall_clock_ms: int
    finish_reason: str


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> JsonDict:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data: JsonDict = json.loads(resp.read().decode("utf-8"))
        return data


def parse_usage(usage: Mapping[str, Any]) -> Usage:
    prompt = int(usage.get("prompt_tokens", 0))
    # DeepSeek reports prompt_cache_hit_tokens; OpenAI-style reports prompt_tokens_details.
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return Usage(
        cache_hit_tokens=int(hit),
        cache_miss_tokens=prompt - int(hit),
        output_tokens=int(usage.get("completion_tokens", 0)),
        reasoning_tokens=int(reasoning or 0),
    )


def complete(
    lane: Lane,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    transport: Transport = urllib_transport,
    env: Mapping[str, str] = os.environ,
    clock: Callable[[], float] = time.monotonic,
    timeout: float = 600.0,
) -> Completion:
    key = env.get(lane.api_key_env)
    if not key:
        raise MissingKeyError(f"{lane.api_key_env} is not set")
    body = {"model": lane.model_id, "messages": messages, "max_tokens": max_tokens, **lane.extra_body}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    started = clock()
    data = transport(
        f"{lane.base_url.rstrip('/')}/chat/completions", headers, json.dumps(body).encode(), timeout
    )
    elapsed_ms = int((clock() - started) * 1000)
    choice = data["choices"][0]
    return Completion(
        text=str(choice["message"].get("content") or ""),
        served_model=str(data.get("model", lane.model_id)),
        usage=parse_usage(data.get("usage", {})),
        wall_clock_ms=elapsed_ms,
        finish_reason=str(choice.get("finish_reason", "")),
    )
