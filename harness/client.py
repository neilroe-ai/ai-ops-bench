"""Lanes: every one remote, BYOK and OpenAI-compatible; paid lanes metered, NVIDIA lanes free.

A lane swap is a base_url and a key. Claude Pro is never a lane (v3 §3).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Billing modes: which money a lane spends. Every budget reads the matching CSV column.
CASH = "cash"  # prepaid balance; cost_usd; monthly/session/lane ceilings
FREE = "free"  # $0; shadow_cost_usd for reporting only
PROMOTIONAL = "promotional"  # provider credit that runs out; credit_usd; credit ceilings
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
    billing: str = CASH
    # Paid lane that prices the shadow cost AND catches throttled/failed free calls.
    shadow_of: str | None = None
    # Parked lanes stay defined (rates, credit rules, tests) but the runner refuses them.
    enabled: bool = True

    @property
    def free(self) -> bool:
        return self.billing == FREE

    @property
    def promotional(self) -> bool:
        return self.billing == PROMOTIONAL


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
    # PARKED (MD-005, 2026-09-17): key blocked by API restrictions; revisit later.
    # Google AI Studio promotional credit. Thinking level set explicitly: default is medium and
    # thinking tokens bill at the output rate. Rates verified; base_url unverified until first call.
    "gemini-3.8-flash": Lane(
        "gemini-3.8-flash",
        "google",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.8-flash",
        "GEMINI_API_KEY",
        extra_body={"reasoning_effort": "medium"},
        billing=PROMOTIONAL,
        enabled=False,
    ),
    # NVIDIA free tier: tried first, falls back to the paid lane in shadow_of.
    # IDs verified against the NVIDIA catalogue 2026-09-17 (V4 Pro and GLM-5.1 were retired).
    "nvidia-deepseek-v4-flash": Lane(
        "nvidia-deepseek-v4-flash",
        "nvidia",
        NVIDIA_BASE_URL,
        "deepseek-ai/deepseek-v4-flash-0731",  # V4 Flash (0731), older than DeepSeek's own V4.1-Flash
        "NVIDIA_API_KEY",
        verified=True,
        billing=FREE,
        shadow_of="deepseek-flash",
    ),
    "nvidia-glm-5.3-flash": Lane(
        "nvidia-glm-5.3-flash",
        "nvidia",
        NVIDIA_BASE_URL,
        "z-ai/glm-5.3-flash",
        "NVIDIA_API_KEY",
        # Thinking off explicitly (GLM defaults to max). Parameter name unverified on NVIDIA:
        # check reasoning_tokens on the smoke run.
        extra_body={"chat_template_kwargs": {"thinking": False}},
        verified=True,
        billing=FREE,
        shadow_of="deepseek-flash",  # nearest priced paid equivalent until paid GLM is priced
    ),
    "nvidia-nemotron-3-ultra": Lane(
        "nvidia-nemotron-3-ultra",
        "nvidia",
        NVIDIA_BASE_URL,
        "nvidia/nemotron-3-ultra-550b-a55b",
        "NVIDIA_API_KEY",
        verified=True,
        billing=FREE,
        shadow_of="deepseek-flash",
    ),
}


class TransportFailure(RuntimeError):
    """Throttled (429), timed out or unreachable. Triggers fallback on free lanes."""


class MissingKeyError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    """Non-retryable provider refusal (4xx). Carries the provider's own message."""


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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data: JsonDict = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        if e.code == 429 or e.code >= 500:
            raise TransportFailure(f"HTTP {e.code}") from e
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise TransportFailure(str(e)) from e


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
