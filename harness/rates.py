"""Price table and rate windows. The single place prices live.

Rule (strategy v3 §0): if a file can enforce it, the file owns it.
Rates are USD per 1M tokens. An unpriced model raises; it is never logged as free.

Weekly research cycle Part A proposes edits to this file as a PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time

PEAK = "peak"
OFF_PEAK = "off_peak"
FLAT = "flat"

# Date the figures below were last checked against the live provider pages.
RATES_VERIFIED_ON = "2026-09-17"


class UnpricedModelError(KeyError):
    """Raised when a (model_id, window) pair has no price. Fails loudly at test time."""


@dataclass(frozen=True)
class Rate:
    cache_hit: float
    cache_miss: float
    output: float


# Keyed on (model_id, rate_window). Source: api-docs.deepseek.com/quick_start/pricing.
# GLM-5.3-Flash, Grok Build 0.1 and Nemotron are deliberately absent until their
# live rates are verified; the governor refuses to dispatch to an unpriced lane.
RATES: dict[tuple[str, str], Rate] = {
    ("deepseek-flash", OFF_PEAK): Rate(cache_hit=0.003, cache_miss=0.15, output=0.60),
    ("deepseek-flash", PEAK): Rate(cache_hit=0.006, cache_miss=0.30, output=1.20),
    ("deepseek-v4-pro", OFF_PEAK): Rate(cache_hit=0.022, cache_miss=0.66, output=1.98),
    ("deepseek-v4-pro", PEAK): Rate(cache_hit=0.044, cache_miss=1.32, output=3.96),
}

# Providers that price by time of day: weekdays (Mon=0..Fri=4), UTC [start, end) spans.
# Providers not listed here use FLAT.
PEAK_WINDOWS: dict[str, tuple[frozenset[int], tuple[tuple[time, time], ...]]] = {
    "deepseek": (
        frozenset({0, 1, 2, 3, 4}),
        ((time(1, 0), time(4, 0)), (time(6, 0), time(10, 0))),
    ),
}


def rate_window(provider: str, at: datetime) -> str:
    """Return PEAK, OFF_PEAK or FLAT for a provider at a timezone-aware instant."""
    if at.tzinfo is None:
        raise ValueError("rate_window needs a timezone-aware datetime")
    if provider not in PEAK_WINDOWS:
        return FLAT
    days, spans = PEAK_WINDOWS[provider]
    utc = at.astimezone(UTC)
    if utc.weekday() in days and any(start <= utc.time() < end for start, end in spans):
        return PEAK
    return OFF_PEAK


def get_rate(model_id: str, window: str) -> Rate:
    try:
        return RATES[(model_id, window)]
    except KeyError:
        raise UnpricedModelError(f"no rate for ({model_id!r}, {window!r})") from None


def cost_usd(
    model_id: str,
    window: str,
    *,
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one call. Reasoning tokens are billed inside output_tokens."""
    rate = get_rate(model_id, window)
    return (
        cache_hit_tokens * rate.cache_hit + cache_miss_tokens * rate.cache_miss + output_tokens * rate.output
    ) / 1_000_000
