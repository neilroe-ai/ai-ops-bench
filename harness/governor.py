"""Spend control, layer 2: refuse to dispatch before the money is spent.

Layer 1 is the prepaid provider balance (auto top-up OFF everywhere).
Invariant (v3 §4): MONTHLY_BUDGET_USD < sum of provider balances.

Hard stops (never overridable): monthly, per-lane.
Soft stop (alert, pause, Neil decides): session — a rolling window across all session names,
so starting a new session name does not reset it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from harness.rates import get_rate

MONTHLY_BUDGET_USD = 15.00
SESSION_BUDGET_USD = 2.00
SESSION_WINDOW_HOURS = 6

# Benchmark scaffolding, not architecture: delete once a workforce lane is named (v3 §8.10).
# A lane missing from this dict has a ceiling of zero. Raise the ceiling, THEN top up.
# Free NVIDIA lanes cost $0 real money, so a zero ceiling never blocks them.
PER_LANE_BUDGET_USD: dict[str, float] = {
    "deepseek-flash": 10.00,
    "deepseek-v4-pro": 2.00,
}


@dataclass(frozen=True)
class Topup:
    provider: str
    amount_usd: float
    date: str  # ISO date as shown on the provider console


# Record every top-up here. Confirm date against the DeepSeek console.
TOPUPS: tuple[Topup, ...] = (Topup(provider="deepseek", amount_usd=20.00, date="2026-08"),)


class BudgetExceededError(RuntimeError):
    """Dispatch refused: a hard ceiling would be crossed. Not overridable."""


class SessionLimitError(RuntimeError):
    """Session ceiling would be crossed. Neil may override with a logged reason."""


@dataclass(frozen=True)
class Spend:
    month_usd: float
    session_usd: float
    lane_usd: float


def provider_balances(topups: Iterable[Topup] = TOPUPS) -> dict[str, float]:
    totals: dict[str, float] = {}
    for t in topups:
        totals[t.provider] = totals.get(t.provider, 0.0) + t.amount_usd
    return totals


def check_invariant(
    monthly_budget: float = MONTHLY_BUDGET_USD,
    topups: Iterable[Topup] = TOPUPS,
) -> None:
    total = sum(provider_balances(topups).values())
    if not monthly_budget < total:
        raise BudgetExceededError(
            f"monthly budget ${monthly_budget:.2f} must be below topped-up total ${total:.2f}"
        )


def estimate_max_cost(model_id: str, window: str, *, input_tokens: int, max_output_tokens: int) -> float:
    """Worst case: every input token a cache miss, output runs to max_tokens."""
    rate = get_rate(model_id, window)  # raises UnpricedModelError for unpriced lanes
    return (input_tokens * rate.cache_miss + max_output_tokens * rate.output) / 1_000_000


def authorize(
    lane: str,
    estimate_usd: float,
    spend: Spend,
    *,
    session_override: bool = False,
    monthly_budget: float = MONTHLY_BUDGET_USD,
    session_budget: float = SESSION_BUDGET_USD,
    per_lane: dict[str, float] | None = None,
) -> None:
    """Hard ceilings are checked first, so an override can never mask them."""
    ceilings = PER_LANE_BUDGET_USD if per_lane is None else per_lane
    hard = (
        ("monthly", spend.month_usd, monthly_budget),
        (f"lane {lane}", spend.lane_usd, ceilings.get(lane, 0.0)),
    )
    for name, spent, ceiling in hard:
        if spent + estimate_usd > ceiling:
            raise BudgetExceededError(
                f"{name} ceiling ${ceiling:.2f}: spent ${spent:.4f} + estimate ${estimate_usd:.4f}"
            )
    if not session_override and spend.session_usd + estimate_usd > session_budget:
        raise SessionLimitError(
            f"session ceiling ${session_budget:.2f} (last {SESSION_WINDOW_HOURS}h): "
            f"spent ${spend.session_usd:.4f} + estimate ${estimate_usd:.4f}"
        )
