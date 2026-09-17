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
from datetime import UTC, date, datetime

from harness.client import LANES
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


# ---- Promotional credit (third money type: not cash, not free — it runs out) ----------
# Source: claude/gemini-credit-assessment.md. Credit spend never enters the cash budget above.
# Google consumes promotional credit BEFORE prepay; once prepay hits $0, credit stops working.
# So credit ceilings stay below the promo balance (prepay is never touched) and prepay stays
# above PREPAY_FLOOR. Confirm expiry, Prepay/Postpay and auto-reload OFF on the billing page.

FX_USD_PER_TWD = 0.0315  # approximate; reconcile against the console in TWD


@dataclass(frozen=True)
class CreditGrant:
    provider: str
    amount_twd: float
    expires: date | None  # None until read from the billing page

    @property
    def amount_usd(self) -> float:
        return self.amount_twd * FX_USD_PER_TWD


CREDIT_GRANTS: tuple[CreditGrant, ...] = (CreditGrant(provider="google", amount_twd=7924.50, expires=None),)
PREPAY_BALANCE_TWD: dict[str, float] = {"google": 354.00}
PREPAY_FLOOR_TWD: dict[str, float] = {"google": 200.00}

# Per-lane credit ceilings: lifetime (below the promo balance) and monthly (below Tier 1's $250 cap).
CREDIT_LANE_CEILING_USD: dict[str, float] = {"gemini-3.8-flash": 230.00}
CREDIT_MONTHLY_CEILING_USD: dict[str, float] = {"gemini-3.8-flash": 150.00}
PROVIDER_MONTHLY_CAP_USD: dict[str, float] = {"google": 250.00}


class BudgetExceededError(RuntimeError):
    """Dispatch refused: a hard ceiling would be crossed. Not overridable."""


class CreditExhaustedError(BudgetExceededError):
    """Credit ceiling, expiry or prepay floor would be crossed. Not overridable."""


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


def estimate_max_cost(
    model_id: str,
    window: str,
    *,
    input_tokens: int,
    max_output_tokens: int,
    at: datetime | None = None,
) -> float:
    """Worst case: every input token a cache miss, output runs to max_tokens."""
    rate = get_rate(model_id, window, at)  # raises UnpricedModelError for unpriced lanes
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


def credit_balance_usd(provider: str) -> float:
    return sum(g.amount_usd for g in CREDIT_GRANTS if g.provider == provider)


def check_credit_invariants() -> None:
    """Credit ceilings sit inside the promo balance; prepay sits above its floor."""
    for lane, ceiling in CREDIT_LANE_CEILING_USD.items():
        provider = lane_provider(lane)
        if not ceiling < credit_balance_usd(provider):
            raise CreditExhaustedError(f"credit ceiling {lane} ${ceiling:.2f} must be below promo balance")
        if not CREDIT_MONTHLY_CEILING_USD.get(lane, 0.0) <= PROVIDER_MONTHLY_CAP_USD.get(provider, 0.0):
            raise CreditExhaustedError(f"monthly credit ceiling {lane} exceeds provider tier cap")
    for provider, floor in PREPAY_FLOOR_TWD.items():
        if not PREPAY_BALANCE_TWD.get(provider, 0.0) > floor:
            raise CreditExhaustedError(
                f"{provider} prepay must stay above NT${floor:.0f} or credit goes inert"
            )


def lane_provider(lane: str) -> str:
    return LANES[lane].provider


def authorize_credit(
    lane: str,
    estimate_usd: float,
    *,
    lifetime_usd: float,
    month_usd: float,
    at: datetime,
) -> None:
    check_credit_invariants()
    provider = lane_provider(lane)
    today = at.astimezone(UTC).date()
    for g in CREDIT_GRANTS:
        if g.provider == provider and g.expires is not None and today >= g.expires:
            raise CreditExhaustedError(f"{provider} promotional credit expired {g.expires}")
    checks = (
        ("credit lifetime", lifetime_usd, CREDIT_LANE_CEILING_USD.get(lane, 0.0)),
        ("credit monthly", month_usd, CREDIT_MONTHLY_CEILING_USD.get(lane, 0.0)),
    )
    for name, spent, ceiling in checks:
        if spent + estimate_usd > ceiling:
            raise CreditExhaustedError(
                f"{name} ceiling {lane} ${ceiling:.2f}: used ${spent:.4f} + estimate ${estimate_usd:.4f}"
            )
