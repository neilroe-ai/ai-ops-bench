"""Dispatch one benchmark task to one lane, single-turn.

    python -m harness.run --lane nvidia-deepseek-v4-pro --task benchmarks/tasks/diagnose.md \
        --session 2026-09-18-baseline --require-window off_peak

Order: breaker -> window -> invariant -> estimate -> authorize (session: alert + pause) -> call
       -> real cost + shadow cost -> log -> blind review packet.
Free lanes that throttle or fail fall back to their paid `shadow_of` lane, logged as such.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness import governor, rates, telemetry
from harness.client import LANES, Completion, Lane, Transport, TransportFailure, complete, urllib_transport

STANDING_INSTRUCTION = (
    "You are working on a real codebase. For diagnose and investigate tasks: do not rewrite "
    "code. Understand and explain first. Cite files and line numbers for every claim."
)

TELEMETRY_PATH = Path("data/telemetry.csv")
REVIEWS_DIR = Path("reviews/pending")

# Circuit breaker: after this many consecutive throttles on a free lane within the window,
# skip straight to the paid fallback until the window passes.
BREAKER_FAILURES = 3
BREAKER_MINUTES = 30
FREE_LANE_TIMEOUT_S = 180.0

ConfirmOverride = Callable[[str], str | None]


class WindowMismatchError(RuntimeError):
    """The benchmark requires one rate window for all lanes (v3 §5)."""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    review_id: str
    lane: str
    cost_usd: float
    shadow_cost_usd: float | None
    completion: Completion
    credit_usd: float = 0.0


def no_prompt(message: str) -> str | None:
    return None


def tty_confirm(message: str) -> str | None:
    """Alert and pause. Non-interactive runs (agents, cron) always stop."""
    print(f"\a\n*** SESSION LIMIT REACHED ***\n{message}", file=sys.stderr)
    if not sys.stdin.isatty():
        print("Not interactive: stopping.", file=sys.stderr)
        return None
    reason = input("Override? Type a reason to continue, or press Enter to stop: ").strip()
    return reason or None


def estimate_input_tokens(text: str) -> int:
    # Conservative: ~3 chars/token over-reserves against the ~4 typical for English/code.
    return len(text) // 3 + 1


def breaker_open(rows: list[telemetry.Row], lane: str, now: datetime) -> bool:
    since = now - timedelta(minutes=BREAKER_MINUTES)
    recent = [r for r in rows if r["lane"] == lane and datetime.fromisoformat(r["timestamp_utc"]) >= since]
    last = recent[-BREAKER_FAILURES:]
    return len(last) == BREAKER_FAILURES and all(r["status"] == "throttled" for r in last)


def shadow_cost(lane: Lane, at: datetime, completion: Completion) -> float | None:
    if not lane.free or lane.shadow_of is None:
        return None
    paid = LANES[lane.shadow_of]
    u = completion.usage
    try:
        return rates.cost_usd(
            paid.model_id,
            rates.rate_window(paid.provider, at),
            at=at,
            cache_hit_tokens=u.cache_hit_tokens,
            cache_miss_tokens=u.cache_miss_tokens,
            output_tokens=u.output_tokens,
        )
    except rates.UnpricedModelError:
        return None


def _blank_row(**kw: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(telemetry.FIELDS, "")
    row.update(kw)
    return row


def run_task(
    *,
    lane_name: str,
    task_path: Path,
    session_id: str,
    max_output_tokens: int = 8000,
    require_window: str | None = None,
    override_reason: str | None = None,
    confirm_override: ConfirmOverride = no_prompt,
    telemetry_path: Path = TELEMETRY_PATH,
    reviews_dir: Path = REVIEWS_DIR,
    transport: Transport = urllib_transport,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    env: dict[str, str] | None = None,
) -> RunResult:
    task_text = task_path.read_text(encoding="utf-8")
    lane = LANES[lane_name]
    rows = telemetry.read(telemetry_path)
    fallback_from = ""
    if lane.free and lane.shadow_of and breaker_open(rows, lane.name, now()):
        fallback_from, lane = lane.name, LANES[lane.shadow_of]

    state = {"override": override_reason}
    try:
        return _attempt(
            lane,
            task_text,
            task_path.stem,
            session_id,
            max_output_tokens,
            require_window,
            state,
            confirm_override,
            telemetry_path,
            reviews_dir,
            transport,
            now,
            env,
            fallback_from,
        )
    except TransportFailure as failure:
        if not (lane.free and lane.shadow_of):
            raise
        at = now()
        telemetry.append(
            telemetry_path,
            _blank_row(
                timestamp_utc=at.isoformat(timespec="seconds"),
                run_id=uuid.uuid4().hex[:12],
                session_id=session_id,
                task_id=task_path.stem,
                lane=lane.name,
                provider=lane.provider,
                model_id=lane.model_id,
                rate_window=rates.FLAT,
                dispatch_hour_utc=at.hour,
                billing_mode=lane.billing,
                cost_usd="0.000000",
                status="throttled",
                error=str(failure),
            ),
        )
        return _attempt(
            LANES[lane.shadow_of],
            task_text,
            task_path.stem,
            session_id,
            max_output_tokens,
            require_window,
            state,
            confirm_override,
            telemetry_path,
            reviews_dir,
            transport,
            now,
            env,
            lane.name,
        )


def _attempt(
    lane: Lane,
    task_text: str,
    task_id: str,
    session_id: str,
    max_output_tokens: int,
    require_window: str | None,
    state: dict[str, str | None],
    confirm_override: ConfirmOverride,
    telemetry_path: Path,
    reviews_dir: Path,
    transport: Transport,
    now: Callable[[], datetime],
    env: dict[str, str] | None,
    fallback_from: str,
) -> RunResult:
    dispatched = now()
    window = rates.rate_window(lane.provider, dispatched)
    if require_window and window not in (require_window, rates.FLAT):
        raise WindowMismatchError(f"{lane.provider} is {window}, benchmark requires {require_window}")

    governor.check_invariant()
    rows = telemetry.read(telemetry_path)
    estimate = governor.estimate_max_cost(
        lane.model_id,
        window,
        input_tokens=estimate_input_tokens(STANDING_INSTRUCTION + task_text),
        max_output_tokens=max_output_tokens,
        at=dispatched,
    )
    # Credit lanes spend no cash: the cash governor sees $0 and credit ceilings see the estimate.
    cash_estimate = 0.0 if lane.promotional else estimate
    if lane.promotional:
        governor.authorize_credit(
            lane.name,
            estimate,
            lifetime_usd=telemetry.credit_lane_spend(rows, lane.name),
            month_usd=telemetry.credit_month_spend(rows, lane.name, dispatched.strftime("%Y-%m")),
            at=dispatched,
        )
    spend = governor.Spend(
        month_usd=telemetry.month_spend(rows, dispatched.strftime("%Y-%m")),
        session_usd=telemetry.window_spend(rows, dispatched, governor.SESSION_WINDOW_HOURS),
        lane_usd=telemetry.lane_spend(rows, lane.name),
    )
    try:
        governor.authorize(lane.name, cash_estimate, spend, session_override=bool(state["override"]))
    except governor.SessionLimitError as limit:
        reason = confirm_override(str(limit))
        if not reason:
            raise
        state["override"] = reason
        governor.authorize(lane.name, cash_estimate, spend, session_override=True)

    completion = complete(
        lane,
        [{"role": "system", "content": STANDING_INSTRUCTION}, {"role": "user", "content": task_text}],
        max_tokens=max_output_tokens,
        transport=transport,
        env=os.environ if env is None else env,
        timeout=FREE_LANE_TIMEOUT_S if lane.free else 600.0,
    )
    u = completion.usage
    list_cost = rates.cost_usd(
        lane.model_id,
        window,
        at=dispatched,
        cache_hit_tokens=u.cache_hit_tokens,
        cache_miss_tokens=u.cache_miss_tokens,
        output_tokens=u.output_tokens,
    )
    credit = list_cost if lane.promotional else 0.0
    cost = 0.0 if lane.promotional else list_cost
    shadow = shadow_cost(lane, dispatched, completion)

    run_id, review_id = uuid.uuid4().hex[:12], uuid.uuid4().hex[:8]
    telemetry.append(
        telemetry_path,
        _blank_row(
            timestamp_utc=dispatched.isoformat(timespec="seconds"),
            run_id=run_id,
            session_id=session_id,
            task_id=task_id,
            review_id=review_id,
            lane=lane.name,
            provider=lane.provider,
            model_id=lane.model_id,
            model_version=completion.served_model,
            rate_window=window,
            dispatch_hour_utc=dispatched.hour,
            billing_mode=lane.billing,
            rate_card_date=rates.rate_card_date(lane.model_id, dispatched),
            credit_usd=f"{credit:.6f}" if lane.promotional else "",
            cache_hit_tokens=u.cache_hit_tokens,
            cache_miss_tokens=u.cache_miss_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.reasoning_tokens,
            wall_clock_ms=completion.wall_clock_ms,
            cost_usd=f"{cost:.6f}",
            shadow_cost_usd="" if shadow is None else f"{shadow:.6f}",
            lane_spend_usd=f"{spend.lane_usd + cost:.6f}",
            fallback_from=fallback_from,
            override_reason=state["override"] or "",
            status="ok" if completion.finish_reason != "length" else "truncated",
        ),
    )

    # Blind packet: task and output only. Lane lives in the CSV, keyed by review_id.
    reviews_dir.mkdir(parents=True, exist_ok=True)
    (reviews_dir / f"{review_id}.md").write_text(
        f"# Review {review_id} — task `{task_id}`\n\n## Task\n\n{task_text}\n\n"
        f"## Output\n\n{completion.text}\n",
        encoding="utf-8",
    )
    return RunResult(run_id, review_id, lane.name, cost, shadow, completion, credit)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", required=True, choices=sorted(LANES))
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--session", required=True)
    p.add_argument("--max-output", type=int, default=8000)
    p.add_argument("--require-window", choices=[rates.PEAK, rates.OFF_PEAK])
    p.add_argument("--override-session", metavar="REASON", help="pre-approve crossing the session ceiling")
    a = p.parse_args(argv)
    try:
        result = run_task(
            lane_name=a.lane,
            task_path=a.task,
            session_id=a.session,
            max_output_tokens=a.max_output,
            require_window=a.require_window,
            override_reason=a.override_session,
            confirm_override=tty_confirm,
        )
    except governor.SessionLimitError:
        print("Stopped at session limit. Nothing dispatched.", file=sys.stderr)
        return 3
    except governor.BudgetExceededError as e:
        print(f"HARD STOP: {e}. Not overridable.", file=sys.stderr)
        return 2
    real = f"${result.cost_usd:.6f}"
    shadow = (
        "" if result.shadow_cost_usd is None else f"  (free; would have cost ${result.shadow_cost_usd:.6f})"
    )
    credit = f"  (credit used ${result.credit_usd:.6f})" if result.credit_usd else ""
    head = f"run {result.run_id}  lane {result.lane}  review {result.review_id}"
    print(f"{head}  real cost {real}{shadow}{credit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
