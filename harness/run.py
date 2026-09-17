"""Dispatch one benchmark task to one lane, single-turn.

    python -m harness.run --lane deepseek-flash --task benchmarks/tasks/diagnose.md \
        --session 2026-09-18-baseline --require-window off_peak

Order: window -> invariant -> estimate -> authorize -> call -> cost -> log -> blind review packet.
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from harness import governor, rates, telemetry
from harness.client import LANES, Completion, Transport, complete, urllib_transport

STANDING_INSTRUCTION = (
    "You are working on a real codebase. For diagnose and investigate tasks: do not rewrite "
    "code. Understand and explain first. Cite files and line numbers for every claim."
)

TELEMETRY_PATH = Path("data/telemetry.csv")
REVIEWS_DIR = Path("reviews/pending")


class WindowMismatchError(RuntimeError):
    """The benchmark requires one rate window for all lanes (v3 §5)."""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    review_id: str
    cost_usd: float
    completion: Completion


def estimate_input_tokens(text: str) -> int:
    # Conservative: ~3 chars/token over-reserves against the ~4 typical for English/code.
    return len(text) // 3 + 1


def run_task(
    *,
    lane_name: str,
    task_path: Path,
    session_id: str,
    max_output_tokens: int = 8000,
    require_window: str | None = None,
    telemetry_path: Path = TELEMETRY_PATH,
    reviews_dir: Path = REVIEWS_DIR,
    transport: Transport = urllib_transport,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    env: dict[str, str] | None = None,
) -> RunResult:
    lane = LANES[lane_name]
    task_text = task_path.read_text(encoding="utf-8")
    task_id = task_path.stem
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
    )
    governor.authorize(
        lane.name,
        estimate,
        governor.Spend(
            month_usd=telemetry.month_spend(rows, dispatched.strftime("%Y-%m")),
            session_usd=telemetry.session_spend(rows, session_id),
            lane_usd=telemetry.lane_spend(rows, lane.name),
        ),
    )

    messages = [
        {"role": "system", "content": STANDING_INSTRUCTION},
        {"role": "user", "content": task_text},
    ]
    completion = complete(
        lane,
        messages,
        max_tokens=max_output_tokens,
        transport=transport,
        env=os.environ if env is None else env,
    )
    u = completion.usage
    cost = rates.cost_usd(
        lane.model_id,
        window,
        cache_hit_tokens=u.cache_hit_tokens,
        cache_miss_tokens=u.cache_miss_tokens,
        output_tokens=u.output_tokens,
    )

    run_id = uuid.uuid4().hex[:12]
    review_id = uuid.uuid4().hex[:8]
    telemetry.append(
        telemetry_path,
        {
            "timestamp_utc": dispatched.isoformat(timespec="seconds"),
            "run_id": run_id,
            "session_id": session_id,
            "task_id": task_id,
            "review_id": review_id,
            "lane": lane.name,
            "provider": lane.provider,
            "model_id": lane.model_id,
            "model_version": completion.served_model,
            "rate_window": window,
            "dispatch_hour_utc": dispatched.hour,
            "cache_hit_tokens": u.cache_hit_tokens,
            "cache_miss_tokens": u.cache_miss_tokens,
            "output_tokens": u.output_tokens,
            "reasoning_tokens": u.reasoning_tokens,
            "wall_clock_ms": completion.wall_clock_ms,
            "cost_usd": f"{cost:.6f}",
            "lane_spend_usd": f"{telemetry.lane_spend(rows, lane.name) + cost:.6f}",
            "status": "ok" if completion.finish_reason != "length" else "truncated",
            "error": "",
        },
    )

    # Blind packet: task and output only. Lane lives in the CSV, keyed by review_id.
    reviews_dir.mkdir(parents=True, exist_ok=True)
    (reviews_dir / f"{review_id}.md").write_text(
        f"# Review {review_id} — task `{task_id}`\n\n## Task\n\n{task_text}\n\n"
        f"## Output\n\n{completion.text}\n",
        encoding="utf-8",
    )
    return RunResult(run_id=run_id, review_id=review_id, cost_usd=cost, completion=completion)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", required=True, choices=sorted(LANES))
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--session", required=True)
    p.add_argument("--max-output", type=int, default=8000)
    p.add_argument("--require-window", choices=[rates.PEAK, rates.OFF_PEAK])
    a = p.parse_args(argv)
    result = run_task(
        lane_name=a.lane,
        task_path=a.task,
        session_id=a.session,
        max_output_tokens=a.max_output,
        require_window=a.require_window,
    )
    print(f"run {result.run_id}  review {result.review_id}  cost ${result.cost_usd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
