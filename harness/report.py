"""Cost per accepted task, and reconciliation.

    python -m harness.report            # lane matrix
    python -m harness.report reconcile  # expected balance per provider vs console

Verdicts file (reviews/verdicts.csv): review_id,verdict,correction_rounds,reviewer,notes
verdict is "accepted" or "rejected". Correction runs reuse the task_id, so their cost
is carried into the accepted task's total.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from harness import governor, telemetry

VERDICTS_PATH = Path("reviews/verdicts.csv")


@dataclass(frozen=True)
class LaneReport:
    lane: str
    runs: int
    reviewed_runs: int
    unreviewed_runs: int
    accepted_tasks: int
    reviewed_cost_usd: float
    cost_per_accepted_task: float | None
    mean_correction_rounds: float | None


def read_verdicts(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["review_id"]: r for r in csv.DictReader(f)}


def lane_matrix(rows: list[telemetry.Row], verdicts: dict[str, dict[str, str]]) -> list[LaneReport]:
    out: list[LaneReport] = []
    for lane in sorted({r["lane"] for r in rows}):
        lane_rows = [r for r in rows if r["lane"] == lane]
        reviewed = [r for r in lane_rows if r["review_id"] in verdicts]
        accepted_tasks = {r["task_id"] for r in reviewed if verdicts[r["review_id"]]["verdict"] == "accepted"}
        # Cost of every reviewed run on a task that was eventually accepted, plus failures.
        cost = sum(float(r["cost_usd"]) for r in reviewed)
        rounds = [int(verdicts[r["review_id"]].get("correction_rounds") or 0) for r in reviewed]
        out.append(
            LaneReport(
                lane=lane,
                runs=len(lane_rows),
                reviewed_runs=len(reviewed),
                unreviewed_runs=len(lane_rows) - len(reviewed),
                accepted_tasks=len(accepted_tasks),
                reviewed_cost_usd=cost,
                cost_per_accepted_task=cost / len(accepted_tasks) if accepted_tasks else None,
                mean_correction_rounds=sum(rounds) / len(rounds) if rounds else None,
            )
        )
    return out


def reconcile(rows: list[telemetry.Row]) -> dict[str, float]:
    """topped up - lifetime spend = expected balance. Lifetime, never month-to-date."""
    spent = telemetry.lifetime_by_provider(rows)
    topped = governor.provider_balances()
    return {p: topped.get(p, 0.0) - spent.get(p, 0.0) for p in sorted(set(topped) | set(spent))}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    rows = telemetry.read(Path("data/telemetry.csv"))
    if args[:1] == ["reconcile"]:
        for provider, balance in reconcile(rows).items():
            print(f"{provider:10s} expected balance ${balance:.4f}  (compare to console)")
        return 0
    for r in lane_matrix(rows, read_verdicts(VERDICTS_PATH)):
        cpat = "n/a" if r.cost_per_accepted_task is None else f"${r.cost_per_accepted_task:.5f}"
        rounds = "n/a" if r.mean_correction_rounds is None else f"{r.mean_correction_rounds:.2f}"
        print(
            f"{r.lane:18s} runs={r.runs} reviewed={r.reviewed_runs} "
            f"UNREVIEWED={r.unreviewed_runs} accepted={r.accepted_tasks} "
            f"cost/accepted={cpat} rounds={rounds}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
