"""The CSV. Canonical record through the lane comparison (telemetry decision §5).

The same file that enforces the budget proves the cost claims.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

FIELDS: tuple[str, ...] = (
    "timestamp_utc",
    "run_id",
    "session_id",
    "task_id",
    "review_id",
    "lane",
    "provider",
    "model_id",
    "model_version",  # served model string: floating aliases like deepseek-flash drift
    "rate_window",
    "dispatch_hour_utc",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "output_tokens",
    "reasoning_tokens",
    "wall_clock_ms",
    "cost_usd",
    "lane_spend_usd",
    "status",
    "error",
)

Row = dict[str, str]


def append(path: Path, row: Mapping[str, object]) -> None:
    missing = set(FIELDS) - set(row)
    extra = set(row) - set(FIELDS)
    if missing or extra:
        raise ValueError(f"row mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in FIELDS})


def read(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _total(rows: Iterable[Row]) -> float:
    return sum(float(r["cost_usd"] or 0) for r in rows)


def month_spend(rows: Iterable[Row], yyyy_mm: str) -> float:
    return _total(r for r in rows if r["timestamp_utc"].startswith(yyyy_mm))


def session_spend(rows: Iterable[Row], session_id: str) -> float:
    return _total(r for r in rows if r["session_id"] == session_id)


def lane_spend(rows: Iterable[Row], lane: str) -> float:
    return _total(r for r in rows if r["lane"] == lane)


def lifetime_by_provider(rows: Iterable[Row]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for r in rows:
        totals[r["provider"]] = totals.get(r["provider"], 0.0) + float(r["cost_usd"] or 0)
    return totals
