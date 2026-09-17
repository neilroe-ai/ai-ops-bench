"""Offline tests. Each guard has a test that fails if the guard is removed."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from harness import governor, rates, report, run, telemetry
from harness.client import LANES, MissingKeyError, complete, parse_usage

# 2026-09-21 is a Monday.
MON = datetime(2026, 9, 21, tzinfo=UTC)


# ---- rates ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (MON.replace(hour=0, minute=59), rates.OFF_PEAK),
        (MON.replace(hour=1), rates.PEAK),
        (MON.replace(hour=3, minute=59), rates.PEAK),
        (MON.replace(hour=4), rates.OFF_PEAK),
        (MON.replace(hour=6), rates.PEAK),
        (MON.replace(hour=10), rates.OFF_PEAK),
        (MON + timedelta(days=5, hours=2), rates.OFF_PEAK),  # Saturday
    ],
)
def test_deepseek_windows(at: datetime, expected: str) -> None:
    assert rates.rate_window("deepseek", at) == expected


def test_window_converts_from_taipei() -> None:
    taipei = timezone(timedelta(hours=8))
    assert rates.rate_window("deepseek", datetime(2026, 9, 21, 9, 30, tzinfo=taipei)) == rates.PEAK
    assert rates.rate_window("deepseek", datetime(2026, 9, 21, 20, 0, tzinfo=taipei)) == rates.OFF_PEAK


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError):
        rates.rate_window("deepseek", datetime(2026, 9, 21, 2))


def test_unlisted_provider_is_flat() -> None:
    assert rates.rate_window("xai", MON) == rates.FLAT


def test_unpriced_model_raises_not_free() -> None:
    with pytest.raises(rates.UnpricedModelError):
        rates.cost_usd("glm-5.3-flash", rates.FLAT, cache_hit_tokens=0, cache_miss_tokens=1, output_tokens=1)


def test_cost_calculation() -> None:
    c = rates.cost_usd(
        "deepseek-flash",
        rates.OFF_PEAK,
        cache_hit_tokens=1_000_000,
        cache_miss_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert c == pytest.approx(0.003 + 0.15 + 0.60)


def test_peak_is_double_off_peak_for_deepseek() -> None:
    for model in ("deepseek-flash", "deepseek-v4-pro"):
        off, peak = rates.get_rate(model, rates.OFF_PEAK), rates.get_rate(model, rates.PEAK)
        assert peak.output == pytest.approx(off.output * 2)
        assert peak.cache_miss == pytest.approx(off.cache_miss * 2)


def test_every_priced_model_has_a_lane() -> None:
    assert {m for m, _ in rates.RATES} <= {lane.model_id for lane in LANES.values()}


# ---- governor ------------------------------------------------------------


def test_invariant_holds_for_committed_config() -> None:
    governor.check_invariant()


def test_invariant_fires_when_budget_reaches_balance() -> None:
    with pytest.raises(governor.BudgetExceededError):
        governor.check_invariant(monthly_budget=20.0, topups=[governor.Topup("deepseek", 20.0, "x")])


def test_per_lane_ceilings_fit_inside_provider_balance() -> None:
    balances = governor.provider_balances()
    by_provider: dict[str, float] = {}
    for lane, ceiling in governor.PER_LANE_BUDGET_USD.items():
        p = LANES[lane].provider
        by_provider[p] = by_provider.get(p, 0.0) + ceiling
    for provider, total in by_provider.items():
        assert total < balances.get(provider, 0.0), provider


@pytest.mark.parametrize(
    "spend",
    [
        governor.Spend(month_usd=14.99, session_usd=0, lane_usd=0),
        governor.Spend(month_usd=0, session_usd=1.99, lane_usd=0),
        governor.Spend(month_usd=0, session_usd=0, lane_usd=9.99),
    ],
)
def test_each_ceiling_refuses(spend: governor.Spend) -> None:
    with pytest.raises(governor.BudgetExceededError):
        governor.authorize("deepseek-flash", 0.02, spend)


def test_unfunded_lane_has_zero_ceiling() -> None:
    with pytest.raises(governor.BudgetExceededError):
        governor.authorize("grok-build-0.1", 0.0001, governor.Spend(0, 0, 0))


def test_authorize_passes_under_ceilings() -> None:
    governor.authorize("deepseek-flash", 0.01, governor.Spend(0, 0, 0))


# ---- client --------------------------------------------------------------


def test_parse_usage_deepseek_shape() -> None:
    u = parse_usage(
        {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 40,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
    )
    assert (u.cache_hit_tokens, u.cache_miss_tokens, u.output_tokens, u.reasoning_tokens) == (40, 60, 50, 20)


def test_parse_usage_openai_shape() -> None:
    u = parse_usage(
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 3}, "completion_tokens": 5}
    )
    assert (u.cache_hit_tokens, u.cache_miss_tokens, u.reasoning_tokens) == (3, 7, 0)


def test_missing_key_raises() -> None:
    with pytest.raises(MissingKeyError):
        complete(LANES["deepseek-flash"], [], max_tokens=1, env={})


# ---- run end to end (fake transport) --------------------------------------


def fake_transport(calls: list[dict[str, Any]]) -> Any:
    def transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> dict[str, Any]:
        calls.append({"url": url})
        return {
            "model": "deepseek-v4.1-flash-0910",
            "choices": [{"message": {"content": "diagnosis"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 0, "completion_tokens": 500},
        }

    return transport


def _task(tmp_path: Path) -> Path:
    t = tmp_path / "diagnose.md"
    t.write_text("Find the bug.", encoding="utf-8")
    return t


def test_run_logs_row_and_blind_packet(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    result = run.run_task(
        lane_name="deepseek-flash",
        task_path=_task(tmp_path),
        session_id="s1",
        telemetry_path=tmp_path / "t.csv",
        reviews_dir=tmp_path / "rev",
        transport=fake_transport(calls),
        now=lambda: MON.replace(hour=12),
        env={"DEEPSEEK_API_KEY": "k"},
    )
    rows = telemetry.read(tmp_path / "t.csv")
    assert len(calls) == 1 and len(rows) == 1
    row = rows[0]
    assert row["model_version"] == "deepseek-v4.1-flash-0910"
    assert row["rate_window"] == rates.OFF_PEAK and row["dispatch_hour_utc"] == "12"
    assert float(row["cost_usd"]) == pytest.approx(result.cost_usd)
    packet = (tmp_path / "rev" / f"{result.review_id}.md").read_text(encoding="utf-8")
    assert "deepseek" not in packet.lower()


def test_run_refused_makes_no_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(governor, "PER_LANE_BUDGET_USD", {"deepseek-flash": 0.0})
    calls: list[dict[str, Any]] = []
    with pytest.raises(governor.BudgetExceededError):
        run.run_task(
            lane_name="deepseek-flash",
            task_path=_task(tmp_path),
            session_id="s1",
            telemetry_path=tmp_path / "t.csv",
            reviews_dir=tmp_path / "rev",
            transport=fake_transport(calls),
            now=lambda: MON.replace(hour=12),
            env={"DEEPSEEK_API_KEY": "k"},
        )
    assert calls == [] and not (tmp_path / "t.csv").exists()


def test_window_requirement_blocks_peak_run(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    with pytest.raises(run.WindowMismatchError):
        run.run_task(
            lane_name="deepseek-flash",
            task_path=_task(tmp_path),
            session_id="s1",
            require_window=rates.OFF_PEAK,
            telemetry_path=tmp_path / "t.csv",
            reviews_dir=tmp_path / "rev",
            transport=fake_transport(calls),
            now=lambda: MON.replace(hour=2),
            env={"DEEPSEEK_API_KEY": "k"},
        )
    assert calls == []


def test_telemetry_rejects_wrong_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        telemetry.append(tmp_path / "t.csv", {"lane": "x"})


# ---- report --------------------------------------------------------------


def _row(review_id: str, task: str, cost: str, lane: str = "deepseek-flash") -> telemetry.Row:
    r = dict.fromkeys(telemetry.FIELDS, "")
    r.update(
        review_id=review_id,
        task_id=task,
        cost_usd=cost,
        lane=lane,
        provider="deepseek",
        timestamp_utc="2026-09-21T12:00:00+00:00",
    )
    return r


def test_cost_per_accepted_task_carries_corrections_and_separates_unreviewed() -> None:
    rows = [_row("a", "diagnose", "0.01"), _row("b", "diagnose", "0.02"), _row("c", "implement", "0.05")]
    verdicts = {
        "a": {"verdict": "rejected", "correction_rounds": "0"},
        "b": {"verdict": "accepted", "correction_rounds": "1"},
    }
    [r] = report.lane_matrix(rows, verdicts)
    assert r.accepted_tasks == 1 and r.unreviewed_runs == 1
    assert r.cost_per_accepted_task == pytest.approx(0.03)


def test_reconcile_uses_lifetime_spend() -> None:
    rows = [_row("a", "t", "1.5"), _row("b", "t", "0.5")]
    rows[1]["timestamp_utc"] = "2026-08-01T00:00:00+00:00"
    assert report.reconcile(rows)["deepseek"] == pytest.approx(18.0)


def test_verdicts_reader(tmp_path: Path) -> None:
    p = tmp_path / "v.csv"
    with p.open("w", newline="") as f:
        csv.writer(f).writerows(
            [
                ["review_id", "verdict", "correction_rounds", "reviewer", "notes"],
                ["a", "accepted", "0", "neil", ""],
            ]
        )
    assert report.read_verdicts(p)["a"]["verdict"] == "accepted"
