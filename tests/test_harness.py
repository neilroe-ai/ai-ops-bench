"""Offline tests. Each guard has a test that fails if the guard is removed."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from harness import governor, rates, report, run, telemetry
from harness.client import LANES, MissingKeyError, TransportFailure, complete, parse_usage

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
    with pytest.raises((governor.BudgetExceededError, governor.SessionLimitError)):
        governor.authorize("deepseek-flash", 0.02, spend)


def test_session_limit_is_soft_and_distinct() -> None:
    spend = governor.Spend(month_usd=0, session_usd=1.99, lane_usd=0)
    with pytest.raises(governor.SessionLimitError):
        governor.authorize("deepseek-flash", 0.02, spend)
    governor.authorize("deepseek-flash", 0.02, spend, session_override=True)


@pytest.mark.parametrize(
    "spend",
    [
        governor.Spend(month_usd=14.99, session_usd=5, lane_usd=0),
        governor.Spend(month_usd=0, session_usd=5, lane_usd=9.99),
    ],
)
def test_override_never_bypasses_hard_ceilings(spend: governor.Spend) -> None:
    with pytest.raises(governor.BudgetExceededError):
        governor.authorize("deepseek-flash", 0.02, spend, session_override=True)


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


# ---- session alert / override ---------------------------------------------


def _seed_spend(path: Path, cost: str, session: str, at: datetime) -> None:
    r = dict.fromkeys(telemetry.FIELDS, "")
    r.update(
        timestamp_utc=at.isoformat(),
        session_id=session,
        lane="deepseek-flash",
        provider="deepseek",
        cost_usd=cost,
        status="ok",
    )
    telemetry.append(path, r)


def _run(tmp_path: Path, calls: list[dict[str, Any]], **kw: Any) -> run.RunResult:
    args: dict[str, Any] = dict(
        lane_name="deepseek-flash",
        task_path=_task(tmp_path),
        session_id="s-new",
        telemetry_path=tmp_path / "t.csv",
        reviews_dir=tmp_path / "rev",
        transport=fake_transport(calls),
        now=lambda: MON.replace(hour=12),
        env={"DEEPSEEK_API_KEY": "k", "NVIDIA_API_KEY": "k"},
    )
    args.update(kw)
    return run.run_task(**args)


def test_session_limit_pauses_and_stops_without_reason(tmp_path: Path) -> None:
    _seed_spend(tmp_path / "t.csv", "1.999", "s-old", MON.replace(hour=11))  # other name: still counts
    calls: list[dict[str, Any]] = []
    prompts: list[str] = []

    def decline(message: str) -> None:
        prompts.append(message)

    with pytest.raises(governor.SessionLimitError):
        _run(tmp_path, calls, confirm_override=decline)
    assert len(prompts) == 1 and calls == []


def test_session_override_proceeds_and_is_logged(tmp_path: Path) -> None:
    _seed_spend(tmp_path / "t.csv", "1.999", "s-old", MON.replace(hour=11))
    calls: list[dict[str, Any]] = []
    _run(tmp_path, calls, confirm_override=lambda m: "finishing diagnose run")
    assert len(calls) == 1
    assert telemetry.read(tmp_path / "t.csv")[-1]["override_reason"] == "finishing diagnose run"


def test_session_window_expires(tmp_path: Path) -> None:
    _seed_spend(tmp_path / "t.csv", "1.999", "s-old", MON.replace(hour=5))  # 7h earlier
    calls: list[dict[str, Any]] = []
    _run(tmp_path, calls)
    assert len(calls) == 1


# ---- NVIDIA free lanes: shadow cost never becomes real money --------------


def test_free_lane_real_cost_zero_shadow_separate(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    result = _run(tmp_path, calls, lane_name="nvidia-deepseek-v4-pro")
    rows = telemetry.read(tmp_path / "t.csv")
    assert result.cost_usd == 0.0 and result.shadow_cost_usd and result.shadow_cost_usd > 0
    assert "integrate.api.nvidia.com" in calls[0]["url"]
    assert telemetry.month_spend(rows, "2026-09") == 0.0
    assert telemetry.lane_spend(rows, "nvidia-deepseek-v4-pro") == 0.0
    assert telemetry.window_spend(rows, MON.replace(hour=13), 6) == 0.0
    assert report.reconcile(rows).get("nvidia", 0.0) == 0.0
    [lane_report] = report.lane_matrix(rows, {})
    assert lane_report.shadow_cost_usd == pytest.approx(result.shadow_cost_usd)


def test_every_free_lane_is_priced_at_zero() -> None:
    for lane in LANES.values():
        if lane.free:
            assert rates.get_rate(lane.model_id, rates.FLAT) == rates.FREE
            assert lane.shadow_of in LANES and not LANES[lane.shadow_of].free


def throttling_transport(calls: list[dict[str, Any]]) -> Any:
    ok = fake_transport(calls)

    def transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> dict[str, Any]:
        if "nvidia" in url:
            calls.append({"url": url})
            raise TransportFailure("HTTP 429")
        result: dict[str, Any] = ok(url, headers, body, timeout)
        return result

    return transport


def test_throttle_falls_back_to_paid_lane(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    result = _run(tmp_path, calls, lane_name="nvidia-deepseek-v4-pro", transport=throttling_transport(calls))
    rows = telemetry.read(tmp_path / "t.csv")
    assert [r["status"] for r in rows] == ["throttled", "ok"]
    assert result.lane == "deepseek-v4-pro" and rows[1]["fallback_from"] == "nvidia-deepseek-v4-pro"
    assert result.cost_usd > 0


def test_breaker_skips_free_lane_after_repeated_throttles(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    for m in (1, 2, 3):
        r = dict.fromkeys(telemetry.FIELDS, "")
        r.update(
            timestamp_utc=MON.replace(hour=11, minute=50 + m).isoformat(),
            lane="nvidia-deepseek-v4-pro",
            provider="nvidia",
            cost_usd="0",
            status="throttled",
        )
        telemetry.append(path, r)
    calls: list[dict[str, Any]] = []
    result = _run(tmp_path, calls, lane_name="nvidia-deepseek-v4-pro")
    assert all("nvidia" not in c["url"] for c in calls)
    assert result.lane == "deepseek-v4-pro"


# ---- Gemini promotional credit: third money type ---------------------------

from harness.client import CASH, FREE, PROMOTIONAL  # noqa: E402


def test_rate_card_switches_on_1_jan_2027() -> None:
    late = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)
    new = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
    assert rates.get_rate("gemini-3.8-flash", rates.FLAT, late).output == 3.75
    assert rates.get_rate("gemini-3.8-flash", rates.FLAT, new).output == 7.50
    assert rates.rate_card_date("gemini-3.8-flash", new) == "2027-01-01"
    assert rates.rate_card_date("deepseek-flash", new) == ""


def test_dated_card_requires_dispatch_time() -> None:
    with pytest.raises(rates.RateCardDateError):
        rates.get_rate("gemini-3.8-flash", rates.FLAT)


def test_every_lane_has_a_known_billing_mode() -> None:
    for lane in LANES.values():
        assert lane.billing in (CASH, FREE, PROMOTIONAL)
        if lane.billing == PROMOTIONAL:
            assert lane.name in governor.CREDIT_LANE_CEILING_USD
            assert lane.name in governor.CREDIT_MONTHLY_CEILING_USD


def test_credit_invariants_hold_for_committed_config() -> None:
    governor.check_credit_invariants()


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("CREDIT_LANE_CEILING_USD", {"gemini-3.8-flash": 10_000.0}),  # would drain promo, then prepay
        ("CREDIT_MONTHLY_CEILING_USD", {"gemini-3.8-flash": 999.0}),  # above Tier 1 cap
        ("PREPAY_BALANCE_TWD", {"google": 100.0}),  # below floor: credit would go inert
    ],
)
def test_credit_invariants_fire(monkeypatch: pytest.MonkeyPatch, attr: str, value: dict[str, float]) -> None:
    monkeypatch.setattr(governor, attr, value)
    with pytest.raises(governor.CreditExhaustedError):
        governor.check_credit_invariants()


def gemini_transport(calls: list[dict[str, Any]]) -> Any:
    def transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> dict[str, Any]:
        calls.append({"url": url, "body": body})
        return {
            "model": "gemini-3.8-flash",
            "choices": [{"message": {"content": "diagnosis"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "completion_tokens_details": {"reasoning_tokens": 200},
            },
        }

    return transport


@pytest.fixture(autouse=True)
def _unpark_gemini_for_credit_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini is parked (MD-005); the credit machinery is still tested against an enabled copy."""
    if "credit" in request.node.name or "expired" in request.node.name:
        import dataclasses

        lanes = dict(LANES)
        lanes["gemini-3.8-flash"] = dataclasses.replace(LANES["gemini-3.8-flash"], enabled=True)
        monkeypatch.setattr(run, "LANES", lanes)


def test_parked_lane_refuses_and_makes_no_call(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    with pytest.raises(run.LaneDisabledError):
        _gemini(tmp_path, calls)
    assert calls == []


def _gemini(tmp_path: Path, calls: list[dict[str, Any]], **kw: Any) -> run.RunResult:
    return _run(
        tmp_path,
        calls,
        lane_name="gemini-3.8-flash",
        transport=gemini_transport(calls),
        env={"GEMINI_API_KEY": "k"},
        **kw,
    )


def test_credit_run_spends_credit_not_cash(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    result = _gemini(tmp_path, calls)
    rows = telemetry.read(tmp_path / "t.csv")
    row = rows[-1]
    assert b'"reasoning_effort": "medium"' in calls[0]["body"]
    assert result.cost_usd == 0.0 and result.credit_usd == pytest.approx((1000 * 0.75 + 500 * 3.75) / 1e6)
    assert row["billing_mode"] == PROMOTIONAL and row["rate_card_date"] == "2026-01-01"
    assert telemetry.month_spend(rows, "2026-09") == 0.0
    assert report.reconcile(rows).get("google", 0.0) == 0.0
    assert telemetry.credit_lane_spend(rows, "gemini-3.8-flash") == pytest.approx(result.credit_usd)
    assert report.reconcile_credits(rows)["google"] == pytest.approx(
        governor.credit_balance_usd("google") - result.credit_usd
    )


def test_credit_ceiling_is_hard_and_makes_no_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(governor, "CREDIT_LANE_CEILING_USD", {"gemini-3.8-flash": 0.0})
    calls: list[dict[str, Any]] = []
    with pytest.raises(governor.CreditExhaustedError):
        _gemini(tmp_path, calls, override_reason="please")
    assert calls == []


def test_expired_credit_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expired = (governor.CreditGrant("google", 7924.50, MON.date()),)
    monkeypatch.setattr(governor, "CREDIT_GRANTS", expired)
    calls: list[dict[str, Any]] = []
    with pytest.raises(governor.CreditExhaustedError):
        _gemini(tmp_path, calls)
    assert calls == []


def test_cost_per_accepted_task_counts_credit_at_list_price() -> None:
    r = _row("a", "diagnose", "0", lane="gemini-3.8-flash")
    r["credit_usd"] = "0.05"
    [lane_report] = report.lane_matrix([r], {"a": {"verdict": "accepted", "correction_rounds": "0"}})
    assert lane_report.cost_per_accepted_task == pytest.approx(0.05)
