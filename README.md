# ai-ops-bench

Which model, for which work, at what cost per **accepted** task — and how fast can that answer change when prices move.

A small, stdlib-only Python harness that dispatches real coding tasks to metered, OpenAI-compatible model lanes, enforces spend ceilings before dispatch, logs every call to CSV, and produces blind review packets for a human gate.

## Quick start

```bash
just setup && just check
cp .env.example .env   # add keys; auto top-up OFF on every provider
set -a; . ./.env; set +a
just run deepseek-flash benchmarks/tasks/diagnose.md 2026-09-baseline
just report
just reconcile          # compare to provider console
```

## Layout

| Path | Owns |
|---|---|
| `harness/rates.py` | Prices keyed on `(model_id, rate_window)`; peak windows. Unpriced → raises |
| `harness/governor.py` | Monthly / session / per-lane ceilings; top-ups; budget < balances invariant |
| `harness/client.py` | Lanes: base_url, model_id, key env, explicit thinking settings |
| `harness/telemetry.py` | CSV schema incl. `model_version`, `rate_window`, `reasoning_tokens`, `wall_clock_ms` |
| `harness/run.py` | One single-turn task → one lane → one row + blind review packet |
| `harness/report.py` | Cost per accepted task (unreviewed separated); lifetime reconciliation |
| `benchmarks/tasks/` | Diagnose, investigate, implement (real Mu Surf tasks) |
| `reviews/` | `pending/` blind packets; `verdicts.csv` human verdicts |
| `research/` | Weekly research cycle procedure, prompt, briefs |
| `docs/model-decisions.md` | Decision log: Ignore · Watch · Benchmark row · Replace |

## What is and isn't checked

| Checked by `just check` | Not checked |
|---|---|
| Rate windows, cost math, unpriced-model refusal | That rates match live provider pages (weekly cycle) |
| Every ceiling refuses; budget < balances; lane ceilings < provider balance | Real API responses (fake transport only) |
| Refused dispatch makes no call and logs nothing | Unverified lanes: GLM, Grok, Nemotron IDs/URLs |
| Blind packets contain no lane name | Review quality |

## Licence

MIT
