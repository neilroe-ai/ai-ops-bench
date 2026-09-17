# AGENTS.md — ai-ops-bench

`CLAUDE.md` points here. This file is the single source of agent rules.

## What this repo is

A benchmark harness answering: **which model, for which work, at what cost per accepted task.**
Reasoning lives in the project doc `ai-coding-agent-strategy-v3.md` (+ addenda). Numbers live in code.

## Route

| You need to… | Go to |
|---|---|
| Change a price or peak window | `harness/rates.py` — cite the provider page + date in the PR |
| Change a spend ceiling or record a top-up | `harness/governor.py` — raise ceiling first, then top up |
| Add or swap a lane | `harness/client.py` — OpenAI-compatible, metered, BYOK only |
| Add a telemetry column | `harness/telemetry.py` + update tests |
| Record a model/subscription decision | `docs/model-decisions.md` |
| Run the weekly research cycle | `research/README.md` |

## Rules

- Stdlib only in `harness/`. Dev tools (ruff, mypy, pytest) via `uv`.
- `just check` passes before any PR. Every guard has a test that fails when the guard is removed; keep it that way.
- A new model enters as a **benchmark row**, never as an adoption. Only the CSV promotes a lane.
- Claude Pro (or any subscription) is never a lane and is never authenticated inside the harness.
- Keep review packets blind: task and output only, no lane or model names.
- Work on a feature branch and open a PR. Neil merges to `main`.
- Never commit `.env` or keys. Never enable auto top-up.
