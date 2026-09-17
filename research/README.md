# Weekly research cycle

Runs every Monday as a scheduled task (prompt: `research/weekly-prompt.md`).
Output: one brief at `research/briefs/YYYY-MM-DD.md`, plus a PR if `rates.py` needs changing.
Neil decides; the decision is logged in `docs/model-decisions.md`.

The defence against price churn is a cheap cost of being wrong, not more research (v3 §2).
So the cycle is short, filtered, and ends in a decision.

## Part A — our lanes

For every lane in `harness/client.py`:

1. Live rates per window vs `harness/rates.py`. Any difference → PR with provider URL + date.
2. Peak windows vs `PEAK_WINDOWS`.
3. Model IDs, aliases, new versions, deprecations. Floating alias moved? Flag it (telemetry `model_version`).
   NVIDIA retires free models without notice (410 Gone): check each `nvidia-*` model ID against the
   NVIDIA catalogue / deprecation notices; propose the replacement ID. Neil can confirm with `just nvidia-models`.
4. Unverified lanes (`verified=False`): can base_url / model_id / thinking parameter now be confirmed?
5. Promotional prices with expiry dates (e.g. Gemini 3.7 Flash, 1 Jan 2027).
6. Governor invariant still holds against recorded top-ups.

## Part B — the landscape

Scan: DeepSeek, Z.ai (GLM), xAI, Google, Anthropic, OpenAI, Moonshot (Kimi), Alibaba (Qwen), NVIDIA, Mistral, plus notable open-weight releases and hosts.

Each candidate passes the filter in order; stop at the first fail:

1. **Metered, BYOK, OpenAI-compatible API?** No → Ignore (Pi can't call it; cost per task unmeasurable).
2. **Coding-capable at workforce or trivial tier?** Published coding benchmarks are priors only.
3. **Cheaper per task than the lane it would compete with?** Account for verbosity/reasoning tokens, not just rate card.
4. **Subscriptions only** — the three standing tests (v3 §7): pooled allowance kills cost-per-task; fixed fee vs $40 budget; consumer plan ≠ API access.

## Brief format

```
# Research brief YYYY-MM-DD
## Part A — lanes (changes only; "no change" if none)
## Part B — candidates
| Item | Filter result | Proposed outcome | Why (one line) | Source |
## Decisions needed from Neil
```

Outcomes: Ignore · Watch · Add as benchmark row · Replace now (CSV evidence only).
