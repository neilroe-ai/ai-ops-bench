# Model & subscription decisions

One row per decision. Outcomes: **Ignore · Watch · Add as benchmark row · Replace now**.
"Replace now" requires our own CSV evidence (cost per accepted task), never a rate card or leaderboard.

| # | Date | Item | Outcome | Why | Evidence |
|---|---|---|---|---|---|
| MD-001 | 2026-09-17 | DeepSeek V4.1-Flash (`deepseek-flash`) | Add as benchmark row — replaces V4 Pro as baseline | ~3.3× cheaper than V4 Pro on every price column; vendor claims better coding, unverified | api-docs.deepseek.com/quick_start/pricing; strategy v3 addendum 01 |
| MD-002 | 2026-09-17 | DeepSeek V4 Pro | Keep as one-task control, then drop | Tests the "Flash beats Pro" claim directly | addendum 01 §2.3 |
| MD-003 | 2026-09-17 | NVIDIA build.nvidia.com free tier (DeepSeek V4 Pro, GLM-5.1, Nemotron 3 Ultra) | Add as benchmark rows — try free first, fall back to paid | $0 real cost, rate-limited best effort; measured as cost avoided (shadow cost), never as spend. Model IDs unverified | build.nvidia.com; v3 §3, §6 |
| MD-004 | 2026-09-17 | Gemini 3.8 Flash on Google AI Studio promotional credit (~NT$7,924 + NT$354 prepay) | Add as benchmark row; fund the agentic run. Not inspector, not trivial bulk | Credit is a deadline wearing a discount. List price $0.75/$3.75 in 2026, $1.50/$7.50 from 1 Jan 2027 — 2027 quotes use 2027 card | claude/gemini-credit-assessment.md; ai.google.dev/gemini-api/docs/pricing (read 2026-09-17) |
