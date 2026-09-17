Run the ai-ops-bench weekly research cycle for Neil (repo: github.com/neilroe-ai/ai-ops-bench).

1. Read `research/README.md`, `harness/rates.py`, `harness/client.py`, `harness/governor.py`, `docs/model-decisions.md`.
2. Part A: verify every lane against live provider pricing/model pages. Use today's date.
3. Part B: scan the landscape for new models, price changes and subscriptions since the last brief. Apply the filter in order.
4. Write `research/briefs/<today>.md` in the brief format. Keep it under one page; cite every price with URL.
5. If rates, windows or model IDs changed: on branch `research/<today>`, edit `harness/rates.py` (and `client.py` if needed), run `just check`, open a PR. Never push to main. Never edit governor ceilings.
6. Do not add rows to `docs/model-decisions.md` — propose them in the brief; Neil decides.
7. Finish with a short message to Neil: changes to our lanes, top candidates with proposed outcome, decisions needed.
