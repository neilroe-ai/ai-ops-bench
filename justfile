# ai-ops-bench — `just setup && just check`

setup:
    uv sync

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy harness tests
    uv run pytest -q

fmt:
    uv run ruff format .

run lane task session:
    uv run python -m harness.run --lane {{lane}} --task {{task}} --session {{session}} --require-window off_peak

report:
    uv run python -m harness.report

reconcile:
    uv run python -m harness.report reconcile
