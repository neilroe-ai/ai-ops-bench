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

# List NVIDIA-hosted model IDs (run on the TUF; confirms the free-lane IDs in rates.py/client.py)
nvidia-models:
    curl -s -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data'] if any(k in m['id'].lower() for k in ('deepseek','glm','kimi','nemotron','devstral','qwen'))]"
