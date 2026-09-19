#!/usr/bin/env python3
"""Latency probe for NVIDIA-hosted lanes.

Runs one fixed prompt N times against each candidate model, measuring
time-to-first-token and total wall time. Writes research/latency-probe.csv.

Usage:  python3 scripts/probe_latency.py [reps]
Env:    NVIDIA_API_KEY
"""
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import UTC, datetime

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEY = os.environ.get("NVIDIA_API_KEY")

MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4-flash-0731",
    "moonshotai/kimi-k3",
]

PROMPT = (
    "Summarise the following in exactly two sentences.\n\n"
    "A queue worker acknowledges the webhook with HTTP 200 before doing any work, "
    "then enqueues the job. A second process drains the queue. If the reply token is "
    "still valid the reply is free; if it has expired the worker falls back to a push "
    "message, which is metered and capped at 80 percent of the monthly quota."
)

OUT = os.path.join("research", "latency-probe.csv")
FIELDS = ["ts", "model", "rep", "ok", "ttft_ms", "total_ms", "completion_tokens", "error"]


def probe(model: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 200,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    t0 = time.monotonic()
    ttft = None
    tokens = 0
    # S310 is suppressed below: URL is the module constant at the top of this
    # file, never caller-supplied, so no file: or custom scheme can reach here.
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                tokens = usage["completion_tokens"]
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    tokens = tokens or 0
    total = time.monotonic() - t0
    return {
        "ttft_ms": round(ttft * 1000) if ttft else "",
        "total_ms": round(total * 1000),
        "completion_tokens": tokens or "",
    }


def main() -> int:
    if not KEY:
        print("NVIDIA_API_KEY is not set", file=sys.stderr)
        return 1
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    results: dict[str, list[float]] = {}

    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for model in MODELS:
            for rep in range(1, reps + 1):
                row = {
                    "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                    "model": model,
                    "rep": rep,
                    "ok": 0,
                    "ttft_ms": "",
                    "total_ms": "",
                    "completion_tokens": "",
                    "error": "",
                }
                try:
                    row.update(probe(model))
                    row["ok"] = 1
                    results.setdefault(model, []).append(float(row["total_ms"]))
                except Exception as exc:  # the probe records failures, never raises
                    row["error"] = f"{type(exc).__name__}: {exc}"[:200]
                w.writerow(row)
                fh.flush()
                print(
                    f"{model:42s} rep {rep}  "
                    f"ttft {row['ttft_ms'] or '-':>6}ms  "
                    f"total {row['total_ms'] or '-':>6}ms  {row['error']}"
                )
                time.sleep(1)

    print(f"\nmedian total_ms ({reps} reps)  ->  {OUT}")
    for model in MODELS:
        vals = results.get(model)
        if vals:
            print(f"  {model:42s} {round(statistics.median(vals)):>6} ms   n={len(vals)}")
        else:
            print(f"  {model:42s}      -   all failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
