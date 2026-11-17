#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send many routing decision requests to a Switchyard server at the same time."""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class Result:
    """One decision request's outcome."""

    status: int | None
    latency_ms: float
    selected_target: str
    selected_model: str
    error: str


def positive_int(value: str) -> int:
    """Parse a command-line integer greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def send(url: str, body: bytes, session_id: str, timeout: float) -> Result:
    """Send one decision request and time it."""
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            "x-switchyard-session-id": session_id,
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decision = json.load(response)
            status = response.status
            selected = decision.get("selected") or {}
            selected_target = selected.get("target", "")
            selected_model = selected.get("model", "")
            error = "" if selected_model else "response has no selected.model"
    except urllib.error.HTTPError as http_error:
        detail = http_error.read().decode("utf-8", "replace")
        status = http_error.code
        selected_target = ""
        selected_model = ""
        error = detail[:200]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as failure:
        status = None
        selected_target = ""
        selected_model = ""
        error = str(failure)
    elapsed = (time.perf_counter() - started) * 1000
    return Result(status, elapsed, selected_target, selected_model, error)


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return a latency percentile from an already sorted list."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:4000/v1/decision")
    parser.add_argument("--model", default="switchyard/qlc", help="route id to decide for")
    parser.add_argument("--input-format", default="openai_chat")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument(
        "--weight-quality",
        type=float,
        default=None,
        help="override the configured quality weight",
    )
    parser.add_argument(
        "--weight-latency",
        type=float,
        default=None,
        help="override the configured latency weight",
    )
    parser.add_argument(
        "--weight-cost",
        type=float,
        default=None,
        help="override the configured cost weight",
    )
    parser.add_argument("--requests", type=positive_int, default=6)
    parser.add_argument("--concurrency", type=positive_int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    """Run the concurrent decision requests and print a summary."""
    args = parse_args()
    request: dict[str, object] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    weights = {
        "weight_quality": args.weight_quality,
        "weight_latency": args.weight_latency,
        "weight_cost": args.weight_cost,
    }
    request.update({key: value for key, value in weights.items() if value is not None})
    body = json.dumps(
        {
            "input_format": args.input_format,
            "request": request,
        }
    ).encode("utf-8")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(send, args.url, body, args.session_id, args.timeout)
            for _ in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started

    for index, result in enumerate(results, start=1):
        target = result.selected_target or "-"
        model = result.selected_model or "-"
        line = (
            f"#{index:<3} {result.status} {result.latency_ms:7.0f} ms "
            f"target={target} model={model}"
        )
        if result.error:
            line += f" error={result.error}"
        print(line)

    latencies = sorted(result.latency_ms for result in results)
    succeeded = sum(
        1 for result in results if result.status == 200 and result.selected_model
    )
    print()
    print(f"requests   {len(results)}")
    print(f"succeeded  {succeeded}")
    print(f"failed     {len(results) - succeeded}")
    print(f"wall       {wall_seconds:.2f} s")
    print(f"throughput {len(results) / wall_seconds:.2f} req/s")
    print(
        f"latency    min {latencies[0]:.0f} ms  "
        f"p50 {percentile(latencies, 0.50):.0f} ms  "
        f"p95 {percentile(latencies, 0.95):.0f} ms  "
        f"max {latencies[-1]:.0f} ms"
    )
    statuses = Counter(str(result.status) for result in results)
    print("statuses   " + "  ".join(f"{code}={count}" for code, count in statuses.items()))
    targets = Counter(result.selected_target for result in results if result.selected_target)
    print("targets    " + "  ".join(f"{name}={count}" for name, count in targets.items()))
    models = Counter(result.selected_model for result in results if result.selected_model)
    print("models     " + "  ".join(f"{name}={count}" for name, count in models.items()))
    if any(value is not None for value in weights.values()):
        applied = "  ".join(
            f"{key}={value}" for key, value in weights.items() if value is not None
        )
        print("weights    " + applied)
    return 0 if succeeded == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
