"""Run a lightweight API load smoke test against Atlas-0.

The default mode uses FastAPI's in-process TestClient so CI can validate core
request paths without starting a server. Pass ``--base-url`` to exercise a
running deployment with the same request mix.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


@dataclass(frozen=True)
class RequestResult:
    """One load-smoke request result."""

    name: str
    status_code: int
    duration_ms: float

    @property
    def ok(self) -> bool:
        """Return whether the response status is considered successful."""
        return 200 <= self.status_code < 300


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="", help="Optional running Atlas-0 base URL.")
    parser.add_argument("--requests", type=int, default=60, help="Total requests to execute.")
    parser.add_argument("--concurrency", type=int, default=8, help="Maximum concurrent workers.")
    parser.add_argument("--p95-budget-ms", type=float, default=250.0, help="Allowed p95 latency.")
    return parser


def _local_client() -> Any:
    from atlas.api.server import app
    from fastapi.testclient import TestClient

    return TestClient(app)


def _request_local(index: int) -> RequestResult:
    client = _local_client()
    started = time.perf_counter()
    if index % 3 == 0:
        response = client.get("/health")
        name = "health"
    elif index % 3 == 1:
        response = client.post("/query", json={"query": "where is the cup?", "max_results": 3})
        name = "query"
    else:
        response = client.post(
            "/product/events",
            json={"event_name": "cta_start_scan", "surface": "load_smoke"},
        )
        name = "product_event"
    return RequestResult(name, response.status_code, (time.perf_counter() - started) * 1000)


def _request_remote(base_url: str, index: int) -> RequestResult:
    import httpx

    started = time.perf_counter()
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        if index % 3 == 0:
            response = client.get("/health")
            name = "health"
        elif index % 3 == 1:
            response = client.post("/query", json={"query": "where is the cup?", "max_results": 3})
            name = "query"
        else:
            response = client.post(
                "/product/events",
                json={"event_name": "cta_start_scan", "surface": "load_smoke"},
            )
            name = "product_event"
    return RequestResult(name, response.status_code, (time.perf_counter() - started) * 1000)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[rank]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    total = max(1, int(args.requests))
    workers = max(1, min(int(args.concurrency), total))
    request_fn = (
        (lambda index: _request_remote(args.base_url.rstrip("/"), index))
        if args.base_url
        else _request_local
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(request_fn, range(total)))

    failures = [result for result in results if not result.ok]
    durations = [result.duration_ms for result in results]
    p95 = _percentile(durations, 95.0)
    mean = statistics.fmean(durations)
    by_name: dict[str, int] = {}
    for result in results:
        by_name[result.name] = by_name.get(result.name, 0) + 1

    print("Atlas-0 API load smoke")
    print(f"requests={total} concurrency={workers} mean_ms={mean:.2f} p95_ms={p95:.2f}")
    print("mix=" + ", ".join(f"{name}:{count}" for name, count in sorted(by_name.items())))

    if failures:
        print(f"FAIL: {len(failures)} request(s) returned non-2xx responses.")
        return 1
    if p95 > float(args.p95_budget_ms):
        print(f"FAIL: p95 {p95:.2f}ms exceeded budget {args.p95_budget_ms:.2f}ms.")
        return 1
    print("PASS: load smoke stayed within the latency budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
