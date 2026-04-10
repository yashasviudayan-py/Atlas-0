#!/usr/bin/env python3
"""Atlas-0 benchmark suite.

Measures and reports latencies for the four performance-sensitive
sub-systems defined in the Phase 2 specification:

1. **IPC read latency** — time to call ``SharedMemReader.get_latest_snapshot()``
   on a temp mmap file written by ``SharedMemWriter``.
2. **VLM inference latency** — wall-clock time for ``VLMEngine.label_region()``
   against a live Ollama instance (skipped if Ollama is unreachable).
3. **Spatial query latency** — FastAPI ``POST /query`` round-trip time using
   the in-process ``TestClient`` (no network overhead).
4. **Scene assessment latency** — single call to
   ``WorldModelAgent._assess_scene()`` with a mock VLM and a small mmap
   snapshot.

Results are printed to stdout as structured JSON and optionally written
to a file via ``--output``.

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 2000 --output results.json
    python scripts/benchmark.py --skip-vlm    # skip live Ollama test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np

# Ensure repo python/ is importable.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stats(samples: list[float]) -> dict[str, float]:
    """Compute mean, p50, p95, p99, min, max in milliseconds.

    Args:
        samples: Latency measurements in **seconds**.

    Returns:
        Dict with keys ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
        ``min_ms``, ``max_ms``.
    """
    arr = sorted(s * 1_000 for s in samples)
    n = len(arr)

    def _pct(p: float) -> float:
        idx = min(int(p / 100.0 * n), n - 1)
        return arr[idx]

    return {
        "mean_ms": sum(arr) / n,
        "p50_ms": _pct(50),
        "p95_ms": _pct(95),
        "p99_ms": _pct(99),
        "min_ms": arr[0],
        "max_ms": arr[-1],
        "samples": n,
    }


def _print_result(name: str, result: dict[str, Any]) -> None:
    """Pretty-print one benchmark result block.

    Args:
        name: Benchmark name.
        result: Stats dict from :func:`_stats` plus any extra fields.
    """
    status = result.get("status", "ok")
    mean = result.get("mean_ms", "—")
    p95 = result.get("p95_ms", "—")
    budget = result.get("budget_ms")
    budget_str = f"  budget={budget:.1f}ms" if budget else ""
    pass_fail = ""
    if budget and isinstance(mean, float):
        pass_fail = "  ✓ PASS" if mean <= budget else "  ✗ FAIL"

    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    if status != "ok":
        print(f"  status : {status}")
        if "reason" in result:
            print(f"  reason : {result['reason']}")
        return
    print(f"  mean   : {mean:.3f} ms{budget_str}{pass_fail}")
    print(f"  p50    : {result.get('p50_ms', 0):.3f} ms")
    print(f"  p95    : {p95:.3f} ms")
    print(f"  p99    : {result.get('p99_ms', 0):.3f} ms")
    print(f"  min    : {result.get('min_ms', 0):.3f} ms")
    print(f"  max    : {result.get('max_ms', 0):.3f} ms")
    print(f"  n      : {result.get('samples', 0)}")


# ── Benchmark 1: IPC read latency ─────────────────────────────────────────────


def bench_ipc(iterations: int) -> dict[str, Any]:
    """Measure SharedMemReader.get_latest_snapshot() latency.

    Creates a temp mmap file with 1 000 random Gaussians using
    ``SharedMemWriter``, then times ``iterations`` reads.

    Args:
        iterations: Number of read calls to time.

    Returns:
        Stats dict with ``budget_ms=5.0`` (Phase 2 spec target).
    """
    from atlas.utils.shared_mem import SharedMemReader, SharedMemWriter

    n_gaussians = 1_000
    with tempfile.NamedTemporaryFile(suffix=".mmap", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        writer = SharedMemWriter(tmp_path, max_gaussians=n_gaussians)
        gs = np.zeros(n_gaussians, dtype=SharedMemWriter.GAUSSIAN_DTYPE)
        rng = np.random.default_rng(42)
        gs["x"] = rng.standard_normal(n_gaussians).astype(np.float32)
        gs["y"] = rng.standard_normal(n_gaussians).astype(np.float32)
        gs["z"] = rng.standard_normal(n_gaussians).astype(np.float32)
        gs["opacity"] = rng.uniform(0.1, 1.0, n_gaussians).astype(np.float32)
        writer.write_snapshot(gs, pose=None, frame_id=1, timestamp_ns=int(time.time_ns()))
        writer.close()

        reader = SharedMemReader(tmp_path, max_gaussians=n_gaussians)
        # Warm-up
        for _ in range(10):
            reader.get_latest_snapshot()

        samples: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            reader.get_latest_snapshot()
            samples.append(time.perf_counter() - t0)
        reader.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    return {**_stats(samples), "budget_ms": 5.0, "gaussian_count": n_gaussians}


# ── Benchmark 2: VLM inference latency ────────────────────────────────────────


async def _bench_vlm_async(iterations: int) -> dict[str, Any]:
    """Async inner for VLM benchmark.

    Args:
        iterations: Number of ``label_region()`` calls.

    Returns:
        Stats dict or a skipped result if Ollama is unreachable.
    """
    import httpx
    from atlas.vlm.inference import VLMConfig, VLMEngine

    # Check if Ollama is reachable.
    ollama_host = os.environ.get("ATLAS_VLM_OLLAMA_HOST", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
    except Exception as exc:
        return {"status": "skipped", "reason": f"Ollama unreachable at {ollama_host}: {exc}"}

    cfg = VLMConfig(ollama_host=ollama_host)
    engine = VLMEngine(cfg)
    await engine.initialize()

    # 1x1 white pixel JPEG as minimal image payload.
    tiny_jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1eB"
        b"\xc4\x1b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xff\xd9"
    )

    # Warm-up
    await engine.label_region(tiny_jpeg)

    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await engine.label_region(tiny_jpeg)
        samples.append(time.perf_counter() - t0)

    await engine.close()
    return {**_stats(samples), "budget_ms": 2_000.0, "model": cfg.model_name}


def bench_vlm(iterations: int) -> dict[str, Any]:
    """Synchronous wrapper for the async VLM benchmark.

    Args:
        iterations: Number of inference calls.

    Returns:
        Stats dict or ``{"status": "skipped", ...}``.
    """
    return asyncio.run(_bench_vlm_async(iterations))


# ── Benchmark 3: Spatial query latency ────────────────────────────────────────


def bench_query(iterations: int) -> dict[str, Any]:
    """Measure POST /query round-trip via FastAPI TestClient.

    Pre-populates the agent with 10 mock objects and 3 risk entries, then
    fires ``iterations`` RISK-type queries.

    Args:
        iterations: Number of query requests.

    Returns:
        Stats dict with ``budget_ms=200.0``.
    """
    from atlas.api.server import _set_agent, app
    from atlas.vlm.inference import SemanticLabel
    from atlas.vlm.region_extractor import BoundingBox
    from atlas.world_model.agent import RiskEntry, WorldModelAgent, WorldModelConfig
    from atlas.world_model.relationships import SemanticObject
    from fastapi.testclient import TestClient

    agent = WorldModelAgent(config=WorldModelConfig(risk_threshold=0.1))
    label_store = agent.label_store

    for i in range(10):
        label_store.update(
            i,
            SemanticLabel(
                label=f"object_{i}",
                material="plastic",
                mass_kg=float(i + 1),
                fragility=0.1 * i,
                friction=0.5,
                confidence=0.9,
            ),
        )
        agent._cached_objects.append(
            SemanticObject(
                object_id=i,
                label=f"object_{i}",
                material="plastic",
                mass_kg=float(i + 1),
                fragility=0.1 * i,
                friction=0.5,
                confidence=0.9,
                bbox=BoundingBox(float(i), 0.0, 0.0, float(i) + 0.5, 0.5, 0.5),
            )
        )

    agent._risks = [
        RiskEntry(
            object_id=9,
            object_label="object_9",
            position=(9.0, 0.25, 0.25),
            risk_score=0.9,
            fragility=0.9,
            mass_kg=10.0,
            description="object_9 (plastic), fragile",
        ),
        RiskEntry(
            object_id=8,
            object_label="object_8",
            position=(8.0, 0.25, 0.25),
            risk_score=0.8,
            fragility=0.8,
            mass_kg=9.0,
            description="object_8 (plastic), fragile",
        ),
        RiskEntry(
            object_id=7,
            object_label="object_7",
            position=(7.0, 0.25, 0.25),
            risk_score=0.7,
            fragility=0.7,
            mass_kg=8.0,
            description="object_7 (plastic)",
        ),
    ]

    _set_agent(agent)

    with TestClient(app) as client:
        # Warm-up
        for _ in range(3):
            client.post("/query", json={"query": "What is the most unstable object?"})

        samples: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            resp = client.post(
                "/query", json={"query": "What is the most unstable object?", "max_results": 5}
            )
            samples.append(time.perf_counter() - t0)
            assert resp.status_code == 200, f"Unexpected status {resp.status_code}"

    return {**_stats(samples), "budget_ms": 200.0}


# ── Benchmark 4: Scene assessment latency ─────────────────────────────────────


async def _bench_assess_async(iterations: int) -> dict[str, Any]:
    """Async inner for scene assessment benchmark.

    Uses a real SharedMemWriter to write a 100-Gaussian snapshot, then
    patches VLMEngine to return a fixed label so the benchmark measures
    pure Python orchestration overhead (not Ollama inference).

    Args:
        iterations: Number of ``_assess_scene()`` calls.

    Returns:
        Stats dict.
    """
    from atlas.utils.shared_mem import SharedMemWriter
    from atlas.vlm.inference import SemanticLabel, VLMEngine
    from atlas.world_model.agent import WorldModelAgent, WorldModelConfig

    n_gaussians = 100
    with tempfile.NamedTemporaryFile(suffix=".mmap", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        writer = SharedMemWriter(tmp_path, max_gaussians=n_gaussians)
        rng = np.random.default_rng(0)
        gs = np.zeros(n_gaussians, dtype=SharedMemWriter.GAUSSIAN_DTYPE)
        gs["x"] = rng.standard_normal(n_gaussians).astype(np.float32)
        gs["y"] = rng.standard_normal(n_gaussians).astype(np.float32)
        gs["z"] = rng.standard_normal(n_gaussians).astype(np.float32)
        gs["opacity"] = rng.uniform(0.3, 1.0, n_gaussians).astype(np.float32)
        writer.write_snapshot(gs, pose=None, frame_id=1, timestamp_ns=int(time.time_ns()))
        writer.close()

        mock_label = SemanticLabel(
            label="box",
            material="cardboard",
            mass_kg=0.5,
            fragility=0.4,
            friction=0.6,
            confidence=0.85,
        )
        mock_vlm = AsyncMock(spec=VLMEngine)
        mock_vlm.label_region = AsyncMock(return_value=mock_label)
        mock_vlm.initialize = AsyncMock()
        mock_vlm.close = AsyncMock()

        agent = WorldModelAgent(
            config=WorldModelConfig(vlm_rate_limit_seconds=0.0),
            vlm_engine=mock_vlm,
        )

        os.environ["ATLAS_MMAP_PATH"] = str(tmp_path)

        # Warm-up
        for _ in range(2):
            await agent._assess_scene()

        samples: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            await agent._assess_scene()
            samples.append(time.perf_counter() - t0)

    finally:
        tmp_path.unlink(missing_ok=True)
        os.environ.pop("ATLAS_MMAP_PATH", None)

    return {**_stats(samples), "gaussian_count": n_gaussians}


def bench_assess(iterations: int) -> dict[str, Any]:
    """Synchronous wrapper for scene assessment benchmark.

    Args:
        iterations: Number of assessment cycles.

    Returns:
        Stats dict.
    """
    return asyncio.run(_bench_assess_async(iterations))


# ── Benchmark 5: Sample walkthrough report pipeline ──────────────────────────


async def _bench_sample_walkthrough_async(iterations: int) -> dict[str, Any]:
    """Measure the sample walkthrough upload-report pipeline."""
    from atlas.api.upload_analysis import analyze_frame_samples, analyze_image_heuristic
    from atlas.utils.video import ExtractedFrame

    fixture_root = _REPO_ROOT / "data" / "sample_walkthrough"
    expected_path = fixture_root / "expected_report.json"
    frame_paths = sorted((fixture_root / "frames").glob("*.jpg"))

    if not expected_path.exists() or not frame_paths:
        return {"status": "skipped", "reason": "sample walkthrough fixture is missing"}

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    frames = [
        ExtractedFrame(index=index, timestamp_s=index * 0.6, image_bytes=path.read_bytes())
        for index, path in enumerate(frame_paths)
    ]

    async def _labeler(content: bytes, _hint: str):
        return analyze_image_heuristic(content)

    await analyze_frame_samples(
        frames,
        filename=expected["fixture_name"],
        source_content_type="image/jpeg",
        labeler=_labeler,
    )

    samples: list[float] = []
    last_result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        last_result = await analyze_frame_samples(
            frames,
            filename=expected["fixture_name"],
            source_content_type="image/jpeg",
            labeler=_labeler,
        )
        samples.append(time.perf_counter() - t0)

    assert last_result is not None
    hazard_codes = [risk["hazard_code"] for risk in last_result.risks]
    fixture_match = (
        last_result.scene_source == expected["scene_source"]
        and len(last_result.objects) >= expected["min_object_count"]
        and len(last_result.risks) >= expected["min_hazard_count"]
        and hazard_codes[0] == expected["top_hazard_code"]
        and all(code in hazard_codes for code in expected["required_hazard_codes"])
    )

    return {
        **_stats(samples),
        "budget_ms": 500.0,
        "fixture_match": fixture_match,
        "object_count": len(last_result.objects),
        "hazard_count": len(last_result.risks),
    }


def bench_sample_walkthrough(iterations: int) -> dict[str, Any]:
    """Synchronous wrapper for the sample walkthrough report benchmark."""
    return asyncio.run(_bench_sample_walkthrough_async(iterations))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and run all benchmarks, emitting JSON results."""
    parser = argparse.ArgumentParser(
        description="Atlas-0 benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Iterations per benchmark (default: 500)",
    )
    parser.add_argument(
        "--vlm-iterations",
        type=int,
        default=5,
        help="Iterations for VLM benchmark (default: 5, slow)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM inference benchmark (requires live Ollama)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this file",
    )
    args = parser.parse_args()

    print("Atlas-0 Benchmark Suite")
    print("=" * 60)

    results: dict[str, Any] = {}

    # 1. IPC read latency
    print("\n[1/5] IPC snapshot read …", flush=True)
    results["ipc_read"] = bench_ipc(args.iterations)
    _print_result("IPC: SharedMemReader.get_latest_snapshot()", results["ipc_read"])

    # 2. VLM inference latency
    if args.skip_vlm:
        results["vlm_inference"] = {"status": "skipped", "reason": "--skip-vlm flag"}
        _print_result("VLM: label_region() — SKIPPED", results["vlm_inference"])
    else:
        print("\n[2/5] VLM inference (requires Ollama) …", flush=True)
        results["vlm_inference"] = bench_vlm(args.vlm_iterations)
        _print_result("VLM: VLMEngine.label_region()", results["vlm_inference"])

    # 3. Spatial query latency
    print("\n[3/5] Spatial query (TestClient) …", flush=True)
    results["spatial_query"] = bench_query(args.iterations)
    _print_result("API: POST /query", results["spatial_query"])

    # 4. Scene assessment latency
    print("\n[4/5] Scene assessment (mock VLM) …", flush=True)
    results["scene_assessment"] = bench_assess(min(args.iterations, 100))
    _print_result("Agent: _assess_scene() [mock VLM]", results["scene_assessment"])

    print("\n[5/5] Sample walkthrough report pipeline …", flush=True)
    results["sample_walkthrough_report"] = bench_sample_walkthrough(min(args.iterations, 20))
    _print_result(
        "Upload: sample walkthrough -> hazard report",
        results["sample_walkthrough_report"],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    budget_checks = {
        "ipc_read": ("IPC read", 5.0),
        "spatial_query": ("Spatial query", 200.0),
        "sample_walkthrough_report": ("Sample walkthrough report", 500.0),
    }
    all_pass = True
    for key, (label, budget) in budget_checks.items():
        r = results.get(key, {})
        mean = r.get("mean_ms")
        if mean is None:
            status = "SKIP"
        elif mean <= budget:
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {label:<25} {status}  (mean={mean:.3f}ms, budget={budget:.1f}ms)")

    sample = results.get("sample_walkthrough_report", {})
    if sample.get("status", "ok") == "ok":
        fixture_status = "PASS" if sample.get("fixture_match") else "FAIL"
        if fixture_status == "FAIL":
            all_pass = False
        print(f"  {'Fixture match':<25} {fixture_status}")

    vlm = results.get("vlm_inference", {})
    if vlm.get("status") == "ok":
        mean = vlm.get("mean_ms", 0)
        budget = 2000.0
        status = "PASS" if mean <= budget else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {'VLM inference':<25} {status}  (mean={mean:.0f}ms, budget={budget:.0f}ms)")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES — check above'}")

    # ── JSON output ───────────────────────────────────────────────────────────
    json_out = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(json_out)
        print(f"\n  Results written to: {args.output}")
    else:
        print(f"\n  JSON results:\n{json_out}")


if __name__ == "__main__":
    main()
