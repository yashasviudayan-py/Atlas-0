#!/usr/bin/env python3
"""Atlas-0 process manager.

Starts the full Atlas-0 stack in the correct order:

1. Rust SLAM pipeline (``cargo run --example live_slam --release``).
2. Waits for the shared-memory file to appear, signalling the SLAM
   pipeline is running and ready.
3. Python FastAPI world-model server (``uvicorn``).

Handles SIGINT / SIGTERM gracefully: sends SIGTERM to both child
processes and waits for them to exit.

Usage::

    python scripts/run_atlas.py [--config configs/custom.toml]
    python scripts/run_atlas.py --no-slam          # Python only
    python scripts/run_atlas.py --no-api           # Rust only
    python scripts/run_atlas.py --dev              # Rust debug build + uvicorn --reload

Environment::

    ATLAS_API_PORT      Override api.port from config.
    ATLAS_MMAP_PATH     Override ipc.mmap_path from config.
    ATLAS_VLM_*         Any ATLAS_ prefix overrides the config section.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so atlas package is importable.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from atlas.utils.config import load_config  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

_SLAM_EXAMPLE = "live_slam"
_API_MODULE = "atlas.api.server:app"
_MMAP_WAIT_TIMEOUT_S = 30.0
_MMAP_POLL_INTERVAL_S = 0.25

# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_cargo() -> str:
    """Return the path to cargo, raising if not found."""
    cargo = os.environ.get("CARGO", "cargo")
    result = subprocess.run([cargo, "--version"], capture_output=True)
    if result.returncode != 0:
        print(
            "[atlas] ERROR: 'cargo' not found. "
            "Install Rust from https://rustup.rs/ or set CARGO env var.",
            file=sys.stderr,
        )
        sys.exit(1)
    return cargo


def _start_slam(
    repo_root: Path,
    config_path: Path,
    release: bool,
) -> subprocess.Popen:  # type: ignore[type-arg]
    """Launch the Rust SLAM pipeline as a subprocess.

    Args:
        repo_root: Repository root directory.
        config_path: Path to TOML config file (passed via ATLAS_CONFIG env var).
        release: If ``True``, build with ``--release``.

    Returns:
        Running :class:`subprocess.Popen` handle.
    """
    cargo = _find_cargo()
    cmd = [cargo, "run", "--example", _SLAM_EXAMPLE]
    if release:
        cmd.append("--release")
    env = {**os.environ, "ATLAS_CONFIG": str(config_path)}
    print(f"[atlas] Starting SLAM: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=repo_root, env=env)


def _wait_for_mmap(mmap_path: Path, timeout_s: float) -> bool:
    """Poll until the shared-memory file appears or timeout expires.

    Args:
        mmap_path: Path to the mmap file created by the Rust SLAM writer.
        timeout_s: Maximum seconds to wait.

    Returns:
        ``True`` if the file appeared within *timeout_s*, ``False`` otherwise.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if mmap_path.exists():
            return True
        time.sleep(_MMAP_POLL_INTERVAL_S)
    return False


def _start_api(
    repo_root: Path,
    config_path: Path,
    host: str,
    port: int,
    reload: bool,
) -> subprocess.Popen:  # type: ignore[type-arg]
    """Launch the FastAPI world-model server via uvicorn.

    Args:
        repo_root: Repository root (added to PYTHONPATH).
        config_path: Path to TOML config file.
        host: Bind host for uvicorn.
        port: Bind port for uvicorn.
        reload: If ``True``, pass ``--reload`` (dev mode).

    Returns:
        Running :class:`subprocess.Popen` handle.
    """
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        _API_MODULE,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")

    python_path = str(repo_root / "python")
    env = {
        **os.environ,
        "PYTHONPATH": python_path,
        "ATLAS_CONFIG": str(config_path),
    }
    print(f"[atlas] Starting API: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=repo_root, env=env)


def _shutdown(
    procs: list[subprocess.Popen],  # type: ignore[type-arg]
    timeout_s: float = 5.0,
) -> None:
    """Send SIGTERM to all processes, then SIGKILL if they don't exit.

    Args:
        procs: List of running processes.
        timeout_s: Grace period before SIGKILL.
    """
    for proc in procs:
        if proc.poll() is None:
            print(f"[atlas] Sending SIGTERM to PID {proc.pid}", flush=True)
            proc.terminate()

    deadline = time.monotonic() + timeout_s
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[atlas] Force-killing PID {proc.pid}", flush=True)
            proc.kill()
            proc.wait()
    print("[atlas] All processes stopped.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse arguments and start the Atlas-0 stack."""
    parser = argparse.ArgumentParser(
        description="Atlas-0 process manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "default.toml",
        help="Path to TOML config file (default: configs/default.toml)",
    )
    parser.add_argument(
        "--no-slam",
        action="store_true",
        help="Skip starting the Rust SLAM pipeline (Python API only)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip starting the Python API server (Rust SLAM only)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: Rust debug build + uvicorn --reload",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mmap_path = Path(
        os.environ.get("ATLAS_MMAP_PATH", os.environ.get("ATLAS_IPC_MMAP_PATH", cfg.ipc.mmap_path))
    )

    procs: list[subprocess.Popen] = []  # type: ignore[type-arg]

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[atlas] Received signal {signum}, shutting down …", flush=True)
        _shutdown(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Start SLAM ────────────────────────────────────────────────────────────
    if not args.no_slam:
        slam_proc = _start_slam(
            repo_root=_REPO_ROOT,
            config_path=args.config,
            release=not args.dev,
        )
        procs.append(slam_proc)

        print(
            f"[atlas] Waiting up to {_MMAP_WAIT_TIMEOUT_S:.0f}s for "
            f"shared memory at {mmap_path} …",
            flush=True,
        )
        if not _wait_for_mmap(mmap_path, _MMAP_WAIT_TIMEOUT_S):
            print(
                f"[atlas] WARNING: shared memory file '{mmap_path}' did not appear "
                f"within {_MMAP_WAIT_TIMEOUT_S:.0f}s. "
                "Starting API anyway — it will degrade gracefully without SLAM data.",
                flush=True,
            )
        else:
            print(f"[atlas] Shared memory ready: {mmap_path}", flush=True)

    # ── Start API ─────────────────────────────────────────────────────────────
    if not args.no_api:
        api_proc = _start_api(
            repo_root=_REPO_ROOT,
            config_path=args.config,
            host=cfg.api.host,
            port=cfg.api.port,
            reload=args.dev,
        )
        procs.append(api_proc)

    if not procs:
        print("[atlas] Nothing to start (--no-slam and --no-api both set).", file=sys.stderr)
        sys.exit(1)

    print(f"[atlas] Stack running. PIDs: {[p.pid for p in procs]}. Press Ctrl+C to stop.")

    # ── Monitor loop ─────────────────────────────────────────────────────────
    try:
        while True:
            for proc in procs:
                rc = proc.poll()
                if rc is not None:
                    print(
                        f"[atlas] Process PID {proc.pid} exited with code {rc}. "
                        "Shutting down stack.",
                        flush=True,
                    )
                    _shutdown([p for p in procs if p is not proc])
                    sys.exit(rc)
            time.sleep(1.0)
    except KeyboardInterrupt:
        _shutdown(procs)


if __name__ == "__main__":
    main()
