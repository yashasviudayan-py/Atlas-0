"""Run a detached Atlas-0 upload worker against durable queued jobs."""

from __future__ import annotations

import argparse
import asyncio

from atlas.api import server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim and process at most one queued job, then exit.",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional stable worker identifier for durable job claims.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    agent = server._get_agent()
    if hasattr(agent, "start"):
        await agent.start()
    try:
        await server.run_detached_upload_worker(
            worker_id=args.worker_id,
            once=bool(args.once),
        )
        return 0
    finally:
        if hasattr(agent, "stop"):
            await agent.stop()


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
