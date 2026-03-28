#!/usr/bin/env python3
"""Regenerate Python protobuf stubs from proto/atlas.proto.

Requires ``grpcio-tools`` (listed in the dev extra of pyproject.toml).
Run from the workspace root::

    uv run python scripts/gen_proto.py

Output is written to ``python/atlas/utils/atlas_pb2.py``.
"""

import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
PROTO_DIR = WORKSPACE_ROOT / "proto"
OUT_DIR = WORKSPACE_ROOT / "python" / "atlas" / "utils"


def main() -> None:
    proto_file = PROTO_DIR / "atlas.proto"
    if not proto_file.exists():
        print(f"ERROR: proto file not found: {proto_file}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        str(proto_file),
    ]
    print("Running:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "ERROR: protoc failed. Install grpcio-tools: pip install grpcio-tools",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    generated = OUT_DIR / "atlas_pb2.py"
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
