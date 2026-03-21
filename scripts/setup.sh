#!/usr/bin/env bash
# Atlas-0 Development Environment Setup
set -euo pipefail

echo "=== Atlas-0 Setup ==="

# Check prerequisites
command -v cargo >/dev/null 2>&1 || { echo "Error: Rust/Cargo not found. Install from https://rustup.rs"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: Python3 not found."; exit 1; }

echo "1/4 Building Rust workspace..."
cargo build --workspace

echo "2/4 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

echo "3/4 Running Rust tests..."
cargo test --workspace

echo "4/4 Running Python tests..."
pytest python/tests/ -v

echo ""
echo "=== Setup complete ==="
echo "Activate the Python env with: source .venv/bin/activate"
