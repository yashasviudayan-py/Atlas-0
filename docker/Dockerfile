# Atlas-0 Development Container
# Multi-stage build: Rust compilation + Python runtime

# Stage 1: Build Rust crates
FROM rust:1.84-bookworm AS rust-builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/

RUN cargo build --release

# Stage 2: Python runtime + Rust binaries
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust binaries
COPY --from=rust-builder /build/target/release/atlas-* /usr/local/bin/

# Install Python dependencies
COPY pyproject.toml .
COPY python/ python/
RUN pip install --no-cache-dir -e ".[ml]"

# Copy configs, schemas, and frontend
COPY configs/ configs/
COPY proto/ proto/
COPY frontend/ frontend/

EXPOSE 8420

# Liveness probe: GET /health must return 200 within 5s.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health')"

CMD ["uvicorn", "atlas.api.server:app", "--host", "0.0.0.0", "--port", "8420"]
