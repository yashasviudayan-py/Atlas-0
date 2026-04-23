# Atlas-0 production-ish container.
# Multi-stage build: Rust compilation + slim Python runtime.

# Stage 1: Build Rust crates
FROM rust:1.95-bookworm AS rust-builder

WORKDIR /build
COPY rust-toolchain.toml ./
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/

RUN cargo build --release

# Stage 2: Python runtime + Rust binaries
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/python \
    ATLAS_CONFIG=/app/configs/default.toml \
    ATLAS_UPLOADS_STORAGE_DIR=/data/uploads

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust binaries
COPY --from=rust-builder /build/target/release/atlas-* /usr/local/bin/

# Install Python dependencies
COPY pyproject.toml .
COPY python/ python/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "."

# Copy runtime assets used by the API, worker, sample report, and static app.
COPY configs/ configs/
COPY data/ data/
COPY frontend/ frontend/
COPY proto/ proto/
COPY scripts/ scripts/

RUN useradd --create-home --uid 10001 atlas \
    && mkdir -p /data/uploads /data/objects \
    && chown -R atlas:atlas /app /data

USER atlas

EXPOSE 8420
VOLUME ["/data"]

# Liveness probe: GET /health must return 200 within 5s.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health')"

CMD ["uvicorn", "atlas.api.server:app", "--host", "0.0.0.0", "--port", "8420"]
