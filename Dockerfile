# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel --outdir /dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the wheel built in stage 1
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Create runtime directories
RUN mkdir -p /app/data/raw /app/data/processed /app/outputs

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false absa
USER absa

ENTRYPOINT ["python", "-m"]
CMD ["absa.train.ate", "--help"]
