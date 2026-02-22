# MLCLI - Production Docker Image
# Multi-stage build for minimal image size

# ============================================
# Builder Stage
# ============================================
FROM python:3.11-slim as builder

LABEL stage="builder"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY pyproject.toml setup.py MANIFEST.in ./
COPY mlcli/ ./mlcli/
COPY README.md LICENSE ./

RUN pip install --upgrade pip setuptools wheel
RUN pip install "." --no-deps
RUN pip install \
    typer[all]>=0.9.0 \
    rich>=13.0.0 \
    pydantic>=2.0.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    xgboost>=1.7.0 \
    pyyaml>=6.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    httpx>=0.24.0 \
    joblib>=1.3.0 \
    numpy>=1.24.0

# ============================================
# Production Stage
# ============================================
FROM python:3.11-slim as production

LABEL maintainer="MLCLI Team" \
      org.opencontainers.image.title="MLCLI" \
      org.opencontainers.image.description="ML Framework for Production" \
      org.opencontainers.image.url="https://github.com/mlcli/mlcli" \
      org.opencontainers.image.source="https://github.com/mlcli/mlcli" \
      org.opencontainers.image.vendor="MLCLI" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/home/mlcli/workspace"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --shell /bin/bash --uid 1000 mlcli

USER mlcli
WORKDIR /home/mlcli/workspace

RUN mkdir -p /home/mlcli/workspace/data/raw

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD mlcli --version || exit 1

EXPOSE 3000

ENTRYPOINT ["mlcli"]
CMD ["--help"]
