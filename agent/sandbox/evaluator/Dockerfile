FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        coreutils \
        curl \
        findutils \
        git \
        golang-go \
        maven \
        nodejs \
        npm \
        openjdk-17-jdk-headless \
        cargo \
        rustc \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY agent/__init__.py /app/agent/__init__.py
COPY agent/config.py /app/agent/config.py
COPY agent/sandbox /app/agent/sandbox

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        "google-auth>=2.35.0" \
        "google-cloud-storage>=2.18.0" \
        "httpx>=0.27.0" \
        "pydantic>=2.9.0" \
        "pydantic-settings>=2.6.0" \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m", "agent.sandbox.evaluator.main"]
