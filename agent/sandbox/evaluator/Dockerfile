FROM python:3.12-slim-bookworm

ARG GITLEAKS_VERSION=8.30.1
ARG SCC_VERSION=3.7.0
ARG HADOLINT_VERSION=2.14.0
ARG TRIVY_VERSION=0.71.0

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
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ARCH="$(dpkg --print-architecture)" \
    && case "${ARCH}" in \
        amd64) GITLEAKS_ARCH="x64"; SCC_ARCH="x86_64"; HADOLINT_ARCH="x86_64"; TRIVY_ARCH="64bit";; \
        arm64) GITLEAKS_ARCH="arm64"; SCC_ARCH="arm64"; HADOLINT_ARCH="arm64"; TRIVY_ARCH="ARM64";; \
        *) echo "Unsupported architecture: ${ARCH}" && exit 1;; \
    esac \
    && curl -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${GITLEAKS_ARCH}.tar.gz" \
        | tar -xz -C /usr/local/bin gitleaks \
    && curl -fsSL "https://github.com/boyter/scc/releases/download/v${SCC_VERSION}/scc_Linux_${SCC_ARCH}.tar.gz" \
        | tar -xz -C /usr/local/bin scc \
    && curl -fsSL "https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-linux-${HADOLINT_ARCH}" \
        -o /usr/local/bin/hadolint \
    && chmod +x /usr/local/bin/hadolint \
    && curl -fsSL "https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-${TRIVY_ARCH}.tar.gz" \
        | tar -xz -C /usr/local/bin trivy \
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
        "checkov>=3.2.0" \
        "interrogate>=1.7.0" \
        "pip-audit>=2.7.3" \
        "semgrep>=1.91.0" \
        "tree-sitter>=0.21.3" \
        "tree-sitter-languages>=1.10.2" \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m", "agent.sandbox.evaluator.main"]
