#!/usr/bin/env bash
# Local dev server: reload only agent/api, skip tests, cap graceful shutdown.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_LEVEL="$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"

exec .venv/bin/uvicorn api.main:app \
  --reload \
  --reload-dir agent \
  --reload-dir api \
  --reload-exclude 'tests/*' \
  --reload-exclude 'scratch/*' \
  --reload-exclude 'tmp_*' \
  --log-level "$LOG_LEVEL" \
  --timeout-graceful-shutdown 10 \
  --timeout-keep-alive 5 \
  --host 0.0.0.0 \
  --port 8080
