# Local dev server: reload only agent/api, skip tests, cap graceful shutdown.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$logLevel = if ($env:LOG_LEVEL) { $env:LOG_LEVEL.ToLower() } else { "info" }

& .\.venv\Scripts\uvicorn.exe api.main:app `
    --reload `
    --reload-dir agent `
    --reload-dir api `
    --log-level $logLevel `
    --timeout-graceful-shutdown 10 `
    --timeout-keep-alive 5 `
    --host 0.0.0.0 `
    --port 8080
