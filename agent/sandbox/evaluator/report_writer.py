"""Write evaluator reports to local files or Google Cloud Storage."""

from __future__ import annotations

from agent.sandbox.models import RepoExecutionReport
from agent.sandbox.report_store import SandboxReportStore, parse_gcs_uri


def write_report(report: RepoExecutionReport, destination: str) -> None:
    """Write a report to ``destination``.

    ``destination`` may be a local path or a ``gs://bucket/object.json`` URI.
    """
    SandboxReportStore().write(report, destination)


_parse_gcs_uri = parse_gcs_uri
