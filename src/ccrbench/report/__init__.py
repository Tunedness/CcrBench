"""JSON and HTML reports, the report model and the CI exit status."""

from ccrbench.report.exit_status import exit_status, suspected_exceeds_limit
from ccrbench.report.html_report import render_html, write_html
from ccrbench.report.json_report import (
    build_report,
    load_schema_file,
    render_json,
    report_exit_code,
    report_json_schema,
    write_json,
)
from ccrbench.report.models import (
    SCHEMA_VERSION,
    ProblemCard,
    Report,
    ReportError,
    ReportWarning,
    RunSources,
    SkippedVariantInfo,
    SuiteInfo,
    Summary,
    ThresholdsUsed,
    ThresholdValue,
    WarningCode,
)

__all__ = [
    "SCHEMA_VERSION",
    "ProblemCard",
    "Report",
    "ReportError",
    "ReportWarning",
    "RunSources",
    "SkippedVariantInfo",
    "SuiteInfo",
    "Summary",
    "ThresholdValue",
    "ThresholdsUsed",
    "WarningCode",
    "build_report",
    "exit_status",
    "load_schema_file",
    "render_html",
    "render_json",
    "report_exit_code",
    "report_json_schema",
    "suspected_exceeds_limit",
    "write_html",
    "write_json",
]
