"""Telemetry collection for MLCLI.

Privacy-first telemetry system for collecting usage data and improving suggestions.

Usage:
    from mlcli.core.telemetry import TelemetryCollector
    
    telemetry = TelemetryCollector(project_dir)
    telemetry.log_command("train", {"model": "xgboost"})
"""

from .collector import (
    TelemetryCollector,
    sanitize_data,
    generate_session_id,
)

__all__ = [
    "TelemetryCollector",
    "sanitize_data",
    "generate_session_id",
]
