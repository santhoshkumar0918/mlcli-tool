"""Privacy-first telemetry collection for MLCLI.

Implements local-only event logging with PII filtering and opt-in cloud sync.

Usage:
    from mlcli.core.telemetry import TelemetryCollector
    
    telemetry = TelemetryCollector(project_dir)
    
    # Log events
    telemetry.log_command("preprocess", {"input": "data.csv"})
    
    # Track suggestions
    session_id = telemetry.log_suggestions_shown(suggestions)
    telemetry.log_suggestion_acted(session_id, suggestion_index=0)
"""

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


PII_FIELDS = {
    "email", "password", "name", "address", "phone", "ssn", "credit_card",
    "api_key", "secret", "token", "credential", "user", "username"
}


def sanitize_data(data: Dict[str, Any], max_depth: int = 5) -> Dict[str, Any]:
    """Remove PII and sanitize data for logging.
    
    Args:
        data: Dictionary to sanitize
        max_depth: Maximum recursion depth
        
    Returns:
        Sanitized dictionary
    """
    if max_depth <= 0:
        return {"_truncated": True}
    
    sanitized = {}
    
    for key, value in data.items():
        key_lower = key.lower()
        
        if any(pii in key_lower for pii in PII_FIELDS):
            sanitized[key] = "[REDACTED]"
            continue
        
        if isinstance(value, dict):
            sanitized[key] = sanitize_data(value, max_depth - 1)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_data(v, max_depth - 1) if isinstance(v, dict) else v
                for v in value[:10]
            ]
        elif isinstance(value, str) and len(value) > 500:
            sanitized[key] = value[:500] + "...[truncated]"
        else:
            sanitized[key] = value
    
    return sanitized


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return uuid.uuid4().hex[:12]


def anonymize_project_path(path: str) -> str:
    """Anonymize file paths by removing user-specific parts."""
    parts = Path(path).parts
    anonymized = []
    for i, part in enumerate(parts):
        if part in ("home", "Users", "Users", "data", "tmp"):
            anonymized.append("~")
        else:
            anonymized.append(part)
    return str(Path(*anonymized))


class TelemetryCollector:
    """Privacy-first telemetry collection system.
    
    All data is stored locally by default. Cloud sync requires explicit opt-in.
    
    Features:
    - Local-only by default
    - PII filtering
    - Anonymous project fingerprinting
    - Suggestion → action tracking
    
    Example:
        telemetry = TelemetryCollector(project_dir)
        
        # Log a command execution
        telemetry.log_command("train", {"model": "xgboost", "accuracy": 0.92})
        
        # Track suggestion interactions
        session_id = telemetry.log_suggestions_shown([
            {"suggestion": "FEATURE_ENGINEERING", "confidence": 0.85}
        ])
        telemetry.log_suggestion_acted(session_id, 0, "executed")
    """
    
    TELEMETRY_DIR = ".mlcli/telemetry"
    EVENTS_FILE = "events.jsonl"
    OUTCOMES_FILE = "outcomes.jsonl"
    CONFIG_FILE = "telemetry_config.json"
    
    def __init__(
        self,
        project_dir: Path,
        enabled: bool = True,
        session_id: Optional[str] = None
    ):
        """Initialize telemetry collector.
        
        Args:
            project_dir: Root directory of the MLCLI project
            enabled: Whether telemetry is enabled
            session_id: Optional session ID (auto-generated if None)
        """
        self.project_dir = Path(project_dir)
        self.enabled = enabled
        
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = generate_session_id()
        
        self.telemetry_dir = self.project_dir / self.TELEMETRY_DIR
        self.events_path = self.telemetry_dir / self.EVENTS_FILE
        self.outcomes_path = self.telemetry_dir / self.OUTCOMES_FILE
        self.config_path = self.telemetry_dir / self.CONFIG_FILE
        
        self.project_fingerprint = self._compute_project_fingerprint()
        
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load telemetry configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "enabled": self.enabled,
            "cloud_sync": False,
            "collect_commands": True,
            "collect_suggestions": True,
            "collect_outcomes": True,
        }
    
    def _save_config(self) -> None:
        """Save telemetry configuration."""
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)
    
    def _compute_project_fingerprint(self) -> str:
        """Compute anonymous project fingerprint."""
        data = f"{self.project_dir.name}:{self.session_id}"
        return sha256(data.encode()).hexdigest()[:16]
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write an event to the event log.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.enabled or not self._config.get("enabled", True):
            return
        
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "project_fingerprint": self.project_fingerprint,
            "event_type": event_type,
            "data": sanitize_data(data),
        }
        
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    
    def log_command(
        self,
        command: str,
        params: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log a command execution.
        
        Args:
            command: Command name (e.g., "preprocess", "train")
            params: Command parameters
            duration_seconds: Execution time
            success: Whether command succeeded
            error: Error message if failed
        """
        if not self._config.get("collect_commands", True):
            return
        
        data = {
            "command": command,
            "params": params or {},
            "duration_seconds": duration_seconds,
            "success": success,
        }
        
        if error:
            data["error"] = error[:200]
        
        self._log_event("command", data)
    
    def log_suggestions_shown(
        self,
        suggestions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log when suggestions are shown to user.
        
        Args:
            suggestions: List of suggestions with confidence scores
            context: Additional context (data profile summary, etc.)
            
        Returns:
            Session ID for tracking follow-up actions
        """
        if not self._config.get("collect_suggestions", True):
            return self.session_id
        
        data = {
            "suggestions": [
                {
                    "suggestion": s.get("suggestion", "unknown"),
                    "confidence": s.get("confidence", 0.0),
                }
                for s in suggestions[:5]
            ],
            "context": sanitize_data(context or {}),
        }
        
        self._log_event("suggestions_shown", data)
        
        return self.session_id
    
    def log_suggestion_acted(
        self,
        session_id: str,
        suggestion_index: int,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log when user acts on a suggestion.
        
        Args:
            session_id: Session ID from log_suggestions_shown
            suggestion_index: Index of the suggestion acted on
            action: Action taken (executed, dismissed, viewed, etc.)
            details: Additional details about the action
        """
        if not self._config.get("collect_suggestions", True):
            return
        
        data = {
            "suggestion_session_id": session_id,
            "suggestion_index": suggestion_index,
            "action": action,
            "details": details or {},
        }
        
        self._log_event("suggestion_action", data)
    
    def log_outcome(
        self,
        artifact_id: str,
        metric_before: float,
        metric_after: float,
        metric_name: str = "accuracy",
        action_taken: Optional[str] = None
    ) -> None:
        """Log the outcome of an action.
        
        Args:
            artifact_id: ID of the affected artifact
            metric_before: Metric value before action
            metric_after: Metric value after action
            metric_name: Name of the metric
            action_taken: Description of action taken
        """
        if not self._config.get("collect_outcomes", True):
            return
        
        outcome = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "project_fingerprint": self.project_fingerprint,
            "artifact_id": artifact_id,
            "metric_name": metric_name,
            "metric_before": metric_before,
            "metric_after": metric_after,
            "metric_delta": metric_after - metric_before,
            "improvement": metric_after > metric_before,
            "action_taken": action_taken,
        }
        
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.outcomes_path, "a") as f:
            f.write(json.dumps(outcome, default=str) + "\n")
    
    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read recent events from the log.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of events (newest first)
        """
        if not self.events_path.exists():
            return []
        
        events = []
        with open(self.events_path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        return list(reversed(events[-limit:]))
    
    def get_outcomes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read recent outcomes from the log.
        
        Args:
            limit: Maximum number of outcomes to return
            
        Returns:
            List of outcomes (newest first)
        """
        if not self.outcomes_path.exists():
            return []
        
        outcomes = []
        with open(self.outcomes_path) as f:
            for line in f:
                if line.strip():
                    outcomes.append(json.loads(line))
        
        return list(reversed(outcomes[-limit:]))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics.
        
        Returns:
            Dictionary with telemetry stats
        """
        events = self.get_events(limit=10000)
        outcomes = self.get_outcomes(limit=10000)
        
        command_counts = {}
        for event in events:
            if event.get("event_type") == "command":
                cmd = event.get("data", {}).get("command", "unknown")
                command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        suggestion_actions = {"executed": 0, "dismissed": 0, "viewed": 0}
        for event in events:
            if event.get("event_type") == "suggestion_action":
                action = event.get("data", {}).get("action", "unknown")
                if action in suggestion_actions:
                    suggestion_actions[action] += 1
        
        improvements = sum(1 for o in outcomes if o.get("improvement", False))
        regressions = len(outcomes) - improvements
        
        return {
            "total_events": len(events),
            "total_outcomes": len(outcomes),
            "command_counts": command_counts,
            "suggestion_actions": suggestion_actions,
            "improvements": improvements,
            "regressions": regressions,
            "improvement_rate": improvements / max(1, len(outcomes)),
        }
    
    def clear(self) -> None:
        """Clear all telemetry data."""
        if self.events_path.exists():
            os.remove(self.events_path)
        if self.outcomes_path.exists():
            os.remove(self.outcomes_path)
    
    def enable(self) -> None:
        """Enable telemetry collection."""
        self.enabled = True
        self._config["enabled"] = True
        self._save_config()
    
    def disable(self) -> None:
        """Disable telemetry collection."""
        self.enabled = False
        self._config["enabled"] = False
        self._save_config()
    
    def export_for_training(self) -> Dict[str, Any]:
        """Export telemetry data for Meta-ML training.
        
        Returns:
            Dictionary with training-ready data
        """
        events = self.get_events(limit=10000)
        outcomes = self.get_outcomes(limit=10000)
        
        suggestion_outcomes = []
        for outcome in outcomes:
            matching_events = [
                e for e in events
                if e.get("event_type") == "suggestions_shown"
                and e.get("session_id") == outcome.get("session_id")
            ]
            
            if matching_events:
                suggestion_event = matching_events[0]
                suggestion_outcomes.append({
                    "suggestions": suggestion_event.get("data", {}).get("suggestions", []),
                    "action_taken": outcome.get("action_taken"),
                    "metric_delta": outcome.get("metric_delta"),
                    "improvement": outcome.get("improvement"),
                })
        
        return {
            "project_fingerprint": self.project_fingerprint,
            "suggestion_outcomes": suggestion_outcomes,
            "total_events": len(events),
            "total_outcomes": len(outcomes),
        }
