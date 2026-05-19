"""Metrics Collector for Jarvis Assistant.

Tracks usage metrics in-process and persists them to SQLite.
Provides aggregated summaries for the KPI dashboard.
"""

import logging
import time
from typing import Optional

from app.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and persists usage metrics.

    Records LLM API calls, tool invocations, and errors.
    Provides aggregated summaries for the dashboard.

    Attributes:
        db_manager: Database manager for persistent metric storage.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def record_llm_call(
        self,
        duration_ms: float,
        success: bool,
        session_id: Optional[str] = None,
    ) -> None:
        """Record an LLM API call with its duration and outcome.

        Args:
            duration_ms: Time taken for the LLM call in milliseconds.
            success: Whether the call succeeded.
            session_id: Optional session identifier.
        """
        try:
            self.db_manager.save_metric_event(
                event_type="llm_call",
                tool_name=None,
                duration_ms=duration_ms,
                success=success,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Failed to record LLM call metric: %s", e)

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        session_id: Optional[str] = None,
    ) -> None:
        """Record a tool invocation metric.

        Args:
            tool_name: Name of the tool (command, search, scan, calendar, email, notes).
            duration_ms: Time taken for the tool call in milliseconds.
            success: Whether the call succeeded.
            session_id: Optional session identifier.
        """
        try:
            self.db_manager.save_metric_event(
                event_type="tool_call",
                tool_name=tool_name,
                duration_ms=duration_ms,
                success=success,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Failed to record tool call metric: %s", e)

    def record_error(
        self,
        error_type: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Record an error event.

        Args:
            error_type: Description of the error type.
            session_id: Optional session identifier.
        """
        try:
            self.db_manager.save_metric_event(
                event_type="error",
                tool_name=error_type,
                duration_ms=None,
                success=False,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Failed to record error metric: %s", e)

    def get_summary(self) -> dict:
        """Return aggregated metrics summary for the dashboard.

        Returns:
            Dict with: total_calls, calls_today, avg_response_ms,
            p95_response_ms, tool_usage, error_rate, active_sessions.
        """
        try:
            return self.db_manager.get_metrics_summary()
        except Exception as e:
            logger.error("Failed to get metrics summary: %s", e)
            return {
                "total_calls": 0,
                "calls_today": 0,
                "avg_response_ms": 0.0,
                "p95_response_ms": 0.0,
                "tool_usage": {},
                "error_rate": 0.0,
                "active_sessions": 0,
            }

    def get_daily_breakdown(self, days: int = 7) -> list[dict]:
        """Return per-day metrics for the last N days.

        Args:
            days: Number of days to look back.

        Returns:
            List of dicts with: date, total_calls, avg_response_ms, error_count.
        """
        try:
            return self.db_manager.get_daily_breakdown(days=days)
        except Exception as e:
            logger.error("Failed to get daily breakdown: %s", e)
            return []
