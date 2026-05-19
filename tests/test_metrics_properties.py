"""Property tests for Metrics Collector.

Tests Properties 22 and 23.
"""

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.database_manager import DatabaseManager
from app.metrics_collector import MetricsCollector


@pytest.fixture
def collector():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    db.initialize()
    mc = MetricsCollector(db)
    yield mc, db, path


# Feature: jarvis-assistant, Property 22: Metrics recording preserves event data
class TestProperty22:
    """Recording an LLM call should be reflected in the summary totals."""

    @given(
        duration_ms=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_metrics_recording_preserves_data(self, duration_ms):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        mc = MetricsCollector(db)
        try:
            mc.record_llm_call(duration_ms=duration_ms, success=True, session_id="s1")
            summary = mc.get_summary()
            assert summary["total_calls"] >= 1
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 23: Tool usage metrics are correctly categorized
class TestProperty23:
    """Recording a tool call with a specific name should appear in tool_usage breakdown."""

    @given(
        tool_name=st.sampled_from(["run_command", "web_search", "network_scan", "calendar", "email", "notes"]),
    )
    @settings(max_examples=100)
    def test_tool_usage_categorization(self, tool_name):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db = DatabaseManager(db_path=path)
        db.initialize()
        mc = MetricsCollector(db)
        try:
            mc.record_tool_call(tool_name=tool_name, duration_ms=50.0, success=True, session_id="s1")
            summary = mc.get_summary()
            assert tool_name in summary["tool_usage"]
            assert summary["tool_usage"][tool_name] >= 1
        finally:
            db.close()
            os.unlink(path)
