"""Property tests for System Monitor.

Tests Property 24.
"""

import os
import platform

import pytest

from app.system_monitor import SystemMonitor


# Feature: jarvis-assistant, Property 24: System metrics return valid ranges
class TestProperty24:
    """System metrics should return values in valid ranges."""

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="System monitor reads from /proc which is Linux-only",
    )
    def test_cpu_in_valid_range(self):
        monitor = SystemMonitor()
        # First call initializes counters
        metrics = monitor.get_metrics()
        # Second call gives meaningful delta
        metrics = monitor.get_metrics()
        assert -1 <= metrics.cpu_percent <= 100

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="System monitor reads from /proc which is Linux-only",
    )
    def test_ram_used_less_than_total(self):
        monitor = SystemMonitor()
        metrics = monitor.get_metrics()
        if metrics.ram_total_mb > 0:
            assert metrics.ram_used_mb <= metrics.ram_total_mb

    def test_disk_used_less_than_total(self):
        """Disk metrics use os.statvfs which works on all platforms."""
        monitor = SystemMonitor(disk_path=".")
        metrics = monitor.get_metrics()
        if metrics.disk_total_gb > 0:
            assert metrics.disk_used_gb <= metrics.disk_total_gb

    def test_disk_percent_in_range(self):
        monitor = SystemMonitor(disk_path=".")
        metrics = monitor.get_metrics()
        if metrics.disk_percent >= 0:
            assert 0 <= metrics.disk_percent <= 100

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="System monitor reads from /proc which is Linux-only",
    )
    def test_ram_percent_in_range(self):
        monitor = SystemMonitor()
        metrics = monitor.get_metrics()
        if metrics.ram_percent >= 0:
            assert 0 <= metrics.ram_percent <= 100

    def test_check_warnings_returns_list(self):
        monitor = SystemMonitor(disk_path=".", ram_warning_percent=0.0, disk_warning_percent=0.0)
        warnings = monitor.check_warnings()
        assert isinstance(warnings, list)


class TestSystemMonitorFallback:
    """Test graceful fallback when /proc is not available (Windows)."""

    @pytest.mark.skipif(
        platform.system() == "Linux",
        reason="Only test fallback on non-Linux",
    )
    def test_returns_negative_on_non_linux(self):
        monitor = SystemMonitor()
        metrics = monitor.get_metrics()
        # On non-Linux, CPU and RAM should return -1
        assert metrics.cpu_percent == -1.0
        assert metrics.ram_used_mb == -1.0
