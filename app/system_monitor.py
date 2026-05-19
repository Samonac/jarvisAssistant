"""System Monitor for Jarvis Assistant.

Reads system resource usage from Linux /proc filesystem and os.statvfs.
No external dependencies required — works directly on Raspberry Pi OS.
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource usage metrics."""

    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    uptime_seconds: float


class SystemMonitor:
    """Reads system resource usage from Linux interfaces.

    Uses /proc/stat, /proc/meminfo, os.statvfs, and /proc/uptime
    to collect metrics without any external dependencies.

    Attributes:
        disk_path: Filesystem path to check disk usage for.
        ram_warning_percent: Threshold for RAM warning.
        disk_warning_percent: Threshold for disk warning.
    """

    def __init__(
        self,
        disk_path: str = "/",
        ram_warning_percent: float = 80.0,
        disk_warning_percent: float = 90.0,
    ):
        self.disk_path = disk_path
        self.ram_warning_percent = ram_warning_percent
        self.disk_warning_percent = disk_warning_percent
        self._prev_cpu_idle: float = 0
        self._prev_cpu_total: float = 0

    def get_metrics(self) -> SystemMetrics:
        """Read current system metrics.

        Returns:
            SystemMetrics dataclass with current resource usage.
        """
        cpu_percent = self._read_cpu_usage()
        ram_used_mb, ram_total_mb, ram_percent = self._read_memory()
        disk_used_gb, disk_total_gb, disk_percent = self._read_disk()
        uptime_seconds = self._read_uptime()

        return SystemMetrics(
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            ram_percent=ram_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_percent=disk_percent,
            uptime_seconds=uptime_seconds,
        )

    def _read_cpu_usage(self) -> float:
        """Parse /proc/stat to calculate CPU usage percentage since last read.

        Uses delta-based calculation between consecutive reads.

        Returns:
            CPU usage percentage (0-100). Returns -1 on error.
        """
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()

            # Format: cpu user nice system idle iowait irq softirq steal guest guest_nice
            parts = line.split()
            if parts[0] != "cpu":
                return -1.0

            values = [int(v) for v in parts[1:]]
            idle = values[3] + values[4]  # idle + iowait
            total = sum(values)

            # Calculate delta
            idle_delta = idle - self._prev_cpu_idle
            total_delta = total - self._prev_cpu_total

            # Update previous values
            self._prev_cpu_idle = idle
            self._prev_cpu_total = total

            if total_delta == 0:
                return 0.0

            cpu_percent = (1.0 - idle_delta / total_delta) * 100.0
            return max(0.0, min(100.0, cpu_percent))

        except (OSError, IOError, ValueError, IndexError) as e:
            logger.warning("Failed to read CPU usage: %s", e)
            return -1.0

    def _read_memory(self) -> tuple[float, float, float]:
        """Parse /proc/meminfo to get memory usage.

        Returns:
            Tuple of (used_mb, total_mb, percent). Returns (-1, -1, -1) on error.
        """
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        value = int(parts[1])  # Value in kB
                        meminfo[key] = value

            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", 0)

            if total_kb == 0:
                return -1.0, -1.0, -1.0

            used_kb = total_kb - available_kb
            total_mb = total_kb / 1024.0
            used_mb = used_kb / 1024.0
            percent = (used_kb / total_kb) * 100.0

            return round(used_mb, 1), round(total_mb, 1), round(percent, 1)

        except (OSError, IOError, ValueError, KeyError) as e:
            logger.warning("Failed to read memory info: %s", e)
            return -1.0, -1.0, -1.0

    def _read_disk(self) -> tuple[float, float, float]:
        """Use os.statvfs (Linux) or shutil.disk_usage (cross-platform) to get disk usage.

        Returns:
            Tuple of (used_gb, total_gb, percent). Returns (-1, -1, -1) on error.
        """
        try:
            if hasattr(os, "statvfs"):
                stat = os.statvfs(self.disk_path)
                total_bytes = stat.f_blocks * stat.f_frsize
                free_bytes = stat.f_bfree * stat.f_frsize
                used_bytes = total_bytes - free_bytes
            else:
                import shutil
                usage = shutil.disk_usage(self.disk_path)
                total_bytes = usage.total
                used_bytes = usage.used
                free_bytes = usage.free

            total_gb = total_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            percent = (used_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0.0

            return round(used_gb, 2), round(total_gb, 2), round(percent, 1)

        except (OSError, IOError) as e:
            logger.warning("Failed to read disk usage: %s", e)
            return -1.0, -1.0, -1.0

    def _read_uptime(self) -> float:
        """Parse /proc/uptime to get system uptime in seconds.

        Returns:
            Uptime in seconds. Returns -1 on error.
        """
        try:
            with open("/proc/uptime", "r") as f:
                line = f.readline()
            uptime_seconds = float(line.split()[0])
            return uptime_seconds
        except (OSError, IOError, ValueError, IndexError) as e:
            logger.warning("Failed to read uptime: %s", e)
            return -1.0

    def check_warnings(self) -> list[str]:
        """Check if resources exceed warning thresholds.

        Returns:
            List of warning messages. Empty if all is well.
        """
        warnings = []
        metrics = self.get_metrics()

        if metrics.ram_percent >= 0 and metrics.ram_percent > self.ram_warning_percent:
            warnings.append(
                f"RAM usage is high: {metrics.ram_percent:.1f}% "
                f"({metrics.ram_used_mb:.0f}MB / {metrics.ram_total_mb:.0f}MB)"
            )

        if metrics.disk_percent >= 0 and metrics.disk_percent > self.disk_warning_percent:
            warnings.append(
                f"Disk usage is high: {metrics.disk_percent:.1f}% "
                f"({metrics.disk_used_gb:.1f}GB / {metrics.disk_total_gb:.1f}GB)"
            )

        return warnings
