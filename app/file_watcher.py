"""File Watcher for Jarvis Assistant.

Monitors the project directory for changes to Python files, .env,
and templates. When a change is detected, restarts the server process.

Uses a simple polling approach (no external dependencies like watchdog)
to keep the footprint minimal on the Raspberry Pi.
"""

import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions to watch
WATCH_EXTENSIONS = {".py", ".env", ".html", ".css", ".js"}

# Directories to ignore
IGNORE_DIRS = {"__pycache__", ".git", ".hypothesis", ".pytest_cache", "venv", "node_modules", "scripts", "plugins", "picture", "oldDB", "uploads", "backups"}


class FileWatcher:
    """Watches project files and triggers a server restart on changes.

    Uses polling (checks every N seconds) to detect file modifications.
    When a change is detected, restarts the current Python process.

    Attributes:
        watch_dir: The directory to monitor.
        poll_interval: Seconds between checks (default 3).
    """

    def __init__(self, watch_dir: str = ".", poll_interval: int = 3):
        self.watch_dir = Path(watch_dir).resolve()
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._file_mtimes: dict[str, str] = {}  # path -> content hash
        self._initial_scan_done = False
        self._start_time = time.time()
        self._restart_pending = False
        self._restart_time = 0.0

    def start(self) -> None:
        """Start the file watcher in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        # Do initial scan to record current state
        self._file_mtimes = self._scan_files()
        self._initial_scan_done = True
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("File watcher started (dir: %s, interval: %ds)", self.watch_dir, self.poll_interval)

    def stop(self) -> None:
        """Stop the file watcher."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _watch_loop(self) -> None:
        """Main polling loop."""
        # Grace period: ignore changes in the first 5 seconds after startup
        # (loading .env, writing .pyc files, etc. can trigger false positives)
        grace_period = 5

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.poll_interval)
            if self._stop_event.is_set():
                break

            # Skip checks during grace period
            if time.time() - self._start_time < grace_period:
                # Re-scan to update baselines during grace period
                self._file_mtimes = self._scan_files()
                continue

            changed = self._check_for_changes()
            if changed:
                logger.info("File change detected: %s — restarting server...", changed)
                self._restart()

    def _scan_files(self) -> dict[str, str]:
        """Scan all watched files and return {path: content_hash} dict.
        
        Uses file content hashing (not mtime) to detect actual changes,
        ignoring reads and access-time updates.
        """
        import hashlib
        hashes = {}
        for root, dirs, files in os.walk(self.watch_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for filename in files:
                ext = Path(filename).suffix
                if ext in WATCH_EXTENSIONS or filename == ".env":
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, "rb") as f:
                            content = f.read()
                        hashes[filepath] = hashlib.md5(content).hexdigest()
                    except (OSError, PermissionError):
                        pass
        return hashes

    def _check_for_changes(self) -> str:
        """Check if any watched file has been modified (content changed).

        Returns the path of the first changed file, or empty string if none.
        """
        current = self._scan_files()

        for filepath, content_hash in current.items():
            if filepath not in self._file_mtimes:
                # New file added
                self._file_mtimes = current
                return filepath
            if content_hash != self._file_mtimes[filepath]:
                # File content actually changed
                self._file_mtimes = current
                return filepath

        # Check for deleted files
        for filepath in list(self._file_mtimes.keys()):
            if filepath not in current:
                self._file_mtimes = current
                return filepath

        self._file_mtimes = current
        return ""

    def _restart(self) -> None:
        """Restart the current Python process with a delay for clients to be notified."""
        logger.info("Restarting Jarvis Assistant in 5 seconds...")

        # Signal the pending restart so the API can inform clients
        self._restart_pending = True
        self._restart_time = time.time() + 5

        # Wait 5 seconds for clients to see the notification
        time.sleep(5)

        if platform.system() == "Windows":
            import subprocess
            subprocess.Popen(
                [sys.executable] + sys.argv,
                cwd=os.getcwd(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            os._exit(0)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)
