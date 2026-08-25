"""Self-discovery of nightly tasks when the queue is empty (Phase 5).

Looks for two categories of low-risk, well-scoped work, each with an
obvious, agent-checkable definition of "done":
- Failing tests (the project's own pytest suite) — done when the test passes.
- TODO/FIXME comments left in the source tree — done when addressed.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME)[:\s]+(.+)", re.IGNORECASE)
FAILED_TEST_PATTERN = re.compile(r"^FAILED\s+(\S+)")
SCAN_EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".hypothesis", "backups", "uploads",
    "picture", "oldDB", ".git", "venv", ".venv", "node_modules",
}
MAX_DISCOVERED_TASKS = 5


def discover_tasks(project_dir: str, command_executor) -> list[str]:
    """Return a small list of candidate task descriptions (possibly empty)."""
    tasks = _discover_failing_tests(command_executor)
    if len(tasks) < MAX_DISCOVERED_TASKS:
        tasks.extend(_discover_todos(project_dir))
    return tasks[:MAX_DISCOVERED_TASKS]


def _discover_failing_tests(command_executor) -> list[str]:
    result = command_executor.execute("python -m pytest -q --tb=no")
    if result.get("blocked") or result.get("timed_out"):
        return []

    tasks = []
    for line in result.get("stdout", "").splitlines():
        match = FAILED_TEST_PATTERN.match(line.strip())
        if match:
            tasks.append(
                f"Fix the failing test: {match.group(1)}. "
                f"Run the test suite yourself to confirm it passes before reporting done."
            )
    return tasks


def _discover_todos(project_dir: str) -> list[str]:
    tasks = []
    root = Path(project_dir)
    for path in root.rglob("*.py"):
        if any(part in SCAN_EXCLUDE_DIRS for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for lineno, line in enumerate(lines, start=1):
            match = TODO_PATTERN.search(line)
            if not match:
                continue
            rel = path.relative_to(root)
            tasks.append(
                f'Address the {match.group(1).upper()} at {rel}:{lineno} — "{match.group(2).strip()}". '
                f"Run the test suite yourself to confirm nothing broke before reporting done."
            )
            if len(tasks) >= MAX_DISCOVERED_TASKS:
                return tasks
    return tasks
