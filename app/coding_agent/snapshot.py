"""Per-task file-tree snapshots for the coding agent (Phase 4).

No git is used here (per project decision — no push rights yet). Snapshots
are plain directory copies under backups/coding_agent/snapshots/<session_id>/,
extending the same non-git backup approach already used by
app.backup_orchestrator.BackupOrchestrator. This lets every task's changes be
fully reverted with a single restore() call, independent of the granular
per-file backups app.coding_agent.tools.CodingAgentTools already makes.
"""

import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# Directories/files never copied into, or touched when restoring from, a snapshot.
EXCLUDED_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".hypothesis",
    "backups", "uploads", "picture", "oldDB",
    "venv", ".venv", "node_modules",
}
EXCLUDED_SUFFIXES = (".db", ".db-shm", ".db-wal", ".pyc")


class TaskSnapshot:
    """Creates/restores/discards a full project-tree snapshot for one task."""

    def __init__(self, project_dir: str, snapshot_root: Optional[str] = None):
        self.project_dir = project_dir
        self.snapshot_root = snapshot_root or os.path.join(project_dir, "backups", "coding_agent", "snapshots")
        os.makedirs(self.snapshot_root, exist_ok=True)

    def create(self, session_id: str) -> str:
        """Copy the current project tree into a fresh snapshot for this session."""
        dest = self._snapshot_dir(session_id)
        if os.path.exists(dest):
            shutil.rmtree(dest)

        shutil.copytree(self.project_dir, dest, ignore=self._ignore)
        logger.info("Created pre-task snapshot for session '%s' at '%s'", session_id, dest)
        return dest

    def restore(self, session_id: str) -> bool:
        """Restore the project tree from this session's snapshot.

        Discards any changes made to non-excluded files/dirs since the
        snapshot was taken. Returns True if a snapshot existed and was
        restored, False otherwise.
        """
        src = self._snapshot_dir(session_id)
        if not os.path.isdir(src):
            logger.warning("No snapshot found for session '%s' — cannot restore.", session_id)
            return False

        for name in os.listdir(self.project_dir):
            if self._is_excluded(name):
                continue
            target = os.path.join(self.project_dir, name)
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            elif os.path.exists(target):
                os.remove(target)

        for name in os.listdir(src):
            src_item = os.path.join(src, name)
            dest_item = os.path.join(self.project_dir, name)
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dest_item)
            else:
                shutil.copy2(src_item, dest_item)

        logger.info("Restored session '%s' from snapshot '%s'", session_id, src)
        return True

    def discard(self, session_id: str) -> None:
        """Delete a session's snapshot once its changes are accepted and no longer needed."""
        dest = self._snapshot_dir(session_id)
        if os.path.isdir(dest):
            shutil.rmtree(dest)

    def _snapshot_dir(self, session_id: str) -> str:
        return os.path.join(self.snapshot_root, session_id)

    def _ignore(self, dir_path: str, names: list[str]) -> set[str]:
        return {n for n in names if self._is_excluded(n)}

    @staticmethod
    def _is_excluded(name: str) -> bool:
        if name in EXCLUDED_NAMES:
            return True
        return any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)
