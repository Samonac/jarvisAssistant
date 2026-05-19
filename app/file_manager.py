"""File Manager for Jarvis Assistant.

Provides safe file operations: read, write, list, and search files
on the Raspberry Pi. Includes a safety sandbox to prevent access
to sensitive system directories.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Directories that are NEVER accessible
BLOCKED_PATHS = {
    "/etc/shadow", "/etc/passwd", "/etc/sudoers",
    "/root", "/proc", "/sys", "/dev",
}

# Prefixes that are blocked
BLOCKED_PREFIXES = [
    "/etc/ssh", "/etc/ssl", "/boot/firmware",
    "/var/log/auth", "/var/log/secure",
]

# Max file size to read (1MB)
MAX_READ_SIZE = 1_048_576

# Max file size to write (512KB)
MAX_WRITE_SIZE = 524_288


class FileManager:
    """Manages file operations with safety restrictions.

    Attributes:
        base_dir: The base directory for file operations (sandbox root).
        extra_dirs: Additional directories that are allowed (e.g., project dir).
        allow_absolute: Whether to allow absolute paths outside base_dir.
    """

    def __init__(self, base_dir: str = "/home/pi", allow_absolute: bool = False, extra_dirs: list[str] = None):
        self.base_dir = Path(base_dir).resolve()
        self.extra_dirs = [Path(d).resolve() for d in (extra_dirs or [])]
        self.allow_absolute = allow_absolute

    def _resolve_path(self, path: str) -> Optional[Path]:
        """Resolve a path, enforcing safety restrictions.

        Returns the resolved Path or None if blocked.
        """
        p = Path(path)

        if not p.is_absolute():
            # For relative paths, prefer the first extra_dir (project dir) if available
            if self.extra_dirs:
                p = self.extra_dirs[0] / p
            else:
                p = self.base_dir / p

        resolved = p.resolve()

        # Check blocked paths
        resolved_str = str(resolved)
        if resolved_str in BLOCKED_PATHS:
            return None
        for prefix in BLOCKED_PREFIXES:
            if resolved_str.startswith(prefix):
                return None

        # Check if within base_dir or any extra allowed directory
        if not self.allow_absolute:
            allowed = False
            try:
                resolved.relative_to(self.base_dir)
                allowed = True
            except ValueError:
                pass

            if not allowed:
                for extra in self.extra_dirs:
                    try:
                        resolved.relative_to(extra)
                        allowed = True
                        break
                    except ValueError:
                        continue

            if not allowed:
                return None

        return resolved

    def read_file(self, path: str) -> dict:
        """Read a file's content.

        Returns dict with 'content' and 'path', or 'error'.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"error": f"Access denied: '{path}' is outside the allowed directory or blocked."}

        if not resolved.exists():
            return {"error": f"File not found: '{path}'"}

        if not resolved.is_file():
            return {"error": f"Not a file: '{path}'"}

        if resolved.stat().st_size > MAX_READ_SIZE:
            return {"error": f"File too large (>{MAX_READ_SIZE // 1024}KB): '{path}'"}

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            return {"content": content, "path": str(resolved), "size": len(content)}
        except PermissionError:
            return {"error": f"Permission denied: '{path}'"}
        except Exception as e:
            return {"error": f"Read error: {e}"}

    def write_file(self, path: str, content: str) -> dict:
        """Write content to a file (creates parent dirs if needed).

        Returns dict with 'path' and 'size', or 'error'.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"error": f"Access denied: '{path}' is outside the allowed directory or blocked."}

        if len(content) > MAX_WRITE_SIZE:
            return {"error": f"Content too large (>{MAX_WRITE_SIZE // 1024}KB)"}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return {"path": str(resolved), "size": len(content), "message": f"Written {len(content)} chars to {resolved.name}"}
        except PermissionError:
            return {"error": f"Permission denied: '{path}'"}
        except Exception as e:
            return {"error": f"Write error: {e}"}

    def list_directory(self, path: str = ".") -> dict:
        """List contents of a directory.

        Returns dict with 'entries' list, or 'error'.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"error": f"Access denied: '{path}'"}

        if not resolved.exists():
            return {"error": f"Directory not found: '{path}'"}

        if not resolved.is_dir():
            return {"error": f"Not a directory: '{path}'"}

        try:
            entries = []
            for item in sorted(resolved.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return {"path": str(resolved), "entries": entries, "count": len(entries)}
        except PermissionError:
            return {"error": f"Permission denied: '{path}'"}
        except Exception as e:
            return {"error": f"List error: {e}"}

    def delete_file(self, path: str) -> dict:
        """Delete a file (not directories).

        Returns dict with 'message' or 'error'.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"error": f"Access denied: '{path}'"}

        if not resolved.exists():
            return {"error": f"File not found: '{path}'"}

        if not resolved.is_file():
            return {"error": f"Not a file (won't delete directories): '{path}'"}

        try:
            resolved.unlink()
            return {"message": f"Deleted: {resolved.name}"}
        except PermissionError:
            return {"error": f"Permission denied: '{path}'"}
        except Exception as e:
            return {"error": f"Delete error: {e}"}

    def search_files(self, pattern: str, path: str = ".") -> dict:
        """Search for files matching a glob pattern.

        Returns dict with 'matches' list, or 'error'.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return {"error": f"Access denied: '{path}'"}

        if not resolved.is_dir():
            return {"error": f"Not a directory: '{path}'"}

        try:
            matches = []
            for item in resolved.rglob(pattern):
                if len(matches) >= 50:  # Limit results
                    break
                matches.append({
                    "path": str(item.relative_to(resolved)),
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return {"matches": matches, "count": len(matches), "pattern": pattern}
        except Exception as e:
            return {"error": f"Search error: {e}"}
