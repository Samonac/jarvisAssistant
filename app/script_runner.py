"""Script Runner for Jarvis Assistant.

Dynamically creates API endpoints for each Python script in the scripts/ folder.
Scripts are exposed as POST /api/scripts/<script_name> endpoints.

Each script can optionally define metadata via comments at the top:
    # DESCRIPTION: Takes a screenshot and saves it
    # ARGS: {"width": "int, optional", "output": "string, optional"}

If no metadata is defined, the script is still exposed with a generic description.
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ScriptRunner:
    """Manages scripts in the scripts/ folder and exposes them as API endpoints.

    Attributes:
        scripts_dir: Path to the scripts directory.
        scripts: Dict of discovered scripts with metadata.
    """

    def __init__(self, scripts_dir: str = "scripts"):
        self.scripts_dir = Path(scripts_dir)
        self.scripts_dir.mkdir(exist_ok=True)
        self.scripts: dict[str, dict] = {}

    def discover_scripts(self) -> int:
        """Scan the scripts/ folder and register all Python scripts.

        Returns the number of scripts discovered.
        """
        self.scripts = {}
        if not self.scripts_dir.exists():
            return 0

        for file in sorted(self.scripts_dir.glob("*.py")):
            if file.name.startswith("_"):
                continue
            metadata = self._extract_metadata(file)
            name = file.stem  # filename without .py
            self.scripts[name] = {
                "name": name,
                "filename": file.name,
                "path": str(file),
                "description": metadata.get("description", f"Execute {file.name}"),
                "args": metadata.get("args", {}),
            }

        logger.info("Discovered %d script(s) in %s", len(self.scripts), self.scripts_dir)
        return len(self.scripts)

    def _extract_metadata(self, file_path: Path) -> dict:
        """Extract metadata from script comments.

        Looks for:
            # DESCRIPTION: ...
            # ARGS: {"param": "type, description"}
        """
        metadata = {}
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            # Extract DESCRIPTION
            desc_match = re.search(r'#\s*DESCRIPTION:\s*(.+)', content)
            if desc_match:
                metadata["description"] = desc_match.group(1).strip()

            # Extract ARGS as JSON
            args_match = re.search(r'#\s*ARGS:\s*(\{.+\})', content)
            if args_match:
                import json
                try:
                    metadata["args"] = json.loads(args_match.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
        return metadata

    def run_script(self, name: str, args: Optional[dict] = None) -> dict:
        """Execute a script by name.

        Args:
            name: Script name (without .py extension).
            args: Optional dict of arguments passed as environment variables.

        Returns:
            Dict with 'stdout', 'stderr', 'return_code', or 'error'.
        """
        if name not in self.scripts:
            return {"error": f"Script '{name}' not found. Available: {', '.join(self.scripts.keys()) or 'none'}"}

        script_info = self.scripts[name]
        script_path = script_info["path"]

        try:
            # Pass args as environment variables prefixed with SCRIPT_
            env = os.environ.copy()
            if args:
                for key, value in args.items():
                    env[f"SCRIPT_{key.upper()}"] = str(value)

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.scripts_dir.parent),  # Run from project root
                env=env,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "script": name,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Script '{name}' timed out after 120 seconds."}
        except Exception as e:
            return {"error": f"Failed to run script '{name}': {e}"}

    def list_scripts(self) -> list[dict]:
        """List all available scripts with metadata."""
        return list(self.scripts.values())

    def get_swagger_paths(self) -> dict:
        """Generate OpenAPI path entries for all discovered scripts."""
        paths = {}
        for name, info in self.scripts.items():
            path = f"/api/scripts/{name}"

            # Build args schema
            properties = {}
            if info["args"]:
                for arg_name, arg_desc in info["args"].items():
                    properties[arg_name] = {"type": "string", "description": str(arg_desc)}

            paths[path] = {
                "post": {
                    "summary": f"Run: {info['description']}",
                    "tags": ["Scripts"],
                    "description": f"Executes `{info['filename']}` from the scripts/ folder.",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": properties if properties else {"args": {"type": "object", "description": "Optional arguments passed as SCRIPT_* env vars"}},
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Script execution result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stdout": {"type": "string"},
                                            "stderr": {"type": "string"},
                                            "return_code": {"type": "integer"},
                                            "script": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        },
                        "404": {"description": "Script not found"},
                    },
                }
            }

        return paths
