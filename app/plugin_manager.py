"""Plugin Manager for Jarvis Assistant.

A simple plugin architecture that allows users to add new tools by:
1. Dropping Python files into a plugins/ directory
2. Uploading plugins via the API

Each plugin is a Python file that defines:
- PLUGIN_NAME: str — the tool name (used in LLM tool calls)
- PLUGIN_DESCRIPTION: str — description for the LLM system prompt
- PLUGIN_ARGS: str — argument format description
- execute(args: dict) -> str — the function that runs when the tool is called
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Plugin:
    """Represents a loaded plugin."""

    def __init__(self, name: str, description: str, args_desc: str, execute_fn):
        self.name = name
        self.description = description
        self.args_desc = args_desc
        self.execute = execute_fn


class PluginManager:
    """Manages loading and executing plugins.

    Plugins are Python files in the plugins/ directory that define:
    - PLUGIN_NAME: str
    - PLUGIN_DESCRIPTION: str
    - PLUGIN_ARGS: str
    - execute(args: dict) -> str

    Attributes:
        plugins_dir: Path to the plugins directory.
        plugins: Dict of loaded plugins keyed by name.
    """

    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: dict[str, Plugin] = {}
        self.plugins_dir.mkdir(exist_ok=True)

    def load_all(self) -> int:
        """Load all plugins from the plugins directory.

        Returns the number of successfully loaded plugins.
        """
        count = 0
        if not self.plugins_dir.exists():
            return 0

        for file in self.plugins_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue
            try:
                self._load_plugin_file(file)
                count += 1
            except Exception as e:
                logger.error("Failed to load plugin '%s': %s", file.name, e)

        logger.info("Loaded %d plugin(s) from %s", count, self.plugins_dir)
        return count

    def _load_plugin_file(self, file_path: Path) -> None:
        """Load a single plugin from a Python file."""
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load module spec from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Validate required attributes
        name = getattr(module, "PLUGIN_NAME", None)
        description = getattr(module, "PLUGIN_DESCRIPTION", None)
        args_desc = getattr(module, "PLUGIN_ARGS", "{}")
        execute_fn = getattr(module, "execute", None)

        if not name:
            raise ValueError(f"Plugin {file_path.name} missing PLUGIN_NAME")
        if not description:
            raise ValueError(f"Plugin {file_path.name} missing PLUGIN_DESCRIPTION")
        if not callable(execute_fn):
            raise ValueError(f"Plugin {file_path.name} missing execute() function")

        plugin = Plugin(name=name, description=description, args_desc=args_desc, execute_fn=execute_fn)
        self.plugins[name] = plugin
        logger.info("Loaded plugin: %s (%s)", name, description)

    def install_plugin(self, name: str, code: str) -> dict:
        """Install a new plugin from Python source code.

        Args:
            name: Plugin filename (without .py extension).
            code: Python source code for the plugin.

        Returns:
            Dict with 'message' or 'error'.
        """
        # Basic safety checks
        dangerous_imports = ["os.system", "subprocess", "shutil.rmtree", "__import__"]
        for danger in dangerous_imports:
            if danger in code:
                return {"error": f"Plugin rejected: contains potentially dangerous code ({danger})"}

        filename = name.replace(" ", "_").lower()
        if not filename.endswith(".py"):
            filename += ".py"

        file_path = self.plugins_dir / filename

        try:
            file_path.write_text(code, encoding="utf-8")
            self._load_plugin_file(file_path)
            return {"message": f"Plugin '{name}' installed and loaded successfully."}
        except Exception as e:
            # Clean up on failure
            if file_path.exists():
                file_path.unlink()
            return {"error": f"Plugin installation failed: {e}"}

    def uninstall_plugin(self, name: str) -> dict:
        """Remove a plugin by name."""
        if name not in self.plugins:
            return {"error": f"Plugin '{name}' not found."}

        # Find and delete the file
        for file in self.plugins_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if getattr(module, "PLUGIN_NAME", None) == name:
                        file.unlink()
                        break
            except Exception:
                continue

        del self.plugins[name]
        return {"message": f"Plugin '{name}' uninstalled."}

    def execute_plugin(self, name: str, args: dict) -> str:
        """Execute a plugin by name.

        Args:
            name: The plugin name.
            args: Arguments dict to pass to the plugin's execute function.

        Returns:
            The plugin's output string.
        """
        if name not in self.plugins:
            return f"Plugin '{name}' not found. Available: {', '.join(self.plugins.keys()) or 'none'}"

        try:
            result = self.plugins[name].execute(args)
            return str(result)
        except Exception as e:
            logger.error("Plugin '%s' execution error: %s", name, e)
            return f"Plugin '{name}' error: {e}"

    def get_tool_descriptions(self) -> list[str]:
        """Get LLM-formatted tool descriptions for all loaded plugins."""
        descriptions = []
        for plugin in self.plugins.values():
            descriptions.append(
                f"- {plugin.name}: {plugin.description}. Args: {plugin.args_desc}"
            )
        return descriptions

    def list_plugins(self) -> list[dict]:
        """List all loaded plugins."""
        return [
            {"name": p.name, "description": p.description, "args": p.args_desc}
            for p in self.plugins.values()
        ]
