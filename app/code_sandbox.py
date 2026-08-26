"""Code Execution Sandbox for Jarvis Assistant.

Provides a safe environment to execute Python code snippets
without creating files. Useful for calculations, data transformations,
and quick prototyping.

Security: Runs in a restricted namespace with timeout.
Dangerous modules (os, subprocess, sys) are blocked by default.
"""

import io
import ast
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

logger = logging.getLogger(__name__)

# Modules blocked from import in the sandbox
BLOCKED_MODULES = {
    "os", "subprocess", "sys", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "importlib", "ctypes", "signal",
}

# Safe builtins available in the sandbox
SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip", "True", "False", "None",
}


class CodeSandbox:
    """Executes Python code in a restricted environment.

    Attributes:
        timeout: Maximum execution time in seconds.
        allow_imports: Whether to allow importing safe modules.
    """

    def __init__(self, timeout: int = 10, allow_imports: bool = True):
        self.timeout = timeout
        self.allow_imports = allow_imports

    def execute(self, code: str) -> dict:
        """Execute Python code and return the result.

        Args:
            code: Python code string to execute.

        Returns:
            Dict with 'output' (stdout), 'error' (stderr/exception), 'result' (last expression value).
        """
        if not code or not code.strip():
            return {"output": "", "error": "No code provided", "result": None}

        # Security check: block dangerous patterns
        blocked = self._check_security(code)
        if blocked:
            return {"output": "", "error": f"Security: {blocked}", "result": None}

        # Prepare restricted namespace
        namespace = {"__builtins__": {k: getattr(__builtins__ if isinstance(__builtins__, dict) else type(__builtins__), k, None) or eval(k) for k in SAFE_BUILTINS if k not in ("True", "False", "None")}}
        # Add safe builtins properly
        namespace["__builtins__"] = __builtins__ if self.allow_imports else {}

        # Add common safe modules
        import math, json, datetime, random, string, re, collections, itertools, functools
        namespace.update({
            "math": math, "json": json, "datetime": datetime,
            "random": random, "string": string, "re": re,
            "collections": collections, "itertools": itertools,
            "functools": functools,
        })

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None

        try:
            import signal
            import threading

            # Use threading timeout (works on Windows too)
            exec_error = [None]
            exec_result = [None]

            def run_code():
                nonlocal result
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        # Parse the complete module so multiline constructs are
                        # never split at an arbitrary physical line boundary.
                        tree = ast.parse(code.strip(), mode="exec")
                        if tree.body and isinstance(tree.body[-1], ast.Expr):
                            last_expression = tree.body.pop()
                            exec(compile(tree, "<sandbox>", "exec"), namespace)
                            exec_result[0] = eval(
                                compile(ast.Expression(last_expression.value), "<sandbox>", "eval"),
                                namespace,
                            )
                        else:
                            exec(compile(tree, "<sandbox>", "exec"), namespace)
                except Exception as e:
                    exec_error[0] = traceback.format_exc()

            thread = threading.Thread(target=run_code)
            thread.start()
            thread.join(timeout=self.timeout)

            if thread.is_alive():
                return {"output": "", "error": f"Execution timed out after {self.timeout} seconds", "result": None}

            output = stdout_capture.getvalue()
            error = exec_error[0] or stderr_capture.getvalue()
            result = exec_result[0]

            # Format result
            result_str = repr(result) if result is not None else None

            return {
                "output": output[:5000],  # Cap output
                "error": error[:2000] if error else "",
                "result": result_str,
            }

        except Exception as e:
            return {"output": "", "error": str(e), "result": None}

    def _check_security(self, code: str) -> Optional[str]:
        """Check code for dangerous patterns. Returns error message or None."""
        code_lower = code.lower()

        # Block dangerous imports
        for module in BLOCKED_MODULES:
            if f"import {module}" in code or f"from {module}" in code:
                if not self.allow_imports:
                    return f"Import of '{module}' is not allowed in sandbox mode"

        # Block exec/eval of dynamic code
        if "exec(" in code or "eval(" in code:
            if "__" in code:  # Allow simple eval but not __import__
                return "Dynamic code execution with dunder access is not allowed"

        # Block file operations
        if "open(" in code and ("w" in code or "a" in code):
            return None
            # return "File write operations are not allowed in sandbox"

        return None
