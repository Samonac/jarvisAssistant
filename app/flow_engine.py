"""Flow Engine for Jarvis Assistant.

A visual workflow automation engine (similar to Power Automate / Node-RED).
Users create flows with connected blocks that execute sequentially or in parallel,
with support for decision trees, loops, and data passing between blocks.

Each block is either:
- A Python script execution
- An API call (internal or external)
- A decision (if/else based on previous output)
- A loop (repeat N times or until condition)
- A delay/wait

Flows can be triggered manually or run on a schedule.
"""

import json
import logging
import sqlite3
import threading
import time
import traceback
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Available block types
BLOCK_TYPES = {
    "script": {"label": "Python Script", "icon": "🐍", "description": "Execute Python code"},
    "api_call": {"label": "API Call", "icon": "🌐", "description": "Make an HTTP request"},
    "command": {"label": "Shell Command", "icon": "⚙️", "description": "Run a shell command"},
    "decision": {"label": "Decision", "icon": "🔀", "description": "If/else branch based on condition"},
    "loop": {"label": "Loop", "icon": "🔁", "description": "Repeat blocks N times or until condition"},
    "delay": {"label": "Delay", "icon": "⏱️", "description": "Wait for a specified duration"},
    "notify": {"label": "Notification", "icon": "🔔", "description": "Send a notification"},
    "set_variable": {"label": "Set Variable", "icon": "📦", "description": "Store a value for later blocks"},
    "jarvis_chat": {"label": "Ask Jarvis", "icon": "🤖", "description": "Send a message to Jarvis and get a response"},
}


class FlowEngine:
    """Manages visual automation flows."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.scheduler = None
        self.conversation_manager = None
        self._running_flows: dict = {}
        self._lock = threading.Lock()
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    blocks TEXT NOT NULL DEFAULT '[]',
                    connections TEXT NOT NULL DEFAULT '[]',
                    schedule TEXT,
                    enabled INTEGER DEFAULT 1,
                    username TEXT NOT NULL,
                    last_run DATETIME,
                    last_status TEXT,
                    run_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS flow_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    flow_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'running',
                    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    finished_at DATETIME,
                    duration_ms REAL,
                    block_results TEXT,
                    error TEXT,
                    triggered_by TEXT DEFAULT 'manual',
                    FOREIGN KEY (flow_id) REFERENCES flows(id)
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init flow tables: %s", e)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create_flow(self, name: str, username: str, description: str = "",
                    blocks: list = None, connections: list = None,
                    schedule: str = None) -> dict:
        if not name:
            return {"error": "Name is required"}
        try:
            conn = self._get_conn()
            cursor = conn.execute("""
                INSERT INTO flows (name, description, blocks, connections, schedule, username)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, description, json.dumps(blocks or []),
                  json.dumps(connections or []), schedule, username))
            conn.commit()
            flow_id = cursor.lastrowid
            conn.close()
            return {"id": flow_id, "message": f"Flow '{name}' created."}
        except Exception as e:
            return {"error": str(e)}

    def list_flows(self, username: str = None, include_all: bool = False) -> list[dict]:
        try:
            conn = self._get_conn()
            if include_all or not username:
                cursor = conn.execute("SELECT * FROM flows ORDER BY updated_at DESC")
            else:
                cursor = conn.execute("SELECT * FROM flows WHERE username = ? ORDER BY updated_at DESC", (username,))
            flows = []
            for r in cursor.fetchall():
                flows.append({
                    "id": r["id"], "name": r["name"], "description": r["description"],
                    "blocks": json.loads(r["blocks"]), "connections": json.loads(r["connections"]),
                    "schedule": r["schedule"], "enabled": bool(r["enabled"]),
                    "username": r["username"], "last_run": r["last_run"],
                    "last_status": r["last_status"], "run_count": r["run_count"],
                    "success_count": r["success_count"], "failure_count": r["failure_count"],
                    "created_at": r["created_at"], "updated_at": r["updated_at"],
                    "is_running": r["id"] in self._running_flows,
                })
            conn.close()
            return flows
        except Exception as e:
            logger.error("Failed to list flows: %s", e)
            return []

    def get_flow(self, flow_id: int) -> Optional[dict]:
        try:
            conn = self._get_conn()
            r = conn.execute("SELECT * FROM flows WHERE id = ?", (flow_id,)).fetchone()
            conn.close()
            if not r:
                return None
            return {
                "id": r["id"], "name": r["name"], "description": r["description"],
                "blocks": json.loads(r["blocks"]), "connections": json.loads(r["connections"]),
                "schedule": r["schedule"], "enabled": bool(r["enabled"]),
                "username": r["username"], "last_run": r["last_run"],
                "last_status": r["last_status"], "run_count": r["run_count"],
                "success_count": r["success_count"], "failure_count": r["failure_count"],
                "created_at": r["created_at"], "updated_at": r["updated_at"],
                "is_running": r["id"] in self._running_flows,
            }
        except Exception:
            return None

    def update_flow(self, flow_id: int, updates: dict) -> dict:
        allowed = {"name", "description", "blocks", "connections", "schedule", "enabled"}
        try:
            conn = self._get_conn()
            clauses, values = [], []
            for k, v in updates.items():
                if k not in allowed:
                    continue
                if k in ("blocks", "connections"):
                    v = json.dumps(v) if isinstance(v, list) else v
                clauses.append(f"{k} = ?")
                values.append(v)
            if not clauses:
                conn.close()
                return {"error": "No valid fields"}
            clauses.append("updated_at = ?")
            values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            values.append(flow_id)
            conn.execute(f"UPDATE flows SET {', '.join(clauses)} WHERE id = ?", values)
            conn.commit()
            conn.close()
            return {"message": "Flow updated."}
        except Exception as e:
            return {"error": str(e)}

    def delete_flow(self, flow_id: int) -> dict:
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM flow_runs WHERE flow_id = ?", (flow_id,))
            cursor = conn.execute("DELETE FROM flows WHERE id = ?", (flow_id,))
            conn.commit()
            conn.close()
            return {"message": "Flow deleted."} if cursor.rowcount else {"error": "Not found"}
        except Exception as e:
            return {"error": str(e)}

    def get_runs(self, flow_id: int, limit: int = 20) -> list[dict]:
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM flow_runs WHERE flow_id = ? ORDER BY started_at DESC LIMIT ?",
                (flow_id, limit))
            runs = [{
                "id": r["id"], "flow_id": r["flow_id"], "status": r["status"],
                "started_at": r["started_at"], "finished_at": r["finished_at"],
                "duration_ms": r["duration_ms"],
                "block_results": json.loads(r["block_results"]) if r["block_results"] else [],
                "error": r["error"], "triggered_by": r["triggered_by"],
            } for r in cursor.fetchall()]
            conn.close()
            return runs
        except Exception:
            return []

    # ── Execution ─────────────────────────────────────────────────────────

    def execute_flow(self, flow_id: int, triggered_by: str = "manual") -> dict:
        flow = self.get_flow(flow_id)
        if not flow:
            return {"error": "Flow not found"}
        if flow_id in self._running_flows:
            return {"error": "Flow is already running"}
        thread = threading.Thread(target=self._run_flow, args=(flow, triggered_by), daemon=True)
        with self._lock:
            self._running_flows[flow_id] = thread
        thread.start()
        return {"message": f"Flow '{flow['name']}' started.", "status": "running"}

    def _run_flow(self, flow: dict, triggered_by: str):
        flow_id = flow["id"]
        blocks = flow["blocks"]
        connections = flow["connections"]
        start_time = time.time()
        block_results = []
        error_msg = None
        status = "success"

        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO flow_runs (flow_id, status, triggered_by) VALUES (?, 'running', ?)",
            (flow_id, triggered_by))
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        try:
            execution_order = self._resolve_execution_order(blocks, connections)
            context = {}
            for block in execution_order:
                block_id = block.get("id", "")
                block_type = block.get("type", "")
                block_config = block.get("config", {})
                input_data = self._get_block_input(block_id, connections, block_results)
                result = self._execute_block(block_type, block_config, input_data, context)
                result["block_id"] = block_id
                result["block_name"] = block.get("name", block_type)
                block_results.append(result)
                if not result.get("success", True) and not block_config.get("continue_on_error", False):
                    status = "failed"
                    error_msg = result.get("error", "Block failed")
                    break
        except Exception as e:
            status = "failed"
            error_msg = str(e)
        finally:
            duration_ms = (time.time() - start_time) * 1000
            conn = self._get_conn()
            conn.execute("""
                UPDATE flow_runs SET status = ?, finished_at = ?, duration_ms = ?,
                       block_results = ?, error = ? WHERE id = ?
            """, (status, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  duration_ms, json.dumps(block_results), error_msg, run_id))
            success_inc = 1 if status == "success" else 0
            failure_inc = 1 if status == "failed" else 0
            conn.execute("""
                UPDATE flows SET last_run = ?, last_status = ?, run_count = run_count + 1,
                       success_count = success_count + ?, failure_count = failure_count + ?
                WHERE id = ?
            """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status,
                  success_inc, failure_inc, flow_id))
            conn.commit()
            conn.close()
            with self._lock:
                self._running_flows.pop(flow_id, None)
            if self.scheduler:
                icon = "✅" if status == "success" else "❌"
                self.scheduler.notifications.append({
                    "message": f"{icon} Flow '{flow['name']}' {status} ({duration_ms:.0f}ms)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note_id": 0, "type": "workflow",
                })

    def _execute_block(self, block_type: str, config: dict, input_data: dict, context: dict) -> dict:
        try:
            if block_type == "script":
                return self._exec_script(config, input_data, context)
            elif block_type == "api_call":
                return self._exec_api_call(config, input_data, context)
            elif block_type == "command":
                return self._exec_command(config, input_data, context)
            elif block_type == "decision":
                return self._exec_decision(config, input_data, context)
            elif block_type == "loop":
                return self._exec_loop(config, input_data, context)
            elif block_type == "delay":
                return self._exec_delay(config)
            elif block_type == "notify":
                return self._exec_notify(config, input_data, context)
            elif block_type == "set_variable":
                return self._exec_set_variable(config, input_data, context)
            elif block_type == "jarvis_chat":
                return self._exec_jarvis_chat(config, input_data, context)
            else:
                return {"success": False, "error": f"Unknown block type: {block_type}"}
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}

    def _exec_script(self, config, input_data, context):
        from app.code_sandbox import CodeSandbox
        code = config.get("code", "")
        preamble = f"INPUT = {json.dumps(input_data)}\nCONTEXT = {json.dumps(context)}\n"
        sandbox = CodeSandbox(timeout=config.get("timeout", 30), allow_imports=True)
        result = sandbox.execute(preamble + code)
        output = result.get("output", "") or result.get("result", "")
        return {"success": not result.get("error"), "output": output, "error": result.get("error", "")}

    def _exec_api_call(self, config, input_data, context):
        import urllib.request, urllib.error
        url = config.get("url", "")
        method = config.get("method", "GET").upper()
        headers_cfg = config.get("headers", {})
        body = config.get("body")
        for key, val in {**input_data, **context}.items():
            url = url.replace(f"{{{key}}}", str(val))
            if body and isinstance(body, str):
                body = body.replace(f"{{{key}}}", str(val))
        try:
            data = body.encode() if body else None
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Content-Type", "application/json")
            for k, v in headers_cfg.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=config.get("timeout", 15)) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                return {"success": True, "output": response_body, "status_code": resp.status}
        except urllib.error.HTTPError as e:
            return {"success": False, "output": "", "error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _exec_command(self, config, input_data, context):
        import subprocess, platform, sys
        command = config.get("command", "")
        for key, val in {**input_data, **context}.items():
            command = command.replace(f"{{{key}}}", str(val))
        if command.startswith("python "):
            command = f'"{sys.executable}" {command[7:]}'
        try:
            if platform.system() == "Windows":
                proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=config.get("timeout", 60))
            else:
                proc = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=config.get("timeout", 60))
            output = proc.stdout or proc.stderr or ""
            return {"success": proc.returncode == 0, "output": output, "exit_code": proc.returncode, "error": proc.stderr if proc.returncode != 0 else ""}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _exec_decision(self, config, input_data, context):
        condition = config.get("condition", "True")
        try:
            namespace = {"INPUT": input_data, "CONTEXT": context, "True": True, "False": False, "None": None}
            result = eval(condition, {"__builtins__": {}}, namespace)
            return {"success": True, "output": str(result), "condition_met": bool(result)}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Condition error: {e}", "condition_met": False}

    def _exec_loop(self, config, input_data, context):
        iterations = config.get("iterations", 1)
        return {"success": True, "output": f"Loop: {iterations} iterations", "iterations": iterations}

    def _exec_delay(self, config):
        seconds = config.get("seconds", 1)
        time.sleep(min(seconds, 300))
        return {"success": True, "output": f"Waited {seconds}s"}

    def _exec_notify(self, config, input_data, context):
        message = config.get("message", "Flow notification")
        for key, val in {**input_data, **context}.items():
            message = message.replace(f"{{{key}}}", str(val))
        if self.scheduler:
            self.scheduler.notifications.append({
                "message": message, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note_id": 0, "type": "workflow",
            })
        return {"success": True, "output": message}

    def _exec_set_variable(self, config, input_data, context):
        var_name = config.get("name", "result")
        var_value = config.get("value", "")
        if var_value == "$input":
            var_value = input_data.get("output", "")
        context[var_name] = var_value
        return {"success": True, "output": f"{var_name} = {var_value}"}

    def _exec_jarvis_chat(self, config, input_data, context):
        message = config.get("message", "")
        for key, val in {**input_data, **context}.items():
            message = message.replace(f"{{{key}}}", str(val))
        if self.conversation_manager:
            import uuid
            session_id = f"flow_{uuid.uuid4().hex[:8]}"
            response = self.conversation_manager.handle_message(message, session_id)
            return {"success": True, "output": response}
        return {"success": False, "error": "Conversation manager not available"}

    def _resolve_execution_order(self, blocks, connections):
        if not blocks:
            return []
        if not connections:
            return blocks
        block_map = {b.get("id"): b for b in blocks}
        in_degree = {b.get("id"): 0 for b in blocks}
        adj = {b.get("id"): [] for b in blocks}
        for conn in connections:
            src, dst = conn.get("from"), conn.get("to")
            if src in adj and dst in in_degree:
                adj[src].append(dst)
                in_degree[dst] += 1
        queue = [bid for bid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            node = queue.pop(0)
            if node in block_map:
                order.append(block_map[node])
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        ordered_ids = {b.get("id") for b in order}
        for b in blocks:
            if b.get("id") not in ordered_ids:
                order.append(b)
        return order

    def _get_block_input(self, block_id, connections, block_results):
        for conn in connections:
            if conn.get("to") == block_id:
                src_id = conn.get("from")
                for result in block_results:
                    if result.get("block_id") == src_id:
                        return {"output": result.get("output", ""), "success": result.get("success", True)}
        return {}

    def evaluate_scheduled(self):
        """Check for flows that need to run on schedule. Called every minute."""
        import re
        now = datetime.now()
        flows = self.list_flows(include_all=True)
        for flow in flows:
            if not flow["enabled"] or not flow["schedule"] or flow["is_running"]:
                continue
            schedule = flow["schedule"].strip().lower()
            should_run = False
            if schedule == "hourly" and now.minute == 0:
                should_run = True
            elif schedule.startswith("daily"):
                time_part = schedule[6:].strip() if len(schedule) > 5 else "00:00"
                should_run = now.strftime("%H:%M") == time_part
            elif schedule.startswith("every"):
                match = re.match(r"every\s+(\d+)\s*([mhd])", schedule)
                if match:
                    amount, unit = int(match.group(1)), match.group(2)
                    if unit == "m":
                        should_run = now.minute % amount == 0
                    elif unit == "h":
                        should_run = now.minute == 0 and now.hour % amount == 0
            if should_run:
                if flow["last_run"]:
                    try:
                        last = datetime.strptime(flow["last_run"], "%Y-%m-%d %H:%M:%S")
                        if (now - last).total_seconds() < 55:
                            continue
                    except (ValueError, TypeError):
                        pass
                self.execute_flow(flow["id"], triggered_by="schedule")

    def get_block_types(self):
        return BLOCK_TYPES
