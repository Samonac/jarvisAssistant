"""Workflow Automation Engine for Jarvis Assistant.

Defines trigger→condition→action pipelines that automate tasks.
Workflows are stored in SQLite and evaluated by the scheduler.

Supported triggers:
- schedule: Cron-like time patterns (daily at HH:MM, weekdays, weekends, every Nm/Nh)
- gps_enter: When a device enters a geographic zone (lat/lon + radius)
- gps_exit: When a device leaves a geographic zone
- event: When a specific system event occurs (e.g., "reminder_fired", "command_completed")

Supported actions:
- notify: Push a notification message
- run_command: Execute a shell command
- briefing: Generate and deliver a daily briefing
- smart_home: Control a smart home device
- note: Create a mental note
- webhook: Call an external URL
"""

import json
import logging
import math
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Manages and evaluates workflow automation rules.

    Attributes:
        db_manager: Database manager for persistent storage.
        scheduler: Reference to the reminder scheduler (for notifications).
        conversation_manager: Reference for executing actions.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.scheduler = None
        self.conversation_manager = None
        self._last_gps_positions: dict = {}  # device_id -> {lat, lon, timestamp}
        self._zone_states: dict = {}  # (workflow_id, device_id) -> bool (inside zone)
        self._lock = threading.Lock()

        # Ensure tables exist
        self._init_tables()

    def _get_thread_connection(self):
        """Get a thread-local SQLite connection for use in background threads."""
        import sqlite3
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create workflow tables if they don't exist."""
        try:
            conn = self._get_thread_connection()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    enabled INTEGER DEFAULT 1,
                    trigger_type TEXT NOT NULL,
                    trigger_config TEXT NOT NULL,
                    conditions TEXT,
                    action_type TEXT NOT NULL,
                    action_config TEXT NOT NULL,
                    username TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_triggered DATETIME,
                    trigger_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS workflow_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id INTEGER NOT NULL,
                    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    result TEXT,
                    success INTEGER DEFAULT 1,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to initialize workflow tables: %s", e)

    # --- CRUD Operations ---

    def create_workflow(self, name: str, trigger_type: str, trigger_config: dict,
                       action_type: str, action_config: dict,
                       description: str = "", conditions: dict = None,
                       username: str = None) -> dict:
        """Create a new workflow rule.

        Args:
            name: Human-readable name for the workflow.
            trigger_type: One of: schedule, gps_enter, gps_exit, event.
            trigger_config: Configuration for the trigger (JSON-serializable).
            action_type: One of: notify, run_command, briefing, smart_home, note, webhook.
            action_config: Configuration for the action (JSON-serializable).
            description: Optional description.
            conditions: Optional conditions that must be met (JSON-serializable).
            username: Owner of the workflow.

        Returns:
            Dict with workflow ID and status.
        """
        valid_triggers = {"schedule", "gps_enter", "gps_exit", "event"}
        valid_actions = {"notify", "run_command", "briefing", "smart_home", "note", "webhook"}

        if trigger_type not in valid_triggers:
            return {"error": f"Invalid trigger type: '{trigger_type}'. Valid: {valid_triggers}"}
        if action_type not in valid_actions:
            return {"error": f"Invalid action type: '{action_type}'. Valid: {valid_actions}"}

        try:
            conn = self._get_thread_connection()
            cursor = conn.execute("""
                INSERT INTO workflows (name, description, trigger_type, trigger_config,
                                       conditions, action_type, action_config, username)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, description, trigger_type,
                json.dumps(trigger_config),
                json.dumps(conditions) if conditions else None,
                action_type, json.dumps(action_config),
                username,
            ))
            conn.commit()
            wf_id = cursor.lastrowid
            conn.close()
            return {"id": wf_id, "message": f"Workflow '{name}' created successfully."}
        except Exception as e:
            logger.error("Failed to create workflow: %s", e)
            return {"error": str(e)}

    def list_workflows(self, username: str = None, include_disabled: bool = False) -> list[dict]:
        """List all workflows, optionally filtered by user.
        
        Thread-safe: uses its own connection to work from any thread.
        """
        try:
            conn = self._get_thread_connection()
            if username:
                query = "SELECT * FROM workflows WHERE username = ?"
                params = [username]
            else:
                query = "SELECT * FROM workflows"
                params = []

            if not include_disabled:
                query += " AND enabled = 1" if "WHERE" in query else " WHERE enabled = 1"

            cursor = conn.execute(query, params)
            workflows = []
            for row in cursor.fetchall():
                workflows.append({
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "enabled": bool(row["enabled"]),
                    "trigger_type": row["trigger_type"],
                    "trigger_config": json.loads(row["trigger_config"]),
                    "conditions": json.loads(row["conditions"]) if row["conditions"] else None,
                    "action_type": row["action_type"],
                    "action_config": json.loads(row["action_config"]),
                    "username": row["username"],
                    "created_at": row["created_at"],
                    "last_triggered": row["last_triggered"],
                    "trigger_count": row["trigger_count"],
                })
            conn.close()
            return workflows
        except Exception as e:
            logger.error("Failed to list workflows: %s", e)
            return []

    def get_workflow(self, workflow_id: int) -> Optional[dict]:
        """Get a single workflow by ID."""
        try:
            conn = self._get_thread_connection()
            row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
            if not row:
                conn.close()
                return None
            result = {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "enabled": bool(row["enabled"]),
                "trigger_type": row["trigger_type"],
                "trigger_config": json.loads(row["trigger_config"]),
                "conditions": json.loads(row["conditions"]) if row["conditions"] else None,
                "action_type": row["action_type"],
                "action_config": json.loads(row["action_config"]),
                "username": row["username"],
                "created_at": row["created_at"],
                "last_triggered": row["last_triggered"],
                "trigger_count": row["trigger_count"],
            }
            conn.close()
            return result
        except Exception as e:
            logger.error("Failed to get workflow: %s", e)
            return None

    def update_workflow(self, workflow_id: int, updates: dict) -> dict:
        """Update a workflow's fields."""
        allowed_fields = {"name", "description", "enabled", "trigger_type",
                          "trigger_config", "action_type", "action_config", "conditions"}
        try:
            conn = self._get_thread_connection()
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key not in allowed_fields:
                    continue
                if key in ("trigger_config", "action_config", "conditions"):
                    value = json.dumps(value) if value is not None else None
                set_clauses.append(f"{key} = ?")
                values.append(value)

            if not set_clauses:
                conn.close()
                return {"error": "No valid fields to update"}

            values.append(workflow_id)
            conn.execute(
                f"UPDATE workflows SET {', '.join(set_clauses)} WHERE id = ?",
                values
            )
            conn.commit()
            conn.close()
            return {"message": "Workflow updated successfully."}
        except Exception as e:
            logger.error("Failed to update workflow: %s", e)
            return {"error": str(e)}

    def delete_workflow(self, workflow_id: int) -> dict:
        """Delete a workflow and its logs."""
        try:
            conn = self._get_thread_connection()
            conn.execute("DELETE FROM workflow_log WHERE workflow_id = ?", (workflow_id,))
            cursor = conn.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
            conn.commit()
            count = cursor.rowcount
            conn.close()
            if count == 0:
                return {"error": "Workflow not found"}
            return {"message": "Workflow deleted."}
        except Exception as e:
            logger.error("Failed to delete workflow: %s", e)
            return {"error": str(e)}

    def get_workflow_logs(self, workflow_id: int, limit: int = 20) -> list[dict]:
        """Get execution logs for a workflow."""
        try:
            conn = self._get_thread_connection()
            cursor = conn.execute(
                "SELECT * FROM workflow_log WHERE workflow_id = ? ORDER BY triggered_at DESC LIMIT ?",
                (workflow_id, limit)
            )
            results = [
                {
                    "id": row["id"],
                    "workflow_id": row["workflow_id"],
                    "triggered_at": row["triggered_at"],
                    "result": row["result"],
                    "success": bool(row["success"]),
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            return results
        except Exception as e:
            logger.error("Failed to get workflow logs: %s", e)
            return []

    # --- Evaluation (called by scheduler) ---

    def evaluate_scheduled(self) -> None:
        """Evaluate all schedule-based workflows. Called every minute by the scheduler."""
        now = datetime.now()
        workflows = self.list_workflows(include_disabled=False)

        for wf in workflows:
            if wf["trigger_type"] != "schedule":
                continue

            if self._matches_schedule(now, wf["trigger_config"]):
                # Check if already triggered this minute
                last = wf.get("last_triggered")
                if last:
                    try:
                        last_dt = datetime.fromisoformat(last)
                        if last_dt.strftime("%Y-%m-%d %H:%M") == now.strftime("%Y-%m-%d %H:%M"):
                            continue  # Already triggered this minute
                    except (ValueError, TypeError):
                        pass

                # Check conditions
                if wf.get("conditions") and not self._check_conditions(wf["conditions"]):
                    continue

                # Execute action
                self._execute_action(wf)

    def evaluate_gps(self, device_id: str, lat: float, lon: float) -> None:
        """Evaluate GPS-based workflows when a device reports its position.

        Called by the location reporting endpoint.
        """
        workflows = self.list_workflows(include_disabled=False)

        for wf in workflows:
            if wf["trigger_type"] not in ("gps_enter", "gps_exit"):
                continue

            config = wf["trigger_config"]
            zone_lat = config.get("latitude", 0)
            zone_lon = config.get("longitude", 0)
            zone_radius = config.get("radius_meters", 100)

            distance = self._haversine(lat, lon, zone_lat, zone_lon)
            inside = distance <= zone_radius

            key = (wf["id"], device_id)
            was_inside = self._zone_states.get(key)

            # Update state
            with self._lock:
                self._zone_states[key] = inside

            # Detect transitions
            if wf["trigger_type"] == "gps_enter" and inside and was_inside is False:
                logger.info("GPS enter trigger: device %s entered zone for workflow '%s'", device_id, wf["name"])
                self._execute_action(wf)
            elif wf["trigger_type"] == "gps_exit" and not inside and was_inside is True:
                logger.info("GPS exit trigger: device %s left zone for workflow '%s'", device_id, wf["name"])
                self._execute_action(wf)

    def evaluate_event(self, event_name: str, event_data: dict = None) -> None:
        """Evaluate event-based workflows when a system event occurs."""
        workflows = self.list_workflows(include_disabled=False)

        for wf in workflows:
            if wf["trigger_type"] != "event":
                continue
            config = wf["trigger_config"]
            if config.get("event_name") == event_name:
                self._execute_action(wf, context=event_data)

    # --- Schedule Matching ---

    def _matches_schedule(self, now: datetime, config: dict) -> bool:
        """Check if the current time matches a schedule trigger config.

        Config options:
            time: "HH:MM" — specific time of day
            days: ["mon","tue",...] or "weekdays" or "weekends" or "daily"
            interval: "every 5m" / "every 2h" — interval-based
        """
        # Interval-based
        interval = config.get("interval")
        if interval:
            return self._matches_interval(now, config)

        # Time-based
        target_time = config.get("time")
        if not target_time:
            return False

        current_time = now.strftime("%H:%M")
        if current_time != target_time:
            return False

        # Check day filter
        days = config.get("days", "daily")
        if days == "daily":
            return True
        elif days == "weekdays":
            return now.weekday() < 5  # Mon-Fri
        elif days == "weekends":
            return now.weekday() >= 5  # Sat-Sun
        elif isinstance(days, list):
            day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            current_day = day_names[now.weekday()]
            return current_day in [d.lower()[:3] for d in days]

        return True

    def _matches_interval(self, now: datetime, config: dict) -> bool:
        """Check if enough time has passed for an interval trigger."""
        import re
        interval = config.get("interval", "")
        match = re.match(r"every\s+(\d+)\s*([mhd])", interval.lower())
        if not match:
            return False

        amount = int(match.group(1))
        unit = match.group(2)

        if unit == "m":
            return now.minute % amount == 0 and now.second < 60
        elif unit == "h":
            return now.minute == 0 and now.hour % amount == 0
        elif unit == "d":
            return now.hour == 0 and now.minute == 0

        return False

    # --- Condition Checking ---

    def _check_conditions(self, conditions: dict) -> bool:
        """Evaluate conditions before executing an action.

        Supported conditions:
            time_range: {"after": "HH:MM", "before": "HH:MM"}
            day_of_week: ["mon", "tue", ...]
        """
        now = datetime.now()

        if "time_range" in conditions:
            tr = conditions["time_range"]
            after = tr.get("after", "00:00")
            before = tr.get("before", "23:59")
            current = now.strftime("%H:%M")
            if not (after <= current <= before):
                return False

        if "day_of_week" in conditions:
            day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            current_day = day_names[now.weekday()]
            allowed = [d.lower()[:3] for d in conditions["day_of_week"]]
            if current_day not in allowed:
                return False

        return True

    # --- Action Execution ---

    def _execute_action(self, workflow: dict, context: dict = None) -> None:
        """Execute a workflow's action."""
        action_type = workflow["action_type"]
        action_config = workflow["action_config"]
        wf_id = workflow["id"]

        logger.info("Executing workflow '%s' (id=%d, action=%s)", workflow["name"], wf_id, action_type)

        result = ""
        success = True

        try:
            if action_type == "notify":
                result = self._action_notify(action_config, workflow)
            elif action_type == "run_command":
                result = self._action_run_command(action_config)
            elif action_type == "briefing":
                result = self._action_briefing(action_config, workflow)
            elif action_type == "smart_home":
                result = self._action_smart_home(action_config)
            elif action_type == "note":
                result = self._action_note(action_config)
            elif action_type == "webhook":
                result = self._action_webhook(action_config)
            else:
                result = f"Unknown action type: {action_type}"
                success = False
        except Exception as e:
            result = f"Error: {e}"
            success = False
            logger.error("Workflow action failed (id=%d): %s", wf_id, e)

        # Update workflow stats
        try:
            conn = self._get_thread_connection()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "UPDATE workflows SET last_triggered = ?, trigger_count = trigger_count + 1 WHERE id = ?",
                (now_str, wf_id)
            )
            conn.execute(
                "INSERT INTO workflow_log (workflow_id, result, success) VALUES (?, ?, ?)",
                (wf_id, result[:500], 1 if success else 0)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to log workflow execution: %s", e)

    def _action_notify(self, config: dict, workflow: dict) -> str:
        """Push a notification message."""
        message = config.get("message", "Workflow triggered.")
        # Variable substitution
        message = message.replace("{time}", datetime.now().strftime("%H:%M"))
        message = message.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        message = message.replace("{workflow_name}", workflow.get("name", ""))

        if self.scheduler:
            self.scheduler.notifications.append({
                "message": message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note_id": 0,
                "type": "workflow",
            })
        return f"Notification sent: {message[:100]}"

    def _action_run_command(self, config: dict) -> str:
        """Execute a shell command."""
        command = config.get("command", "")
        if not command:
            return "No command specified"

        if self.conversation_manager and self.conversation_manager.command_executor:
            result = self.conversation_manager.command_executor.execute(command)
            output = result.get("stdout", "") or result.get("stderr", "") or "(no output)"

            # Optionally notify with output
            if config.get("notify_output", False) and self.scheduler:
                self.scheduler.notifications.append({
                    "message": f"⚙️ Workflow command completed:\n`{command}`\nOutput: {output[:200]}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note_id": 0,
                    "type": "workflow",
                })
            return output[:500]
        return "Command executor not available"

    def _action_briefing(self, config: dict, workflow: dict) -> str:
        """Generate and deliver a daily briefing."""
        # Import here to avoid circular imports
        from app.daily_briefing import DailyBriefing
        briefing = DailyBriefing(self.db_manager)
        if self.conversation_manager:
            briefing.weather_client = self.conversation_manager.weather_client
            briefing.calendar_client = self.conversation_manager.calendar_client
            briefing.notes_manager = self.conversation_manager.notes_manager
            briefing.metrics_collector = self.conversation_manager.metrics_collector

        username = workflow.get("username")
        text = briefing.generate(username)

        if self.scheduler:
            self.scheduler.notifications.append({
                "message": text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note_id": 0,
                "type": "briefing",
            })
        return "Briefing delivered"

    def _action_smart_home(self, config: dict) -> str:
        """Control a smart home device."""
        if not self.conversation_manager or not self.conversation_manager.smart_home:
            return "Smart home not configured"

        sh = self.conversation_manager.smart_home
        action = config.get("action", "turn_on")
        entity_id = config.get("entity_id", "")

        if action == "turn_on":
            brightness = config.get("brightness")
            result = sh.turn_on(entity_id, brightness=brightness)
        elif action == "turn_off":
            result = sh.turn_off(entity_id)
        elif action == "set_color":
            rgb = config.get("rgb", [255, 255, 255])
            result = sh.set_color(entity_id, rgb)
        else:
            result = {"error": f"Unknown smart home action: {action}"}

        return str(result)

    def _action_note(self, config: dict) -> str:
        """Create a mental note."""
        if not self.conversation_manager or not self.conversation_manager.notes_manager:
            return "Notes manager not available"

        content = config.get("content", "Workflow-generated note")
        content = content.replace("{time}", datetime.now().strftime("%H:%M"))
        content = content.replace("{date}", datetime.now().strftime("%Y-%m-%d"))

        note_id = self.db_manager.save_note(content=content, category=config.get("category"))
        return f"Note created (ID: {note_id})"

    def _action_webhook(self, config: dict) -> str:
        """Call an external webhook URL."""
        import urllib.request
        import urllib.error

        url = config.get("url", "")
        method = config.get("method", "POST").upper()
        payload = config.get("payload")

        if not url:
            return "No webhook URL specified"

        try:
            data = json.dumps(payload).encode() if payload else None
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return f"Webhook {method} {url} → {resp.status}"
        except urllib.error.URLError as e:
            return f"Webhook error: {e}"
        except Exception as e:
            return f"Webhook error: {e}"

    # --- Utility ---

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two GPS coordinates."""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
