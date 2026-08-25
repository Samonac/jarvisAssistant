"""Background autopilot orchestrator for the nightly self-improvement mode (Phase 5).

Runs entirely in-process (no external cron/systemd), because its trigger
condition needs live application state: the current time AND whether the
user has been actively chatting recently. A background thread polls
periodically; each poll either does nothing (outside the window, recent
chat activity, or disabled) or processes exactly one task end-to-end using
the same building blocks as the interactive coding agent (Phases 2-4):
CodingAgentLoop + TaskSnapshot + apply_verification_gate.

Safety posture:
- Every task gets a pre-task snapshot; anything that isn't fully verified
  (self-reported "done" AND the verification command passes) is rolled back.
- A task that needs to ask_user is parked as "needs_clarification" rather
  than guessed at or retried in a loop — no human is present overnight.
- A UI/UX-affecting change is verified but held back from being considered
  finished until the user explicitly confirms it (see confirm_task()).
- A cycle is skipped entirely (no task started) if RAM/disk usage is already
  above a safe threshold (Pi Zero 2W has 512MB RAM total) — see
  _has_sufficient_resources(). Each task also gets a hard wall-clock budget
  (max_task_seconds) so a stuck/looping model can't run all night.

Nightly memory consolidation:
- Independent of the `enabled` (autonomous coding) toggle, once per calendar
  day (during the same quiet-hours window) the day's conversations are
  summarized per user into the knowledge base via MemoryConsolidator — see
  app.memory_consolidator. This is what lets Jarvis "remember" past
  exchanges and get used to each user's typical requirements over time.
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Optional

from app.agent_session import AgentSessionConfig
from app.autopilot.discovery import discover_tasks
from app.autopilot.task_queue import AutopilotTaskQueue
from app.coding_agent.loop import CodingAgentLoop
from app.coding_agent.session_store import AgentSessionStore
from app.coding_agent.snapshot import TaskSnapshot
from app.coding_agent.tools import build_default_tools
from app.coding_agent.verify import apply_verification_gate
from app.memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_START_HOUR = 2
DEFAULT_WINDOW_END_HOUR = 6
DEFAULT_INACTIVITY_MINUTES = 60
DEFAULT_POLL_INTERVAL_SECONDS = 300
DEFAULT_MAX_TASK_SECONDS = 600  # 10 minutes — generous for CPU-only Pi inference, still bounded
DEFAULT_RAM_LIMIT_PERCENT = 85.0
DEFAULT_DISK_LIMIT_PERCENT = 95.0
DEFAULT_CPU_TEMP_LIMIT_CELSIUS = 80.0  # Pi SoCs typically start throttling around 80-85C

# Heuristic for "this task probably needs a human to eyeball the result".
UI_KEYWORDS = ("template", "static", " ui ", " ux ", "frontend", "css", "html", "layout", "button", "page")


class AutopilotManager:
    """Coordinates the nightly autonomous coding cycle."""

    def __init__(
        self,
        config,
        db_manager,
        project_dir: str,
        notes_manager=None,
        system_monitor=None,
        kb_manager=None,
        window_start_hour: int = DEFAULT_WINDOW_START_HOUR,
        window_end_hour: int = DEFAULT_WINDOW_END_HOUR,
        inactivity_minutes: int = DEFAULT_INACTIVITY_MINUTES,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        max_task_seconds: int = DEFAULT_MAX_TASK_SECONDS,
        ram_limit_percent: float = DEFAULT_RAM_LIMIT_PERCENT,
        disk_limit_percent: float = DEFAULT_DISK_LIMIT_PERCENT,
        cpu_temp_limit_celsius: float = DEFAULT_CPU_TEMP_LIMIT_CELSIUS,
    ):
        self.config = config
        self.project_dir = project_dir
        self.notes_manager = notes_manager
        self.system_monitor = system_monitor
        self.window_start_hour = window_start_hour
        self.window_end_hour = window_end_hour
        self.inactivity_minutes = inactivity_minutes
        self.poll_interval_seconds = poll_interval_seconds
        self.max_task_seconds = max_task_seconds
        self.ram_limit_percent = ram_limit_percent
        self.disk_limit_percent = disk_limit_percent
        self.cpu_temp_limit_celsius = cpu_temp_limit_celsius

        self.task_queue = AutopilotTaskQueue(db_manager)
        self.session_store = AgentSessionStore(db_manager)
        self.snapshot = TaskSnapshot(project_dir)

        self.memory_consolidator: Optional[MemoryConsolidator] = None
        if kb_manager is not None:
            try:
                from app.llm_client import create_failover_client
                llm_client = create_failover_client(config)
                self.memory_consolidator = MemoryConsolidator(db_manager, kb_manager, llm_client)
            except Exception as e:
                logger.warning("Could not initialize memory consolidator: %s", e)

        self.enabled = False
        self.last_activity: datetime = datetime.now()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- background thread lifecycle (called once at app startup/shutdown) ----

    def start_thread(self) -> None:
        """Start the background poll thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Autopilot background thread started (poll every %ds)", self.poll_interval_seconds)

    def stop_thread(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            try:
                self.run_cycle_if_due()
            except Exception as e:
                logger.error("Autopilot cycle failed: %s", e)

    # ---- chat/API-facing controls ----

    def enable(self) -> None:
        with self._lock:
            self.enabled = True
        logger.info("Autopilot mode enabled")

    def disable(self) -> None:
        with self._lock:
            self.enabled = False
        logger.info("Autopilot mode disabled")

    def record_activity(self) -> None:
        """Called on every chat message — resets the inactivity clock."""
        with self._lock:
            self.last_activity = datetime.now()

    def status_dict(self) -> dict:
        with self._lock:
            enabled = self.enabled
            last_activity = self.last_activity
        now = datetime.now()
        return {
            "enabled": enabled,
            "window": f"{self.window_start_hour:02d}:00-{self.window_end_hour:02d}:00",
            "in_window_now": self._in_window(now),
            "last_activity": last_activity.isoformat(),
            "minutes_since_activity": round((now - last_activity).total_seconds() / 60, 1),
            "queued_tasks": len(self.task_queue.list_tasks(status="queued")),
            "awaiting_confirmation": len(self.task_queue.list_tasks(status="awaiting_user_confirmation")),
            "resource_check": self._insufficient_resources_reason(),
            "memory_consolidation_pending": (
                self.memory_consolidator.is_due(now) if self.memory_consolidator else None
            ),
        }

    # ---- core cycle ----

    def run_cycle_if_due(self, force: bool = False) -> dict:
        """Process at most one thing this cycle, if conditions are met (or always, if force=True).

        Gating order:
          1. Quiet-hours window + chat inactivity (skipped entirely if force=True)
          2. Resource guard (RAM/disk/CPU temp) — ALWAYS applies, even with force=True,
             since it's a hardware safety limit, not a scheduling preference.
          3. Nightly memory consolidation, if due — independent of the `enabled`
             (autonomous coding) toggle, since it carries no code-change risk.
          4. The `enabled` toggle then gates whether a coding task may run.
        """
        now = datetime.now()

        if not force:
            with self._lock:
                last_activity = self.last_activity
            if not self._in_window(now):
                return {"ran": False, "reason": "outside_window"}
            if now - last_activity < timedelta(minutes=self.inactivity_minutes):
                return {"ran": False, "reason": "recent_chat_activity"}

        resource_reason = self._insufficient_resources_reason()
        if resource_reason:
            logger.warning("Autopilot cycle skipped: %s", resource_reason)
            return {"ran": False, "reason": "insufficient_resources", "detail": resource_reason}

        if self.memory_consolidator and self.memory_consolidator.is_due(now):
            result = self.memory_consolidator.consolidate(now)
            logger.info("Nightly memory consolidation ran: %s", result)
            return {"ran": True, "action": "memory_consolidation", **result}

        if not force:
            with self._lock:
                enabled = self.enabled
            if not enabled:
                return {"ran": False, "reason": "disabled"}

        task_row = self.task_queue.next_queued()
        if task_row is None:
            task_row = self._discover_and_enqueue()
            if task_row is None:
                return {"ran": False, "reason": "no_tasks"}

        return self._run_task(task_row)

    def _in_window(self, now: datetime) -> bool:
        return self.window_start_hour <= now.hour < self.window_end_hour

    def _insufficient_resources_reason(self) -> Optional[str]:
        """Check RAM/disk usage against the configured limits.

        Returns None if resources are fine (or unreadable — fail open, since
        a -1 sentinel from SystemMonitor means "couldn't measure", not "0% used").
        """
        if self.system_monitor is None:
            return None
        try:
            metrics = self.system_monitor.get_metrics()
        except Exception as e:
            logger.warning("Autopilot resource check failed, proceeding anyway: %s", e)
            return None

        if 0 <= metrics.ram_percent and metrics.ram_percent > self.ram_limit_percent:
            return f"RAM usage {metrics.ram_percent:.1f}% exceeds limit {self.ram_limit_percent:.1f}%"
        if 0 <= metrics.disk_percent and metrics.disk_percent > self.disk_limit_percent:
            return f"Disk usage {metrics.disk_percent:.1f}% exceeds limit {self.disk_limit_percent:.1f}%"
        cpu_temp = getattr(metrics, "cpu_temp_celsius", -1.0)
        if 0 <= cpu_temp and cpu_temp > self.cpu_temp_limit_celsius:
            return f"CPU temperature {cpu_temp:.1f}°C exceeds limit {self.cpu_temp_limit_celsius:.1f}°C"
        return None

    def _discover_and_enqueue(self) -> Optional[dict]:
        tools = build_default_tools(self.config, self.project_dir)
        discovered = discover_tasks(self.project_dir, tools.command_executor)
        if not discovered:
            return None
        task_id = self.task_queue.add_task(discovered[0], source="discovered")
        return self.task_queue.get(task_id)

    def _run_task(self, task_row: dict) -> dict:
        task_id = task_row["id"]
        task_text = task_row["task"]
        self.task_queue.update_status(task_id, "in_progress")

        try:
            session = AgentSessionConfig(self.config, effort="standard")
        except (ValueError, NotImplementedError) as e:
            # No usable LLM configured — the safe fallback is to do nothing risky.
            self.task_queue.update_status(task_id, "skipped", notes=str(e))
            return {"ran": True, "task_id": task_id, "status": "skipped", "reason": str(e)}

        tools = build_default_tools(self.config, self.project_dir)
        session_id = uuid.uuid4().hex
        self.snapshot.create(session_id)
        loop = CodingAgentLoop(session, tools, self.project_dir)
        deadline = datetime.now() + timedelta(seconds=self.max_task_seconds)

        try:
            state = loop.start(task_text, session_id=session_id, deadline=deadline)
        except Exception as e:
            logger.error("Autopilot task #%s failed: %s", task_id, e)
            self.snapshot.restore(session_id)
            self.task_queue.update_status(task_id, "rolled_back", session_id=session_id, notes=str(e))
            return {"ran": True, "task_id": task_id, "status": "rolled_back", "reason": str(e)}

        if state.status == "awaiting_user":
            # No human present overnight — park it rather than guessing an answer.
            self.snapshot.restore(session_id)
            self.task_queue.update_status(
                task_id, "needs_clarification", session_id=session_id,
                notes=f"Needs clarification: {state.pending_question}",
            )
            self.session_store.save(state)
            return {"ran": True, "task_id": task_id, "status": "needs_clarification"}

        if state.status in ("max_iterations_reached", "time_budget_exceeded", "error"):
            # Incomplete/errored work is never left lying around unattended.
            self.snapshot.restore(session_id)
            self.task_queue.update_status(
                task_id, "rolled_back", session_id=session_id, notes=state.error or state.summary
            )
            self.session_store.save(state)
            return {"ran": True, "task_id": task_id, "status": "rolled_back"}

        state = apply_verification_gate(state, tools, self.snapshot, task_row.get("verify_command"))
        self.session_store.save(state)

        if state.status == "rolled_back":
            self.task_queue.update_status(task_id, "rolled_back", session_id=session_id, notes=state.error)
            return {"ran": True, "task_id": task_id, "status": "rolled_back"}

        # state.status == "done" and verified — but UI/UX changes still need a human look.
        if _touches_ui(task_text, state):
            self.task_queue.update_status(
                task_id, "awaiting_user_confirmation", session_id=session_id, notes=state.summary
            )
            self._propose_confirmation(task_id, task_text)
            return {"ran": True, "task_id": task_id, "status": "awaiting_user_confirmation"}

        self.task_queue.update_status(task_id, "done", session_id=session_id, notes=state.summary)
        return {"ran": True, "task_id": task_id, "status": "done"}

    def confirm_task(self, task_id: int, accepted: bool) -> dict:
        """Finalize a task that was left awaiting_user_confirmation.

        Args:
            task_id: The autopilot task id.
            accepted: True to keep the change (snapshot discarded, confirmed_done),
                False to reject it (snapshot restored, rolled_back).
        """
        task_row = self.task_queue.get(task_id)
        if task_row is None:
            return {"error": f"No task found with id {task_id}"}
        if task_row["status"] != "awaiting_user_confirmation":
            return {"error": f"Task {task_id} is not awaiting confirmation (status: '{task_row['status']}')."}

        session_id = task_row["session_id"]
        if accepted:
            self.snapshot.discard(session_id)
            self.task_queue.update_status(task_id, "confirmed_done")
            return {"task_id": task_id, "status": "confirmed_done"}

        self.snapshot.restore(session_id)
        self.task_queue.update_status(task_id, "rolled_back", notes="Rejected by user during confirmation.")
        return {"task_id": task_id, "status": "rolled_back"}

    def _propose_confirmation(self, task_id: int, task_text: str) -> None:
        """Best-effort: leave a note asking the user to review a UI/UX change."""
        if not self.notes_manager:
            return
        try:
            self.notes_manager.add_note(
                content=(
                    f'Autopilot finished a UI/UX-affecting task overnight (#{task_id}): "{task_text}". '
                    f"It passed verification but needs your visual confirmation before it's considered done."
                ),
                category="autopilot",
            )
        except Exception as e:
            logger.warning("Failed to create confirmation note for task #%s: %s", task_id, e)


def _touches_ui(task_text: str, state) -> bool:
    """Heuristic: does this task look like it changed something a human should eyeball?"""
    lower_task = f" {task_text.lower()} "
    if any(keyword in lower_task for keyword in UI_KEYWORDS):
        return True
    for record in state.history:
        if record.tool in ("write_file", "delete_file"):
            path = str(record.args.get("path", "")).lower()
            if "templates" in path or "static" in path:
                return True
    return False
