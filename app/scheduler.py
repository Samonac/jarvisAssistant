"""Background Scheduler for Jarvis Assistant.

Runs a background thread that checks for due notes every 10 seconds
and pushes reminders to the frontend. Tracks unacknowledged reminders
so they are re-raised at every user prompt until explicitly dismissed.
"""

import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Background scheduler that checks for due notes and fires reminders.

    Unacknowledged reminders are kept in a separate queue and re-injected
    into the conversation context at every prompt until the user acknowledges
    them (by saying "done", "ok", "acknowledged", etc.).

    Attributes:
        db_manager: Database manager for querying due notes.
        check_interval: Seconds between checks (default 10).
        notifications: Pending one-shot notification messages for the frontend.
        unacknowledged: Active reminders awaiting user acknowledgment.
    """

    def __init__(self, db_manager, llm_client=None, check_interval: int = 10):
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.check_interval = check_interval

        # One-shot notifications for the frontend polling endpoint
        self.notifications: deque = deque(maxlen=50)

        # Unacknowledged reminders: {note_id: {message, content, note_id}}
        self.unacknowledged: dict[int, dict] = {}

        # Track the last reminder that was actually displayed to the user
        self.last_displayed_note_id: Optional[int] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired_note_ids: set = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Reminder scheduler started (interval: %ds)", self.check_interval)

    def stop(self) -> None:
        """Stop the background scheduler thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_pending_notifications(self) -> list[dict]:
        """Retrieve and clear all pending one-shot notifications for the frontend.
        
        Also tracks the last displayed note_id so acknowledgment targets the right one.
        """
        notifications = []
        with self._lock:
            while self.notifications:
                try:
                    notif = self.notifications.popleft()
                    notifications.append(notif)
                    # Track the last one displayed
                    if notif.get("note_id"):
                        self.last_displayed_note_id = notif["note_id"]
                except IndexError:
                    break
        return notifications

    def get_unacknowledged_reminders(self) -> list[dict]:
        """Return all reminders that have not yet been acknowledged."""
        with self._lock:
            return list(self.unacknowledged.values())

    def acknowledge(self, note_id: int) -> bool:
        """Mark a reminder as acknowledged, removing it from the active queue
        and marking it as completed in the database.

        Args:
            note_id: The note ID to acknowledge.

        Returns:
            True if the reminder was found and acknowledged.
        """
        with self._lock:
            if note_id in self.unacknowledged:
                del self.unacknowledged[note_id]
                # Permanently mark as fired so recurring notes don't re-fire
                self._fired_note_ids.add(note_id)
                # Also mark as completed in the database so it won't fire again on restart
                try:
                    self.db_manager.complete_note(note_id)
                except Exception as e:
                    logger.warning("Failed to complete note #%d in DB: %s", note_id, e)
                logger.info("Reminder #%d acknowledged and completed", note_id)
                return True
        return False

    def acknowledge_all(self) -> int:
        """Acknowledge all pending reminders and mark them completed in DB. Returns count."""
        with self._lock:
            count = len(self.unacknowledged)
            for note_id in list(self.unacknowledged.keys()):
                self._fired_note_ids.add(note_id)
                try:
                    self.db_manager.complete_note(note_id)
                except Exception as e:
                    logger.warning("Failed to complete note #%d in DB: %s", note_id, e)
            self.unacknowledged.clear()
        return count

    def _run(self) -> None:
        """Main scheduler loop — checks every check_interval seconds."""
        self._tick_count = 0
        while not self._stop_event.is_set():
            try:
                self._check_due_notes()

                # Every 6 ticks (~60s with 10s interval), check briefings and workflows
                self._tick_count += 1
                if self._tick_count % 6 == 0:
                    self._check_briefings_and_workflows()
            except Exception as e:
                logger.error("Scheduler error: %s", e)
            self._stop_event.wait(timeout=self.check_interval)

    def _check_briefings_and_workflows(self) -> None:
        """Check daily briefings and evaluate scheduled workflows (every ~60s).
        
        Uses direct SQLite connections since this runs in the scheduler thread,
        not the main Flask thread.
        """
        try:
            if hasattr(self, '_daily_briefing') and self._daily_briefing:
                self._daily_briefing.check_and_deliver(self)
        except Exception as e:
            logger.warning("Briefing check error: %s", e)

        try:
            if hasattr(self, '_workflow_engine') and self._workflow_engine:
                self._workflow_engine.evaluate_scheduled()
        except Exception as e:
            logger.warning("Workflow evaluation error: %s", e)

        try:
            # Cron jobs (every minute)
            if hasattr(self, '_cron_manager') and self._cron_manager:
                self._cron_manager.evaluate_jobs()
        except Exception as e:
            logger.warning("Cron evaluation error: %s", e)

        try:
            # Flow engine scheduled flows (every minute)
            if hasattr(self, '_flow_engine') and self._flow_engine:
                self._flow_engine.evaluate_scheduled()
        except Exception as e:
            logger.warning("Flow engine evaluation error: %s", e)

        try:
            # Log analyzer (every ~30s, but called every 60s here)
            if hasattr(self, '_log_analyzer') and self._log_analyzer:
                self._log_analyzer.check_logs()
        except Exception as e:
            logger.warning("Log analyzer error: %s", e)

        try:
            # Backup orchestrator (checks if backup is due)
            if hasattr(self, '_backup_orchestrator') and self._backup_orchestrator:
                self._backup_orchestrator.check_and_backup()
        except Exception as e:
            logger.warning("Backup check error: %s", e)

        try:
            # Contextual suggestions (every 10 minutes = every 10th call)
            if hasattr(self, '_suggestions_engine') and self._suggestions_engine:
                if not hasattr(self, '_suggestion_counter'):
                    self._suggestion_counter = 0
                self._suggestion_counter += 1
                if self._suggestion_counter % 10 == 0:
                    self._run_suggestions()
        except Exception as e:
            logger.warning("Suggestions check error: %s", e)

    def _run_suggestions(self) -> None:
        """Run contextual suggestions analysis for all active users."""
        if not hasattr(self, '_suggestions_engine') or not self._suggestions_engine:
            return
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_manager.db_path)
            conn.row_factory = sqlite3.Row
            # Get users who have been active in the last hour
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            cursor = conn.execute(
                "SELECT DISTINCT username FROM user_activity WHERE timestamp >= ? AND username IS NOT NULL",
                (cutoff,)
            )
            active_users = [row["username"] for row in cursor.fetchall()]
            conn.close()

            for username in active_users:
                self._suggestions_engine.deliver_suggestions(username)
        except Exception as e:
            logger.warning("Suggestions delivery error: %s", e)

    def _check_due_notes(self) -> None:
        """Check for notes that are now due, execute commands, and generate reminders.
        Also auto-expires notes past their expires_at time.
        
        IMPORTANT: Due-date check runs BEFORE auto-expire to ensure reminders
        fire even if expires_at is set to the same time or shortly after due_date.
        """
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            conn.row_factory = sqlite3.Row

            # Check columns for command/recurrence/expires support
            cursor = conn.execute("PRAGMA table_info(notes)")
            columns = {row[1] for row in cursor.fetchall()}
            has_expires = "expires_at" in columns
            has_command = "command" in columns
            has_recurrence = "recurrence" in columns

            # FIRST: Check for due notes and fire reminders
            if has_command and has_recurrence:
                query = """
                    SELECT id, content, category, due_date, command, recurrence
                    FROM notes
                    WHERE status = 'active'
                      AND due_date IS NOT NULL
                      AND due_date <= ?
                """
            else:
                query = """
                    SELECT id, content, category, due_date
                    FROM notes
                    WHERE status = 'active'
                      AND due_date IS NOT NULL
                      AND due_date <= ?
                """

            cursor = conn.execute(query, (now_str,))
            due_notes = cursor.fetchall()

            if not due_notes:
                # Log occasionally to confirm scheduler is running
                logger.debug("Scheduler check: no due notes (checked at %s)", now_str)

            for note in due_notes:
                note_id = note["id"]
                if note_id in self._fired_note_ids:
                    continue

                # Double-check the note is still active (might have been completed between query and now)
                verify = conn.execute("SELECT status FROM notes WHERE id = ?", (note_id,)).fetchone()
                if not verify or verify["status"] != "active":
                    self._fired_note_ids.add(note_id)
                    continue

                self._fired_note_ids.add(note_id)

                content = note["content"]
                category = note["category"] or "Reminder"
                due_date = note["due_date"]
                command = note["command"] if has_command and "command" in note.keys() else None
                recurrence = note["recurrence"] if has_recurrence and "recurrence" in note.keys() else None

                # Execute command if specified
                command_output = ""
                if command:
                    command_output = self._execute_scheduled_command(command)

                # Generate reminder message
                if command and command_output:
                    reminder_msg = (
                        f"Scheduled task executed, Sir.\n"
                        f"Note: \"{content}\"\n"
                        f"Command: `{command}`\n"
                        f"Output:\n{command_output[:500]}\n\n"
                        f"Please reply \"done\" to acknowledge."
                    )
                else:
                    reminder_msg = self._generate_reminder(content, category, due_date)

                reminder = {
                    "message": reminder_msg,
                    "timestamp": now_str,
                    "note_id": note_id,
                    "content": content,
                    "type": "reminder",
                }

                with self._lock:
                    self.notifications.append(reminder)
                    self.unacknowledged[note_id] = reminder

                logger.info("Reminder fired for note #%d: %s", note_id, content[:50])

                # Handle recurrence: reschedule the note
                if recurrence:
                    next_due = self._calculate_next_due(now, recurrence)
                    if next_due:
                        conn.execute(
                            "UPDATE notes SET due_date = ? WHERE id = ?",
                            (next_due.strftime("%Y-%m-%d %H:%M:%S"), note_id),
                        )
                        conn.commit()
                        self._fired_note_ids.discard(note_id)
                        logger.info("Recurring note #%d rescheduled to %s", note_id, next_due)
                    else:
                        # Rescheduling failed — keep it fired (don't loop)
                        logger.warning("Could not reschedule note #%d (invalid recurrence: '%s'). Note will not repeat.", note_id, recurrence)

            # AFTER firing reminders: auto-expire notes past their expires_at
            if has_expires:
                expired = conn.execute(
                    "UPDATE notes SET status = 'completed', completed_at = ? "
                    "WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at <= ?",
                    (now_str, now_str),
                )
                if expired.rowcount > 0:
                    conn.commit()
                    logger.info("Auto-expired %d note(s)", expired.rowcount)

            conn.close()

        except Exception as e:
            logger.error("Error checking due notes: %s", e)

    def _execute_scheduled_command(self, command: str) -> str:
        """Execute a scheduled command and return its output."""
        import subprocess
        import platform as _platform
        import sys

        # Replace bare "python" with actual executable
        if command.startswith("python ") or command.startswith("python3 "):
            prefix = "python3 " if command.startswith("python3 ") else "python "
            script_part = command[len(prefix):]
            command = f'"{sys.executable}" {script_part}'

        try:
            if _platform.system() == "Windows":
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=60
                )
            else:
                result = subprocess.run(
                    command, shell=True, executable="/bin/bash",
                    capture_output=True, text=True, timeout=60
                )
            output = result.stdout or result.stderr or "(no output)"
            logger.info("Scheduled command executed: '%s' → exit %d", command[:50], result.returncode)
            return output.strip()
        except subprocess.TimeoutExpired:
            return "(command timed out after 60s)"
        except Exception as e:
            logger.error("Scheduled command error: %s", e)
            return f"(error: {e})"

    def _calculate_next_due(self, from_time: datetime, recurrence: str) -> Optional[datetime]:
        """Calculate the next due time based on recurrence pattern."""
        from datetime import timedelta as _td
        rec = recurrence.lower().strip()

        if rec == "hourly":
            return from_time + _td(hours=1)
        elif rec == "daily":
            return from_time + _td(days=1)
        elif rec == "weekly":
            return from_time + _td(weeks=1)
        elif rec.startswith("every "):
            part = rec[6:].strip()
            # Handle "Nm" placeholder (literal N)
            import re as _re
            placeholder = _re.match(r'^n([mhd])$', part, _re.IGNORECASE)
            if placeholder:
                part = "1" + placeholder.group(1)
            try:
                if part.endswith("m"):
                    return from_time + _td(minutes=int(part[:-1]))
                elif part.endswith("h"):
                    return from_time + _td(hours=int(part[:-1]))
                elif part.endswith("d"):
                    return from_time + _td(days=int(part[:-1]))
                else:
                    return from_time + _td(minutes=int(part))
            except ValueError:
                logger.warning("Invalid recurrence value: '%s'", recurrence)
                return None

        logger.warning("Unknown recurrence pattern: '%s'", recurrence)
        return None

    def _generate_reminder(self, content: str, category: str, due_date: str) -> str:
        """Generate a Jarvis-style reminder message using LLM or fallback template."""
        if self.llm_client:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are J.A.R.V.I.S. Generate a brief, polite reminder in your "
                            "formal British style. Address the user as 'Sir'. "
                            "End with: 'Please reply \"done\" to acknowledge this reminder.' "
                            "Keep it to 2-3 sentences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Remind me about (category: {category}): {content}",
                    },
                ]
                response = self.llm_client.chat(messages)
                if response and "unavailable" not in response.lower():
                    return response
            except Exception as e:
                logger.warning("LLM reminder generation failed: %s", e)

        return (
            f"Pardon the interruption, Sir. A scheduled reminder requires your attention: "
            f"\"{content}\" (Due: {due_date}). "
            f"Please reply \"done\" to acknowledge this reminder."
        )
