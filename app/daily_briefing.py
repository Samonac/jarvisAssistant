"""Daily Briefing for Jarvis Assistant.

Generates a morning summary combining weather, calendar, notes, and metrics.
Scheduled to run at a configurable time each day.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DailyBriefing:
    """Generates and delivers daily briefing summaries.

    Combines data from multiple sources:
    - Weather (current + forecast)
    - Calendar events (next 12 hours)
    - Active notes/reminders
    - System metrics (yesterday's usage)
    - Custom sections (user-configurable)

    Attributes:
        weather_client: Weather integration.
        calendar_client: Calendar integration.
        notes_manager: Notes manager.
        metrics_collector: Metrics collector.
        db_manager: Database manager.
        config: App configuration.
    """

    def __init__(self, db_manager, config=None):
        self.db_manager = db_manager
        self.config = config
        self.weather_client = None
        self.calendar_client = None
        self.notes_manager = None
        self.metrics_collector = None
        self.llm_client = None

        # Default settings (overridden by user preferences)
        self.default_settings = {
            "enabled": True,
            "time": "08:00",  # 24h format
            "include_weather": True,
            "include_calendar": True,
            "include_notes": True,
            "include_metrics": True,
            "include_quote": True,
            "city": None,  # Uses default from weather config
        }

    def get_settings(self, username: str = None) -> dict:
        """Get briefing settings for a user (from DB or defaults)."""
        settings = dict(self.default_settings)
        if username and self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                # Ensure table exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS briefing_settings (
                        username TEXT PRIMARY KEY,
                        enabled INTEGER DEFAULT 1,
                        time TEXT DEFAULT '08:00',
                        include_weather INTEGER DEFAULT 1,
                        include_calendar INTEGER DEFAULT 1,
                        include_notes INTEGER DEFAULT 1,
                        include_metrics INTEGER DEFAULT 1,
                        include_quote INTEGER DEFAULT 1,
                        city TEXT
                    )
                """)
                row = conn.execute(
                    "SELECT * FROM briefing_settings WHERE username = ?", (username,)
                ).fetchone()
                if row:
                    settings["enabled"] = bool(row["enabled"])
                    settings["time"] = row["time"]
                    settings["include_weather"] = bool(row["include_weather"])
                    settings["include_calendar"] = bool(row["include_calendar"])
                    settings["include_notes"] = bool(row["include_notes"])
                    settings["include_metrics"] = bool(row["include_metrics"])
                    settings["include_quote"] = bool(row["include_quote"])
                    settings["city"] = row["city"]
            except Exception as e:
                logger.warning("Failed to load briefing settings: %s", e)
        return settings

    def save_settings(self, username: str, settings: dict) -> bool:
        """Save briefing settings for a user."""
        if not self.db_manager:
            return False
        try:
            conn = self.db_manager._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS briefing_settings (
                    username TEXT PRIMARY KEY,
                    enabled INTEGER DEFAULT 1,
                    time TEXT DEFAULT '08:00',
                    include_weather INTEGER DEFAULT 1,
                    include_calendar INTEGER DEFAULT 1,
                    include_notes INTEGER DEFAULT 1,
                    include_metrics INTEGER DEFAULT 1,
                    include_quote INTEGER DEFAULT 1,
                    city TEXT
                )
            """)
            conn.execute("""
                INSERT OR REPLACE INTO briefing_settings
                (username, enabled, time, include_weather, include_calendar,
                 include_notes, include_metrics, include_quote, city)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                username,
                1 if settings.get("enabled", True) else 0,
                settings.get("time", "08:00"),
                1 if settings.get("include_weather", True) else 0,
                1 if settings.get("include_calendar", True) else 0,
                1 if settings.get("include_notes", True) else 0,
                1 if settings.get("include_metrics", True) else 0,
                1 if settings.get("include_quote", True) else 0,
                settings.get("city"),
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to save briefing settings: %s", e)
            return False

    def generate(self, username: str = None, honorific: str = "Sir") -> str:
        """Generate the daily briefing text.

        Args:
            username: User to generate briefing for (loads their settings).
            honorific: How to address the user.

        Returns:
            Formatted briefing string.
        """
        settings = self.get_settings(username)
        now = datetime.now()
        sections = []

        # Header
        greeting = self._get_greeting(now, honorific)
        sections.append(greeting)
        sections.append(f"📅 {now.strftime('%A, %B %d, %Y')}\n")

        # Weather
        if settings["include_weather"] and self.weather_client:
            weather_section = self._get_weather_section(settings.get("city"))
            if weather_section:
                sections.append(weather_section)

        # Calendar
        if settings["include_calendar"] and self.calendar_client:
            calendar_section = self._get_calendar_section()
            if calendar_section:
                sections.append(calendar_section)

        # Notes & Reminders
        if settings["include_notes"] and self.notes_manager:
            notes_section = self._get_notes_section()
            if notes_section:
                sections.append(notes_section)

        # Metrics
        if settings["include_metrics"] and self.metrics_collector:
            metrics_section = self._get_metrics_section()
            if metrics_section:
                sections.append(metrics_section)

        # Motivational quote (simple rotation)
        if settings["include_quote"]:
            sections.append(self._get_quote())

        # Footer
        sections.append(f"\n— Your faithful assistant, J.A.R.V.I.S.")

        return "\n".join(sections)

    def _get_greeting(self, now: datetime, honorific: str) -> str:
        """Generate time-appropriate greeting."""
        hour = now.hour
        if hour < 12:
            return f"☀️ Good morning, {honorific}. Here is your daily briefing."
        elif hour < 17:
            return f"🌤️ Good afternoon, {honorific}. Here is your briefing."
        else:
            return f"🌙 Good evening, {honorific}. Here is your briefing."

    def _get_weather_section(self, city: Optional[str] = None) -> Optional[str]:
        """Get weather summary for the briefing."""
        try:
            if not self.weather_client or not self.weather_client.is_configured():
                return None

            current = self.weather_client.get_current(city)
            if "error" in current:
                return None

            forecast = self.weather_client.get_forecast(city, days=1)
            today = {}
            if "error" not in forecast and forecast.get("forecasts"):
                today = forecast["forecasts"][0]

            lines = [
                "🌡️ **Weather**",
                f"  {current['city']}: {current['description']}, {current['temperature']}°C "
                f"(feels like {current['feels_like']}°C)",
                f"  Humidity: {current['humidity']}% | Wind: {current['wind_speed']} km/h",
            ]
            if today:
                lines.append(
                    f"  Today: High {today.get('temp_max')}°C / Low {today.get('temp_min')}°C, "
                    f"Rain chance: {today.get('chance_of_rain', 0)}%"
                )
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.warning("Weather section failed: %s", e)
            return None

    def _get_calendar_section(self) -> Optional[str]:
        """Get upcoming calendar events for the briefing."""
        try:
            events = self.calendar_client.get_upcoming_events(hours_ahead=12)
            if not events:
                return "📆 **Calendar**\n  No events scheduled for today.\n"

            lines = ["📆 **Calendar**"]
            for event in events[:5]:  # Max 5 events
                time_str = event.get("start", "")
                if "T" in time_str:
                    time_str = time_str.split("T")[1][:5]
                lines.append(f"  • {time_str} — {event.get('title', 'Untitled')}")
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.warning("Calendar section failed: %s", e)
            return None

    def _get_notes_section(self) -> Optional[str]:
        """Get active notes/reminders for the briefing."""
        try:
            notes = self.notes_manager.get_all_active()
            if not notes:
                return "📝 **Notes & Reminders**\n  No active notes.\n"

            # Separate due-today from general
            now = datetime.now()
            due_today = []
            general = []
            for note in notes:
                if note.get("due_date"):
                    try:
                        due = datetime.fromisoformat(note["due_date"])
                        if due.date() == now.date():
                            due_today.append(note)
                        else:
                            general.append(note)
                    except (ValueError, TypeError):
                        general.append(note)
                else:
                    general.append(note)

            lines = [f"📝 **Notes & Reminders** ({len(notes)} active)"]
            if due_today:
                lines.append("  Due today:")
                for n in due_today[:5]:
                    due_time = ""
                    if n.get("due_date") and "T" in str(n["due_date"]):
                        due_time = f" at {str(n['due_date']).split('T')[1][:5]}"
                    elif n.get("due_date") and " " in str(n["due_date"]):
                        due_time = f" at {str(n['due_date']).split(' ')[1][:5]}"
                    lines.append(f"    ⏰ {n['content'][:60]}{due_time}")
            if general:
                lines.append(f"  General notes: {len(general)} active")
                for n in general[:3]:
                    lines.append(f"    • {n['content'][:60]}")

            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.warning("Notes section failed: %s", e)
            return None

    def _get_metrics_section(self) -> Optional[str]:
        """Get yesterday's usage metrics for the briefing."""
        try:
            summary = self.metrics_collector.get_summary()
            if not summary or summary.get("total_calls", 0) == 0:
                return None

            lines = [
                "📊 **Usage Summary**",
                f"  Yesterday: {summary.get('calls_today', 0)} interactions",
                f"  Avg response: {summary.get('avg_response_ms', 0):.0f}ms",
                f"  Active sessions: {summary.get('active_sessions', 0)}",
            ]
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.warning("Metrics section failed: %s", e)
            return None

    def _get_quote(self) -> str:
        """Get a rotating motivational/informational quote."""
        import hashlib
        quotes = [
            "\"The best way to predict the future is to create it.\" — Peter Drucker",
            "\"Simplicity is the ultimate sophistication.\" — Leonardo da Vinci",
            "\"First, solve the problem. Then, write the code.\" — John Johnson",
            "\"Any sufficiently advanced technology is indistinguishable from magic.\" — Arthur C. Clarke",
            "\"The only way to do great work is to love what you do.\" — Steve Jobs",
            "\"In the middle of difficulty lies opportunity.\" — Albert Einstein",
            "\"Talk is cheap. Show me the code.\" — Linus Torvalds",
            "\"Stay hungry, stay foolish.\" — Steve Jobs",
            "\"Measure what is measurable, and make measurable what is not so.\" — Galileo",
            "\"The advance of technology is based on making it fit in so that you don't really even notice it.\" — Bill Gates",
            "\"Intelligence is the ability to adapt to change.\" — Stephen Hawking",
            "\"Shall I compare thee to a summer's day? Thou art more lovely and more temperate.\" — Shakespeare",
        ]
        # Rotate based on day of year
        day_hash = int(hashlib.md5(datetime.now().strftime("%Y-%m-%d").encode()).hexdigest(), 16)
        idx = day_hash % len(quotes)
        return f"\n💡 {quotes[idx]}"

    def check_and_deliver(self, scheduler) -> None:
        """Check if it's time to deliver the briefing and push notification.

        Called by the scheduler every minute. Checks all users' briefing times.
        Uses its own SQLite connection since this runs in the scheduler thread.
        """
        import sqlite3

        now = datetime.now()
        current_time = now.strftime("%H:%M")

        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("""
                CREATE TABLE IF NOT EXISTS briefing_settings (
                    username TEXT PRIMARY KEY,
                    enabled INTEGER DEFAULT 1,
                    time TEXT DEFAULT '08:00',
                    include_weather INTEGER DEFAULT 1,
                    include_calendar INTEGER DEFAULT 1,
                    include_notes INTEGER DEFAULT 1,
                    include_metrics INTEGER DEFAULT 1,
                    include_quote INTEGER DEFAULT 1,
                    city TEXT
                )
            """)
            # Also track last delivery to avoid duplicates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS briefing_log (
                    username TEXT,
                    delivered_date TEXT,
                    PRIMARY KEY (username, delivered_date)
                )
            """)

            cursor = conn.execute(
                "SELECT username, time FROM briefing_settings WHERE enabled = 1"
            )
            for row in cursor.fetchall():
                username = row["username"]
                briefing_time = row["time"]

                if current_time == briefing_time:
                    # Check if already delivered today
                    today = now.strftime("%Y-%m-%d")
                    already = conn.execute(
                        "SELECT 1 FROM briefing_log WHERE username = ? AND delivered_date = ?",
                        (username, today)
                    ).fetchone()
                    if already:
                        continue

                    # Generate and deliver
                    briefing_text = self.generate(username)
                    logger.info("Delivering daily briefing to user: %s", username)

                    # Push as notification
                    if scheduler:
                        scheduler.notifications.append({
                            "message": briefing_text,
                            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "note_id": 0,
                            "type": "briefing",
                        })

                    # Mark as delivered
                    conn.execute(
                        "INSERT OR IGNORE INTO briefing_log (username, delivered_date) VALUES (?, ?)",
                        (username, today)
                    )
                    conn.commit()

            conn.close()
        except Exception as e:
            logger.error("Briefing delivery check failed: %s", e)
