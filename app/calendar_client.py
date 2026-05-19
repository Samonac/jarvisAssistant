"""Calendar Client for Jarvis Assistant.

Connects to calendar services to manage events. Uses a provider abstraction
to support Google Calendar and CalDAV.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

from app.config import Config

logger = logging.getLogger(__name__)


class CalendarProvider(ABC):
    """Abstract base class for calendar providers."""

    @abstractmethod
    def get_events(self, start: datetime, end: datetime) -> list[dict]:
        """Retrieve events within the given time range.

        Returns:
            List of event dicts with: event_id, title, start, end, description, location.
        """
        pass

    @abstractmethod
    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        description: str = "",
    ) -> dict:
        """Create a new calendar event.

        Returns:
            Dict representing the created event.
        """
        pass

    @abstractmethod
    def update_event(self, event_id: str, **kwargs) -> dict:
        """Update an existing event's fields.

        Returns:
            Dict representing the updated event.
        """
        pass

    @abstractmethod
    def delete_event(self, event_id: str) -> bool:
        """Delete an event by ID.

        Returns:
            True on success.
        """
        pass


class GoogleCalendarProvider(CalendarProvider):
    """Calendar provider using Google Calendar API with OAuth2 credentials.

    Requires a service account credentials JSON file.
    """

    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self._service = None

    def _get_service(self):
        """Lazily initialize the Google Calendar API service."""
        if self._service is not None:
            return self._service

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/calendar"],
            )
            self._service = build("calendar", "v3", credentials=credentials)
            return self._service
        except ImportError:
            raise RuntimeError(
                "Google Calendar API libraries not installed. "
                "Install google-api-python-client and google-auth."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Calendar: {e}")

    def get_events(self, start: datetime, end: datetime) -> list[dict]:
        service = self._get_service()
        try:
            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start.isoformat() + "Z",
                    timeMax=end.isoformat() + "Z",
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])
            return [
                {
                    "event_id": e.get("id", ""),
                    "title": e.get("summary", "Untitled"),
                    "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
                    "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date", "")),
                    "description": e.get("description", ""),
                    "location": e.get("location", ""),
                }
                for e in events
            ]
        except Exception as e:
            logger.error("Google Calendar get_events error: %s", e)
            raise

    def create_event(
        self, title: str, start: datetime, end: datetime, description: str = ""
    ) -> dict:
        service = self._get_service()
        try:
            event_body = {
                "summary": title,
                "description": description,
                "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
            }
            event = service.events().insert(calendarId="primary", body=event_body).execute()
            return {
                "event_id": event.get("id", ""),
                "title": event.get("summary", title),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "description": description,
            }
        except Exception as e:
            logger.error("Google Calendar create_event error: %s", e)
            raise

    def update_event(self, event_id: str, **kwargs) -> dict:
        service = self._get_service()
        try:
            event = service.events().get(calendarId="primary", eventId=event_id).execute()
            if "title" in kwargs:
                event["summary"] = kwargs["title"]
            if "description" in kwargs:
                event["description"] = kwargs["description"]
            if "start" in kwargs:
                event["start"] = {"dateTime": kwargs["start"].isoformat(), "timeZone": "UTC"}
            if "end" in kwargs:
                event["end"] = {"dateTime": kwargs["end"].isoformat(), "timeZone": "UTC"}

            updated = service.events().update(
                calendarId="primary", eventId=event_id, body=event
            ).execute()
            return {
                "event_id": updated.get("id", event_id),
                "title": updated.get("summary", ""),
                "start": updated.get("start", {}).get("dateTime", ""),
                "end": updated.get("end", {}).get("dateTime", ""),
                "description": updated.get("description", ""),
            }
        except Exception as e:
            logger.error("Google Calendar update_event error: %s", e)
            raise

    def delete_event(self, event_id: str) -> bool:
        service = self._get_service()
        try:
            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return True
        except Exception as e:
            logger.error("Google Calendar delete_event error: %s", e)
            raise


class CalDAVProvider(CalendarProvider):
    """Calendar provider using CalDAV protocol.

    Works with any CalDAV-compatible server (Nextcloud, Radicale, etc.).
    """

    def __init__(self, url: str, username: str, password: str):
        self.url = url
        self.username = username
        self.password = password
        self._client = None
        self._calendar = None

    def _get_calendar(self):
        """Lazily initialize the CalDAV client and get the principal calendar."""
        if self._calendar is not None:
            return self._calendar

        try:
            import caldav

            self._client = caldav.DAVClient(
                url=self.url, username=self.username, password=self.password
            )
            principal = self._client.principal()
            calendars = principal.calendars()
            if not calendars:
                raise RuntimeError("No calendars found on the CalDAV server.")
            self._calendar = calendars[0]
            return self._calendar
        except ImportError:
            raise RuntimeError(
                "CalDAV library not installed. Install the 'caldav' package."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CalDAV server: {e}")

    def get_events(self, start: datetime, end: datetime) -> list[dict]:
        calendar = self._get_calendar()
        try:
            events = calendar.date_search(start=start, end=end, expand=True)
            results = []
            for event in events:
                vevents = event.vobject_instance.vevent_list if hasattr(event.vobject_instance, 'vevent_list') else [event.vobject_instance.vevent]
                for vevent in vevents:
                    results.append({
                        "event_id": str(event.url),
                        "title": str(vevent.summary.value) if hasattr(vevent, "summary") else "Untitled",
                        "start": vevent.dtstart.value.isoformat() if hasattr(vevent, "dtstart") else "",
                        "end": vevent.dtend.value.isoformat() if hasattr(vevent, "dtend") else "",
                        "description": str(vevent.description.value) if hasattr(vevent, "description") else "",
                        "location": str(vevent.location.value) if hasattr(vevent, "location") else "",
                    })
            return results
        except Exception as e:
            logger.error("CalDAV get_events error: %s", e)
            raise

    def create_event(
        self, title: str, start: datetime, end: datetime, description: str = ""
    ) -> dict:
        calendar = self._get_calendar()
        try:
            import uuid

            uid = str(uuid.uuid4())
            vcal = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Jarvis Assistant//EN
BEGIN:VEVENT
UID:{uid}
DTSTART:{start.strftime('%Y%m%dT%H%M%SZ')}
DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}
SUMMARY:{title}
DESCRIPTION:{description}
END:VEVENT
END:VCALENDAR"""
            event = calendar.save_event(vcal)
            return {
                "event_id": str(event.url),
                "title": title,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "description": description,
            }
        except Exception as e:
            logger.error("CalDAV create_event error: %s", e)
            raise

    def update_event(self, event_id: str, **kwargs) -> dict:
        calendar = self._get_calendar()
        try:
            event = calendar.event_by_url(event_id)
            vevent = event.vobject_instance.vevent
            if "title" in kwargs:
                vevent.summary.value = kwargs["title"]
            if "description" in kwargs:
                vevent.description.value = kwargs["description"]
            if "start" in kwargs:
                vevent.dtstart.value = kwargs["start"]
            if "end" in kwargs:
                vevent.dtend.value = kwargs["end"]
            event.save()
            return {
                "event_id": event_id,
                "title": str(vevent.summary.value) if hasattr(vevent, "summary") else "",
                "start": vevent.dtstart.value.isoformat() if hasattr(vevent, "dtstart") else "",
                "end": vevent.dtend.value.isoformat() if hasattr(vevent, "dtend") else "",
                "description": str(vevent.description.value) if hasattr(vevent, "description") else "",
            }
        except Exception as e:
            logger.error("CalDAV update_event error: %s", e)
            raise

    def delete_event(self, event_id: str) -> bool:
        calendar = self._get_calendar()
        try:
            event = calendar.event_by_url(event_id)
            event.delete()
            return True
        except Exception as e:
            logger.error("CalDAV delete_event error: %s", e)
            raise


class CalendarClient:
    """High-level calendar facade that delegates to the configured provider.

    Attributes:
        provider: The underlying calendar provider.
        reminder_window_minutes: How far ahead to look for reminders.
    """

    def __init__(self, provider: CalendarProvider, reminder_window_minutes: int = 15):
        self.provider = provider
        self.reminder_window_minutes = reminder_window_minutes

    def get_upcoming_events(self, hours_ahead: int = 24) -> list[dict]:
        """Get events in the next N hours."""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)
        return self.provider.get_events(now, end)

    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        description: str = "",
    ) -> dict:
        """Create a new event."""
        return self.provider.create_event(title, start, end, description)

    def modify_event(self, event_id: str, **kwargs) -> dict:
        """Modify an existing event."""
        return self.provider.update_event(event_id, **kwargs)

    def delete_event(self, event_id: str) -> bool:
        """Delete an event."""
        return self.provider.delete_event(event_id)

    def get_due_reminders(self) -> list[dict]:
        """Get events starting within the reminder window."""
        now = datetime.utcnow()
        window_end = now + timedelta(minutes=self.reminder_window_minutes)
        return self.provider.get_events(now, window_end)


def create_calendar_provider(config: Config) -> Optional[CalendarProvider]:
    """Factory function to create the appropriate calendar provider.

    Returns None if no calendar provider is configured.
    """
    if not config.calendar_provider:
        return None

    provider_type = config.calendar_provider.lower().strip()

    if provider_type == "google":
        if not config.google_credentials_path:
            logger.warning("Google Calendar configured but GOOGLE_CREDENTIALS_PATH not set.")
            return None
        return GoogleCalendarProvider(config.google_credentials_path)
    elif provider_type == "caldav":
        if not config.caldav_url or not config.caldav_username or not config.caldav_password:
            logger.warning("CalDAV configured but connection settings incomplete.")
            return None
        return CalDAVProvider(config.caldav_url, config.caldav_username, config.caldav_password)
    else:
        logger.warning("Unknown calendar provider: '%s'", config.calendar_provider)
        return None
