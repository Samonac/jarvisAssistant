"""Smart Home Controller for Jarvis Assistant.

Connects to Home Assistant via its REST API to control lights, switches,
and other entities. Supports Philips Hue lights through HA integration.

Can also connect directly to a Philips Hue Bridge if no Home Assistant is available.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 10


class SmartHomeController:
    """Controls smart home devices via Home Assistant REST API.

    Attributes:
        ha_url: Home Assistant base URL (e.g., http://192.168.1.100:8123)
        ha_token: Long-lived access token for Home Assistant.
    """

    def __init__(self, ha_url: str, ha_token: str):
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token
        self._headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.ha_url and self.ha_token)

    def get_states(self, domain: Optional[str] = None) -> list[dict]:
        """Get all entity states, optionally filtered by domain (e.g., 'light', 'switch')."""
        try:
            resp = requests.get(
                f"{self.ha_url}/api/states",
                headers=self._headers,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            entities = resp.json()
            if domain:
                entities = [e for e in entities if e.get("entity_id", "").startswith(f"{domain}.")]
            return [
                {
                    "entity_id": e["entity_id"],
                    "state": e["state"],
                    "name": e.get("attributes", {}).get("friendly_name", e["entity_id"]),
                    "brightness": e.get("attributes", {}).get("brightness"),
                }
                for e in entities
            ]
        except Exception as e:
            logger.error("HA get_states error: %s", e)
            return []

    def turn_on(self, entity_id: str, brightness: Optional[int] = None) -> dict:
        """Turn on an entity (light, switch, etc.)."""
        data = {"entity_id": entity_id}
        if brightness is not None:
            data["brightness"] = max(0, min(255, brightness))
        return self._call_service("homeassistant", "turn_on", data)

    def turn_off(self, entity_id: str) -> dict:
        """Turn off an entity."""
        return self._call_service("homeassistant", "turn_off", {"entity_id": entity_id})

    def toggle(self, entity_id: str) -> dict:
        """Toggle an entity."""
        return self._call_service("homeassistant", "toggle", {"entity_id": entity_id})

    def set_brightness(self, entity_id: str, brightness: int) -> dict:
        """Set brightness for a light (0-255)."""
        return self.turn_on(entity_id, brightness=brightness)

    def set_color(self, entity_id: str, rgb: list[int]) -> dict:
        """Set RGB color for a light."""
        data = {"entity_id": entity_id, "rgb_color": rgb}
        return self._call_service("light", "turn_on", data)

    def list_lights(self) -> list[dict]:
        """List all light entities with their current state."""
        return self.get_states(domain="light")

    def list_switches(self) -> list[dict]:
        """List all switch entities."""
        return self.get_states(domain="switch")

    def _call_service(self, domain: str, service: str, data: dict) -> dict:
        """Call a Home Assistant service."""
        try:
            resp = requests.post(
                f"{self.ha_url}/api/services/{domain}/{service}",
                headers=self._headers,
                json=data,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return {"success": True, "message": f"Service {domain}.{service} called successfully."}
            elif resp.status_code == 401:
                return {"success": False, "message": "Home Assistant authentication failed. Check HA_TOKEN."}
            elif resp.status_code == 404:
                return {"success": False, "message": f"Entity or service not found: {domain}.{service}"}
            else:
                return {"success": False, "message": f"HA error {resp.status_code}: {resp.text[:100]}"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": "Cannot connect to Home Assistant. Check HA_URL."}
        except requests.exceptions.Timeout:
            return {"success": False, "message": "Home Assistant request timed out."}
        except Exception as e:
            logger.error("HA service call error: %s", e)
            return {"success": False, "message": f"Smart home error: {e}"}
