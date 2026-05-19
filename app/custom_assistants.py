"""Custom Assistants Manager for Jarvis Assistant.

Allows users to create their own AI assistants with custom system prompts
and inference parameters. Each assistant is a named configuration that can
be selected in the chat interface.

Features:
- Create/edit/delete custom assistants
- Per-assistant system prompt (fully editable)
- Per-assistant inference parameters (temperature, top_p, max_tokens, etc.)
- Per-user assistants (users see only their own + shared ones)
- Default "Jarvis" assistant always available
- Switch between assistants mid-conversation
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Default inference parameters for new assistants
DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.1,
}


class CustomAssistantsManager:
    """Manages custom AI assistant configurations.

    Attributes:
        db_manager: Database manager for persistent storage.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS custom_assistants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    system_prompt TEXT NOT NULL,
                    inference_params TEXT NOT NULL,
                    icon TEXT DEFAULT '🤖',
                    shared INTEGER DEFAULT 0,
                    username TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init custom assistants tables: %s", e)

    def create_assistant(self, name: str, system_prompt: str, username: str,
                         description: str = "", inference_params: dict = None,
                         icon: str = "🤖", shared: bool = False) -> dict:
        """Create a new custom assistant.

        Args:
            name: Display name for the assistant.
            system_prompt: The full system prompt text.
            username: Owner of the assistant.
            description: Short description of what this assistant does.
            inference_params: Custom inference parameters (or defaults).
            icon: Emoji icon for the assistant.
            shared: If True, all users can use this assistant.

        Returns:
            Dict with assistant ID or error.
        """
        if not name or not name.strip():
            return {"error": "Name is required."}
        if not system_prompt or not system_prompt.strip():
            return {"error": "System prompt is required."}

        params = inference_params or dict(DEFAULT_INFERENCE_PARAMS)
        # Validate params
        params = self._validate_params(params)

        try:
            conn = self._get_conn()
            cursor = conn.execute("""
                INSERT INTO custom_assistants (name, description, system_prompt, inference_params,
                                               icon, shared, username)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name.strip(), description.strip(), system_prompt,
                  json.dumps(params), icon, 1 if shared else 0, username))
            conn.commit()
            assistant_id = cursor.lastrowid
            conn.close()
            return {"id": assistant_id, "message": f"Assistant '{name}' created."}
        except Exception as e:
            return {"error": str(e)}

    def list_assistants(self, username: str) -> list[dict]:
        """List assistants available to a user (own + shared).

        Args:
            username: Current user.

        Returns:
            List of assistant dicts.
        """
        try:
            conn = self._get_conn()
            cursor = conn.execute("""
                SELECT * FROM custom_assistants
                WHERE username = ? OR shared = 1
                ORDER BY name ASC
            """, (username,))
            assistants = []
            for r in cursor.fetchall():
                assistants.append({
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "system_prompt": r["system_prompt"],
                    "inference_params": json.loads(r["inference_params"]),
                    "icon": r["icon"],
                    "shared": bool(r["shared"]),
                    "username": r["username"],
                    "is_owner": r["username"] == username,
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                })
            conn.close()
            return assistants
        except Exception as e:
            logger.error("Failed to list assistants: %s", e)
            return []

    def get_assistant(self, assistant_id: int) -> Optional[dict]:
        """Get a single assistant by ID."""
        try:
            conn = self._get_conn()
            r = conn.execute("SELECT * FROM custom_assistants WHERE id = ?", (assistant_id,)).fetchone()
            conn.close()
            if not r:
                return None
            return {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "system_prompt": r["system_prompt"],
                "inference_params": json.loads(r["inference_params"]),
                "icon": r["icon"],
                "shared": bool(r["shared"]),
                "username": r["username"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        except Exception:
            return None

    def update_assistant(self, assistant_id: int, updates: dict, username: str, is_admin: bool = False) -> dict:
        """Update an assistant's configuration.

        Args:
            assistant_id: ID of the assistant to update.
            updates: Fields to update.
            username: Current user (must be owner or admin).
            is_admin: If True, can edit any assistant.

        Returns:
            Success or error dict.
        """
        allowed = {"name", "description", "system_prompt", "inference_params", "icon", "shared"}
        try:
            conn = self._get_conn()
            # Check ownership (admins can edit any)
            row = conn.execute("SELECT username FROM custom_assistants WHERE id = ?", (assistant_id,)).fetchone()
            if not row:
                conn.close()
                return {"error": "Assistant not found."}
            if row["username"] != username and not is_admin:
                conn.close()
                return {"error": "You can only edit your own assistants."}

            clauses, values = [], []
            for k, v in updates.items():
                if k not in allowed:
                    continue
                if k == "inference_params":
                    v = json.dumps(self._validate_params(v if isinstance(v, dict) else json.loads(v)))
                elif k == "shared":
                    v = 1 if v else 0
                clauses.append(f"{k} = ?")
                values.append(v)

            if not clauses:
                conn.close()
                return {"error": "No valid fields to update."}

            clauses.append("updated_at = ?")
            values.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            values.append(assistant_id)

            conn.execute(f"UPDATE custom_assistants SET {', '.join(clauses)} WHERE id = ?", values)
            conn.commit()
            conn.close()
            return {"message": "Assistant updated."}
        except Exception as e:
            return {"error": str(e)}

    def delete_assistant(self, assistant_id: int, username: str, is_admin: bool = False) -> dict:
        """Delete an assistant (owner or admin only)."""
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT username FROM custom_assistants WHERE id = ?", (assistant_id,)).fetchone()
            if not row:
                conn.close()
                return {"error": "Assistant not found."}
            if row["username"] != username and not is_admin:
                conn.close()
                return {"error": "You can only delete your own assistants."}

            conn.execute("DELETE FROM custom_assistants WHERE id = ?", (assistant_id,))
            conn.commit()
            conn.close()
            return {"message": "Assistant deleted."}
        except Exception as e:
            return {"error": str(e)}

    def get_default_system_prompt(self) -> str:
        """Get the default Jarvis system prompt for use as a template."""
        from app.conversation_manager import JARVIS_SYSTEM_PROMPT_TEMPLATE
        return JARVIS_SYSTEM_PROMPT_TEMPLATE

    def _validate_params(self, params: dict) -> dict:
        """Validate and clamp inference parameters."""
        validated = dict(DEFAULT_INFERENCE_PARAMS)
        if "temperature" in params:
            validated["temperature"] = max(0.0, min(2.0, float(params["temperature"])))
        if "top_p" in params:
            validated["top_p"] = max(0.0, min(1.0, float(params["top_p"])))
        if "max_tokens" in params:
            validated["max_tokens"] = max(64, min(4096, int(params["max_tokens"])))
        if "frequency_penalty" in params:
            validated["frequency_penalty"] = max(-2.0, min(2.0, float(params["frequency_penalty"])))
        if "presence_penalty" in params:
            validated["presence_penalty"] = max(-2.0, min(2.0, float(params["presence_penalty"])))
        return validated
