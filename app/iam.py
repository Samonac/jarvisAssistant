"""Identity and Access Management (IAM) for Jarvis Assistant.

Manages users, roles, and endpoint-level permissions.
Stores data in the SQLite database.
"""

import logging
from datetime import datetime
from typing import Optional

from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

# Default permission groups with descriptions
PERMISSION_GROUPS = {
    "chat": {
        "description": "Conversational AI",
        "endpoints": ["/chat", "/chat-with-context"],
    },
    "sessions": {
        "description": "View & manage conversation history",
        "endpoints": ["/api/sessions", "/api/search"],
    },
    "notes": {
        "description": "Mental notes & reminders",
        "endpoints": ["/api/db/notes"],
    },
    "commands": {
        "description": "Execute system commands",
        "endpoints": ["/api/scripts"],
    },
    "config": {
        "description": "View & modify configuration",
        "endpoints": ["/api/config", "/api/inference"],
    },
    "monitoring": {
        "description": "System metrics & dashboard",
        "endpoints": ["/api/metrics", "/api/system", "/dashboard"],
    },
    "database": {
        "description": "Direct database access",
        "endpoints": ["/api/db"],
    },
    "bluetooth": {
        "description": "Bluetooth device management",
        "endpoints": ["/api/bluetooth", "/bluetooth"],
    },
    "files": {
        "description": "File management & uploads",
        "endpoints": ["/api/upload", "/api/fetch-url"],
    },
    "plugins": {
        "description": "Plugin management",
        "endpoints": ["/api/plugins"],
    },
    "devices": {
        "description": "Device location & permissions",
        "endpoints": ["/api/device"],
    },
    "admin": {
        "description": "Server restart, IAM, full access",
        "endpoints": ["/api/restart", "/api/iam", "/iam"],
    },
}

# Default roles
DEFAULT_ROLES = {
    "admin": {
        "description": "Full access to all features",
        "permissions": list(PERMISSION_GROUPS.keys()),
    },
    "user": {
        "description": "Standard user — chat, notes, sessions",
        "permissions": ["chat", "sessions", "notes", "monitoring", "devices"],
    },
    "viewer": {
        "description": "Read-only — can view but not modify",
        "permissions": ["chat", "sessions", "monitoring"],
    },
}


class IAMManager:
    """Manages users, roles, and permissions.

    Stores user/role data in SQLite via the database manager.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.users: dict[str, dict] = {}
        self.roles: dict[str, dict] = dict(DEFAULT_ROLES)
        self._initialize_tables()
        self._load_from_db()

    def _initialize_tables(self):
        """Create IAM tables if they don't exist."""
        conn = self.db_manager._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS iam_users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS iam_roles (
                name TEXT PRIMARY KEY,
                description TEXT,
                permissions TEXT
            );
        """)
        conn.commit()

    def _load_from_db(self):
        """Load users and roles from the database."""
        conn = self.db_manager._get_connection()

        # Load users
        try:
            cursor = conn.execute("SELECT username, password_hash, role, created_at, last_login, active FROM iam_users")
            for row in cursor.fetchall():
                self.users[row["username"]] = {
                    "username": row["username"],
                    "password_hash": row["password_hash"],
                    "role": row["role"],
                    "created_at": row["created_at"],
                    "last_login": row["last_login"],
                    "active": bool(row["active"]),
                }
        except Exception:
            pass

        # Load custom roles
        try:
            cursor = conn.execute("SELECT name, description, permissions FROM iam_roles")
            for row in cursor.fetchall():
                import json
                self.roles[row["name"]] = {
                    "description": row["description"] or "",
                    "permissions": json.loads(row["permissions"]) if row["permissions"] else [],
                }
        except Exception:
            pass

    def create_user(self, username: str, password: str, role: str = "user") -> dict:
        """Create a new user."""
        if username in self.users:
            return {"error": f"User '{username}' already exists"}
        if role not in self.roles:
            return {"error": f"Role '{role}' does not exist"}

        password_hash = generate_password_hash(password)
        conn = self.db_manager._get_connection()
        conn.execute(
            "INSERT INTO iam_users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()

        self.users[username] = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": None,
            "active": True,
        }
        return {"success": True, "message": f"User '{username}' created with role '{role}'"}

    def update_user(self, username: str, updates: dict) -> dict:
        """Update user properties (role, active status, password)."""
        if username not in self.users:
            return {"error": f"User '{username}' not found"}

        conn = self.db_manager._get_connection()
        if "role" in updates:
            if updates["role"] not in self.roles:
                return {"error": f"Role '{updates['role']}' does not exist"}
            conn.execute("UPDATE iam_users SET role = ? WHERE username = ?", (updates["role"], username))
            self.users[username]["role"] = updates["role"]

        if "active" in updates:
            conn.execute("UPDATE iam_users SET active = ? WHERE username = ?", (1 if updates["active"] else 0, username))
            self.users[username]["active"] = updates["active"]

        if "password" in updates:
            ph = generate_password_hash(updates["password"])
            conn.execute("UPDATE iam_users SET password_hash = ? WHERE username = ?", (ph, username))
            self.users[username]["password_hash"] = ph

        conn.commit()
        return {"success": True, "message": f"User '{username}' updated"}

    def delete_user(self, username: str) -> dict:
        """Delete a user."""
        if username not in self.users:
            return {"error": f"User '{username}' not found"}
        conn = self.db_manager._get_connection()
        conn.execute("DELETE FROM iam_users WHERE username = ?", (username,))
        conn.commit()
        del self.users[username]
        return {"success": True, "message": f"User '{username}' deleted"}

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Authenticate a user. Returns user dict or None."""
        user = self.users.get(username)
        if not user or not user["active"]:
            return None
        if check_password_hash(user["password_hash"], password):
            # Update last login
            conn = self.db_manager._get_connection()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("UPDATE iam_users SET last_login = ? WHERE username = ?", (now, username))
            conn.commit()
            user["last_login"] = now
            return user
        return None

    def get_user_permissions(self, username: str) -> list[str]:
        """Get the permission list for a user based on their role."""
        user = self.users.get(username)
        if not user:
            return []
        role = self.roles.get(user["role"], {})
        return role.get("permissions", [])

    def has_permission(self, username: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        perms = self.get_user_permissions(username)
        return permission in perms or "admin" in perms

    def list_users(self) -> list[dict]:
        """List all users (without password hashes)."""
        return [
            {
                "username": u["username"],
                "role": u["role"],
                "created_at": u["created_at"],
                "last_login": u["last_login"],
                "active": u["active"],
            }
            for u in self.users.values()
        ]

    def list_roles(self) -> list[dict]:
        """List all roles with their permissions."""
        return [
            {"name": name, "description": r["description"], "permissions": r["permissions"]}
            for name, r in self.roles.items()
        ]

    def update_role(self, name: str, permissions: list[str], description: str = "") -> dict:
        """Create or update a role."""
        import json
        conn = self.db_manager._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO iam_roles (name, description, permissions) VALUES (?, ?, ?)",
            (name, description, json.dumps(permissions)),
        )
        conn.commit()
        self.roles[name] = {"description": description, "permissions": permissions}
        return {"success": True, "message": f"Role '{name}' updated"}

    def delete_role(self, name: str) -> dict:
        """Delete a custom role (cannot delete default roles)."""
        if name in ("admin", "user", "viewer"):
            return {"error": "Cannot delete default roles"}
        conn = self.db_manager._get_connection()
        conn.execute("DELETE FROM iam_roles WHERE name = ?", (name,))
        conn.commit()
        self.roles.pop(name, None)
        return {"success": True, "message": f"Role '{name}' deleted"}

    @staticmethod
    def get_permission_groups() -> dict:
        """Return all available permission groups with descriptions."""
        return PERMISSION_GROUPS
