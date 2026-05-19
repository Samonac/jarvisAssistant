"""User Preferences Manager for Jarvis Assistant.

Stores per-user settings like honorific, inference parameters,
voice profile data, and other personalization options.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PREFERENCES = {
    "honorific": "Sir",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 1024,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "tts_voice": "",
    "tts_rate": 0.92,
    "tts_pitch": 0.85,
    "stt_language": "en-GB",
    "voice_enrolled": False,
}


class UserPreferencesManager:
    """Manages per-user preferences stored in SQLite.

    Each user has their own set of preferences that override global defaults.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._initialize_table()

    def _initialize_table(self):
        """Create the user_preferences table if it doesn't exist."""
        conn = self.db_manager._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                username TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (username, key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                username TEXT PRIMARY KEY,
                profile_data TEXT,
                enrolled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                sample_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()

    def get_preferences(self, username: str) -> dict:
        """Get all preferences for a user (merged with defaults)."""
        prefs = dict(DEFAULT_PREFERENCES)
        conn = self.db_manager._get_connection()
        cursor = conn.execute(
            "SELECT key, value FROM user_preferences WHERE username = ?",
            (username,),
        )
        for row in cursor.fetchall():
            key = row["key"]
            value = row["value"]
            # Type-cast based on default type
            if key in DEFAULT_PREFERENCES:
                default_type = type(DEFAULT_PREFERENCES[key])
                try:
                    if default_type == bool:
                        value = value.lower() in ("true", "1", "yes")
                    elif default_type == float:
                        value = float(value)
                    elif default_type == int:
                        value = int(value)
                except (ValueError, AttributeError):
                    pass
            prefs[key] = value
        return prefs

    def set_preference(self, username: str, key: str, value) -> None:
        """Set a single preference for a user."""
        conn = self.db_manager._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (username, key, value) VALUES (?, ?, ?)",
            (username, key, str(value)),
        )
        conn.commit()

    def set_preferences(self, username: str, prefs: dict) -> None:
        """Set multiple preferences at once."""
        for key, value in prefs.items():
            self.set_preference(username, key, value)

    def get_honorific(self, username: str) -> str:
        """Get the user's preferred honorific."""
        prefs = self.get_preferences(username)
        return prefs.get("honorific", "Sir")

    def get_inference_params(self, username: str) -> dict:
        """Get the user's inference parameters."""
        prefs = self.get_preferences(username)
        return {
            "temperature": prefs["temperature"],
            "top_p": prefs["top_p"],
            "max_tokens": prefs["max_tokens"],
            "frequency_penalty": prefs["frequency_penalty"],
            "presence_penalty": prefs["presence_penalty"],
        }

    # ── Voice Profile Management ───────────────────────────────────────────

    def save_voice_profile(self, username: str, profile_data: dict, sample_count: int = 0) -> None:
        """Save a user's voice profile (features extracted from enrollment samples)."""
        conn = self.db_manager._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO voice_profiles (username, profile_data, sample_count) VALUES (?, ?, ?)",
            (username, json.dumps(profile_data), sample_count),
        )
        conn.commit()
        self.set_preference(username, "voice_enrolled", "true")

    def get_voice_profile(self, username: str) -> Optional[dict]:
        """Get a user's voice profile."""
        conn = self.db_manager._get_connection()
        cursor = conn.execute(
            "SELECT profile_data, sample_count FROM voice_profiles WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row and row["profile_data"]:
            return {
                "profile": json.loads(row["profile_data"]),
                "sample_count": row["sample_count"],
            }
        return None

    def get_all_voice_profiles(self) -> dict:
        """Get all enrolled voice profiles (for speaker identification)."""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("SELECT username, profile_data FROM voice_profiles")
        profiles = {}
        for row in cursor.fetchall():
            if row["profile_data"]:
                profiles[row["username"]] = json.loads(row["profile_data"])
        return profiles

    def is_voice_enrolled(self, username: str) -> bool:
        """Check if a user has completed voice enrollment."""
        profile = self.get_voice_profile(username)
        return profile is not None and profile.get("sample_count", 0) >= 3


class FaceProfileManager:
    """Manages facial recognition profiles for biometric authentication.

    Stores face descriptor vectors (128-dimensional embeddings) from multiple
    angles (front, up, down, left, right) for each user.

    Face detection and feature extraction happens client-side using face-api.js.
    Only the numeric descriptors are sent to the server — no images are stored.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._initialize_table()

    def _initialize_table(self):
        """Create the face_profiles table if it doesn't exist."""
        conn = self.db_manager._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_profiles (
                username TEXT NOT NULL,
                angle TEXT NOT NULL,
                descriptor TEXT NOT NULL,
                enrolled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (username, angle)
            )
        """)
        conn.commit()

    def save_face_descriptor(self, username: str, angle: str, descriptor: list[float]) -> None:
        """Save a face descriptor for a specific angle.

        Args:
            username: The user's username.
            angle: One of 'front', 'up', 'down', 'left', 'right'.
            descriptor: 128-dimensional face descriptor vector.
        """
        conn = self.db_manager._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO face_profiles (username, angle, descriptor) VALUES (?, ?, ?)",
            (username, angle, json.dumps(descriptor)),
        )
        conn.commit()

    def get_face_profile(self, username: str) -> dict:
        """Get all face descriptors for a user.

        Returns dict of {angle: descriptor_list}.
        """
        conn = self.db_manager._get_connection()
        cursor = conn.execute(
            "SELECT angle, descriptor FROM face_profiles WHERE username = ?",
            (username,),
        )
        profile = {}
        for row in cursor.fetchall():
            profile[row["angle"]] = json.loads(row["descriptor"])
        return profile

    def is_face_enrolled(self, username: str) -> bool:
        """Check if a user has completed face enrollment (all 5 angles)."""
        profile = self.get_face_profile(username)
        required_angles = {"front", "up", "down", "left", "right"}
        return required_angles.issubset(set(profile.keys()))

    def get_all_face_profiles(self) -> dict:
        """Get all enrolled face profiles for identification.

        Returns {username: {angle: descriptor}}.
        """
        conn = self.db_manager._get_connection()
        cursor = conn.execute("SELECT username, angle, descriptor FROM face_profiles")
        profiles = {}
        for row in cursor.fetchall():
            username = row["username"]
            if username not in profiles:
                profiles[username] = {}
            profiles[username][row["angle"]] = json.loads(row["descriptor"])
        return profiles

    def identify_face(self, descriptor: list[float], threshold: float = 0.6) -> Optional[tuple]:
        """Identify a face by comparing against all enrolled profiles.

        Args:
            descriptor: 128-dimensional face descriptor to match.
            threshold: Maximum Euclidean distance for a match (lower = stricter).

        Returns:
            Tuple of (username, distance) or None if no match.
        """
        profiles = self.get_all_face_profiles()
        best_match = None
        best_distance = float('inf')

        for username, angles in profiles.items():
            for angle, enrolled_desc in angles.items():
                dist = self._euclidean_distance(descriptor, enrolled_desc)
                if dist < best_distance:
                    best_distance = dist
                    best_match = username

        if best_match and best_distance <= threshold:
            return (best_match, best_distance)
        return None

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        if len(a) != len(b):
            return float('inf')
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
