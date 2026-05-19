"""Authentication Manager for Jarvis Assistant.

Handles user authentication for the web frontend using Flask sessions
and password hashing. Simple username/password validation against
environment-configured credentials.
"""

import functools
import logging

from flask import session, redirect, url_for, request
from werkzeug.security import generate_password_hash, check_password_hash

from app.config import Config

logger = logging.getLogger(__name__)


class AuthManager:
    """Manages authentication for the web frontend.

    Uses Flask sessions for login state and werkzeug for password hashing.

    Attributes:
        username: The configured admin username.
        password_hash: Hashed version of the configured password.
        enabled: Whether authentication is enabled (requires WEB_PASSWORD to be set).
    """

    def __init__(self, config: Config):
        self.username = config.web_username
        self.enabled = bool(config.web_password)

        if self.enabled:
            self.password_hash = generate_password_hash(config.web_password)
        else:
            self.password_hash = ""
            logger.warning(
                "WEB_PASSWORD not set. Web frontend authentication is disabled. "
                "Set WEB_PASSWORD to enable authentication."
            )

    def authenticate(self, username: str, password: str) -> bool:
        """Validate credentials.

        Args:
            username: Submitted username.
            password: Submitted password.

        Returns:
            True if credentials are valid, False otherwise.
        """
        if not self.enabled:
            return False

        if username != self.username:
            return False

        return check_password_hash(self.password_hash, password)

    def create_session(self, username: str) -> None:
        """Store user info in Flask session after successful login.

        Args:
            username: The authenticated username.
        """
        session["authenticated"] = True
        session["username"] = username

    def destroy_session(self) -> None:
        """Clear the Flask session (logout)."""
        session.clear()

    def is_authenticated(self) -> bool:
        """Check if the current request has a valid session.

        Returns:
            True if the user is authenticated.
        """
        if not self.enabled:
            return True  # Auth disabled = always authenticated
        return session.get("authenticated", False)

    def login_required(self, f):
        """Decorator to protect routes.

        Redirects to /login if no valid session exists.
        API routes (starting with /api/) return 401 instead of redirecting.
        """

        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.is_authenticated():
                if request.path.startswith("/api/"):
                    from flask import jsonify
                    return jsonify({"error": "Authentication required"}), 401
                return redirect("/login")
            return f(*args, **kwargs)

        return decorated_function
