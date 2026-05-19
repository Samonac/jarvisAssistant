"""Jarvis Assistant - A lightweight Flask-based personal assistant for Raspberry Pi."""

import logging
import os
import uuid

from flask import Flask

from app.config import Config, load_config

logger = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> Flask:
    """Flask application factory.

    Creates a minimal Flask app with session support, single-threaded design,
    and startup logging of provider and capabilities.

    Args:
        config: Optional Config instance. If None, loads from environment.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # Load config if not provided
    if config is None:
        config = load_config()

    # Store config on app for access in routes
    app.config["JARVIS_CONFIG"] = config

    # Flask session configuration
    app.secret_key = config.secret_key

    # Register routes
    from app.routes import register_routes

    register_routes(app)

    # Log startup information
    _log_startup(config)

    return app


def _log_startup(config: Config) -> None:
    """Log provider and capabilities at startup."""
    capabilities = _get_capabilities(config)
    logger.info(
        "Jarvis Assistant starting - Provider: %s, Capabilities: %s",
        config.llm_provider,
        capabilities,
    )


def _get_capabilities(config: Config) -> list[str]:
    """Determine available capabilities based on configuration."""
    capabilities = ["chat", "command_execution", "web_search", "network_scan"]

    if config.calendar_provider:
        capabilities.append("calendar")

    if config.smtp_host:
        capabilities.append("email")

    return capabilities
