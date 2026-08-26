"""Configuration management for Jarvis Assistant.

Reads and validates environment variables at startup.
Uses python-dotenv for .env file support.
"""

import os
import sys
import secrets
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Default Linux-dangerous command blocklist
DEFAULT_BLOCKLIST = [
    "rm -rf /",
    "rm -rf /*",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "dd if=",
    "mkfs",
    ":(){ :|:& };:",
    "> /dev/sda",
    "chmod -R 777 /",
    "mv / ",
    "wget|sh",
    "curl|sh",
    "format",
]


class ConfigError(Exception):
    """Raised when a required configuration variable is missing."""

    pass


class Config:
    """Application configuration loaded from environment variables.

    Required variables:
        - LLM_API_KEY: API key for the LLM provider
        - LLM_PROVIDER: Name of the LLM provider ("groq", "huggingface", or "gemini")

    Optional variables with defaults are documented in .env.example.
    """

    # Cloud credentials are optional when an explicit gateway is configured.
    REQUIRED_VARS = ["LLM_API_KEY", "LLM_PROVIDER"]

    def __init__(self):
        # Load .env file if present
        load_dotenv()

        # Gateway URL takes precedence; cloud keys are optional fallbacks.
        gateway_url = os.environ.get("LLM_GATEWAY_URL", "http://192.168.1.52:8000")

        if not os.environ.get("LLM_GATEWAY_URL"):
            self._require("LLM_API_KEY")
            self._require("LLM_PROVIDER")

        # LLM_PROVIDER / LLM_API_KEY default gracefully so the gateway-only
        # setup doesn't require them to be set.
        self.llm_api_key: str = os.environ.get("LLM_API_KEY", "")
        self.llm_provider: str = os.environ.get("LLM_PROVIDER", "gateway")
        _ = gateway_url  # consumed later via os.environ in llm_client

        # Optional per-provider API keys for failover
        # Discovers ALL keys matching the pattern: PROVIDER_API_KEY, PROVIDER_API_KEY_2, PROVIDER_API_KEY_3, etc.
        self.groq_api_keys: list[str] = self._discover_keys("GROQ_API_KEY")
        self.huggingface_api_keys: list[str] = self._discover_keys("HUGGINGFACE_API_KEY")
        self.gemini_api_keys: list[str] = self._discover_keys("GEMINI_API_KEY")

        # Legacy single-key compat (still works)
        self.groq_api_key: str | None = self.groq_api_keys[0] if self.groq_api_keys else None
        self.huggingface_api_key: str | None = self.huggingface_api_keys[0] if self.huggingface_api_keys else None
        self.gemini_api_key: str | None = self.gemini_api_keys[0] if self.gemini_api_keys else None

        # Streaming chat responses (token-by-token SSE)
        # Set to true to enable streaming; requires gateway running
        self.chat_streaming: bool = os.environ.get("CHAT_STREAMING", "false").lower() in ("1", "true", "yes")

        # Agent/Autopilot mode: when enabled, Jarvis can plan and execute
        # multi-step tool sequences. When disabled, all queries use the fast
        # single-shot path (lower latency, no planning overhead).
        self.agent_mode_enabled: bool = os.environ.get("AGENT_MODE", "true").lower() in ("1", "true", "yes")

        # Local LLM Gateway URL (FastAPI on the Mac)
        self.llm_gateway_url: str = os.environ.get(
            "LLM_GATEWAY_URL", "http://192.168.1.52:8000"
        )

        # Failover order (comma-separated provider names, e.g., "gateway,groq,huggingface")
        self.llm_failover_order: list[str] = [
            p.strip() for p in os.environ.get("LLM_FAILOVER_ORDER", "").split(",") if p.strip()
        ]

        # Server settings
        self.port: int = int(os.environ.get("PORT", "5000"))

        # Command execution settings
        self.command_blocklist: list[str] = self._parse_blocklist(
            os.environ.get("COMMAND_BLOCKLIST", "")
        )
        self.command_timeout: int = int(os.environ.get("COMMAND_TIMEOUT", "60"))
        self.scan_timeout: int = int(os.environ.get("SCAN_TIMEOUT", "120"))

        # Conversation settings
        self.max_history_pairs: int = int(os.environ.get("MAX_HISTORY_PAIRS", "10"))

        # Calendar settings
        self.calendar_provider: str | None = os.environ.get("CALENDAR_PROVIDER")
        self.caldav_url: str | None = os.environ.get("CALDAV_URL")
        self.caldav_username: str | None = os.environ.get("CALDAV_USERNAME")
        self.caldav_password: str | None = os.environ.get("CALDAV_PASSWORD")
        self.google_credentials_path: str | None = os.environ.get(
            "GOOGLE_CREDENTIALS_PATH"
        )
        self.reminder_window_minutes: int = int(
            os.environ.get("REMINDER_WINDOW_MINUTES", "15")
        )

        # Email settings
        self.smtp_host: str | None = os.environ.get("SMTP_HOST")
        self.smtp_port: int = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_username: str | None = os.environ.get("SMTP_USERNAME")
        self.smtp_password: str | None = os.environ.get("SMTP_PASSWORD")
        self.smtp_from_address: str | None = os.environ.get("SMTP_FROM_ADDRESS")

        # Database settings
        self.database_path: str = os.environ.get("DATABASE_PATH", "jarvis.db")
        self.retention_days: int = int(os.environ.get("RETENTION_DAYS", "30"))

        # Web frontend settings
        self.web_username: str = os.environ.get("WEB_USERNAME", "admin")
        self.web_password: str = os.environ.get("WEB_PASSWORD", "")
        self.secret_key: str = self._get_or_create_secret_key()

        # System monitoring thresholds
        self.ram_warning_percent: float = float(
            os.environ.get("RAM_WARNING_PERCENT", "80.0")
        )
        self.disk_warning_percent: float = float(
            os.environ.get("DISK_WARNING_PERCENT", "90.0")
        )

        # Weather settings
        self.weather_api_key: str | None = os.environ.get("WEATHER_API_KEY")
        self.weather_default_city: str = os.environ.get("WEATHER_DEFAULT_CITY", "Paris")

        # User preference
        self.user_honorific: str = os.environ.get("USER_HONORIFIC", "")

    def _require(self, var_name: str) -> str:
        """Read a required environment variable or raise ConfigError."""
        value = os.environ.get(var_name)
        if not value:
            raise ConfigError(
                f"Required environment variable '{var_name}' is not set"
            )
        return value

    def _get_or_create_secret_key(self) -> str:
        """Get SECRET_KEY from env, or generate and persist one.

        If no SECRET_KEY is set, generates a random one and writes it to .env
        so sessions survive server restarts.
        """
        key = os.environ.get("SECRET_KEY", "")
        if key:
            return key

        # Generate a stable key and persist it
        key = secrets.token_hex(32)
        try:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "SECRET_KEY=" in content:
                    import re
                    content = re.sub(r"SECRET_KEY=.*", f"SECRET_KEY={key}", content)
                else:
                    content = content.rstrip() + f"\nSECRET_KEY={key}\n"
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info("Generated and persisted SECRET_KEY to .env")
        except Exception as e:
            logger.warning("Could not persist SECRET_KEY to .env: %s", e)

        return key

    def _discover_keys(self, prefix: str) -> list[str]:
        """Discover all API keys matching a prefix pattern.

        Looks for: PREFIX, PREFIX_2, PREFIX_3, ... PREFIX_N
        Returns a list of all non-empty values found.
        """
        keys = []
        # Check the base key (e.g., HUGGINGFACE_API_KEY)
        base = os.environ.get(prefix, "")
        if base:
            keys.append(base)
        # Check numbered variants (e.g., HUGGINGFACE_API_KEY_2, _3, ...)
        for i in range(2, 100):  # Support up to 99 keys per provider
            val = os.environ.get(f"{prefix}_{i}", "")
            if val:
                keys.append(val)
            else:
                break  # Stop at first gap
        return keys

    def _parse_blocklist(self, raw: str) -> list[str]:
        """Parse a comma-separated blocklist string into a list.

        If the environment variable is empty or not set, returns the default
        Linux-dangerous command blocklist.
        """
        if not raw.strip():
            return list(DEFAULT_BLOCKLIST)
        return [pattern.strip() for pattern in raw.split(",") if pattern.strip()]


def load_config() -> Config:
    """Load configuration, logging errors and exiting if required vars are missing."""
    try:
        return Config()
    except ConfigError as e:
        logger.error(str(e))
        sys.exit(1)
