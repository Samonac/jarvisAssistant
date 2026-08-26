"""Main entry point for Jarvis Assistant.

Creates and starts the Flask app with all components wired together.
Binds to 0.0.0.0 for LAN access on the Raspberry Pi.
Single-threaded to minimize memory usage.
"""

import logging
import os
import re
import sys

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class _QuietPollingAccessFilter(logging.Filter):
    """Hide successful health-style polling requests from the access log."""

    _SUCCESSFUL_POLL = re.compile(
        r'"(?:GET|HEAD) /api/(?:notifications|restart/status|chat-progress/) HTTP/[^" ]+" 200(?:\s|$)'
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not self._SUCCESSFUL_POLL.search(record.getMessage())


def main():
    """Initialize all components and start the Flask server."""
    from app.config import load_config
    from app.database_manager import DatabaseManager
    from app.llm_client import create_llm_client, create_failover_client
    from app.conversation_manager import ConversationManager
    from app.notes_manager import NotesManager
    from app.command_executor import CommandExecutor
    from app.web_searcher import WebSearcher
    from app.network_scanner import NetworkScanner
    from app.email_client import EmailClient
    from app.calendar_client import CalendarClient, create_calendar_provider
    from app.metrics_collector import MetricsCollector
    from app.system_monitor import SystemMonitor
    from app.auth import AuthManager
    from app.weather_client import WeatherClient
    from app.smart_home import SmartHomeController
    from app.file_manager import FileManager
    from app.ssh_client import SSHClient
    from app.plugin_manager import PluginManager
    from app.wake_word import WakeWordDetector
    from app.file_watcher import FileWatcher
    from app import create_app

    # Load configuration
    config = load_config()

    # Determine project directory (used for CWD, file access, etc.)
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize database
    db_manager = DatabaseManager(db_path=config.database_path)
    db_manager.initialize()
    logger.info("Database initialized at: %s", config.database_path)

    # Prune old conversations on startup
    pruned = db_manager.prune_old_conversations(retention_days=config.retention_days)
    if pruned > 0:
        logger.info("Pruned %d old conversation records", pruned)

    # Initialize LLM client (with automatic failover if multiple providers configured)
    llm_client = create_failover_client(config)

    # Initialize tools
    command_executor = CommandExecutor(
        blocklist=config.command_blocklist, timeout=config.command_timeout, cwd=project_dir
    )
    web_searcher = WebSearcher()
    network_scanner = NetworkScanner(timeout=config.scan_timeout)
    email_client = EmailClient(config)
    notes_manager = NotesManager(
        db_manager, reminder_window_minutes=config.reminder_window_minutes
    )
    metrics_collector = MetricsCollector(db_manager)
    system_monitor = SystemMonitor(
        ram_warning_percent=config.ram_warning_percent,
        disk_warning_percent=config.disk_warning_percent,
    )

    # Initialize calendar (optional)
    calendar_client = None
    calendar_provider = create_calendar_provider(config)
    if calendar_provider:
        calendar_client = CalendarClient(
            calendar_provider,
            reminder_window_minutes=config.reminder_window_minutes,
        )
        logger.info("Calendar integration enabled: %s", config.calendar_provider)

    # Initialize conversation manager with all tools
    conversation_manager = ConversationManager(
        llm_client=llm_client,
        config=config,
        db_manager=db_manager,
    )
    conversation_manager.notes_manager = notes_manager
    conversation_manager.command_executor = command_executor
    conversation_manager.web_searcher = web_searcher
    conversation_manager.network_scanner = network_scanner
    conversation_manager.email_client = email_client
    conversation_manager.calendar_client = calendar_client
    conversation_manager.metrics_collector = metrics_collector

    # Initialize weather client (optional)
    weather_client = None
    if config.weather_api_key:
        weather_client = WeatherClient(config.weather_api_key, config.weather_default_city)
        conversation_manager.weather_client = weather_client
        logger.info("Weather integration enabled (default city: %s)", config.weather_default_city)

    # Initialize smart home (optional)
    ha_url = os.environ.get("HA_URL", "")
    ha_token = os.environ.get("HA_TOKEN", "")
    if ha_url and ha_token:
        smart_home = SmartHomeController(ha_url, ha_token)
        conversation_manager.smart_home = smart_home
        logger.info("Smart home integration enabled (HA: %s)", ha_url)

    # Initialize file manager
    file_base_dir = os.environ.get("FILE_BASE_DIR", os.path.expanduser("~"))
    uploads_dir = os.path.join(project_dir, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    file_manager = FileManager(base_dir=file_base_dir, extra_dirs=[project_dir, uploads_dir])
    conversation_manager.file_manager = file_manager
    logger.info("File management enabled (base: %s, project: %s)", file_base_dir, project_dir)

    # Initialize SSH client (optional)
    ssh_hosts_raw = os.environ.get("SSH_HOSTS", "")
    if ssh_hosts_raw:
        import json as _json
        try:
            ssh_hosts = _json.loads(ssh_hosts_raw)
            ssh_client = SSHClient(hosts=ssh_hosts)
            conversation_manager.ssh_client = ssh_client
            logger.info("SSH integration enabled (%d host(s))", len(ssh_hosts))
        except Exception as e:
            logger.warning("Failed to parse SSH_HOSTS: %s", e)

    # Initialize plugin manager
    plugins_dir = os.environ.get("PLUGINS_DIR", "plugins")
    plugin_manager = PluginManager(plugins_dir=plugins_dir)
    plugin_manager.load_all()
    conversation_manager.plugin_manager = plugin_manager

    # Initialize script runner (scans scripts/ folder for API endpoints)
    from app.script_runner import ScriptRunner
    script_runner = ScriptRunner(scripts_dir=os.path.join(project_dir, "scripts"))
    script_runner.discover_scripts()

    # Initialize wake word detector (optional, requires hardware)
    wake_word_enabled = os.environ.get("WAKE_WORD_ENABLED", "").lower() in ("1", "true", "yes")
    wake_word_detector = None
    if wake_word_enabled:
        def on_wake():
            logger.info("Wake word detected! Activating...")
            # Push a notification to the frontend
            if scheduler:
                scheduler.notifications.append({
                    "message": "At your service, Sir. I heard you call.",
                    "timestamp": "",
                    "note_id": 0,
                    "type": "wake_word",
                })

        wake_word_detector = WakeWordDetector(threshold=0.5, on_wake=on_wake)
        if wake_word_detector.start():
            logger.info("Wake word detection active — say 'Hey Jarvis'")
        else:
            logger.warning("Wake word detection unavailable (missing dependencies)")

    # Initialize auth manager
    auth_manager = AuthManager(config)

    # Initialize reminder scheduler
    from app.scheduler import ReminderScheduler

    scheduler = ReminderScheduler(
        db_manager=db_manager,
        llm_client=llm_client,
        check_interval=10,  # Check every 10 seconds for tighter timing
    )
    scheduler.start()
    logger.info("Reminder scheduler started")

    # Create Flask app
    app = create_app(config)

    # Store components on app for route access
    app.config["CONFIG"] = config
    app.config["CONVERSATION_MANAGER"] = conversation_manager
    app.config["METRICS_COLLECTOR"] = metrics_collector
    app.config["SYSTEM_MONITOR"] = system_monitor
    app.config["AUTH_MANAGER"] = auth_manager
    app.config["DB_MANAGER"] = db_manager
    app.config["NOTES_MANAGER"] = notes_manager
    app.config["PLUGIN_MANAGER"] = plugin_manager
    app.config["SCRIPT_RUNNER"] = script_runner

    # Initialize Bluetooth manager
    from app.bluetooth_manager import BluetoothManager
    bluetooth_manager = BluetoothManager()
    app.config["BLUETOOTH_MANAGER"] = bluetooth_manager

    # Initialize WiFi manager
    from app.wifi_manager import WiFiManager
    wifi_manager = WiFiManager(db_manager, network_scanner=network_scanner)
    app.config["WIFI_MANAGER"] = wifi_manager

    # Initialize IAM manager
    from app.iam import IAMManager
    iam_manager = IAMManager(db_manager)
    app.config["IAM_MANAGER"] = iam_manager
    # Create default admin user if no users exist
    if not iam_manager.users:
        iam_manager.create_user(config.web_username or "admin", config.web_password or "admin", "admin")
        logger.info("Default admin user created")

    # Initialize user preferences manager
    from app.user_preferences import UserPreferencesManager, FaceProfileManager
    user_prefs_manager = UserPreferencesManager(db_manager)
    app.config["USER_PREFS_MANAGER"] = user_prefs_manager
    conversation_manager.user_prefs_manager = user_prefs_manager

    # Initialize face profile manager
    face_profile_manager = FaceProfileManager(db_manager)
    app.config["FACE_PROFILE_MANAGER"] = face_profile_manager

    # Initialize knowledge base manager
    from app.knowledge_base import KnowledgeBaseManager
    kb_manager = KnowledgeBaseManager(db_manager)
    app.config["KB_MANAGER"] = kb_manager
    conversation_manager.kb_manager = kb_manager
    app.config["SCHEDULER"] = scheduler

    # Initialize daily briefing
    from app.daily_briefing import DailyBriefing
    daily_briefing = DailyBriefing(db_manager, config)
    daily_briefing.weather_client = weather_client
    daily_briefing.calendar_client = calendar_client
    daily_briefing.notes_manager = notes_manager
    daily_briefing.metrics_collector = metrics_collector
    daily_briefing.llm_client = llm_client
    app.config["DAILY_BRIEFING"] = daily_briefing

    # Initialize workflow engine
    from app.workflow_engine import WorkflowEngine
    workflow_engine = WorkflowEngine(db_manager)
    workflow_engine.scheduler = scheduler
    workflow_engine.conversation_manager = conversation_manager
    conversation_manager.workflow_engine = workflow_engine
    app.config["WORKFLOW_ENGINE"] = workflow_engine

    # Initialize contextual suggestions engine
    from app.contextual_suggestions import ContextualSuggestions
    suggestions_engine = ContextualSuggestions(db_manager)
    suggestions_engine.scheduler = scheduler
    conversation_manager.suggestions_engine = suggestions_engine
    app.config["SUGGESTIONS_ENGINE"] = suggestions_engine

    # Initialize cron job manager
    from app.cron_manager import CronManager
    cron_manager = CronManager(db_manager)
    app.config["CRON_MANAGER"] = cron_manager

    # Initialize log analyzer
    from app.log_analyzer import LogAnalyzer
    log_analyzer = LogAnalyzer(db_manager)
    log_analyzer.scheduler = scheduler
    app.config["LOG_ANALYZER"] = log_analyzer

    # Initialize service manager
    from app.service_manager import ServiceManager
    ssh_client_ref = conversation_manager.ssh_client if hasattr(conversation_manager, 'ssh_client') else None
    service_manager = ServiceManager(ssh_client=ssh_client_ref)
    app.config["SERVICE_MANAGER"] = service_manager

    # Initialize backup orchestrator
    from app.backup_orchestrator import BackupOrchestrator
    backup_orchestrator = BackupOrchestrator(db_manager, project_dir=project_dir)
    backup_orchestrator.scheduler = scheduler
    app.config["BACKUP_ORCHESTRATOR"] = backup_orchestrator

    # Initialize custom assistants manager
    from app.custom_assistants import CustomAssistantsManager
    custom_assistants = CustomAssistantsManager(db_manager)
    app.config["CUSTOM_ASSISTANTS"] = custom_assistants

    # Initialize flow engine (visual workflow builder)
    from app.flow_engine import FlowEngine
    flow_engine = FlowEngine(db_manager)
    flow_engine.scheduler = scheduler
    flow_engine.conversation_manager = conversation_manager
    app.config["FLOW_ENGINE"] = flow_engine

    # Initialize autopilot (nightly self-improvement mode) — disabled by default,
    # enabled/paused/stopped via chat ("start/pause/stop autopilot mode") or the API.
    from app.autopilot.manager import AutopilotManager
    autopilot_manager = AutopilotManager(
        config=config,
        db_manager=db_manager,
        project_dir=project_dir,
        notes_manager=notes_manager,
        system_monitor=system_monitor,
        kb_manager=kb_manager,
    )
    autopilot_manager.start_thread()
    conversation_manager.autopilot_manager = autopilot_manager
    app.config["AUTOPILOT_MANAGER"] = autopilot_manager

    # Give the conversation manager access to the scheduler for acknowledgment
    conversation_manager.scheduler = scheduler

    # Wire daily briefing and workflow engine into the scheduler
    scheduler._daily_briefing = daily_briefing
    scheduler._workflow_engine = workflow_engine
    scheduler._suggestions_engine = suggestions_engine
    scheduler._cron_manager = cron_manager
    scheduler._log_analyzer = log_analyzer
    scheduler._backup_orchestrator = backup_orchestrator
    scheduler._flow_engine = flow_engine

    # Start file watcher for auto-restart on code changes
    auto_reload = os.environ.get("AUTO_RELOAD", "true").lower() in ("1", "true", "yes")
    if auto_reload:
        # Only watch the app/ directory (not root) to avoid restarts from user-created files
        app_dir = os.path.join(project_dir, "app")
        file_watcher = FileWatcher(watch_dir=app_dir, poll_interval=3)
        file_watcher.start()
        app.config["FILE_WATCHER"] = file_watcher

    logging.getLogger("werkzeug").addFilter(_QuietPollingAccessFilter())

    # Progress polling requires the chat request and progress request to be
    # served concurrently while a Gateway call is in progress.
    logger.info(
        "Starting Jarvis Assistant on 0.0.0.0:%d (threaded)", config.port
    )
    app.run(
        host="0.0.0.0",
        port=config.port,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
