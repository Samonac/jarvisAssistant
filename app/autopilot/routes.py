"""HTTP routes for the autopilot (nightly self-improvement) mode (Phase 5).

Independent of both app.routes (Jarvis chat) and app.coding_agent.routes
(interactive coding agent) — this is control-plane surface for the
autonomous mode: status, start/pause/stop, task queue management, and
confirming UI/UX-affecting changes.
"""

import logging
import os

from flask import Flask, jsonify, request

from app.autopilot.manager import AutopilotManager
from app.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# app/autopilot/routes.py -> app/autopilot -> app -> project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_autopilot_manager(app: Flask) -> AutopilotManager:
    """Get (or lazily create) the AutopilotManager, reusing DB_MANAGER/NOTES_MANAGER if present."""
    manager = app.config.get("AUTOPILOT_MANAGER")
    if manager is not None:
        return manager

    config = app.config["JARVIS_CONFIG"]
    db_manager = app.config.get("DB_MANAGER")
    if db_manager is None:
        db_manager = DatabaseManager(db_path=config.database_path)
        db_manager.initialize()

    manager = AutopilotManager(
        config=config,
        db_manager=db_manager,
        project_dir=PROJECT_DIR,
        notes_manager=app.config.get("NOTES_MANAGER"),
        system_monitor=app.config.get("SYSTEM_MONITOR"),
        kb_manager=app.config.get("KB_MANAGER"),
    )
    app.config["AUTOPILOT_MANAGER"] = manager
    return manager


def register_autopilot_routes(app: Flask) -> None:
    """Register the /api/autopilot/* endpoints on the Flask app."""

    @app.route("/api/autopilot/status", methods=["GET"])
    def autopilot_status():
        return jsonify(_get_autopilot_manager(app).status_dict())

    @app.route("/api/autopilot/start", methods=["POST"])
    def autopilot_start():
        _get_autopilot_manager(app).enable()
        return jsonify(_get_autopilot_manager(app).status_dict())

    @app.route("/api/autopilot/pause", methods=["POST"])
    def autopilot_pause():
        _get_autopilot_manager(app).disable()
        return jsonify(_get_autopilot_manager(app).status_dict())

    @app.route("/api/autopilot/stop", methods=["POST"])
    def autopilot_stop():
        _get_autopilot_manager(app).disable()
        return jsonify(_get_autopilot_manager(app).status_dict())

    @app.route("/api/autopilot/run-now", methods=["POST"])
    def autopilot_run_now():
        """Manually trigger one cycle immediately, bypassing the window/activity checks."""
        result = _get_autopilot_manager(app).run_cycle_if_due(force=True)
        return jsonify(result)

    @app.route("/api/autopilot/tasks", methods=["GET"])
    def autopilot_list_tasks():
        status = request.args.get("status")
        return jsonify(_get_autopilot_manager(app).task_queue.list_tasks(status=status))

    @app.route("/api/autopilot/tasks", methods=["POST"])
    def autopilot_add_task():
        """Queue a new user-submitted task (goes to the front of the queue)."""
        data = request.get_json(silent=True)
        if data is None or not data.get("task"):
            return jsonify({"error": "Missing required field: 'task'"}), 400

        manager = _get_autopilot_manager(app)
        task_id = manager.task_queue.add_task(
            data["task"], source="user", verify_command=data.get("verify_command")
        )
        return jsonify(manager.task_queue.get(task_id)), 201

    @app.route("/api/autopilot/tasks/<int:task_id>/confirm", methods=["POST"])
    def autopilot_confirm_task(task_id):
        """Finalize a task left awaiting_user_confirmation. Body: {"accepted": true|false}."""
        data = request.get_json(silent=True)
        if data is None or "accepted" not in data:
            return jsonify({"error": "Missing required field: 'accepted'"}), 400

        result = _get_autopilot_manager(app).confirm_task(task_id, bool(data["accepted"]))
        if "error" in result:
            status_code = 404 if "No task found" in result["error"] else 400
            return jsonify(result), status_code
        return jsonify(result)
