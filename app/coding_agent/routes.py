"""HTTP routes for the interactive coding-agent mode (Phases 2-3).

Independent of app.routes' Jarvis chat endpoints — a separate `/api/agent/*`
surface with its own tool set (file I/O + shell) and its own session
configuration (provider/model/effort), per the project decision to keep the
coding agent as its own interaction mode alongside the existing chat.
"""

import logging
import os
import uuid

from flask import Flask, jsonify, request

from app.agent_session import AgentSessionConfig
from app.coding_agent.loop import CodingAgentLoop
from app.coding_agent.session_store import AgentSessionStore
from app.coding_agent.snapshot import TaskSnapshot
from app.coding_agent.tools import CodingAgentTools, build_default_tools
from app.coding_agent.verify import apply_verification_gate
from app.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# app/coding_agent/routes.py -> app/coding_agent -> app -> project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_session_store(app: Flask) -> AgentSessionStore:
    """Get (or lazily create) the AgentSessionStore, reusing DB_MANAGER if present."""
    store = app.config.get("CODING_AGENT_SESSION_STORE")
    if store is not None:
        return store

    db_manager = app.config.get("DB_MANAGER")
    if db_manager is None:
        config = app.config["JARVIS_CONFIG"]
        db_manager = DatabaseManager(db_path=config.database_path)
        db_manager.initialize()

    store = AgentSessionStore(db_manager)
    app.config["CODING_AGENT_SESSION_STORE"] = store
    return store


def _get_snapshot(app: Flask) -> TaskSnapshot:
    """Get (or lazily create) the TaskSnapshot manager for this app instance."""
    snapshot = app.config.get("CODING_AGENT_SNAPSHOT")
    if snapshot is None:
        snapshot = TaskSnapshot(PROJECT_DIR)
        app.config["CODING_AGENT_SNAPSHOT"] = snapshot
    return snapshot


def _build_tools(config) -> CodingAgentTools:
    """Build a fresh, project-sandboxed tool registry for one loop run."""
    return build_default_tools(config, PROJECT_DIR)


def register_coding_agent_routes(app: Flask) -> None:
    """Register the /api/agent/* endpoints on the Flask app."""

    @app.route("/api/agent/run", methods=["POST"])
    def agent_run():
        """Start a new coding-agent task; runs until it pauses, finishes, or errors.

        A self-reported "done" is only accepted if the verification command
        (default: the project's pytest suite) passes — otherwise the task's
        changes are rolled back to the pre-task snapshot automatically.

        Body: {"task": "<instruction>", "provider": "<optional>",
               "model": "<optional>", "effort": "quick|standard|deep",
               "verify_command": "<optional shell command>"}
        """
        data = request.get_json(silent=True)
        if data is None or not data.get("task"):
            return jsonify({"error": "Missing required field: 'task'"}), 400

        config = app.config["JARVIS_CONFIG"]

        try:
            session = AgentSessionConfig(
                config,
                provider=data.get("provider"),
                model=data.get("model"),
                effort=data.get("effort", "standard"),
            )
        except (ValueError, NotImplementedError) as e:
            return jsonify({"error": str(e)}), 400

        session_id = uuid.uuid4().hex
        snapshot = _get_snapshot(app)
        snapshot.create(session_id)

        tools = _build_tools(config)
        loop = CodingAgentLoop(session, tools, PROJECT_DIR)

        try:
            state = loop.start(data["task"], session_id=session_id)
        except Exception as e:
            logger.error("Coding agent run failed: %s", e)
            return jsonify({"error": f"Agent run failed: {e}"}), 500

        state = apply_verification_gate(state, tools, snapshot, data.get("verify_command"))
        _get_session_store(app).save(state)

        result = state.to_dict()
        result["session"] = session.to_dict()
        return jsonify(result)

    @app.route("/api/agent/resume", methods=["POST"])
    def agent_resume():
        """Resume a paused (awaiting_user) session with the user's answer.

        Body: {"session_id": "<id>", "answer": "<the user's reply>",
               "verify_command": "<optional shell command>"}
        """
        data = request.get_json(silent=True)
        if data is None or not data.get("session_id") or not data.get("answer"):
            return jsonify({"error": "Missing required fields: 'session_id' and 'answer'"}), 400

        store = _get_session_store(app)
        state = store.load(data["session_id"])
        if state is None:
            return jsonify({"error": f"No session found with id '{data['session_id']}'"}), 404

        if state.status != "awaiting_user":
            return jsonify({
                "error": f"Session '{state.session_id}' is not awaiting a user answer "
                         f"(current status: '{state.status}')."
            }), 400

        config = app.config["JARVIS_CONFIG"]
        try:
            session = AgentSessionConfig(config, provider=state.provider, model=state.model, effort=state.effort)
        except (ValueError, NotImplementedError) as e:
            return jsonify({"error": str(e)}), 400

        tools = _build_tools(config)
        loop = CodingAgentLoop(session, tools, PROJECT_DIR)

        try:
            state = loop.resume(state, data["answer"])
        except Exception as e:
            logger.error("Coding agent resume failed: %s", e)
            return jsonify({"error": f"Agent resume failed: {e}"}), 500

        state = apply_verification_gate(state, tools, _get_snapshot(app), data.get("verify_command"))
        store.save(state)

        result = state.to_dict()
        result["session"] = session.to_dict()
        return jsonify(result)

    @app.route("/api/agent/sessions/<session_id>", methods=["GET"])
    def agent_session_status(session_id):
        """Return the current status/transcript for a coding-agent session."""
        state = _get_session_store(app).load(session_id)
        if state is None:
            return jsonify({"error": f"No session found with id '{session_id}'"}), 404
        return jsonify(state.to_dict())

    @app.route("/api/agent/rollback", methods=["POST"])
    def agent_rollback():
        """Manually roll back a session's changes to its pre-task snapshot.

        Available at any point, per the project's requirement that a task's
        changes can always be reverted to the prior version of the code.

        Body: {"session_id": "<id>"}
        """
        data = request.get_json(silent=True)
        if data is None or not data.get("session_id"):
            return jsonify({"error": "Missing required field: 'session_id'"}), 400

        store = _get_session_store(app)
        state = store.load(data["session_id"])
        if state is None:
            return jsonify({"error": f"No session found with id '{data['session_id']}'"}), 404

        restored = _get_snapshot(app).restore(state.session_id)
        if not restored:
            return jsonify({"error": f"No snapshot available to restore for session '{state.session_id}'."}), 409

        state.status = "rolled_back"
        state.error = "Manually rolled back by user request."
        store.save(state)
        return jsonify(state.to_dict())

