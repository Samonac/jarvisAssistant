"""Conversation Manager for Jarvis Assistant.

Orchestrates the full conversation flow:
- Loads session history from the database
- Injects relevant mental notes into LLM context
- Sends messages to the LLM client
- Routes tool calls to the appropriate executor
- Persists messages back to the database
- Enforces the 10-pair history limit
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

from app.llm_client import LLMClient
from app.config import Config
from app.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

JARVIS_SYSTEM_PROMPT_TEMPLATE = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), the personal AI assistant. You speak with a formal, refined British accent and manner. You are witty, precise, and unfailingly polite. You ALWAYS address the user as "{user_honorific}". NEVER switch between "Sir" and "Ma'am" — use ONLY "{user_honorific}" consistently in every response.

===== SYSTEM CLOCK (AUTHORITATIVE) =====
The current date and time is: {current_datetime}
This is the ONLY correct time. NEVER use any other date or time. If the user asks what time it is, use THIS value. If you need to calculate a future time (e.g., "in 5 minutes"), add to THIS value.
=========================================

You have access to the following tools. When you need to use a tool, respond ONLY with a JSON object in this exact format (no other text):
{{"tool": "<tool_name>", "args": {{<arguments>}}}}

Available tools:
- get_time: Get the current date and time from the system clock. Args: {{}}
- get_weather: Get current weather for a city. Args: {{"city": "<city name, optional>"}}
- get_forecast: Get weather forecast for the next few days. Args: {{"city": "<city name, optional>", "days": <integer, default 3>}}
- run_command: Execute a Linux bash command. Args: {{"command": "<bash command>"}}
- web_search: Search the web for information. Args: {{"query": "<search query>"}}
- network_scan: Scan the local network for connected devices. Args: {{}}
- add_note: Save a mental note or reminder. Args: {{"content": "<note text>", "due_date": "<ISO datetime or null>", "category": "<category or null>", "command": "<command to execute when due, or null>", "recurrence": "<recurrence pattern: hourly/daily/weekly/every Nm/every Nh, or null>", "expires_at": "<ISO datetime after which note auto-clears, or null>"}}
- get_notes: Retrieve all active mental notes. Args: {{}}
- complete_note: Mark a note as done. Args: {{"note_id": <integer>}}
- clear_all_notes: Clear/delete ALL active notes. Args: {{}}
- search_notes: Search notes by keyword. Args: {{"query": "<search term>"}}
- get_calendar_events: Get upcoming calendar events. Args: {{"hours_ahead": <integer, default 24>}}
- create_calendar_event: Create a new calendar event. Args: {{"title": "<title>", "start": "<ISO datetime>", "end": "<ISO datetime>", "description": "<optional>"}}
- send_email: Compose and send an email (will ask for confirmation). Args: {{"to": "<email>", "subject": "<subject>", "body": "<body>"}}

SMART HOME TOOLS:
- list_lights: List all smart home lights and their states. Args: {{}}
- turn_on_light: Turn on a light. Args: {{"entity_id": "<entity_id>", "brightness": <0-255, optional>}}
- turn_off_light: Turn off a light. Args: {{"entity_id": "<entity_id>"}}
- set_light_color: Set a light's color. Args: {{"entity_id": "<entity_id>", "rgb": [<r>, <g>, <b>]}}

FILE MANAGEMENT TOOLS:
- read_file: Read a file's content. Args: {{"path": "<file path>"}}
- write_file: Write content to a file. Args: {{"path": "<file path>", "content": "<text>"}}
- list_files: List directory contents. Args: {{"path": "<directory path, default .>"}}
- search_files: Search for files by pattern. Args: {{"pattern": "<glob pattern>", "path": "<directory, default .>"}}
- delete_file: Delete a file. Args: {{"path": "<file path>"}}
- ocr_image: Extract text from an image file using OCR. Args: {{"path": "<image file path>"}}

NOTE: You CAN read and edit your own configuration files (like .env) in the project directory. Use read_file and write_file with the full path to modify settings. The project directory is accessible.
IMPORTANT: When creating files, folders, or scripts, ALWAYS use the project directory as the base. NEVER use absolute root paths like "/picture/" or "C:/picture/". Instead use relative paths (e.g., "picture/") which will resolve to the project directory. All file operations are relative to the project directory.

SSH TOOLS:
- ssh_execute: Execute a command on a remote machine. Args: {{"host": "<host name>", "command": "<command>"}}
- ssh_list_hosts: List configured SSH hosts. Args: {{}}
- ssh_add_host: Add a new SSH host to the configuration. Args: {{"name": "<friendly name>", "host": "<hostname or IP>", "port": <port, default 22>, "username": "<user>", "password": "<password, optional>", "key_path": "<path to key, optional>"}}

CODE EXECUTION:
- run_code: Execute Python code in a sandboxed environment (no file creation needed). Args: {{"code": "<python code>"}}
  Safe for calculations, data transformations, and quick prototyping. Has access to: math, json, datetime, random, re, collections, itertools, functools.

DAILY BRIEFING:
- get_briefing: Generate the daily briefing (weather + calendar + notes + metrics). Args: {{}}

WORKFLOW AUTOMATION:
- list_workflows: List all active automation workflows. Args: {{}}
- create_workflow: Create a new automation workflow. Args: {{"name": "<name>", "trigger_type": "<schedule|gps_enter|gps_exit|event>", "trigger_config": {{}}, "action_type": "<notify|run_command|briefing|smart_home|note|webhook>", "action_config": {{}}}}

AUTOPILOT (nightly self-improvement mode):
- autopilot_control: Start, pause, or stop the nightly autonomous coding mode, or check its status. Args: {{"action": "<start|pause|stop|status>"}}

IMPORTANT RULES:
1. When the user asks for the current time, ALWAYS use the get_time tool or refer to the SYSTEM CLOCK above. NEVER guess or invent a time.
2. When the user mentions relative times like "in 5 minutes", "tomorrow", "next week", calculate from the SYSTEM CLOCK value above.
3. If no tool is needed, respond naturally as Jarvis. Be concise but thorough.
4. When presenting tool results, summarise them in your characteristic style.
5. NEVER fabricate or invent tool outputs. If you need to use a tool, respond ONLY with the JSON tool call. Do NOT write fake command outputs, fake file contents, or fake SSH results.
6. If a tool returns an error, report the EXACT error to the user. Do NOT make up a successful result.
7. You can ONLY know the result of a command AFTER calling the tool. Do NOT pretend you already ran it.
8. EDITING FILES: When asked to "modify", "adapt", "update", "change", or "add to" a file, you MUST:
   a) First use read_file to get the current content
   b) Then use write_file to save the modified version
   c) Only AFTER writing should you run_command to execute it (if asked)
   Do NOT skip the read/write steps and just run the file."""


def _get_system_prompt(honorific: str = "Sir") -> str:
    """Build the system prompt with the current date/time and honorific injected."""
    import os
    now = datetime.now()
    current_dt = now.strftime("%Y-%m-%d %H:%M:%S (%A)")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt = JARVIS_SYSTEM_PROMPT_TEMPLATE.format(current_datetime=current_dt, user_honorific=honorific)
    # Inject project path info
    prompt += f"\n\nPROJECT DIRECTORY: {project_dir}\nYour .env config file is at: {project_dir}/.env\nALL files and folders you create MUST be inside this project directory. Use relative paths or prepend this directory path."
    return prompt


# Single source of truth for all known tool names.
# Update this set whenever a new tool is added.
KNOWN_TOOLS = {
    # Core
    "get_time",
    # Weather
    "get_weather", "get_forecast",
    # System
    "run_command",
    # Web
    "web_search",
    # Network
    "network_scan",
    # Notes
    "add_note", "get_notes", "complete_note", "search_notes", "clear_all_notes",
    # Calendar
    "get_calendar_events", "create_calendar_event",
    # Email
    "send_email",
    # Smart Home
    "list_lights", "turn_on_light", "turn_off_light", "set_light_color",
    # File Management
    "read_file", "write_file", "list_files", "search_files", "delete_file", "ocr_image",
    # SSH
    "ssh_execute", "ssh_list_hosts", "ssh_add_host",
    # Code Sandbox
    "run_code",
    # Daily Briefing
    "get_briefing",
    # Workflows
    "list_workflows", "create_workflow",
    # Autopilot
    "autopilot_control",
}


class ConversationManager:
    """Manages conversation flow, history, and tool routing.

    Attributes:
        llm_client: The LLM provider client.
        config: Application configuration.
        db_manager: Database manager for persistent history.
        notes_manager: Notes manager (injected after construction).
        command_executor: Command executor tool.
        web_searcher: Web search tool.
        network_scanner: Network scanner tool.
        calendar_client: Calendar integration (optional).
        email_client: Email client (optional).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Config,
        db_manager: DatabaseManager,
    ):
        self.llm_client = llm_client
        self.config = config
        self.db_manager = db_manager

        # Tools — injected after construction
        self.notes_manager = None
        self.command_executor = None
        self.web_searcher = None
        self.network_scanner = None
        self.calendar_client = None
        self.email_client = None
        self.metrics_collector = None
        self.weather_client = None
        self.smart_home = None
        self.file_manager = None
        self.ssh_client = None
        self.plugin_manager = None
        self.kb_manager = None

        # Pending email confirmations keyed by session_id
        self._pending_emails: dict[str, dict] = {}
        self.scheduler = None  # Injected after construction
        self._last_agent_steps: list[dict] = []  # Last agent execution steps
        self.user_prefs_manager = None  # Injected after construction
        self.workflow_engine = None  # Injected after construction
        self.code_sandbox = None  # Injected after construction
        self.suggestions_engine = None  # Injected after construction
        self.autopilot_manager = None  # Injected after construction

    def handle_message(self, message: str, session_id: str) -> str:
        """Process a user message and return the assistant response.

        Loads history from DB, injects relevant notes, calls the LLM,
        handles any tool calls, and persists the exchange.

        Args:
            message: The user's input message.
            session_id: Unique session identifier.

        Returns:
            The assistant's response string.
        """
        import time

        start_time = time.time()

        # Any chat activity resets the autopilot inactivity clock.
        if self.autopilot_manager:
            self.autopilot_manager.record_activity()

        # Check for pending email confirmation
        if session_id in self._pending_emails:
            response = self._handle_email_confirmation(message, session_id)
            self._persist_exchange(session_id, message, response)
            return response

        # Check if user is acknowledging a reminder
        # ONLY trigger if Jarvis' last message was actually a reminder notification
        ack_keywords = {"done", "acknowledged", "got it", "noted"}
        ack_all_keywords = {"clear all", "dismiss all", "acknowledge all"}
        msg_lower = message.strip().lower().rstrip(".,!")

        # Check if the last assistant message in this session was a reminder
        last_msg_was_reminder = False
        history = self.db_manager.get_history(session_id, max_pairs=1)
        if history:
            last_assistant = [m for m in history if m["role"] == "assistant"]
            if last_assistant:
                last_content = last_assistant[-1]["content"].lower()
                last_msg_was_reminder = (
                    "reminder acknowledged" in last_content
                    or "pending reminder" in last_content
                    or 'reply "done"' in last_content
                    or "reply \"done\"" in last_content
                    or "🔔" in last_assistant[-1]["content"]
                )

        # Also consider if a notification was just displayed (no DB record for those)
        if not last_msg_was_reminder and self.scheduler:
            last_msg_was_reminder = (
                self.scheduler.last_displayed_note_id is not None
                and len(self.scheduler.get_unacknowledged_reminders()) > 0
            )

        if last_msg_was_reminder and self.scheduler and msg_lower in ack_all_keywords:
            count = self.scheduler.acknowledge_all()
            if count > 0:
                response = (
                    f"Very good, Sir. All {count} reminder{'s' if count > 1 else ''} "
                    f"{'have' if count > 1 else 'has'} been cleared."
                )
                self._persist_exchange(session_id, message, response)
                return response
        elif last_msg_was_reminder and self.scheduler and msg_lower in ack_keywords:
            unacked = self.scheduler.get_unacknowledged_reminders()
            if unacked:
                # Acknowledge the last reminder that was actually displayed to the user
                target_id = self.scheduler.last_displayed_note_id
                target = None
                if target_id and target_id in self.scheduler.unacknowledged:
                    target = self.scheduler.unacknowledged[target_id]
                else:
                    # Fallback: use the first unacknowledged one
                    target = unacked[0]
                    target_id = target["note_id"]

                self.scheduler.acknowledge(target_id)
                remaining = len(unacked) - 1
                content = target.get("content", "reminder")
                if remaining > 0:
                    response = (
                        f"Very good, Sir. Reminder acknowledged: \"{content}\". "
                        f"You still have {remaining} pending reminder{'s' if remaining > 1 else ''}."
                    )
                else:
                    response = (
                        f"Very good, Sir. Reminder acknowledged: \"{content}\". "
                        f"No further reminders pending."
                    )
                self._persist_exchange(session_id, message, response)
                return response

        # Intercept "adapt/modify file then run" requests — handle programmatically
        # because small LLMs can't reliably orchestrate read→write→run sequences
        file_edit_result = self._try_file_edit_and_run(message, session_id)
        if file_edit_result:
            self._persist_exchange(session_id, message, file_edit_result)
            return file_edit_result

        # Auto-detect time-related queries and inject fresh time
        time_keywords = ["what time", "current time", "what's the time", "what date", "current date", "today's date"]
        if any(kw in message.lower() for kw in time_keywords):
            # Force a get_time tool call to ensure accurate response
            time_result = self._tool_get_time()
            # Build messages with time context prepended
            messages = self._build_messages(session_id, message)
            messages.insert(-1, {"role": "system", "content": f"[SYSTEM CLOCK RESULT]: {time_result}"})
            raw_response = self.llm_client.chat(messages)
            # Record metric
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics_collector:
                self.metrics_collector.record_llm_call(duration_ms=duration_ms, success=True, session_id=session_id)
            # Check for tool call in response
            tool_call = self._parse_tool_call(raw_response)
            if tool_call:
                response = self._execute_tool_and_summarize(tool_call, messages, session_id)
            else:
                response = raw_response
            self._persist_exchange(session_id, message, response)
            return response

        # Build message list for LLM
        messages = self._build_messages(session_id, message)

        # Check if this query needs multi-step agent processing
        from app.agent import AgentExecutor
        agent = AgentExecutor(self)
        plan = agent.should_use_agent(message, messages)

        if plan:
            # Multi-step agent mode
            logger.info("Agent mode activated for: %s", message[:50])
            result = agent.execute(message, session_id, messages, plan)
            response = result["response"]
            # Store the thinking steps for the frontend
            self._last_agent_steps = result.get("steps", [])
            self._persist_exchange(session_id, message, response)
            return response

        # Normal single-shot mode
        # Call LLM
        raw_response = self.llm_client.chat(messages)

        # Debug: log what the LLM actually returned
        logger.debug("LLM raw response (first 200 chars): %s", raw_response[:200])

        # Record LLM call metric
        duration_ms = (time.time() - start_time) * 1000
        if self.metrics_collector:
            self.metrics_collector.record_llm_call(
                duration_ms=duration_ms,
                success=True,
                session_id=session_id,
            )

        # Check if the response is a tool call
        tool_call = self._parse_tool_call(raw_response)
        if tool_call:
            # SPECIAL CASE: If the response contains a code block AND a run_command,
            # the LLM is trying to show a file edit + run it, but only the run_command
            # was parsed. We need to write the file FIRST, then run the command.
            if (tool_call.get("tool") == "run_command" and
                ("```python" in raw_response.lower() or "```py" in raw_response.lower()) and
                any(kw in raw_response.lower() for kw in ["adapted", "modified", "updated", "here is the", "here's the"])):

                # Extract the code and determine the file path
                code = self._extract_code_from_response(raw_response)
                file_path = self._guess_file_path_from_message(message)

                if code and file_path:
                    logger.info("Detected file edit + run pattern. Writing file first: %s", file_path)
                    # Write the file first
                    write_result = self._execute_tool("write_file", {"path": file_path, "content": code}, session_id)
                    logger.info("File write result: %s", write_result[:100])
                    # Then execute the run_command
                    response = self._execute_tool_and_summarize(tool_call, messages, session_id)
                else:
                    response = self._execute_tool_and_summarize(tool_call, messages, session_id)
            else:
                response = self._execute_tool_and_summarize(
                    tool_call, messages, session_id
                )
        else:
            # Detect if the LLM fabricated a response instead of calling a tool
            # If the response contains code blocks or fake outputs but no tool was called,
            # it's likely hallucinating. Check if it SHOULD have called a tool.
            if self._looks_like_fabricated_output(raw_response):
                # Force a second attempt with a stricter prompt
                logger.warning("Detected likely fabricated output, forcing tool call retry")
                retry_messages = messages + [
                    {"role": "assistant", "content": raw_response},
                    {"role": "user", "content": (
                        "STOP. You just wrote out code/output in your response instead of actually "
                        "using a tool. You MUST use the tools to make real changes. "
                        "If you need to edit a file, call write_file with the full new content. "
                        "If you need to run a command, call run_command. "
                        "Respond ONLY with the JSON tool call. No prose, no code blocks."
                    )},
                ]
                retry_response = self.llm_client.chat(retry_messages)
                retry_tool_call = self._parse_tool_call(retry_response)
                if retry_tool_call:
                    response = self._execute_tool_and_summarize(
                        retry_tool_call, messages, session_id
                    )
                else:
                    # Give up — return the original response with a warning
                    response = raw_response
            elif self._should_have_called_tool(message, raw_response):
                # The user clearly asked for a tool action but the LLM just responded with text
                logger.warning("LLM should have called a tool but didn't. Forcing tool call. Message: '%s'", message[:80])
                forced_tool = self._force_tool_call(message)
                if forced_tool:
                    logger.info("Forced tool call: %s", forced_tool)
                    response = self._execute_tool_and_summarize(forced_tool, messages, session_id)
                else:
                    logger.warning("Could not determine forced tool call for: '%s'", message[:80])
                    response = raw_response
            else:
                response = raw_response

        # Persist the exchange
        self._persist_exchange(session_id, message, response)

        return response

    def _build_messages(self, session_id: str, new_message: str) -> list[dict]:
        """Build the full message list for the LLM.

        Structure:
          1. System prompt (Jarvis personality)
          2. Relevant mental notes (if any)
          3. Conversation history (last 10 pairs from DB)
          4. New user message

        Args:
            session_id: Session identifier for history lookup.
            new_message: The new user message to append.

        Returns:
            List of message dicts with 'role' and 'content'.
        """
        # Use custom system prompt if a custom assistant is active
        if hasattr(self, '_custom_system_prompt') and self._custom_system_prompt:
            # Inject current datetime into custom prompt if it has the placeholder
            import os
            now = datetime.now()
            current_dt = now.strftime("%Y-%m-%d %H:%M:%S (%A)")
            prompt = self._custom_system_prompt
            prompt = prompt.replace("{current_datetime}", current_dt)
            prompt = prompt.replace("{user_honorific}", self._get_user_honorific())
            messages = [{"role": "system", "content": prompt}]
        else:
            messages = [{"role": "system", "content": _get_system_prompt(self._get_user_honorific())}]

        # Inject unacknowledged reminders — re-raise until user says "done"
        if self.scheduler:
            pending = self.scheduler.get_unacknowledged_reminders()
            if pending:
                reminder_lines = ["[UNACKNOWLEDGED REMINDERS — mention these to the user and ask them to say \"done\" to clear them]"]
                for r in pending:
                    reminder_lines.append(f"  - Note #{r['note_id']}: {r['content']}")
                messages.append({
                    "role": "system",
                    "content": "\n".join(reminder_lines),
                })

        # Inject relevant notes into system context
        if self.notes_manager:
            relevant_notes = self.notes_manager.get_relevant_notes(new_message)
            timely_notes = self.notes_manager.get_timely_notes()
            all_notes = {n["id"]: n for n in relevant_notes + timely_notes}
            if all_notes:
                notes_text = self.notes_manager.format_for_context(
                    list(all_notes.values())
                )
                messages.append(
                    {
                        "role": "system",
                        "content": f"[ACTIVE NOTES & REMINDERS]\n{notes_text}",
                    }
                )

        # Inject relevant knowledge base context
        if self.kb_manager:
            try:
                from flask import session as flask_session
                kb_username = flask_session.get("username", "")
                if kb_username:
                    kb_context = self.kb_manager.get_context_for_message(kb_username, new_message)
                    if kb_context:
                        messages.append({"role": "system", "content": kb_context})
            except RuntimeError:
                pass  # Outside request context

        # Load history from database
        history = self.db_manager.get_history(
            session_id, max_pairs=self.config.max_history_pairs
        )
        for record in history:
            messages.append(
                {"role": record["role"], "content": record["content"]}
            )

        # Append the new user message
        messages.append({"role": "user", "content": new_message})

        return messages

    def _should_have_called_tool(self, message: str, response: str) -> bool:
        """Detect if the user clearly asked for a tool action but the LLM just responded with text."""
        msg_lower = message.lower()

        # User sent what looks like Python code to execute
        if self._looks_like_code_to_run(message):
            # LLM should have called run_code but instead fabricated output or an error
            if "run_code" not in response and "Tool 'run_code'" not in response:
                return True

        # User asked to run code explicitly: "run the following code: ..."
        if re.search(r'(?:run|execute|try|launch)\s+(?:the\s+|this\s+)?(?:following\s+)?(?:code|script|python)', msg_lower):
            if "run_code" not in response and "Output:" not in response and "Result:" not in response:
                return True

        # User explicitly asked to run a command
        if any(kw in msg_lower for kw in [
            "run the", "run command", "execute", "run dir", "run ls",
            "run python", "run pip", "run npm", '"dir"', "'dir'",
            "run a command", "launch", "run it",
        ]):
            if "command output:" not in response.lower() and "Tool 'run_command'" not in response:
                return True

        # User asked to read/show a file
        if any(kw in msg_lower for kw in ["show me", "read the file", "show the content", "cat "]):
            if "File:" not in response and "content" not in response[:50].lower():
                return True

        # User asked about their notes/reminders
        if any(kw in msg_lower for kw in ["what notes", "what mental notes", "show notes", "list notes", "current notes", "my notes", "active notes"]):
            if "Note saved with ID" not in response:
                return True

        # User asked to clear/delete all notes
        if any(kw in msg_lower for kw in ["clear all notes", "delete all notes", "remove all notes", "clear all mental notes"]):
            if "have been cleared" not in response:
                return True

        # User asked to create a note/reminder but no tool was called
        # BUT NOT if they're asking about existing notes (what/show/list/current)
        if any(kw in msg_lower for kw in [
            "remind me", "make a note", "create a note", "add a note",
            "set a reminder", "new note",
            "make a recurring", "create a recurring", "recurring note",
            "schedule a", "schedule the",
        ]):
            # Exclude queries about existing notes
            asking_keywords = ["what", "show", "list", "current", "do you have", "are there", "how many"]
            if not any(ask in msg_lower for ask in asking_keywords):
                if "Note saved with ID" not in response and "note_id" not in response:
                    return True

        return False

    def _looks_like_code_to_run(self, message: str) -> bool:
        """Detect if a message IS Python code the user wants executed.

        This should return True only when the message itself is code,
        NOT when it's a natural language instruction that mentions code.

        Heuristics:
        - Contains print() calls
        - Contains import statements
        - Contains variable assignments with = (but not == or !=)
        - Contains function definitions
        - Multiple lines with Python-like syntax
        - Starts with common code patterns
        """
        msg = message.strip()
        msg_lower = msg.lower()

        # Skip if it's clearly a natural language instruction wrapping code
        # e.g., "run the following code: print(10+10)"
        if re.match(r'(?:run|execute|try|launch|test)\s+(?:the\s+)?(?:following\s+)?(?:code|script|python|this)', msg_lower):
            return False

        # Skip if it's clearly a natural language question
        question_starters = ["what", "how", "why", "when", "where", "who", "can you",
                             "could you", "please", "tell me", "explain", "is there",
                             "do you", "does", "will", "should"]
        first_word = msg_lower.split()[0] if msg_lower.split() else ""
        if first_word in question_starters:
            return False

        # Skip if it starts with common conversational patterns
        if re.match(r'^(hey|hi|hello|jarvis|sir|ok|yes|no|thanks|thank)\b', msg_lower):
            return False

        # Strong indicators of code
        code_indicators = 0

        if "print(" in msg:
            code_indicators += 2
        if re.search(r'^(import|from)\s+\w+', msg, re.MULTILINE):
            code_indicators += 2
        if re.search(r'^def\s+\w+\s*\(', msg, re.MULTILINE):
            code_indicators += 2
        if re.search(r'^for\s+\w+\s+in\s+', msg, re.MULTILINE):
            code_indicators += 2
        if re.search(r'^while\s+', msg, re.MULTILINE):
            code_indicators += 1
        if re.search(r'^\w+\s*=\s*[^=]', msg, re.MULTILINE):
            code_indicators += 1
        if re.search(r'\w+\.\w+\(', msg):  # method calls like math.factorial()
            code_indicators += 1
        if "```" in msg:  # Code block markers
            code_indicators += 2
        if re.search(r'#\s*\w+', msg):  # Python comments
            code_indicators += 1

        # If the message is short and contains print() AND starts with code (not prose)
        if "print(" in msg and len(msg) < 200:
            # Make sure it actually starts like code, not like "run the following code: print(...)"
            if re.match(r'^(print|import|from|def|for|while|if|class|try|\w+\s*=)', msg):
                return True

        return code_indicators >= 3

    def _force_tool_call(self, message: str) -> Optional[dict]:
        """Create a forced tool call based on the user's message when the LLM failed to do so."""
        msg_lower = message.lower().strip().rstrip(".")

        # PRIORITY: Extract code from natural language instructions like:
        # "run the following code: print(10+10)"
        # "execute this code: import math; print(math.pi)"
        # "run this: print('hello')"
        # "execute this python: 2+2"
        code_instruction_match = re.search(
            r'(?:run|execute|try|launch)\s+(?:the\s+|this\s+)?(?:following\s+)?(?:code|script|python)?\s*[:\-]\s*["\u201c`]?(.+?)["\u201d`]?\s*$',
            message.strip(), re.IGNORECASE | re.DOTALL
        )
        if code_instruction_match:
            code = code_instruction_match.group(1).strip().strip('"\'`\u201c\u201d')
            if code:
                return {"tool": "run_code", "args": {"code": code}}

        # Also handle: "run the following code :\n<code lines>"
        code_block_match = re.search(
            r'(?:run|execute|try|launch)\s+(?:the\s+|this\s+)?(?:following\s+)?(?:code|script|python)?\s*[:\-]\s*\n(.+)',
            message.strip(), re.IGNORECASE | re.DOTALL
        )
        if code_block_match:
            code = code_block_match.group(1).strip()
            # Strip markdown code block markers if present
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                code = "\n".join(lines)
            if code:
                return {"tool": "run_code", "args": {"code": code}}

        # If the message IS pure code (no natural language wrapper), force run_code
        if self._looks_like_code_to_run(message):
            code = message.strip()
            # Strip markdown code block markers if present
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                code = "\n".join(lines)
            return {"tool": "run_code", "args": {"code": code}}

        # Extract command from quotes: run the "dir" command / run 'ls -la'
        quoted_match = re.search(r'(?:run|execute)\s+(?:the\s+)?(?:following\s+)?(?:command\s*[:\-]?\s*)?["\u201c\u201d\'`]([^"\u201c\u201d\'`]+)["\u201c\u201d\'`]', msg_lower)
        if quoted_match:
            return {"tool": "run_command", "args": {"command": quoted_match.group(1).strip()}}

        # "run the following command : python test.py" / "run this command: ls -la"
        following_match = re.search(r'(?:run|execute)\s+(?:the\s+)?(?:following|this)\s+command\s*[:\-]\s*(.+)', msg_lower)
        if following_match:
            command = following_match.group(1).strip()
            if command:
                return {"tool": "run_command", "args": {"command": command}}

        # "run the dir command" / "run dir" / "execute dir"
        run_match = re.search(
            r'(?:run|execute)\s+(?:the\s+)?(\w[\w\s\-./]*?)(?:\s+command)?\s*$',
            msg_lower
        )
        if run_match:
            command = run_match.group(1).strip()
            if command and command not in ("the", "a", "this", "it", "following"):
                return {"tool": "run_command", "args": {"command": command}}

        # "show me test.py" / "read test.py"
        file_match = re.search(r'(?:show|read|cat|display)\s+(?:me\s+)?(?:the\s+)?(?:content\s+of\s+)?(\S+\.\w+)', msg_lower)
        if file_match:
            return {"tool": "read_file", "args": {"path": file_match.group(1)}}

        # "what notes do you have" / "show my notes" / "list notes"
        if any(kw in msg_lower for kw in ["what notes", "what mental notes", "show notes", "list notes", "current notes", "my notes", "active notes", "do you have"]):
            if "note" in msg_lower or "reminder" in msg_lower:
                return {"tool": "get_notes", "args": {}}

        # "clear all notes" / "delete all notes"
        if any(kw in msg_lower for kw in ["clear all notes", "delete all notes", "remove all notes", "clear all mental notes"]):
            return {"tool": "clear_all_notes", "args": {}}

        # "remind me to X in Y minutes" / "make a note to X"
        # But NOT "what notes do you have" / "show me my notes"
        asking_keywords = ["what", "show", "list", "current", "do you have", "are there", "how many"]
        note_creation_keywords = [
            "remind me", "make a note", "create a note", "add a note",
            "set a reminder", "new note", "mental note",
            "make a recurring", "create a recurring", "recurring note",
            "schedule a", "schedule the",
        ]
        if any(kw in msg_lower for kw in note_creation_keywords):
            if any(ask in msg_lower for ask in asking_keywords):
                # This is a query, not a creation request — force get_notes instead
                return {"tool": "get_notes", "args": {}}
            import re as _re
            from datetime import timedelta

            now = datetime.now()

            # Extract time: "in X minutes/hours"
            due_date = None
            time_match = _re.search(r'in\s+(\d+)\s*(minute|min|hour|hr|second|sec)s?', msg_lower)
            if time_match:
                amount = int(time_match.group(1))
                unit = time_match.group(2)
                if unit in ("minute", "min"):
                    due_date = (now + timedelta(minutes=amount)).isoformat()
                elif unit in ("hour", "hr"):
                    due_date = (now + timedelta(hours=amount)).isoformat()
                elif unit in ("second", "sec"):
                    due_date = (now + timedelta(seconds=amount)).isoformat()

            # "starting now" means due_date = now
            if not due_date and "starting now" in msg_lower:
                due_date = now.isoformat()

            # Extract recurrence: "every minute", "every hour", "every 5 minutes", "daily"
            recurrence = None
            rec_match = _re.search(r'every\s+(\d+\s*)?(minute|min|hour|hr|day|second|sec)s?', msg_lower)
            if rec_match:
                amount_str = (rec_match.group(1) or "1").strip()
                unit = rec_match.group(2)
                amount = int(amount_str)
                if unit in ("minute", "min"):
                    recurrence = f"every {amount}m"
                elif unit in ("hour", "hr"):
                    recurrence = f"every {amount}h"
                elif unit in ("day",):
                    recurrence = f"every {amount}d"
                elif unit in ("second", "sec"):
                    recurrence = f"every {amount}m"  # Minimum granularity: 1 min
                # If recurring and no due_date, start now
                if not due_date:
                    due_date = now.isoformat()
            elif "daily" in msg_lower:
                recurrence = "daily"
                if not due_date:
                    due_date = now.isoformat()
            elif "hourly" in msg_lower:
                recurrence = "hourly"
                if not due_date:
                    due_date = now.isoformat()
            elif "weekly" in msg_lower:
                recurrence = "weekly"
                if not due_date:
                    due_date = now.isoformat()

            # Extract command: "run X" / "execute X" / "launches the command X"
            command = None
            # Pattern: "launches/runs the command "python test.py""
            cmd_quoted = _re.search(r'(?:run|execute|launch(?:es)?)\s+(?:the\s+)?command\s*[:\s]*["\']([^"\']+)["\']', msg_lower)
            if cmd_quoted:
                command = cmd_quoted.group(1).strip()
            else:
                cmd_match = _re.search(r'(?:run|execute|launch(?:es)?)\s+(?:the\s+)?(?:command\s*[:\s]*)?(.+?)(?:\s+(?:every|daily|hourly|weekly|starting|in\s+\d|until|,))', msg_lower)
                if cmd_match:
                    command = cmd_match.group(1).strip().rstrip(",.")
                    if command.endswith(" script"):
                        command = command[:-7].strip()
                    if command.endswith(".py") and not command.startswith("python"):
                        command = f"python {command}"
                else:
                    # Try simpler pattern: "run test.py" anywhere in message
                    cmd_match2 = _re.search(r'(?:run|execute|launch(?:es)?)\s+(?:the\s+)?(?:command\s*[:\s]*)?["\']?(\S+\.py)["\']?', msg_lower)
                    if cmd_match2:
                        script = cmd_match2.group(1)
                        command = f"python {script}"

            # Clean up content
            clean_content = _re.sub(r',?\s*in\s+\d+\s*(minute|min|hour|hr|second|sec)s?', '', message).strip()
            clean_content = _re.sub(r'^(remind me to|make a (?:new )?(?:mental )?note to|create a note to|set a reminder to|add a note to)\s*', '', clean_content, flags=_re.IGNORECASE).strip()
            if not clean_content:
                clean_content = message

            return {"tool": "add_note", "args": {
                "content": clean_content,
                "due_date": due_date,
                "category": None,
                "command": command,
                "recurrence": recurrence,
                "expires_at": None,
            }}

        return None

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from a markdown code block in the LLM response."""
        match = re.search(r'```(?:python|py)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _get_user_honorific(self) -> str:
        """Get the current user's preferred honorific."""
        try:
            from flask import session as flask_session
            username = flask_session.get("username", "")
            if username and self.user_prefs_manager:
                return self.user_prefs_manager.get_honorific(username)
        except RuntimeError:
            pass
        return self.config.user_honorific or "Sir"

    def _guess_file_path_from_message(self, message: str) -> str:
        """Try to determine the target file path from the user's message."""
        import os
        # Look for explicit file names in the message
        match = re.search(r'(\S+\.py)\b', message)
        if match:
            filename = match.group(1)
            # If it's just a filename (no path), prepend the project dir
            if not os.path.sep in filename and "/" not in filename:
                project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                return os.path.join(project_dir, filename)
            return filename
        return ""

    def _normalize_recurrence(self, recurrence: str) -> str:
        """Normalize any recurrence string to a standard format the scheduler understands.

        Handles:
        - "every M", "every min", "every minute", "every 1 minute" → "every 1m"
        - "every H", "every hr", "every hour", "every 2 hours" → "every 1h" / "every 2h"
        - "every D", "every day", "every 3 days" → "daily" / "every 3d"
        - "every S", "every sec", "every 30 seconds" → "every 1m" (min granularity)
        - "every W", "every week" → "weekly"
        - "hourly", "daily", "weekly" → pass through
        - "every 5m", "every 2h" → pass through (already correct)
        """
        r = recurrence.lower().strip()

        # Already in correct format
        if r in ("hourly", "daily", "weekly"):
            return r
        if re.match(r'^every \d+[mhd]$', r):
            return r

        # Handle LLM sending literal "N" as placeholder: "every Nm" → "every 1m"
        placeholder_match = re.match(r'^every\s+n([mhds])$', r, re.IGNORECASE)
        if placeholder_match:
            unit = placeholder_match.group(1).lower()
            return f"every 1{unit}"

        # "every <N> <unit>" or "every <unit>"
        match = re.match(r'every\s+(\d+)?\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|s|sec|secs|second|seconds|w|week|weeks)\s*$', r)
        if match:
            amount = int(match.group(1)) if match.group(1) else 1
            unit = match.group(2)[0]  # First char: m, h, d, s, w
            if unit == 's':
                # Minimum granularity is 1 minute
                return "every 1m"
            elif unit == 'w':
                return "weekly" if amount == 1 else f"every {amount * 7}d"
            elif unit == 'm':
                return f"every {amount}m"
            elif unit == 'h':
                return f"every {amount}h"
            elif unit == 'd':
                return "daily" if amount == 1 else f"every {amount}d"

        # Single letter shortcuts: "every M" → minute, "every H" → hour, "every D" → day
        single_map = {'m': 'every 1m', 'h': 'every 1h', 'd': 'daily', 'w': 'weekly', 's': 'every 1m'}
        if r.startswith("every ") and len(r) == 7 and r[-1].lower() in single_map:
            return single_map[r[-1].lower()]

        # Fallback: return as-is and let the scheduler handle it
        logger.warning("Could not normalize recurrence: '%s'", recurrence)
        return recurrence

    def _validate_note_command(self, command: str) -> Optional[str]:
        """Validate that a command looks like a real shell command.

        Rejects commands that are clearly natural language fragments
        misinterpreted by the LLM as commands.

        Returns:
            The command if valid, None if it's not a real command.
        """
        if not command or not command.strip():
            return None

        cmd = command.strip()
        cmd_lower = cmd.lower()

        # Reject if it's just 1-2 common English words that happen to be Unix commands
        # but are clearly used in natural language context
        false_positive_words = {
            "make", "make the", "make a", "make it", "make sure",
            "find", "find the", "find a", "find out",
            "test", "test the", "test it", "test a",
            "time", "time to", "time the",
            "watch", "watch the", "watch a",
            "top", "top of", "top the",
            "head", "head to", "head the",
            "tail", "tail the",
            "sort", "sort the", "sort out",
            "cut", "cut the", "cut it",
            "date", "date the",
            "touch", "touch the", "touch a",
            "kill", "kill the", "kill it",
            "man", "man the",
            "more", "more of", "more the",
            "less", "less of", "less the",
            "which", "which the", "which is",
            "who", "who the", "who is",
            "yes", "no",
        }
        if cmd_lower in false_positive_words:
            logger.debug("Rejected false-positive command: '%s'", cmd)
            return None

        # Reject if it looks like a natural language phrase (3+ words, no special chars)
        words = cmd.split()
        if len(words) >= 3 and not any(c in cmd for c in "/.\\|><=;$&(){}[]"):
            # Check if it starts with a known command prefix
            known_prefixes = {
                "python", "python3", "pip", "npm", "node", "bash", "sh", "zsh",
                "git", "docker", "curl", "wget", "apt", "yum", "brew",
                "systemctl", "service", "sudo", "cat", "ls", "dir", "cd",
                "mkdir", "rm", "cp", "mv", "chmod", "chown", "grep", "awk",
                "sed", "echo", "export", "source", "ssh", "scp", "rsync",
                "tar", "zip", "unzip", "ping", "traceroute", "nmap",
                "powershell", "cmd", "wsl",
            }
            first_word = words[0].lower().rstrip(".")
            if first_word not in known_prefixes:
                logger.debug("Rejected natural-language command: '%s'", cmd)
                return None

        # Reject if it's too short and doesn't contain a path or extension
        if len(cmd) < 3:
            return None

        # Reject if it's just a single common word without arguments
        if len(words) == 1 and cmd_lower in {"make", "find", "test", "time", "watch", "top", "head", "tail", "sort", "cut", "date", "touch", "kill", "more", "less", "which", "who"}:
            return None

        return cmd

    def _extract_command_from_text(self, text: str) -> Optional[str]:
        """Extract a command from natural language text.

        Looks for:
        - Quoted strings that look like commands: "python test.py", 'ls -la'
        - Common command patterns: python X, npm X, pip X, bash X, etc.
        - File references with executable extensions: script.py, run.sh
        """
        # First: look for quoted commands (most reliable)
        quoted = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', text)
        if quoted:
            candidate = quoted.group(1).strip()
            # If it looks like a command (has a space or ends with an extension)
            if ' ' in candidate or re.search(r'\.\w{1,4}$', candidate):
                return candidate

        # Single quotes
        squoted = re.search(r"'([^']+)'", text)
        if squoted:
            candidate = squoted.group(1).strip()
            if ' ' in candidate or re.search(r'\.\w{1,4}$', candidate):
                return candidate

        # Look for "command: X" or "command X" patterns
        cmd_label = re.search(r'command[:\s]+["\']?([^"\',.]+)["\']?', text.lower())
        if cmd_label:
            candidate = cmd_label.group(1).strip()
            if candidate and len(candidate) > 2:
                return candidate

        # Look for script files first (before exec patterns to avoid "sh" matching in "backup.sh")
        script_match = re.search(r'\b(\S+\.(py|sh|bash|js|rb|pl))\b', text)
        if script_match:
            script = script_match.group(1)
            ext = script_match.group(2)
            if ext == 'py':
                return f"python {script}"
            elif ext in ('sh', 'bash'):
                return f"bash {script}"
            return script

        # Look for common executable patterns (only if no script file found)
        exec_match = re.search(r'\b(python3?|pip|npm|node|ruby|perl|java|go run|cargo run|make)\s+(\S+)', text.lower())
        if exec_match:
            return exec_match.group(0).strip()

        return None

    def _try_file_edit_and_run(self, message: str, session_id: str) -> Optional[str]:
        """Intercept 'adapt/modify file then run' requests and handle programmatically.

        Detects patterns like:
        - "Adapt test.py to add X, then run it"
        - "Modify script.py to do Y and execute it"

        Orchestrates: read_file → LLM generates new code → write_file → run_command

        Returns the final response string, or None if this isn't a file-edit request.
        """
        msg_lower = message.lower()

        # Check if this is a file-edit-and-run request
        edit_keywords = ["adapt", "modify", "update", "change", "edit", "add to", "rewrite"]
        run_keywords = ["then run", "and run", "then execute", "and execute", "run it", "execute it"]

        has_edit = any(kw in msg_lower for kw in edit_keywords)
        has_run = any(kw in msg_lower for kw in run_keywords)
        file_path = self._guess_file_path_from_message(message)

        if not (has_edit and file_path):
            return None

        logger.info("File edit interceptor triggered for: %s", file_path)

        # Step 1: Read the current file
        if not self.file_manager:
            return "File management is not available, Sir."

        read_result = self.file_manager.read_file(file_path)
        if "error" in read_result:
            return f"I was unable to read the file, Sir: {read_result['error']}"

        current_content = read_result["content"]

        # Step 2: Ask the LLM to generate the modified code
        edit_messages = [
            {
                "role": "system",
                "content": (
                    "You are a code editor. The user wants to modify a Python file. "
                    "Given the current file content and the user's instructions, "
                    "output ONLY the complete new file content. "
                    "No explanations, no markdown code fences, no prose. "
                    "Just the raw Python code that should replace the file."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current content of {file_path}:\n\n{current_content}\n\n"
                    f"User's instruction: {message}\n\n"
                    f"Output the complete modified file content:"
                ),
            },
        ]

        new_content = self.llm_client.chat(edit_messages)

        # Clean up: remove any markdown fences the LLM might have added despite instructions
        new_content = new_content.strip()
        if new_content.startswith("```python"):
            new_content = new_content[len("```python"):].strip()
        if new_content.startswith("```py"):
            new_content = new_content[len("```py"):].strip()
        if new_content.startswith("```"):
            new_content = new_content[3:].strip()
        if new_content.endswith("```"):
            new_content = new_content[:-3].strip()

        # Validate: the new content should look like Python code
        if not new_content or len(new_content) < 10:
            return "I was unable to generate the modified code, Sir. The LLM returned an empty response."

        # Step 3: Write the modified file
        write_result = self.file_manager.write_file(file_path, new_content)
        if "error" in write_result:
            return f"I was unable to write the file, Sir: {write_result['error']}"

        logger.info("File written successfully: %s (%d chars)", file_path, len(new_content))

        # Step 4: Run the file if requested
        run_output = ""
        if has_run:
            import os
            filename = os.path.basename(file_path)
            run_result = self.command_executor.execute(f"python {filename}")
            if run_result.get("blocked"):
                run_output = f"\n\nExecution blocked: {run_result.get('blocked_reason', '')}"
            elif run_result.get("timed_out"):
                run_output = "\n\nExecution timed out."
            else:
                stdout = run_result.get("stdout", "")
                stderr = run_result.get("stderr", "")
                run_output = f"\n\nExecution output:\n{stdout}"
                if stderr:
                    run_output += f"\nErrors:\n{stderr}"

        # Step 5: Summarize for the user
        summary_messages = [
            {"role": "system", "content": "You are J.A.R.V.I.S. Summarize what was done in your characteristic formal British style. Address the user as 'Sir'. Be concise."},
            {"role": "user", "content": (
                f"I modified the file '{file_path}' as requested and saved it. "
                f"The changes were: {message}{run_output}"
            )},
        ]
        summary = self.llm_client.chat(summary_messages)

        # If summary looks like a fallback error, use a template
        if "unavailable" in summary.lower() or len(summary) < 10:
            summary = f"Very good, Sir. I have modified {os.path.basename(file_path)} as requested and saved the changes."
            if run_output:
                summary += f"\n{run_output}"

        return summary

    def _looks_like_fabricated_output(self, response: str) -> bool:
        """Detect if the LLM fabricated terminal/command output or file edits
        instead of calling a tool.

        Heuristics:
        - Contains code blocks (```) with system-like content (fake terminal output)
        - Contains code blocks with what looks like a full script (fake file edit)
        - Contains SSH-like output patterns
        """
        # Fake terminal output
        fake_terminal = (
            "```" in response and any(kw in response for kw in [
                "PID", "USER", "COMMAND", "systemd", "kthread",
                "total,", "running,", "sleeping,",
                "load average:", "KiB Mem", "KiB Swap",
                "drwx", "rwx", "-rw-",
                "PING", "bytes from", "icmp_seq",
            ])
        )

        # Narrating running a command with fake output
        fake_narration = (
            ("running the" in response.lower() or "here is the output" in response.lower())
            and "```" in response
        )

        # Fake file edit: LLM shows a code block with a full script instead of calling write_file
        # Detect: response contains a python/code block AND mentions "adapted" or "modified" or "updated"
        fake_file_edit = (
            "```python" in response.lower() or "```py" in response.lower()
        ) and any(kw in response.lower() for kw in [
            "adapted", "modified", "updated", "here is the", "here's the",
            "the adapted", "the modified", "the updated",
        ])

        return fake_terminal or fake_narration or fake_file_edit

    def _trim_history(self, session_id: str) -> None:
        """Trim in-memory history to MAX_HISTORY_PAIRS.

        The database handles this via the max_pairs parameter in get_history.
        This method is a no-op for the DB-backed implementation but is kept
        for interface compatibility and testing.
        """
        pass  # DB-backed: trimming is done at read time via LIMIT

    def _parse_tool_call(self, response: str) -> Optional[dict]:
        """Parse a tool call from the LLM response.

        Supports multiple formats:
        1. JSON: {"tool": "name", "args": {...}}
        2. Function-call style: tool_name({"key": "value"})
        3. Function-call style: tool_name({key: value})

        Returns a dict with 'tool' and 'args' keys, or None if not a tool call.
        """
        stripped = response.strip()

        # Format 1: Direct JSON with "tool" and "args" keys
        if stripped.startswith("{"):
            # First try direct parse
            try:
                data = json.loads(stripped)
                if "tool" in data and "args" in data:
                    return data
                if len(data) == 1:
                    key = list(data.keys())[0]
                    known_tools = KNOWN_TOOLS
                    if key in known_tools and isinstance(data[key], dict):
                        return {"tool": key, "args": data[key]}
            except (json.JSONDecodeError, ValueError):
                pass

            # Try fixing literal newlines in strings (LLM sometimes outputs actual newlines)
            try:
                fixed = stripped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                data = json.loads(fixed)
                if isinstance(data, dict):
                    if "tool" in data and "args" in data:
                        return data
                    if len(data) == 1:
                        key = list(data.keys())[0]
                        if key in KNOWN_TOOLS and isinstance(data[key], dict):
                            return {"tool": key, "args": data[key]}
            except (json.JSONDecodeError, ValueError):
                pass

            # Try ast.literal_eval for Python-style dicts with single quotes
            try:
                import ast
                data = ast.literal_eval(stripped)
                if isinstance(data, dict):
                    if "tool" in data and "args" in data:
                        return data
                    if len(data) == 1:
                        key = list(data.keys())[0]
                        if key in KNOWN_TOOLS and isinstance(data[key], dict):
                            return {"tool": key, "args": data[key]}
            except (ValueError, SyntaxError):
                pass

        # Format 1b: Extract {"tool": ..., "args": {...}} from within prose text
        # Tries json.loads from every { position, expanding until valid JSON is found
        if "{" in stripped:
            idx = 0
            while idx < len(stripped):
                pos = stripped.find("{", idx)
                if pos == -1:
                    break
                # Try progressively longer substrings from this position
                # Find all } positions after this {
                for j in range(pos + 1, len(stripped)):
                    if stripped[j] == '}':
                        candidate = stripped[pos:j+1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict):
                                if "tool" in data and "args" in data:
                                    return data
                                if len(data) == 1:
                                    key = list(data.keys())[0]
                                    if key in KNOWN_TOOLS and isinstance(data[key], dict):
                                        return {"tool": key, "args": data[key]}
                        except (json.JSONDecodeError, ValueError):
                            # Try ast.literal_eval for Python-style dicts with single quotes
                            try:
                                import ast
                                data = ast.literal_eval(candidate)
                                if isinstance(data, dict):
                                    if "tool" in data and "args" in data:
                                        return data
                                    if len(data) == 1:
                                        key = list(data.keys())[0]
                                        if key in KNOWN_TOOLS and isinstance(data[key], dict):
                                            return {"tool": key, "args": data[key]}
                            except (ValueError, SyntaxError):
                                pass
                            continue
                idx = pos + 1

        # Format 1d: {"tool_name": "value"} or {"tool_name": {...}} embedded in prose
        # Matches patterns like: {"run_command": "ls -la"} or {"run_command": {"command": "ls"}}
        for tool_name in KNOWN_TOOLS:
            # Match {"tool_name": "string_value"}
            pattern = r'\{"' + re.escape(tool_name) + r'"\s*:\s*"([^"]*(?:\\.[^"]*)*)"\}'
            match = re.search(pattern, stripped)
            if match:
                value = match.group(1).replace('\\"', '"')
                # For run_command, the string value IS the command
                if tool_name == "run_command":
                    return {"tool": tool_name, "args": {"command": value}}
                elif tool_name == "web_search":
                    return {"tool": tool_name, "args": {"query": value}}
                else:
                    return {"tool": tool_name, "args": {"value": value}}

            # Match {"tool_name": {nested_json}}
            pattern2 = r'\{"' + re.escape(tool_name) + r'"\s*:\s*(\{[^}]*\})\}'
            match2 = re.search(pattern2, stripped)
            if match2:
                try:
                    args = json.loads(match2.group(1))
                    if isinstance(args, dict):
                        return {"tool": tool_name, "args": args}
                except (json.JSONDecodeError, ValueError):
                    pass

        # Format 2: Function-call style — tool_name({...}) or tool_name(...)
        func_match = re.match(r'^(\w+)\s*\(\s*(\{.*\})\s*\)\s*$', stripped, re.DOTALL)
        if func_match:
            tool_name = func_match.group(1)
            args_str = func_match.group(2)
            try:
                args = json.loads(args_str)
                if isinstance(args, dict):
                    return {"tool": tool_name, "args": args}
            except (json.JSONDecodeError, ValueError):
                pass

        # Format 2b: Function-call style embedded in a longer response
        func_match = re.search(r'(\w+)\s*\(\s*(\{[^)]*\})\s*\)', stripped)
        if func_match:
            tool_name = func_match.group(1)
            args_str = func_match.group(2)
            # Only match known tool names to avoid false positives
            known_tools = KNOWN_TOOLS
            if tool_name in known_tools:
                try:
                    args = json.loads(args_str)
                    if isinstance(args, dict):
                        return {"tool": tool_name, "args": args}
                except (json.JSONDecodeError, ValueError):
                    pass

        # Format 3: Just the tool name with no args (e.g., "get_notes()")
        bare_match = re.match(r'^(\w+)\s*\(\s*\)\s*$', stripped)
        if bare_match:
            tool_name = bare_match.group(1)
            known_tools = KNOWN_TOOLS
            if tool_name in known_tools:
                return {"tool": tool_name, "args": {}}

        # Format 4: tool_name {json} (space-separated, no parentheses)
        # e.g., "add_note {"content": "...", "due_date": "..."}"
        space_match = re.match(r'^(\w+)\s+(\{.*\})\s*$', stripped, re.DOTALL)
        if space_match:
            tool_name = space_match.group(1)
            args_str = space_match.group(2)
            known_tools = KNOWN_TOOLS
            if tool_name in known_tools:
                try:
                    args = json.loads(args_str)
                    if isinstance(args, dict):
                        return {"tool": tool_name, "args": args}
                except (json.JSONDecodeError, ValueError):
                    pass

        # Format 4b: tool_name {json} embedded in a longer response
        space_embed_match = re.search(r'(\w+)\s+(\{[^}]+\})', stripped)
        if space_embed_match:
            tool_name = space_embed_match.group(1)
            args_str = space_embed_match.group(2)
            known_tools = KNOWN_TOOLS
            if tool_name in known_tools:
                try:
                    args = json.loads(args_str)
                    if isinstance(args, dict):
                        return {"tool": tool_name, "args": args}
                except (json.JSONDecodeError, ValueError):
                    pass

        return None

    def _execute_tool_and_summarize(
        self, tool_call: dict, messages: list[dict], session_id: str
    ) -> str:
        """Execute a tool call and ask the LLM to summarize the result.

        Args:
            tool_call: Dict with 'tool' and 'args' keys.
            messages: The original message list sent to the LLM.
            session_id: Current session ID.

        Returns:
            The LLM's summarized response incorporating the tool output.
        """
        tool_name = tool_call.get("tool", "")
        args = tool_call.get("args", {})

        # Record tool call metric
        if self.metrics_collector:
            self.metrics_collector.record_tool_call(
                tool_name=tool_name,
                duration_ms=0,
                success=True,
                session_id=session_id,
            )

        tool_output = self._execute_tool(tool_name, args, session_id)

        # Track activity for contextual suggestions
        if self.suggestions_engine:
            try:
                username = None
                try:
                    from flask import session as flask_session
                    username = flask_session.get("username")
                except RuntimeError:
                    pass
                if username:
                    if tool_name == "run_command":
                        self.suggestions_engine.track_command(username, args.get("command", ""))
                    else:
                        self.suggestions_engine.track_tool_call(username, tool_name, args)
            except Exception:
                pass  # Never let tracking break the main flow

        # Log the tool output for debugging
        logger.info("Tool '%s' output: %s", tool_name, tool_output[:200])

        # Ask LLM to summarize the tool output in Jarvis style
        summarize_messages = messages + [
            {"role": "assistant", "content": json.dumps(tool_call)},
            {
                "role": "user",
                "content": (
                    f"The tool '{tool_name}' has returned the following REAL data. "
                    f"You MUST use these exact values in your response. "
                    f"Do NOT use placeholders like [summary] or [percentage]. "
                    f"Do NOT invent or fabricate any data that is not in the output below. "
                    f"If the output contains an error, report the error honestly.\n\n"
                    f"ACTUAL TOOL OUTPUT:\n{tool_output}\n\n"
                    f"Now present this information to me as Jarvis would, using ONLY the real values above."
                ),
            },
        ]

        summary = self.llm_client.chat(summarize_messages)
        return summary

    def _execute_tool(self, tool_name: str, args: dict, session_id: str) -> str:
        """Route a tool call to the appropriate executor.

        Args:
            tool_name: Name of the tool to invoke.
            args: Arguments dict for the tool.
            session_id: Current session ID (needed for email confirmation).

        Returns:
            String output from the tool.
        """
        try:
            if tool_name == "get_time":
                return self._tool_get_time()
            elif tool_name == "get_weather":
                return self._tool_get_weather(args)
            elif tool_name == "get_forecast":
                return self._tool_get_forecast(args)
            elif tool_name == "run_command":
                return self._tool_run_command(args)
            elif tool_name == "web_search":
                return self._tool_web_search(args)
            elif tool_name == "network_scan":
                return self._tool_network_scan()
            elif tool_name == "add_note":
                return self._tool_add_note(args)
            elif tool_name == "get_notes":
                return self._tool_get_notes()
            elif tool_name == "complete_note":
                return self._tool_complete_note(args)
            elif tool_name == "search_notes":
                return self._tool_search_notes(args)
            elif tool_name == "clear_all_notes":
                return self._tool_clear_all_notes()
            elif tool_name == "get_calendar_events":
                return self._tool_get_calendar_events(args)
            elif tool_name == "create_calendar_event":
                return self._tool_create_calendar_event(args)
            elif tool_name == "send_email":
                return self._tool_send_email(args, session_id)
            # Smart Home
            elif tool_name == "list_lights":
                return self._tool_list_lights()
            elif tool_name == "turn_on_light":
                return self._tool_turn_on_light(args)
            elif tool_name == "turn_off_light":
                return self._tool_turn_off_light(args)
            elif tool_name == "set_light_color":
                return self._tool_set_light_color(args)
            # File Management
            elif tool_name == "read_file":
                return self._tool_read_file(args)
            elif tool_name == "write_file":
                return self._tool_write_file(args)
            elif tool_name == "list_files":
                return self._tool_list_files(args)
            elif tool_name == "search_files":
                return self._tool_search_files(args)
            elif tool_name == "delete_file":
                return self._tool_delete_file(args)
            elif tool_name == "ocr_image":
                return self._tool_ocr_image(args)
            # SSH
            elif tool_name == "ssh_execute":
                return self._tool_ssh_execute(args)
            elif tool_name == "ssh_list_hosts":
                return self._tool_ssh_list_hosts()
            elif tool_name == "ssh_add_host":
                return self._tool_ssh_add_host(args)
            # Code Sandbox
            elif tool_name == "run_code":
                return self._tool_run_code(args)
            # Daily Briefing
            elif tool_name == "get_briefing":
                return self._tool_get_briefing()
            # Workflows
            elif tool_name == "list_workflows":
                return self._tool_list_workflows()
            elif tool_name == "create_workflow":
                return self._tool_create_workflow(args)
            elif tool_name == "autopilot_control":
                return self._tool_autopilot_control(args)
            else:
                # Check plugins
                if self.plugin_manager and tool_name in self.plugin_manager.plugins:
                    return self.plugin_manager.execute_plugin(tool_name, args)
                return f"Unknown tool: '{tool_name}'. I'm afraid I don't have that capability, Sir."
        except Exception as e:
            logger.error("Tool execution error (%s): %s", tool_name, e)
            if self.metrics_collector:
                self.metrics_collector.record_error(
                    error_type=f"tool_error_{tool_name}", session_id=session_id
                )
            return f"I encountered an error while executing '{tool_name}': {e}"

    # --- Tool implementations ---

    def _tool_get_time(self) -> str:
        now = datetime.now()
        return (
            f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} "
            f"({now.strftime('%A, %B %d, %Y')})"
        )

    def _tool_get_weather(self, args: dict) -> str:
        if not self.weather_client or not self.weather_client.is_configured():
            return "Weather service is not configured, Sir. Please set the WEATHER_API_KEY environment variable."
        city = args.get("city")

        # Fetch current conditions
        current = self.weather_client.get_current(city)
        if "error" in current:
            return f"Weather error: {current['error']}"

        # Also fetch today's forecast to get precipitation chance
        forecast = self.weather_client.get_forecast(city, days=1)
        today = {}
        if "error" not in forecast and forecast.get("forecasts"):
            today = forecast["forecasts"][0]

        lines = [
            f"Weather in {current['city']}, {current['country']}:",
            f"  Condition: {current['description']}",
            f"  Temperature: {current['temperature']}°C (feels like {current['feels_like']}°C)",
            f"  Humidity: {current['humidity']}%",
            f"  Wind: {current['wind_speed']} km/h ({current.get('wind_dir', '')})",
            f"  Visibility: {current.get('visibility', 'N/A')} km",
            f"  UV Index: {current.get('uv_index', 'N/A')}",
            f"  Local time: {current.get('local_time', 'N/A')}",
        ]

        if today:
            lines += [
                f"  Today's high/low: {today.get('temp_max')}°C / {today.get('temp_min')}°C",
                f"  Chance of rain: {today.get('chance_of_rain', 0)}%",
                f"  Total precipitation: {today.get('precipitation', 0)} mm",
            ]
            if today.get("sunrise"):
                lines.append(f"  Sunrise: {today['sunrise']}  Sunset: {today.get('sunset', 'N/A')}")

        return "\n".join(lines)

    def _tool_get_forecast(self, args: dict) -> str:
        if not self.weather_client or not self.weather_client.is_configured():
            return "Weather service is not configured, Sir. Please set the WEATHER_API_KEY environment variable."
        city = args.get("city")
        days = args.get("days", 3)
        result = self.weather_client.get_forecast(city, days=days)
        if "error" in result:
            return f"Forecast error: {result['error']}"
        lines = [f"Forecast for {result['city']}, {result['country']}:"]
        for day in result.get("forecasts", []):
            chance_of_rain = day.get('chance_of_rain', 0)
            precip = day.get('precipitation', 0)
            lines.append(
                f"  {day['date']}: {day['description']}, "
                f"Low {day['temp_min']}°C / High {day['temp_max']}°C, "
                f"Chance of rain: {chance_of_rain}%, "
                f"Precipitation: {precip} mm, "
                f"Max wind: {day['wind_max']} km/h"
            )
            if day.get("sunrise"):
                lines.append(f"    Sunrise: {day['sunrise']}  Sunset: {day.get('sunset', 'N/A')}")
        return "\n".join(lines)

    def _tool_run_command(self, args: dict) -> str:
        if not self.command_executor:
            return "Command execution is not available, Sir."
        command = args.get("command", "")
        if not command:
            return "No command was specified, Sir."
        result = self.command_executor.execute(command)
        if result.get("blocked"):
            return f"I'm afraid that command is not permitted, Sir. {result.get('blocked_reason', '')}"
        if result.get("timed_out"):
            return "The command exceeded the time limit and was terminated, Sir."
        output = result.get("stdout", "") or result.get("stderr", "") or "(no output)"
        return f"Command output:\n{output}"

    def _tool_web_search(self, args: dict) -> str:
        if not self.web_searcher:
            return "Web search is not available, Sir."
        query = args.get("query", "")
        if not query:
            return "No search query was specified, Sir."
        results = self.web_searcher.search(query)
        if not results:
            return f"I found no results for '{query}', Sir."
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   {r.get('url', '')}")
            lines.append(f"   {r.get('snippet', '')}\n")
        return "\n".join(lines)

    def _tool_network_scan(self) -> str:
        if not self.network_scanner:
            return "Network scanning is not available, Sir."
        devices = self.network_scanner.scan()
        if not devices:
            return "No devices were discovered on the network, Sir."
        lines = [f"Network scan complete. Found {len(devices)} device(s):\n"]
        for d in devices:
            hostname = d.get("hostname") or "unknown"
            lines.append(
                f"  IP: {d.get('ip', 'N/A')}  MAC: {d.get('mac', 'N/A')}  Host: {hostname}"
            )
        return "\n".join(lines)

    def _tool_add_note(self, args: dict) -> str:
        if not self.notes_manager:
            return "Note management is not available, Sir."
        content = args.get("content", "")
        if not content:
            return "No note content was specified, Sir."
        due_date = args.get("due_date")
        category = args.get("category")
        command = args.get("command")
        recurrence = args.get("recurrence")
        expires_at_str = args.get("expires_at")

        # Validate command: reject if it doesn't look like a real shell command
        if command:
            command = self._validate_note_command(command)

        due_dt = None
        if due_date:
            try:
                due_dt = datetime.fromisoformat(due_date)
            except (ValueError, TypeError):
                pass

        expires_dt = None
        if expires_at_str:
            try:
                expires_dt = datetime.fromisoformat(expires_at_str)
            except (ValueError, TypeError):
                pass

        # If no due_date was set but the content/args suggest a time, log a warning
        if not due_dt and due_date:
            logger.warning("Could not parse due_date '%s' for note: %s", due_date, content[:50])

        # Auto-set due_date to now if recurrence is set but no due_date
        if recurrence and not due_dt:
            due_dt = datetime.now()
            logger.info("Auto-setting due_date to now for recurring note")

        # Normalize recurrence: convert any natural language pattern to a standard format
        if recurrence:
            recurrence = self._normalize_recurrence(recurrence)

        # If command is None but content mentions a command, try to extract it
        if not command and content:
            command = self._extract_command_from_text(content)

        note_id = self.notes_manager.add_note(
            content, due_date=due_dt, category=category,
            command=command, recurrence=recurrence, expires_at=expires_dt
        )

        parts = [f"Note saved with ID {note_id}: \"{content}\""]
        if command:
            parts.append(f"Command to execute: `{command}`")
        if recurrence:
            parts.append(f"Recurrence: {recurrence}")
        if due_dt:
            parts.append(f"Due: {due_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            parts.append("WARNING: No due date was set. The reminder will not fire automatically.")
        if expires_dt:
            parts.append(f"Expires: {expires_dt.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.info("Note #%d created: content='%s', due=%s, command=%s, recurrence=%s",
                    note_id, content[:50], due_dt, command, recurrence)
        return "\n".join(parts)

    def _tool_get_notes(self) -> str:
        if not self.notes_manager:
            return "Note management is not available, Sir."
        notes = self.notes_manager.get_all_active()
        if not notes:
            return "Your mind is clear, Sir. There are no pending notes."
        return self.notes_manager.format_for_context(notes)

    def _tool_complete_note(self, args: dict) -> str:
        if not self.notes_manager:
            return "Note management is not available, Sir."
        note_id = args.get("note_id")
        if note_id is None:
            return "No note ID was specified, Sir."
        note_id = int(note_id)
        success = self.notes_manager.complete_note(note_id)
        if success:
            # Also tell the scheduler to stop firing this note
            if self.scheduler:
                self.scheduler._fired_note_ids.add(note_id)
                with self.scheduler._lock:
                    self.scheduler.unacknowledged.pop(note_id, None)
            return f"Note {note_id} has been marked as completed, Sir."
        return f"I could not find an active note with ID {note_id}, Sir."

    def _tool_search_notes(self, args: dict) -> str:
        if not self.notes_manager:
            return "Note management is not available, Sir."
        query = args.get("query", "")
        if not query:
            return "No search query was specified, Sir."
        results = self.notes_manager.search(query)
        if not results:
            return f"No notes found matching '{query}', Sir."
        return self.notes_manager.format_for_context(results)

    def _tool_clear_all_notes(self) -> str:
        """Clear/complete ALL active notes."""
        if not self.notes_manager:
            return "Note management is not available, Sir."
        notes = self.notes_manager.get_all_active()
        if not notes:
            return "There are no active notes to clear, Sir."
        count = 0
        for note in notes:
            self.notes_manager.complete_note(note["id"])
            # Also tell scheduler to stop firing
            if self.scheduler:
                self.scheduler._fired_note_ids.add(note["id"])
                with self.scheduler._lock:
                    self.scheduler.unacknowledged.pop(note["id"], None)
            count += 1
        return f"All {count} active note(s) have been cleared, Sir."

    def _tool_get_calendar_events(self, args: dict) -> str:
        if not self.calendar_client:
            return "Calendar integration is not configured, Sir."
        hours_ahead = args.get("hours_ahead", 24)
        try:
            events = self.calendar_client.get_upcoming_events(hours_ahead=hours_ahead)
            if not events:
                return f"Your calendar is clear for the next {hours_ahead} hours, Sir."
            lines = [f"Upcoming events (next {hours_ahead}h):\n"]
            for e in events:
                lines.append(
                    f"  - {e.get('title', 'Untitled')} at {e.get('start', 'N/A')}"
                )
                if e.get("description"):
                    lines.append(f"    {e['description']}")
            return "\n".join(lines)
        except Exception as e:
            return f"I was unable to retrieve your calendar events, Sir: {e}"

    def _tool_create_calendar_event(self, args: dict) -> str:
        if not self.calendar_client:
            return "Calendar integration is not configured, Sir."
        try:
            from datetime import datetime
            title = args.get("title", "Untitled Event")
            start = datetime.fromisoformat(args["start"])
            end = datetime.fromisoformat(args["end"])
            description = args.get("description", "")
            event = self.calendar_client.create_event(title, start, end, description)
            return f"Calendar event created: '{event.get('title', title)}' on {args['start']}"
        except KeyError as e:
            return f"Missing required field for calendar event: {e}"
        except Exception as e:
            return f"I was unable to create the calendar event, Sir: {e}"

    def _tool_send_email(self, args: dict, session_id: str) -> str:
        if not self.email_client:
            return "Email is not configured, Sir. Please set the SMTP environment variables."
        if not self.email_client.is_configured():
            return "Email sending is not configured, Sir. Please set the SMTP environment variables."

        to = args.get("to", "")
        subject = args.get("subject", "")
        body = args.get("body", "")

        if not to or not subject or not body:
            return "I need a recipient, subject, and body to send an email, Sir."

        draft = self.email_client.compose_draft(to, subject, body)
        self._pending_emails[session_id] = {
            "draft": draft,
            "to": to,
            "subject": subject,
            "body": body,
        }

        return (
            f"I've prepared the following email for your review, Sir:\n\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Body:\n{body}\n\n"
            f"Shall I send this? Please reply 'yes' to confirm or 'no' to cancel."
        )

    # --- Smart Home Tools ---

    def _tool_list_lights(self) -> str:
        if not self.smart_home or not self.smart_home.is_configured():
            return "Smart home is not configured, Sir. Set HA_URL and HA_TOKEN."
        lights = self.smart_home.list_lights()
        if not lights:
            return "No lights found in your smart home system, Sir."
        lines = [f"Found {len(lights)} light(s):"]
        for l in lights:
            state = l["state"]
            brightness = f" (brightness: {l['brightness']})" if l.get("brightness") else ""
            lines.append(f"  {l['name']} [{l['entity_id']}]: {state}{brightness}")
        return "\n".join(lines)

    def _tool_turn_on_light(self, args: dict) -> str:
        if not self.smart_home or not self.smart_home.is_configured():
            return "Smart home is not configured, Sir."
        entity_id = args.get("entity_id", "")
        brightness = args.get("brightness")
        result = self.smart_home.turn_on(entity_id, brightness=brightness)
        return result.get("message", "Done.")

    def _tool_turn_off_light(self, args: dict) -> str:
        if not self.smart_home or not self.smart_home.is_configured():
            return "Smart home is not configured, Sir."
        entity_id = args.get("entity_id", "")
        result = self.smart_home.turn_off(entity_id)
        return result.get("message", "Done.")

    def _tool_set_light_color(self, args: dict) -> str:
        if not self.smart_home or not self.smart_home.is_configured():
            return "Smart home is not configured, Sir."
        entity_id = args.get("entity_id", "")
        rgb = args.get("rgb", [255, 255, 255])
        result = self.smart_home.set_color(entity_id, rgb)
        return result.get("message", "Done.")

    # --- File Management Tools ---

    def _tool_read_file(self, args: dict) -> str:
        if not self.file_manager:
            return "File management is not available, Sir."
        path = args.get("path", "")
        if not path:
            return "No file path specified, Sir."
        result = self.file_manager.read_file(path)
        if "error" in result:
            return f"File error: {result['error']}"
        return f"File: {result['path']} ({result['size']} chars)\n\n{result['content']}"

    def _tool_write_file(self, args: dict) -> str:
        if not self.file_manager:
            return "File management is not available, Sir."
        path = args.get("path", "")
        content = args.get("content", "")
        if not path or not content:
            return "Both 'path' and 'content' are required, Sir."
        result = self.file_manager.write_file(path, content)
        if "error" in result:
            return f"File error: {result['error']}"
        return result.get("message", "File written successfully.")

    def _tool_list_files(self, args: dict) -> str:
        if not self.file_manager:
            return "File management is not available, Sir."
        path = args.get("path", ".")
        result = self.file_manager.list_directory(path)
        if "error" in result:
            return f"File error: {result['error']}"
        lines = [f"Directory: {result['path']} ({result['count']} items)"]
        for entry in result["entries"]:
            icon = "📁" if entry["type"] == "dir" else "📄"
            size = f" ({entry['size']} bytes)" if entry.get("size") is not None else ""
            lines.append(f"  {icon} {entry['name']}{size}")
        return "\n".join(lines)

    def _tool_search_files(self, args: dict) -> str:
        if not self.file_manager:
            return "File management is not available, Sir."
        pattern = args.get("pattern", "*")
        path = args.get("path", ".")
        result = self.file_manager.search_files(pattern, path)
        if "error" in result:
            return f"Search error: {result['error']}"
        if not result["matches"]:
            return f"No files matching '{pattern}' found, Sir."
        lines = [f"Found {result['count']} match(es) for '{pattern}':"]
        for m in result["matches"]:
            lines.append(f"  {m['path']} ({m['type']})")
        return "\n".join(lines)

    def _tool_delete_file(self, args: dict) -> str:
        if not self.file_manager:
            return "File management is not available, Sir."
        path = args.get("path", "")
        if not path:
            return "No file path specified, Sir."
        result = self.file_manager.delete_file(path)
        if "error" in result:
            return f"Delete error: {result['error']}"
        return result.get("message", "File deleted.")

    def _tool_ocr_image(self, args: dict) -> str:
        """Extract text from an image file using OCR."""
        path = args.get("path", "")
        if not path:
            return "No image path specified, Sir."
        import os
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Search in multiple locations
        candidates = [
            path,
            os.path.join(project_dir, path),
            os.path.join(project_dir, "uploads", path),
            os.path.join(project_dir, "uploads", os.path.basename(path)),
            os.path.join(project_dir, "scripts", path),
        ]

        full_path = None
        for candidate in candidates:
            if os.path.isfile(candidate):
                full_path = candidate
                break

        if not full_path:
            return f"Image file not found: '{path}'. Checked project dir, uploads/, and scripts/."
        try:
            with open(full_path, "rb") as f:
                from app.context_extractor import _extract_image_ocr
                import io

                class FakeStorage:
                    def __init__(self, data):
                        self._data = io.BytesIO(data)
                    def read(self):
                        return self._data.read()
                    def seek(self, pos):
                        self._data.seek(pos)

                content = f.read()
                storage = FakeStorage(content)
                text = _extract_image_ocr(storage)
                return f"OCR result from '{path}':\n{text}"
        except Exception as e:
            return f"OCR error: {e}"

    # --- SSH Tools ---

    def _tool_ssh_execute(self, args: dict) -> str:
        if not self.ssh_client or not self.ssh_client.is_configured():
            return "SSH is not configured, Sir. Set SSH_HOSTS in your environment."
        host = args.get("host", "")
        command = args.get("command", "")
        if not host or not command:
            return "Both 'host' and 'command' are required, Sir."
        result = self.ssh_client.execute(host, command)
        if "error" in result:
            return f"SSH error: {result['error']}"
        output = result.get("stdout", "") or result.get("stderr", "") or "(no output)"
        return f"SSH [{result.get('host', host)}] (exit code {result.get('return_code', '?')}):\n{output}"

    def _tool_ssh_list_hosts(self) -> str:
        if not self.ssh_client or not self.ssh_client.is_configured():
            return "SSH is not configured, Sir. No hosts available. You can add one using the ssh_add_host tool."
        hosts = self.ssh_client.list_hosts()
        lines = [f"Configured SSH hosts ({len(hosts)}):"]
        for h in hosts:
            lines.append(f"  {h['name']} — {h['username']}@{h['host']}:{h['port']}")
        return "\n".join(lines)

    def _tool_ssh_add_host(self, args: dict) -> str:
        """Add a new SSH host to the live config and persist to .env."""
        import json as _json
        import os

        name = args.get("name", "")
        host = args.get("host", "")
        if not name or not host:
            return "Both 'name' and 'host' are required to add an SSH host, Sir."

        new_host = {
            "name": name,
            "host": host,
            "port": args.get("port", 22),
            "username": args.get("username", "pi"),
        }
        if args.get("password"):
            new_host["password"] = args["password"]
        if args.get("key_path"):
            new_host["key_path"] = args["key_path"]

        # Update live SSH client
        if not self.ssh_client:
            from app.ssh_client import SSHClient
            self.ssh_client = SSHClient(hosts=[new_host])
        else:
            self.ssh_client.hosts[name.lower()] = new_host

        # Persist to .env file
        try:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env_path = os.path.join(project_dir, ".env")

            # Read current .env
            env_content = ""
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    env_content = f.read()

            # Build the full hosts list
            all_hosts = list(self.ssh_client.hosts.values())
            hosts_json = _json.dumps(all_hosts)

            # Update or add SSH_HOSTS line
            if "SSH_HOSTS=" in env_content:
                import re
                env_content = re.sub(r"SSH_HOSTS=.*", f"SSH_HOSTS={hosts_json}", env_content)
            else:
                env_content = env_content.rstrip() + f"\nSSH_HOSTS={hosts_json}\n"

            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)

            return (
                f"SSH host '{name}' added successfully, Sir. "
                f"Connection: {new_host['username']}@{host}:{new_host['port']}. "
                f"Configuration saved to .env."
            )
        except Exception as e:
            return f"Host added to live config but failed to save to .env: {e}"

    def _tool_run_code(self, args: dict) -> str:
        """Execute Python code in the sandbox."""
        from app.code_sandbox import CodeSandbox

        code = args.get("code", "")
        if not code:
            return "No code was provided, Sir."

        sandbox = CodeSandbox(timeout=10, allow_imports=True)
        result = sandbox.execute(code)

        parts = []
        if result.get("output"):
            parts.append(f"Output:\n{result['output']}")
        if result.get("result"):
            parts.append(f"Result: {result['result']}")
        if result.get("error"):
            parts.append(f"Error:\n{result['error']}")

        if not parts:
            parts.append("Code executed successfully (no output).")

        return "\n".join(parts)

    def _tool_get_briefing(self) -> str:
        """Generate the daily briefing."""
        from app.daily_briefing import DailyBriefing

        briefing = DailyBriefing(self.db_manager, self.config)
        briefing.weather_client = self.weather_client
        briefing.calendar_client = self.calendar_client
        briefing.notes_manager = self.notes_manager
        briefing.metrics_collector = self.metrics_collector

        # Get username from Flask session
        username = None
        honorific = self._get_user_honorific()
        try:
            from flask import session as flask_session
            username = flask_session.get("username")
        except RuntimeError:
            pass

        return briefing.generate(username, honorific)

    def _tool_list_workflows(self) -> str:
        """List all active workflows."""
        if not hasattr(self, 'workflow_engine') or not self.workflow_engine:
            return "Workflow engine is not available, Sir."

        username = None
        try:
            from flask import session as flask_session
            username = flask_session.get("username")
        except RuntimeError:
            pass

        workflows = self.workflow_engine.list_workflows(username=username, include_disabled=True)
        if not workflows:
            return "No workflows configured yet, Sir. You can create one with the create_workflow tool."

        lines = [f"Active workflows ({len(workflows)}):"]
        for wf in workflows:
            status = "✓" if wf["enabled"] else "✗"
            triggered = f" (last: {wf['last_triggered']})" if wf.get("last_triggered") else ""
            lines.append(
                f"  {status} #{wf['id']} \"{wf['name']}\" — "
                f"{wf['trigger_type']} → {wf['action_type']}{triggered}"
            )
        return "\n".join(lines)

    def _tool_create_workflow(self, args: dict) -> str:
        """Create a new automation workflow."""
        if not hasattr(self, 'workflow_engine') or not self.workflow_engine:
            return "Workflow engine is not available, Sir."

        name = args.get("name", "")
        trigger_type = args.get("trigger_type", "")
        trigger_config = args.get("trigger_config", {})
        action_type = args.get("action_type", "")
        action_config = args.get("action_config", {})

        if not name or not trigger_type or not action_type:
            return "I need at minimum a name, trigger_type, and action_type to create a workflow, Sir."

        username = None
        try:
            from flask import session as flask_session
            username = flask_session.get("username")
        except RuntimeError:
            pass

        result = self.workflow_engine.create_workflow(
            name=name,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            action_type=action_type,
            action_config=action_config,
            description=args.get("description", ""),
            conditions=args.get("conditions"),
            username=username,
        )

        if "error" in result:
            return f"Failed to create workflow: {result['error']}"
        return f"Workflow '{name}' created successfully (ID: {result['id']}), Sir."

    def _tool_autopilot_control(self, args: dict) -> str:
        """Start, pause, stop, or report on the nightly autopilot mode."""
        if not self.autopilot_manager:
            return "The autopilot mode is not available, Sir."

        action = str(args.get("action", "")).strip().lower()

        if action == "start":
            self.autopilot_manager.enable()
            return (
                "Autopilot mode is now enabled, Sir. It will only act during its "
                "02:00-06:00 window, and only after an hour of no chat activity."
            )
        if action in ("pause", "stop"):
            self.autopilot_manager.disable()
            return f"Autopilot mode has been {'paused' if action == 'pause' else 'stopped'}, Sir."
        if action == "status":
            status = self.autopilot_manager.status_dict()
            return (
                f"Autopilot is currently {'enabled' if status['enabled'] else 'disabled'}, Sir. "
                f"Window: {status['window']} (in window now: {status['in_window_now']}). "
                f"Queued tasks: {status['queued_tasks']}. "
                f"Awaiting your confirmation: {status['awaiting_confirmation']}."
            )
        return f"Unknown autopilot action: '{action}'. Use start, pause, stop, or status, Sir."

    def _handle_email_confirmation(self, message: str, session_id: str) -> str:
        """Handle user confirmation or cancellation of a pending email."""
        pending = self._pending_emails.get(session_id)
        if not pending:
            return "There is no pending email to confirm, Sir."

        msg_lower = message.strip().lower()
        if msg_lower in ("yes", "y", "send", "confirm", "ok", "sure"):
            draft = pending["draft"]
            del self._pending_emails[session_id]
            result = self.email_client.send(draft)
            if result.get("success"):
                return f"Email sent successfully to {pending['to']}, Sir."
            return f"I was unable to send the email, Sir: {result.get('message', 'Unknown error')}"
        else:
            del self._pending_emails[session_id]
            return "Very well, Sir. The email has been discarded."

    def _persist_exchange(
        self, session_id: str, user_message: str, assistant_response: str
    ) -> None:
        """Save the user message and assistant response to the database."""
        try:
            # Get current username from Flask session if available
            username = None
            try:
                from flask import session as flask_session
                username = flask_session.get("username")
            except RuntimeError:
                pass  # Outside request context

            self.db_manager.save_message(session_id, "user", user_message, username=username)
            self.db_manager.save_message(session_id, "assistant", assistant_response, username=username)

            # Auto-generate session title after first exchange
            self._maybe_generate_title(session_id, user_message)
        except Exception as e:
            logger.error("Failed to persist conversation exchange: %s", e)

    def _maybe_generate_title(self, session_id: str, user_message: str) -> None:
        """Generate a short title for the session if one doesn't exist yet.

        Only generates on the first message of a session to avoid repeated LLM calls.
        Uses the LLM to create a concise 3-6 word title.
        """
        try:
            # Check if title already exists
            sessions = self.db_manager.get_sessions(limit=100)
            for s in sessions:
                if s["session_id"] == session_id:
                    if s.get("title"):
                        return  # Already has a title
                    break

            # Check message count — only generate on first exchange
            history = self.db_manager.get_history(session_id, max_pairs=2)
            if len(history) > 4:  # More than 2 pairs = not the first exchange
                return

            # Generate title using LLM
            title_messages = [
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (3-6 words max) for a conversation "
                        "that starts with the following message. Return ONLY the title, "
                        "nothing else. No quotes, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": user_message[:200]},
            ]
            title = self.llm_client.chat(title_messages)

            # Clean up the title
            title = title.strip().strip('"\'').strip()
            if len(title) > 60:
                title = title[:57] + "..."

            if title and "unavailable" not in title.lower():
                self.db_manager.save_session_title(session_id, title)
                logger.info("Generated title for session %s: %s", session_id[:8], title)
        except Exception as e:
            logger.debug("Title generation failed (non-critical): %s", e)
