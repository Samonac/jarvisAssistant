"""Agent Module for Jarvis Assistant.

Implements a multi-step reasoning agent that can:
1. Analyze the user's query and create a plan
2. Execute multiple tools in sequence
3. Gather and aggregate results
4. Generate a final cohesive response

The "thinking" process is visible to the user via step-by-step updates.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 8  # Maximum tool calls per query to prevent infinite loops

PLANNING_PROMPT = """You are J.A.R.V.I.S. analyzing a user request to determine the best approach.

Given the user's message, decide if this requires:
A) A simple direct response (no tools needed)
B) A single tool call
C) A multi-step plan with multiple tool calls

If it requires a multi-step plan (option C), respond with a JSON plan:
{{"plan": true, "steps": ["<step 1 description>", "<step 2 description>", ...], "tools_needed": ["<tool1>", "<tool2>", ...]}}

If it's simple (option A or B), respond with:
{{"plan": false}}

ONLY respond with the JSON object, nothing else.

Examples of multi-step queries:
- "What's the weather and do I have any meetings today?" → needs get_weather + get_calendar_events
- "Save the current network scan results to a file" → needs network_scan + write_file
- "Check if my server is running and what the disk usage is" → needs ssh_execute (multiple commands)
- "Turn off all the lights and set a reminder to turn them on at 7am" → needs list_lights + turn_off_light + add_note
- "Research the best restaurants in Paris and save the results" → needs web_search + write_file
- "Modify test.py to add X, then run it" → needs read_file + write_file + run_command (ALWAYS read first, write the changes, THEN run)
- "Update the config file and restart" → needs read_file + write_file + run_command
- "Add a function to script.py" → needs read_file + write_file
- "Write a Python script that does X and save it" → needs write_file (persist to disk)
- "Write a script to do X, save it, and run it" → needs write_file + run_command (persist THEN execute)
- "Create a script that does X and execute it" → needs write_file + run_command

CRITICAL RULES:
1. When asked to "modify", "adapt", "update", "change", or "edit" a file and then run it, you MUST plan: read_file → write_file → run_command. NEVER skip the read/write steps.
2. When the user asks to "write", "create", or "save" a script/file AND run/execute it, you MUST plan: write_file → run_command. The file MUST be persisted to disk with write_file first, then executed with run_command. Do NOT use run_code for this — run_code is only for quick in-memory calculations that don't need to be saved.
3. run_code is ONLY appropriate when the user wants a quick calculation or data transformation WITHOUT saving to a file. If the user mentions "save", "write", "create a file", or "persist", you MUST use write_file instead.

Examples of simple queries:
- "Hello, how are you?" → simple conversation
- "What time is it?" → single tool call
- "Turn off the bedroom light" → single tool call
- "What is 2+2?" → single tool call (run_code for quick math)"""


class AgentStep:
    """Represents one step in the agent's execution."""

    def __init__(self, step_type: str, description: str):
        self.step_type = step_type  # "thinking", "tool_call", "result", "final"
        self.description = description
        self.tool_name: Optional[str] = None
        self.tool_args: Optional[dict] = None
        self.tool_output: Optional[str] = None
        self.timestamp = datetime.now().strftime("%H:%M:%S")


class AgentExecutor:
    """Executes multi-step agent plans with visible thinking.

    The agent loop:
    1. Asks the LLM if a plan is needed
    2. If yes, executes tools step by step
    3. After each tool, asks the LLM if more steps are needed
    4. Aggregates all results into a final response

    Attributes:
        conversation_manager: The parent conversation manager (for tool access).
        llm_client: The LLM client for generating responses.
    """

    def __init__(self, conversation_manager):
        self.cm = conversation_manager
        self.llm_client = conversation_manager.llm_client

    @staticmethod
    def _profile_messages(messages: list[dict], profile: Optional[str]) -> list[dict]:
        """Add an internal profile hint without changing model-visible content."""
        marked = [dict(message) for message in messages]
        if marked and profile:
            marked[0]["_profile"] = profile
        return marked

    @staticmethod
    def _is_quick_inventory_request(message: str) -> bool:
        """Identify file-inventory requests that do not need deep reasoning."""
        text = message.lower()
        return (
            any(word in text for word in ("enumerate", "list", "show"))
            and "file" in text
            and not any(word in text for word in ("improve", "analyze", "review"))
        )

    def should_use_agent(self, message: str, messages: list[dict]) -> Optional[dict]:
        """Determine if the query needs multi-step agent processing.

        Returns a plan dict if agent mode is needed, None otherwise.
        """
        planning_messages = [
            {"role": "system", "content": PLANNING_PROMPT},
            {"role": "user", "content": message},
        ]
        profile = "fast" if self._is_quick_inventory_request(message) else None

        try:
            response = self.llm_client.chat(self._profile_messages(planning_messages, profile))
            stripped = response.strip()

            # Try to parse as JSON
            if stripped.startswith("{"):
                plan = json.loads(stripped)
                if plan.get("plan") is True and plan.get("steps"):
                    plan = self._normalize_plan(plan)
                    return plan
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Planning parse failed (will use normal flow): %s", e)

        return None

    @staticmethod
    def _normalize_plan(plan: dict) -> dict:
        """Keep planning metadata from being mistaken for executable tools."""
        from app.conversation_manager import KNOWN_TOOLS

        normalized = dict(plan)
        requested = plan.get("tools_needed", [])
        normalized["tools_needed"] = [
            tool for tool in requested if tool in KNOWN_TOOLS
        ]
        normalized["unsupported_tools"] = [
            tool for tool in requested if tool not in KNOWN_TOOLS
        ]
        return normalized

    def execute(self, message: str, session_id: str, messages: list[dict], plan: dict) -> dict:
        """Execute a multi-step agent plan.

        Args:
            message: The user's original message.
            session_id: Current session ID.
            messages: The full message context.
            plan: The plan dict from should_use_agent.

        Returns:
            Dict with 'response' (final text) and 'steps' (list of AgentStep dicts).
        """
        steps = []
        tool_results = []
        planned_steps = plan.get("steps", [])
        tools_needed = set(plan.get("tools_needed", []))
        profile = "fast" if self._is_quick_inventory_request(message) else None

        # Step 1: Show the plan
        steps.append({
            "type": "thinking",
            "description": f"Planning approach: {len(planned_steps)} step(s) identified",
            "details": planned_steps,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

        # Step 2: Execute tools iteratively
        iteration = 0
        context_so_far = ""

        while iteration < MAX_AGENT_STEPS:
            iteration += 1

            # Ask LLM what to do next
            next_action = self._get_next_action(
                message, messages, tool_results, context_so_far, planned_steps, profile,
                tools_needed=tools_needed,
            )

            if next_action is None:
                # LLM decided no more tools needed
                break

            if next_action.get("done"):
                # Enforce that critical planned tools were actually called
                missing = self._get_missing_critical_tools(tools_needed, tool_results)
                if missing:
                    # Force the first missing critical tool instead of finishing
                    forced = self._force_missing_tool_action(
                        missing, message, tool_results
                    )
                    if forced:
                        next_action = forced
                    else:
                        # Cannot recover content from prior results; ask LLM
                        # to produce the tool call properly
                        nudge = self._nudge_for_missing_tool(
                            missing[0], message, messages, tool_results, planned_steps, profile,
                            tools_needed=tools_needed,
                        )
                        if nudge and not nudge.get("done"):
                            next_action = nudge
                        else:
                            break
                elif not tool_results:
                    # No tools called yet and nothing critical missing — try
                    # the first inspection fallback
                    next_action = self._first_inspection_action(message, planned_steps)
                    if next_action is None:
                        break
                else:
                    break

            tool_name = next_action.get("tool", "")
            tool_args = next_action.get("args", {})

            if tool_name == "run_command":
                tool_args = self._resolve_run_command(tool_args, tool_results)

            # Validate write_file content is not a placeholder
            if tool_name == "write_file":
                content = tool_args.get("content", "")
                if self._is_placeholder_content(content):
                    logger.warning(
                        "write_file called with placeholder content (%d chars). "
                        "Attempting to recover real content.", len(content)
                    )
                    # First: check if the LLM's last response had a ```python block
                    # with the actual code (common pattern: LLM shows code in python
                    # block but puts placeholder in the JSON write_file call)
                    real_content = None
                    if hasattr(self, '_last_llm_response') and self._last_llm_response:
                        extracted = self._extract_code_block(self._last_llm_response)
                        if extracted and len(extracted) > 50 and not self._is_placeholder_content(extracted):
                            real_content = extracted
                            logger.info(
                                "Recovered real content from ```python block (%d chars)",
                                len(real_content),
                            )

                    # Second: ask LLM to generate actual content
                    if not real_content:
                        real_content = self._generate_file_content(
                            message, messages, tool_args.get("path", ""), tool_results, profile
                        )

                    if real_content:
                        tool_args = {**tool_args, "content": real_content}
                    else:
                        logger.error("Could not generate real file content. Skipping write_file.")
                        continue

            if tool_results and tool_results[-1].get("tool") == tool_name and tool_results[-1].get("args") == tool_args:
                return {
                    "response": self._recovery_prompt(
                        tool_name,
                        "The agent proposed the same action more than once without progress.",
                    ),
                    "steps": steps,
                    "pending_recovery": {
                        "message": message,
                        "messages": messages,
                        "tool_results": tool_results,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    },
                }

            if not tool_name:
                break

            # Record the step
            steps.append({
                "type": "tool_call",
                "description": f"Calling: {tool_name}",
                "tool": tool_name,
                "args": tool_args,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

            # Execute the tool
            tool_output = self.cm._execute_tool(tool_name, tool_args, session_id)

            # Record the result
            steps.append({
                "type": "result",
                "description": f"Result from {tool_name}",
                "output": tool_output[:500],  # Truncate for display
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

            tool_results.append({
                "tool": tool_name,
                "args": tool_args,
                "output": tool_output,
            })

            context_so_far += f"\n[{tool_name}]: {tool_output}\n"

            if self._tool_output_failed(tool_output):
                dependency = self._missing_python_dependency(tool_output)
                if (tool_name in ("run_code", "run_command")) and dependency:
                    install_command = f"python -m pip install {dependency}"
                    steps.append({
                        "type": "tool_call",
                        "description": f"Installing missing Python dependency: {dependency}",
                        "tool": "run_command",
                        "args": {"command": install_command},
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                    install_output = self.cm._execute_tool(
                        "run_command", {"command": install_command}, session_id
                    )
                    steps.append({
                        "type": "result",
                        "description": "Dependency installation result",
                        "output": install_output[:500],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                    tool_results.append({
                        "tool": "run_command",
                        "args": {"command": install_command},
                        "output": install_output,
                    })
                    context_so_far += f"\n[run_command]: {install_output}\n"
                    if not self._tool_output_failed(install_output):
                        # Retry the original command/code after installing
                        retry_output = self.cm._execute_tool(
                            tool_name, tool_args, session_id
                        )
                        steps.append({
                            "type": "tool_call",
                            "description": f"Retrying {tool_name} after dependency installation",
                            "tool": tool_name,
                            "args": tool_args,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })
                        steps.append({
                            "type": "result",
                            "description": f"Retry result from {tool_name}",
                            "output": retry_output[:500],
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        })
                        tool_results.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "output": retry_output,
                        })
                        context_so_far += f"\n[{tool_name}]: {retry_output}\n"
                        if not self._tool_output_failed(retry_output):
                            continue
                        tool_output = retry_output
                    else:
                        return {
                            "response": self._recovery_prompt("run_command", install_output),
                            "steps": steps,
                            "pending_recovery": {
                                "message": message,
                                "messages": messages,
                                "tool_results": tool_results,
                                "tool_name": "run_command",
                                "tool_args": {"command": install_command},
                            },
                        }
                if tool_name == "run_command" and self._needs_file_discovery(tool_output):
                    discovery_output = self.cm._execute_tool(
                        "list_files", {"path": "."}, session_id
                    )
                    steps.append({
                        "type": "tool_call",
                        "description": "Calling: list_files to locate the missing script",
                        "tool": "list_files",
                        "args": {"path": "."},
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                    steps.append({
                        "type": "result",
                        "description": "Result from list_files",
                        "output": discovery_output[:500],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                    tool_results.append({
                        "tool": "list_files",
                        "args": {"path": "."},
                        "output": discovery_output,
                    })
                    context_so_far += f"\n[list_files]: {discovery_output}\n"
                    continue
                return {
                    "response": self._recovery_prompt(tool_name, tool_output),
                    "steps": steps,
                    "pending_recovery": {
                        "message": message,
                        "messages": messages,
                        "tool_results": tool_results,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    },
                }

        # Step 3: Generate final aggregated response
        steps.append({
            "type": "thinking",
            "description": "Composing final response from gathered information",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

        final_response = self._generate_final_response(message, messages, tool_results)

        steps.append({
            "type": "final",
            "description": "Response ready",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

        return {
            "response": final_response,
            "steps": steps,
        }

    @staticmethod
    def _tool_output_failed(output: str) -> bool:
        """Recognize tool failures even when the tool returns formatted text."""
        lowered = output.lower()
        return (
            lowered.startswith(("error:", "file error:", "command failed"))
            or "is not recognized as an internal or external command" in lowered
            or "can't open file" in lowered
            or "no such file or directory" in lowered
        )

    @staticmethod
    def _needs_file_discovery(output: str) -> bool:
        lowered = output.lower()
        return "can't open file" in lowered or "no such file or directory" in lowered

    @staticmethod
    def _missing_python_dependency(output: str) -> Optional[str]:
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", output)
        if not match:
            return None
        return match.group(1).split(".")[0]

    @staticmethod
    def _resolve_run_command(tool_args: dict, tool_results: list[dict]) -> dict:
        """Resolve vague run requests against the most recently written file."""
        command = str(tool_args.get("command", "")).strip().lower()
        if command not in {
            "file", "the file", "it", "run it", "execute it",
            "it anyway", "run it anyway", "execute it anyway", "try it anyway",
            "the script", "script",
        }:
            return tool_args

        for result in reversed(tool_results):
            if result.get("tool") != "write_file":
                continue
            path = result.get("args", {}).get("path", "")
            from app.command_executor import command_for_file

            resolved_command = command_for_file(path)
            if resolved_command:
                return {**tool_args, "command": resolved_command}
        return tool_args

    def _first_inspection_action(
        self, message: str, planned_steps: Optional[list[str]] = None
    ) -> Optional[dict]:
        """Force the first real observation when an agent prematurely says done."""
        path = self.cm._guess_file_path_from_message(message)
        if path:
            return {"tool": "read_file", "args": {"path": path}}
        if any(word in message.lower() for word in ("file", "folder", "directory", "project")):
            return {"tool": "list_files", "args": {"path": "."}}
        if any(word in message.lower() for word in ("application", "app", "window")):
            return {"tool": "run_command", "args": {"command": "tasklist"}}
        return None

    @staticmethod
    def _recovery_prompt(tool_name: str, tool_output: str) -> str:
        """Offer actionable recovery choices after a tool failure."""
        if tool_name == "run_code":
            return (
                "The code could not be executed, Sir.\n\n"
                "[1] I will repair the code and retry it\n"
                "[2] Retry the original code\n"
                "[3] Cancel this operation\n\n"
                f"Error details:\n{tool_output[:1200]}\n\n"
                "Reply with 1, 2, or 3."
            )
        return (
            f"The tool '{tool_name}' failed, Sir.\n\n"
            "[1] Retry the operation\n"
            "[2] Cancel this operation\n\n"
            f"Error details:\n{tool_output[:1200]}\n\n"
            "Reply with 1 or 2."
        )

    def resume_recovery(self, choice: str, pending: dict, session_id: str) -> str:
        """Apply a user's recovery choice and retry the failed operation."""
        choice = choice.strip().lower()
        if choice in {"3", "cancel", "cancel this operation"}:
            return "Understood, Sir. The operation has been cancelled."

        tool_name = pending["tool_name"]
        tool_args = dict(pending["tool_args"])
        if tool_name == "run_code" and choice in {"1", "repair", "fix"}:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You repair Python code. Return only the corrected Python code "
                        "inside a ```python``` block. Do not explain anything."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original code:\n```python\n{tool_args.get('code', '')}\n```\n\n"
                        f"Execution error:\n{pending['tool_results'][-1]['output']}"
                    ),
                },
            ]
            repaired = self.llm_client.chat(repair_messages)
            repaired_code = self._extract_code_block(repaired)
            if not repaired_code:
                return "I could not produce a corrected version, Sir. Please choose 2 to retry or 3 to cancel."
            tool_args["code"] = repaired_code
        elif choice not in {"1", "2", "retry", "retry the original code", "retry the operation"}:
            return "Please choose one of the numbered recovery options, Sir."

        output = self.cm._execute_tool(tool_name, tool_args, session_id)
        if output.startswith("Error:"):
            return self._recovery_prompt(tool_name, output)

        results = list(pending["tool_results"])
        results[-1] = {"tool": tool_name, "args": tool_args, "output": output}
        return self._generate_final_response(
            pending["message"], pending["messages"], results
        )

    def _get_next_action(
        self, message: str, messages: list[dict],
        tool_results: list[dict], context_so_far: str,
        planned_steps: list[str], profile: Optional[str],
        tools_needed: Optional[set] = None,
    ) -> Optional[dict]:
        """Ask the LLM what tool to call next, or if we're done.

        Returns:
            Dict with 'tool' and 'args', or {'done': True}, or None.
        """
        results_summary = ""
        if tool_results:
            parts = []
            for r in tool_results:
                parts.append(f"[{r['tool']}]: {r['output'][:300]}")
            results_summary = "\n".join(parts)

        # Build a constraint clause when write_file is planned but not yet called
        file_persistence_constraint = ""
        if tools_needed and "write_file" in tools_needed:
            already_called = {r["tool"] for r in tool_results}
            if "write_file" not in already_called:
                file_persistence_constraint = (
                    "\n\nCRITICAL CONSTRAINT: The plan requires persisting a file to disk using write_file. "
                    "Do NOT use run_code to execute the script in-memory. You MUST call write_file to save "
                    "the file first, then use run_command to execute it. run_code is NOT a substitute for "
                    "write_file + run_command when the user asked to create/save a file."
                )

        next_action_prompt = f"""You are executing a multi-step plan for the user's request: "{message}"

    Only use these executable tools: read_file, write_file, list_files, search_files, run_command, run_code, web_search, network_scan, get_time, get_weather, get_forecast, add_note, get_notes, complete_note, clear_all_notes, search_notes, get_calendar_events, create_calendar_event, send_email, list_lights, turn_on_light, turn_off_light, set_light_color, ssh_execute, ssh_list_hosts, ssh_add_host.
    Never invent tool names. If an operation has no specialized tool, use run_command with the appropriate platform command or create a script using write_file and then execute it with run_command.

Tools already called and their results:
{results_summary if results_summary else "(none yet)"}

Planned steps that must be completed:
{chr(10).join(f"- {step}" for step in planned_steps)}{file_persistence_constraint}

Based on the user's original request and the results gathered so far, what should be done next?

IMPORTANT: If you need to edit/modify a file, you MUST call write_file with the FULL new file content. Do NOT just show the code — actually call the tool.

If you need to call another tool, respond with ONLY:
{{"tool": "<tool_name>", "args": {{<arguments>}}}}

Only respond with done after the required tool steps have actually been performed and their real results are above:
{{"done": true}}

Do NOT include any other text. Do NOT show code blocks. ONLY the JSON tool call or done."""

        action_messages = messages[:1] + [  # Keep system prompt
            {"role": "user", "content": next_action_prompt}
        ]

        try:
            response = self.llm_client.chat(self._profile_messages(action_messages, profile))
            self._last_llm_response = response  # Store for content recovery
            stripped = response.strip()

            # Try to parse JSON — handle cases where LLM adds trailing text
            if stripped.startswith("{"):
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    # Try extracting just the first JSON object
                    data = self._extract_first_json_object(stripped)
                    if data is None:
                        # Fall through to code block extraction below
                        data = None

                if data is not None:
                    from app.conversation_manager import KNOWN_TOOLS
                    if data.get("tool") not in KNOWN_TOOLS and not data.get("done"):
                        logger.warning("Replacing unsupported agent tool: %s", data.get("tool"))
                        return {"tool": "list_files", "args": {"path": "."}}

                    # Prevent run_code when write_file is planned but not yet called
                    if (data.get("tool") == "run_code" and tools_needed
                            and "write_file" in tools_needed):
                        already_called = {r["tool"] for r in tool_results}
                        if "write_file" not in already_called:
                            logger.info(
                                "Agent attempted run_code but write_file is planned and "
                                "not yet called. Redirecting to write_file."
                            )
                            # Extract the code from run_code args and redirect to write_file
                            code = data.get("args", {}).get("code", "")
                            file_path = self._guess_write_file_path(message, tool_results)
                            if code and file_path:
                                return {"tool": "write_file", "args": {"path": file_path, "content": code}}
                            # If we can't determine a path, let the LLM try again
                            # by not returning done
                            return {"tool": "write_file", "args": {"path": file_path or "script.py", "content": code}}

                    return data

            # Check if the LLM returned a code block instead of a tool call
            if "```" in response:
                # First: try to extract a JSON tool call from ```json blocks
                # Prefer write_file if it's in the plan and not yet called
                preferred = None
                if tools_needed and "write_file" in tools_needed:
                    already_called = {r["tool"] for r in tool_results}
                    if "write_file" not in already_called:
                        preferred = "write_file"
                json_block = self._extract_json_from_code_block(response, preferred_tool=preferred)
                if json_block and json_block.get("tool"):
                    logger.info("Agent: extracted JSON tool call from code block: %s", json_block.get("tool"))
                    return json_block

                # Second: extract code for write_file
                code = self._extract_code_block(response)
                if code and tool_results:
                    # Find the file path from previous read_file results
                    file_path = self._find_file_path(tool_results, message)
                    if file_path:
                        logger.info("Agent: extracted code block, forcing write_file to %s", file_path)
                        return {"tool": "write_file", "args": {"path": file_path, "content": code}}
                elif code and tools_needed and "write_file" in tools_needed:
                    # No prior tool results, but write_file is planned — use the code
                    file_path = self._guess_write_file_path(message, tool_results)
                    if file_path:
                        logger.info("Agent: extracted code block for planned write_file to %s", file_path)
                        return {"tool": "write_file", "args": {"path": file_path, "content": code}}

            # Also try parsing tool calls from within prose
            from app.conversation_manager import KNOWN_TOOLS
            import re
            for tool_name in KNOWN_TOOLS:
                pattern = r'\{"' + re.escape(tool_name) + r'"\s*:\s*"([^"]*(?:\\.[^"]*)*)"\}'
                match = re.search(pattern, stripped)
                if match:
                    value = match.group(1).replace('\\"', '"')
                    if tool_name == "run_command":
                        return {"tool": tool_name, "args": {"command": value}}
                    elif tool_name == "write_file":
                        return {"tool": tool_name, "args": {"path": value, "content": ""}}

        except json.JSONDecodeError as e:
            logger.debug("Next action JSON parse failed: %s", e)
        except Exception as e:
            logger.warning("Next action unexpected error: %s", e, exc_info=True)

        return {"done": True}

    def _extract_code_block(self, response: str) -> str:
        """Extract code from a markdown code block in the response.

        Handles both properly closed blocks (```python...```) and
        truncated/unclosed blocks where the LLM response was cut off.
        """
        import re
        # First try: properly closed code block
        match = re.search(r'```(?:python|py)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Second try: unclosed code block (truncated LLM response)
        # Only match if it starts with ```python (require language tag to avoid
        # matching prose that happens to start with ```)
        unclosed = re.search(r'```(?:python|py)\s*\n(.+)', response, re.DOTALL)
        if unclosed:
            code = unclosed.group(1).strip()
            # Remove any trailing prose that clearly isn't code
            # (e.g., "### Steps to Execute" or "1. Run the script")
            lines = code.split("\n")
            code_lines = []
            for line in lines:
                # Stop at markdown headers, numbered lists after blank line, etc.
                if re.match(r'^#{1,4}\s', line):
                    break
                if re.match(r'^\d+\.\s+\*\*', line):  # "1. **Step**"
                    break
                code_lines.append(line)
            # Remove trailing blank lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            result = "\n".join(code_lines).strip()
            if len(result) > 30:  # Only use if substantial
                return result

        return ""

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict]:
        """Extract the first valid JSON object from text that may have trailing content."""
        # Find the matching closing brace by counting braces
        depth = 0
        in_string = False
        escape_next = False
        start = text.index("{")

        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                if in_string:
                    escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _extract_json_from_code_block(self, response: str, preferred_tool: Optional[str] = None) -> Optional[dict]:
        """Extract a JSON tool call from ```json code blocks.

        If preferred_tool is set, looks for that tool among all blocks first.
        Otherwise returns the first valid tool call found.
        """
        import re
        blocks = re.findall(r'```(?:json)?\s*\n(.*?)```', response, re.DOTALL)
        if not blocks:
            return None

        all_calls = []
        for block_str in blocks:
            json_str = block_str.strip()
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and data.get("tool"):
                    all_calls.append(data)
            except json.JSONDecodeError:
                # Try extracting first JSON object from the block
                if json_str.startswith("{"):
                    extracted = self._extract_first_json_object(json_str)
                    if extracted and extracted.get("tool"):
                        all_calls.append(extracted)

        if not all_calls:
            return None

        # If a preferred tool is specified, prioritize it
        if preferred_tool:
            for call in all_calls:
                if call.get("tool") == preferred_tool:
                    # For write_file, reject calls with placeholder content
                    if preferred_tool == "write_file":
                        content = call.get("args", {}).get("content", "")
                        if self._is_placeholder_content(content):
                            logger.info(
                                "Skipping write_file JSON block with placeholder content"
                            )
                            continue
                    return call

        # Return the first valid tool call
        return all_calls[0]

    @staticmethod
    def _get_missing_critical_tools(tools_needed: set, tool_results: list[dict]) -> list[str]:
        """Return planned tools that are critical and have not been called yet.

        Critical tools are those whose absence means the task is incomplete:
        - write_file: the user explicitly asked to save/create a file
        """
        # Tools that MUST be called if they appear in the plan
        CRITICAL_TOOLS = {"write_file"}
        already_called = {r["tool"] for r in tool_results}
        missing = [
            tool for tool in tools_needed
            if tool in CRITICAL_TOOLS and tool not in already_called
        ]
        return missing

    def _force_missing_tool_action(
        self, missing_tools: list[str], message: str, tool_results: list[dict]
    ) -> Optional[dict]:
        """Create an action to call the first missing critical tool.

        When the agent tries to declare 'done' but critical planned tools haven't
        been called, this forces the next required tool call.
        """
        tool = missing_tools[0]
        if tool == "write_file":
            # Try to find content from a prior run_code call or from the message
            file_path = self._guess_write_file_path(message, tool_results)
            content = self._extract_write_content_from_results(tool_results, message)
            if file_path and content:
                logger.info(
                    "Enforcing planned write_file (agent tried to skip it): %s",
                    file_path,
                )
                return {"tool": "write_file", "args": {"path": file_path, "content": content}}
        return None

    def _nudge_for_missing_tool(
        self,
        missing_tool: str,
        message: str,
        messages: list[dict],
        tool_results: list[dict],
        planned_steps: list[str],
        profile: Optional[str],
        tools_needed: Optional[set] = None,
    ) -> Optional[dict]:
        """Ask the LLM explicitly to produce the missing tool call.

        Called when the agent declared 'done' but a critical tool (e.g. write_file)
        was never called and we couldn't synthesize the call ourselves.
        """
        results_summary = ""
        if tool_results:
            parts = []
            for r in tool_results:
                parts.append(f"[{r['tool']}]: {r['output'][:300]}")
            results_summary = "\n".join(parts)

        file_path = self._guess_write_file_path(message, tool_results)

        if missing_tool == "write_file":
            nudge_prompt = f"""The user asked: "{message}"

The plan requires saving a file to disk using write_file, but you never did it.
You MUST now produce the write_file tool call with COMPLETE, REAL, WORKING code.

File path to use: {file_path}

Previous tool results (if any):
{results_summary if results_summary else "(none)"}

RESPOND WITH ONLY THIS JSON (no other text, no markdown, no explanation):
{{"tool": "write_file", "args": {{"path": "{file_path}", "content": "<THE COMPLETE PYTHON SCRIPT HERE>"}}}}

RULES:
- The "content" field must contain the FULL working Python script, NOT a placeholder.
- Include all imports, classes, functions, and a main block.
- The code must be a real implementation based on what the user asked for.
- Escape any quotes inside the content string with backslash.
- Do NOT output anything other than the JSON object."""
        else:
            nudge_prompt = f"""The user asked: "{message}"

The plan requires calling '{missing_tool}', but you never did it.
You MUST now produce the '{missing_tool}' tool call.

Previous tool results (if any):
{results_summary if results_summary else "(none)"}

Respond with ONLY the JSON tool call. No other text."""

        action_messages = messages[:1] + [
            {"role": "user", "content": nudge_prompt}
        ]

        try:
            response = self.llm_client.chat(self._profile_messages(action_messages, profile))
            stripped = response.strip()

            # Try to parse as JSON directly
            if stripped.startswith("{"):
                data = json.loads(stripped)
                if data.get("tool") == missing_tool:
                    # Validate that write_file has real content (not a placeholder)
                    if missing_tool == "write_file":
                        content = data.get("args", {}).get("content", "")
                        if content and len(content) > 20 and "<" not in content[:30]:
                            logger.info(
                                "Nudge successful: LLM produced %s call (%d chars)",
                                missing_tool, len(content),
                            )
                            return data
                        else:
                            logger.warning(
                                "Nudge returned write_file with placeholder/empty content, "
                                "falling through to code block extraction."
                            )
                    else:
                        logger.info("Nudge successful: LLM produced %s call", missing_tool)
                        return data

            # If the LLM returned a code block, extract it for write_file
            if missing_tool == "write_file" and "```" in response:
                code = self._extract_code_block(response)
                if code and len(code) > 20:
                    path = file_path or self._guess_write_file_path(message, tool_results)
                    logger.info("Nudge: extracted code block for write_file to %s (%d chars)", path, len(code))
                    return {"tool": "write_file", "args": {"path": path, "content": code}}

            # Last resort: the LLM returned prose with code inline (no fences)
            # Try to extract anything that looks like Python code
            if missing_tool == "write_file" and len(stripped) > 100:
                # Check if the response itself IS code (starts with import, def, class, #)
                first_line = stripped.split("\n")[0].strip()
                if re.match(r'^(import |from |#|def |class |""")', first_line):
                    logger.info("Nudge: response appears to be raw code, using as write_file content")
                    path = file_path or self._guess_write_file_path(message, tool_results)
                    return {"tool": "write_file", "args": {"path": path, "content": stripped}}

        except json.JSONDecodeError:
            # JSON parse failed — try to extract code from the response
            if missing_tool == "write_file":
                code = self._extract_code_block(response)
                if code and len(code) > 20:
                    path = file_path or self._guess_write_file_path(message, tool_results)
                    logger.info("Nudge (JSON failed): extracted code block for write_file to %s", path)
                    return {"tool": "write_file", "args": {"path": path, "content": code}}
        except Exception as e:
            logger.debug("Nudge failed: %s", e)

        return None

    def _guess_write_file_path(self, message: str, tool_results: list[dict]) -> str:
        """Determine the target file path for a write_file call.

        Checks prior tool results (read_file, write_file attempts) and the message.
        """
        import re
        # Check if any previous tool call referenced a path
        for r in reversed(tool_results):
            if r["tool"] in ("read_file", "write_file") and r.get("args", {}).get("path"):
                return r["args"]["path"]

        # Try to extract from the user's message
        match = re.search(r'(\S+\.(?:py|js|ts|sh|rb|pl|java|c|cpp|go|rs))\b', message)
        if match:
            return match.group(1)

        # Default fallback for Python scripts
        msg_lower = message.lower()
        if "python" in msg_lower or "script" in msg_lower:
            return "script.py"

        return "output.txt"

    @staticmethod
    def _extract_write_content_from_results(tool_results: list[dict], message: str) -> str:
        """Extract file content from prior tool results (e.g., code from run_code args).

        When the agent used run_code instead of write_file, the code is in the
        run_code args and we can recover it.
        """
        # Look for run_code calls — the code arg is what should have been written
        for r in reversed(tool_results):
            if r["tool"] == "run_code" and r.get("args", {}).get("code"):
                return r["args"]["code"]
        return ""

    @staticmethod
    def _is_placeholder_content(content: str) -> bool:
        """Detect if write_file content is a placeholder rather than real code."""
        if not content or len(content.strip()) < 20:
            return True
        stripped = content.strip()
        # Common placeholder patterns
        placeholder_markers = [
            "<paste", "<insert", "<your", "<code", "<script", "<the ",
            "...", "# TODO", "# placeholder",
        ]
        if any(marker in stripped.lower() for marker in placeholder_markers):
            # But only if the content is short (long files with TODO comments are OK)
            if len(stripped) < 200:
                return True
        # If content is just a single short line, it's likely a placeholder
        lines = [l for l in stripped.split("\n") if l.strip()]
        if len(lines) <= 2 and len(stripped) < 100:
            return True
        # Detect stub scripts where all function bodies are just 'pass' or comments
        # Count function/method definitions vs actual implementation lines
        import re
        func_defs = re.findall(r'^(?:    )?def \w+', stripped, re.MULTILINE)
        pass_lines = re.findall(r'^\s+pass\s*$', stripped, re.MULTILINE)
        if func_defs and len(pass_lines) >= len(func_defs):
            # Every function is a stub — this is not real implementation
            return True
        return False

    def _generate_file_content(
        self, message: str, messages: list[dict], file_path: str,
        tool_results: list[dict], profile: Optional[str],
    ) -> Optional[str]:
        """Ask the LLM to generate actual file content for a write_file call.

        Called when the agent produced a write_file call with placeholder content.
        """
        results_context = ""
        if tool_results:
            parts = []
            for r in tool_results:
                parts.append(f"[{r['tool']}]: {r['output'][:200]}")
            results_context = "\n".join(parts)

        gen_prompt = f"""The user asked: "{message}"

You need to write the COMPLETE content for the file "{file_path}".

{f"Context from previous steps:{chr(10)}{results_context}" if results_context else ""}

Generate the FULL, WORKING Python script. Include:
- All necessary imports
- Complete class/function implementations
- A main block (if __name__ == "__main__")
- Proper error handling
- Comments explaining key sections

Output ONLY the Python code. No markdown fences. No explanations. Just the raw code."""

        gen_messages = messages[:1] + [
            {"role": "user", "content": gen_prompt}
        ]

        try:
            response = self.llm_client.chat(self._profile_messages(gen_messages, profile))
            code = response.strip()

            # Strip markdown fences if the LLM added them despite instructions
            if code.startswith("```"):
                code = self._extract_code_block(response) or code
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            # Validate: must look like real code
            if len(code) > 50 and not self._is_placeholder_content(code):
                logger.info(
                    "Generated real file content for %s (%d chars)", file_path, len(code)
                )
                return code
            else:
                logger.warning(
                    "Generated content still looks like a placeholder (%d chars)", len(code)
                )
        except Exception as e:
            logger.error("Failed to generate file content: %s", e)

        return None

    def _find_file_path(self, tool_results: list[dict], message: str) -> str:
        """Try to determine the target file path from previous tool results or the message."""
        import re
        # Check if read_file was called previously — use that path
        for r in tool_results:
            if r["tool"] == "read_file" and r.get("args", {}).get("path"):
                return r["args"]["path"]
        # Try to extract from the message
        match = re.search(r'(\S+\.py)\b', message)
        if match:
            return match.group(1)
        return ""

    def _generate_final_response(
        self, message: str, messages: list[dict], tool_results: list[dict]
    ) -> str:
        """Generate the final aggregated response using all tool results."""
        results_text = ""
        if tool_results:
            parts = []
            for r in tool_results:
                parts.append(f"--- {r['tool']} ---\n{r['output']}")
            results_text = "\n\n".join(parts)

        # Build a file path confirmation clause for the prompt
        written_files = [
            r["args"].get("path", "")
            for r in tool_results
            if r["tool"] == "write_file" and not r["output"].startswith("File error:")
        ]
        file_path_clause = ""
        if written_files:
            paths = ", ".join(written_files)
            file_path_clause = (
                f"\n\nFiles confirmed written to disk: {paths}. "
                "If the user asks where a file was saved, report ONLY these confirmed paths."
            )
        elif any("write_file" in step for step in (r["tool"] for r in tool_results)):
            file_path_clause = (
                "\n\nNOTE: A write_file call was attempted but may have failed. "
                "Do NOT fabricate a file path. State that the file was not successfully created "
                "if the tool output indicates failure."
            )

        final_prompt = (
            f"You executed multiple tools to answer the user's request: \"{message}\"\n\n"
            f"Here are ALL the real results gathered:\n\n{results_text}\n\n"
            f"Now compose a single, cohesive response as Jarvis using ALL the real data above. "
            f"Treat command results as authoritative: never claim a command succeeded or a file was created "
            f"unless the tool output explicitly confirms success. "
            f"Use the actual values — do NOT use placeholders like [summary] or [percentage]. "
            f"Be thorough but concise. Address the user as 'Sir'.{file_path_clause}"
        )

        final_messages = messages[:1] + [  # Keep system prompt
            {"role": "user", "content": final_prompt}
        ]

        profile = "fast" if self._is_quick_inventory_request(message) else None
        return self.llm_client.chat(self._profile_messages(final_messages, profile))
