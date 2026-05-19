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

CRITICAL: When asked to "modify", "adapt", "update", "change", or "edit" a file and then run it, you MUST plan: read_file → write_file → run_command. NEVER skip the read/write steps.

Examples of simple queries:
- "Hello, how are you?" → simple conversation
- "What time is it?" → single tool call
- "Turn off the bedroom light" → single tool call"""


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

    def should_use_agent(self, message: str, messages: list[dict]) -> Optional[dict]:
        """Determine if the query needs multi-step agent processing.

        Returns a plan dict if agent mode is needed, None otherwise.
        """
        planning_messages = [
            {"role": "system", "content": PLANNING_PROMPT},
            {"role": "user", "content": message},
        ]

        try:
            response = self.llm_client.chat(planning_messages)
            stripped = response.strip()

            # Try to parse as JSON
            if stripped.startswith("{"):
                plan = json.loads(stripped)
                if plan.get("plan") is True and plan.get("steps"):
                    return plan
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Planning parse failed (will use normal flow): %s", e)

        return None

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
            next_action = self._get_next_action(message, messages, tool_results, context_so_far)

            if next_action is None:
                # LLM decided no more tools needed
                break

            if next_action.get("done"):
                # LLM says we have enough info
                break

            tool_name = next_action.get("tool", "")
            tool_args = next_action.get("args", {})

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

    def _get_next_action(
        self, message: str, messages: list[dict],
        tool_results: list[dict], context_so_far: str
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

        next_action_prompt = f"""You are executing a multi-step plan for the user's request: "{message}"

Tools already called and their results:
{results_summary if results_summary else "(none yet)"}

Based on the user's original request and the results gathered so far, what should be done next?

IMPORTANT: If you need to edit/modify a file, you MUST call write_file with the FULL new file content. Do NOT just show the code — actually call the tool.

If you need to call another tool, respond with ONLY:
{{"tool": "<tool_name>", "args": {{<arguments>}}}}

If you have gathered enough information to answer the user, respond with ONLY:
{{"done": true}}

Do NOT include any other text. Do NOT show code blocks. ONLY the JSON tool call or done."""

        action_messages = messages[:1] + [  # Keep system prompt
            {"role": "user", "content": next_action_prompt}
        ]

        try:
            response = self.llm_client.chat(action_messages)
            stripped = response.strip()

            if stripped.startswith("{"):
                data = json.loads(stripped)
                return data

            # Check if the LLM returned a code block instead of a tool call
            # If so, extract the code and create a write_file tool call
            if "```" in response:
                code = self._extract_code_block(response)
                if code and tool_results:
                    # Find the file path from previous read_file results
                    file_path = self._find_file_path(tool_results, message)
                    if file_path:
                        logger.info("Agent: extracted code block, forcing write_file to %s", file_path)
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

        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Next action parse failed: %s", e)

        return {"done": True}

    def _extract_code_block(self, response: str) -> str:
        """Extract code from a markdown code block in the response."""
        import re
        match = re.search(r'```(?:python|py)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

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

        final_prompt = (
            f"You executed multiple tools to answer the user's request: \"{message}\"\n\n"
            f"Here are ALL the real results gathered:\n\n{results_text}\n\n"
            f"Now compose a single, cohesive response as Jarvis using ALL the real data above. "
            f"Use the actual values — do NOT use placeholders like [summary] or [percentage]. "
            f"Be thorough but concise. Address the user as 'Sir'."
        )

        final_messages = messages[:1] + [  # Keep system prompt
            {"role": "user", "content": final_prompt}
        ]

        return self.llm_client.chat(final_messages)
