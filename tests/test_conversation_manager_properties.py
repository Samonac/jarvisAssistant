"""Property tests for Conversation Manager.

Tests Properties 2, 3, 8, and 9.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.config import Config
from app.database_manager import DatabaseManager
from app.conversation_manager import ConversationManager, _get_system_prompt


def _make_config():
    os.environ["LLM_API_KEY"] = "test-key"
    os.environ["LLM_PROVIDER"] = "groq"
    return Config()


def _make_manager(db_path):
    config = _make_config()
    db = DatabaseManager(db_path=db_path)
    db.initialize()
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "Very good, Sir."
    mgr = ConversationManager(llm_client=mock_llm, config=config, db_manager=db)
    return mgr, db


# Feature: jarvis-assistant, Property 2: System prompt is always first
class TestProperty2:
    """The message array sent to the LLM always has the Jarvis system prompt first."""

    @given(
        message=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
    )
    @settings(max_examples=100)
    def test_system_prompt_always_first(self, message):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mgr, db = _make_manager(path)
        try:
            mgr.handle_message(message, "test-session")
            # Find the main chat call (the one with the Jarvis system prompt)
            main_call = None
            for call in mgr.llm_client.chat.call_args_list:
                msgs = call[0][0]
                if msgs and "J.A.R.V.I.S." in msgs[0].get("content", ""):
                    main_call = msgs
                    break
            assert main_call is not None, "No main chat call found"
            assert main_call[0]["role"] == "system"
            assert "J.A.R.V.I.S." in main_call[0]["content"]
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 3: Conversation history is included in LLM context
class TestProperty3:
    """For sessions with existing history, history messages are included in LLM context."""

    @given(
        num_prior=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_history_included_in_context(self, num_prior):
        """History messages are included in the LLM context."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mgr, db = _make_manager(path)
        try:
            session_id = "history-session"
            for i in range(num_prior):
                db.save_message(session_id, "user", f"prior-{i}")
                db.save_message(session_id, "assistant", f"reply-{i}")

            mgr.handle_message("new message", session_id)

            # Verify the LLM was called at least once
            assert mgr.llm_client.chat.call_count >= 1

            # Check that history content appears in at least one LLM call
            all_call_contents = str(mgr.llm_client.chat.call_args_list)
            assert "prior-0" in all_call_contents
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 8: History grows by one pair per interaction
class TestProperty8:
    """After processing a message, history has exactly one more pair."""

    @given(
        message=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=100)
    def test_history_grows_by_one_pair(self, message):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mgr, db = _make_manager(path)
        try:
            session_id = "growth-session"
            before = db.get_history(session_id, max_pairs=20)
            before_count = len(before)

            mgr.handle_message(message, session_id)

            after = db.get_history(session_id, max_pairs=20)
            after_count = len(after)

            # Should have exactly 2 more messages (1 user + 1 assistant)
            assert after_count == before_count + 2
        finally:
            db.close()
            os.unlink(path)


# Feature: jarvis-assistant, Property 9: History trimming preserves bounds and system prompt
class TestProperty9:
    """History retrieval never exceeds 10 pairs and system prompt is always first in context."""

    def test_history_trimming_at_limit(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mgr, db = _make_manager(path)
        try:
            session_id = "trim-session"
            # Add 15 pairs (exceeds 10-pair limit)
            for i in range(15):
                db.save_message(session_id, "user", f"msg-{i}")
                db.save_message(session_id, "assistant", f"reply-{i}")

            mgr.handle_message("new message", session_id)

            # Find the main chat call (the one with the Jarvis system prompt)
            main_call = None
            for call in mgr.llm_client.chat.call_args_list:
                msgs = call[0][0]
                if msgs and "J.A.R.V.I.S." in msgs[0].get("content", ""):
                    main_call = msgs
                    break

            assert main_call is not None
            assert main_call[0]["role"] == "system"
            assert "J.A.R.V.I.S." in main_call[0]["content"]

            # History portion should be limited (10 pairs = 20 messages max)
            history_portion = main_call[1:-1]
            assert len(history_portion) <= 20
        finally:
            db.close()
            os.unlink(path)
