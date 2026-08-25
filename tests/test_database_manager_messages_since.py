"""Unit tests for DatabaseManager.get_messages_since (used by MemoryConsolidator)."""

from datetime import datetime, timedelta

import pytest

from app.database_manager import DatabaseManager


@pytest.fixture
def db(tmp_path):
    dbm = DatabaseManager(db_path=str(tmp_path / "test.db"))
    dbm.initialize()
    return dbm


class TestGetMessagesSince:
    def test_groups_messages_by_username(self, db):
        db.save_message("s1", "user", "hi alice", username="alice")
        db.save_message("s2", "user", "hi bob", username="bob")

        grouped = db.get_messages_since(datetime.now() - timedelta(days=1))

        assert set(grouped.keys()) == {"alice", "bob"}
        assert grouped["alice"][0]["content"] == "hi alice"
        assert grouped["bob"][0]["content"] == "hi bob"

    def test_anonymous_sessions_grouped_under_empty_string(self, db):
        db.save_message("s1", "user", "anon message", username=None)

        grouped = db.get_messages_since(datetime.now() - timedelta(days=1))

        assert "" in grouped
        assert grouped[""][0]["content"] == "anon message"

    def test_excludes_messages_before_cutoff(self, db):
        db.save_message("s1", "user", "hi", username="alice")

        grouped = db.get_messages_since(datetime.now() + timedelta(days=1))

        assert grouped == {}

    def test_messages_ordered_chronologically_within_user(self, db):
        db.save_message("s1", "user", "first", username="alice")
        db.save_message("s1", "assistant", "second", username="alice")
        db.save_message("s1", "user", "third", username="alice")

        grouped = db.get_messages_since(datetime.now() - timedelta(days=1))

        contents = [m["content"] for m in grouped["alice"]]
        assert contents == ["first", "second", "third"]

    def test_empty_database_returns_empty_dict(self, db):
        assert db.get_messages_since(datetime.now() - timedelta(days=1)) == {}
