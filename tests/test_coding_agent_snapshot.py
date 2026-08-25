"""Unit tests for TaskSnapshot (app.coding_agent.snapshot)."""

import os

import pytest

from app.coding_agent.snapshot import TaskSnapshot


@pytest.fixture
def project_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "a.py").write_text("original a\n", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "b.py").write_text("original b\n", encoding="utf-8")
    # Excluded dirs/files should never be captured or touched.
    (d / "backups").mkdir()
    (d / "backups" / "should_not_be_snapshotted.txt").write_text("noop", encoding="utf-8")
    (d / "jarvis.db").write_text("binary-ish db content", encoding="utf-8")
    return d


@pytest.fixture
def snapshot(project_dir):
    return TaskSnapshot(str(project_dir))


class TestCreateAndRestore:
    def test_restore_reverts_modified_file(self, snapshot, project_dir):
        snapshot.create("s1")
        (project_dir / "a.py").write_text("modified a\n", encoding="utf-8")

        restored = snapshot.restore("s1")

        assert restored is True
        assert (project_dir / "a.py").read_text(encoding="utf-8") == "original a\n"

    def test_restore_reverts_nested_file(self, snapshot, project_dir):
        snapshot.create("s1")
        (project_dir / "sub" / "b.py").write_text("modified b\n", encoding="utf-8")

        snapshot.restore("s1")

        assert (project_dir / "sub" / "b.py").read_text(encoding="utf-8") == "original b\n"

    def test_restore_removes_newly_created_file(self, snapshot, project_dir):
        snapshot.create("s1")
        (project_dir / "brand_new.py").write_text("i should not survive\n", encoding="utf-8")

        snapshot.restore("s1")

        assert not (project_dir / "brand_new.py").exists()

    def test_restore_brings_back_deleted_file(self, snapshot, project_dir):
        snapshot.create("s1")
        (project_dir / "a.py").unlink()

        snapshot.restore("s1")

        assert (project_dir / "a.py").read_text(encoding="utf-8") == "original a\n"

    def test_restore_missing_snapshot_returns_false(self, snapshot):
        assert snapshot.restore("never-created") is False

    def test_excluded_dirs_are_never_snapshotted(self, snapshot, project_dir):
        dest = snapshot.create("s1")
        assert not os.path.exists(os.path.join(dest, "backups"))
        assert not os.path.exists(os.path.join(dest, "jarvis.db"))

    def test_excluded_dirs_survive_restore_untouched(self, snapshot, project_dir):
        snapshot.create("s1")
        (project_dir / "backups" / "new_file_during_task.txt").write_text("added mid-task", encoding="utf-8")

        snapshot.restore("s1")

        # Excluded paths are left alone by restore (not snapshotted, not deleted).
        assert (project_dir / "backups" / "should_not_be_snapshotted.txt").exists()
        assert (project_dir / "backups" / "new_file_during_task.txt").exists()


class TestDiscard:
    def test_discard_removes_snapshot_directory(self, snapshot, project_dir):
        dest = snapshot.create("s1")
        assert os.path.isdir(dest)

        snapshot.discard("s1")

        assert not os.path.isdir(dest)

    def test_discard_of_nonexistent_snapshot_is_a_noop(self, snapshot):
        snapshot.discard("never-created")  # must not raise
