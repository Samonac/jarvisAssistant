"""Backup Orchestrator for Jarvis Assistant.

Scheduled backups of the Jarvis database, configs, and knowledge base.
Supports local and remote backup destinations. Monitors backup health
and provides a disaster recovery plan.

Features:
- Scheduled automatic backups (configurable frequency)
- Backup targets: local directory, remote via SCP/SSH
- Backup contents: SQLite DB, .env, knowledge base, scripts, plugins
- Backup history with size/duration tracking
- Integrity verification (SHA256 checksums)
- Disaster recovery plan generation
- Heartbeat monitoring (detect unresponsive instance)
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class BackupOrchestrator:
    """Manages automated backups and disaster recovery.

    Attributes:
        db_manager: Database manager.
        project_dir: Root directory of the Jarvis project.
        backup_dir: Local backup storage directory.
    """

    def __init__(self, db_manager, project_dir: str = None):
        self.db_manager = db_manager
        self.project_dir = project_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.backup_dir = os.path.join(self.project_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)

        self.scheduler = None
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS backup_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backup_type TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    file_path TEXT,
                    size_bytes INTEGER,
                    duration_seconds REAL,
                    checksum TEXT,
                    status TEXT DEFAULT 'success',
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS backup_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS heartbeat (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_beat DATETIME,
                    instance_id TEXT,
                    version TEXT
                );
            """)
            # Initialize heartbeat
            conn.execute("""
                INSERT OR IGNORE INTO heartbeat (id, last_beat, instance_id, version)
                VALUES (1, ?, ?, '1.0')
            """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), platform.node()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init backup tables: %s", e)

    # ── Configuration ─────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Get backup configuration."""
        defaults = {
            "enabled": "true",
            "frequency": "daily",  # hourly, daily, weekly
            "time": "03:00",  # When to run daily backups
            "retention_days": "30",
            "remote_enabled": "false",
            "remote_host": "",
            "remote_path": "",
            "remote_user": "",
            "include_uploads": "false",
        }
        try:
            conn = self._get_conn()
            cursor = conn.execute("SELECT key, value FROM backup_config")
            for row in cursor.fetchall():
                defaults[row["key"]] = row["value"]
            conn.close()
        except Exception:
            pass
        return defaults

    def save_config(self, config: dict) -> dict:
        """Save backup configuration."""
        try:
            conn = self._get_conn()
            for key, value in config.items():
                conn.execute(
                    "INSERT OR REPLACE INTO backup_config (key, value) VALUES (?, ?)",
                    (key, str(value))
                )
            conn.commit()
            conn.close()
            return {"message": "Backup configuration saved."}
        except Exception as e:
            return {"error": str(e)}

    # ── Backup Execution ──────────────────────────────────────────────────

    def run_backup(self, backup_type: str = "full") -> dict:
        """Execute a backup.

        Args:
            backup_type: "full" (everything), "db" (database only), "config" (configs only).

        Returns:
            Dict with backup details or error.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"jarvis_backup_{backup_type}_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        os.makedirs(backup_path, exist_ok=True)

        files_backed_up = []
        errors = []

        try:
            # Always backup the database
            if backup_type in ("full", "db"):
                db_result = self._backup_database(backup_path)
                if db_result.get("success"):
                    files_backed_up.append(db_result["file"])
                else:
                    errors.append(f"DB: {db_result.get('error')}")

            # Backup configs
            if backup_type in ("full", "config"):
                config_files = [".env", ".env.example", "requirements.txt"]
                for cf in config_files:
                    src = os.path.join(self.project_dir, cf)
                    if os.path.exists(src):
                        dst = os.path.join(backup_path, cf)
                        shutil.copy2(src, dst)
                        files_backed_up.append(cf)

            # Backup scripts and plugins
            if backup_type == "full":
                for folder in ["scripts", "plugins"]:
                    src_dir = os.path.join(self.project_dir, folder)
                    if os.path.isdir(src_dir):
                        dst_dir = os.path.join(backup_path, folder)
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                        files_backed_up.append(f"{folder}/")

                # Optionally backup uploads
                config = self.get_config()
                if config.get("include_uploads") == "true":
                    uploads_dir = os.path.join(self.project_dir, "uploads")
                    if os.path.isdir(uploads_dir):
                        dst_dir = os.path.join(backup_path, "uploads")
                        shutil.copytree(uploads_dir, dst_dir, dirs_exist_ok=True)
                        files_backed_up.append("uploads/")

            # Create archive
            archive_path = shutil.make_archive(backup_path, "zip", backup_path)
            # Remove uncompressed folder
            shutil.rmtree(backup_path)

            # Calculate checksum
            checksum = self._sha256(archive_path)
            size = os.path.getsize(archive_path)
            duration = time.time() - start_time

            # Record in history
            self._record_backup(backup_type, "local", archive_path, size, duration, checksum,
                                "success" if not errors else "partial", "; ".join(errors) or None)

            # Copy to remote if configured
            config = self.get_config()
            remote_result = None
            if config.get("remote_enabled") == "true" and config.get("remote_host"):
                remote_result = self._push_to_remote(archive_path, config)

            # Cleanup old backups
            self._cleanup_old_backups(int(config.get("retention_days", 30)))

            return {
                "success": True,
                "archive": archive_path,
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 2),
                "duration_seconds": round(duration, 2),
                "checksum": checksum,
                "files": files_backed_up,
                "errors": errors,
                "remote": remote_result,
            }

        except Exception as e:
            duration = time.time() - start_time
            self._record_backup(backup_type, "local", "", 0, duration, "", "failed", str(e))
            return {"error": str(e)}

    def _backup_database(self, dest_dir: str) -> dict:
        """Safely backup the SQLite database using the backup API."""
        db_path = self.db_manager.db_path
        if not os.path.exists(db_path):
            return {"error": "Database file not found"}

        dest_path = os.path.join(dest_dir, "jarvis.db")
        try:
            # Use SQLite backup API for consistency
            src_conn = sqlite3.connect(db_path)
            dst_conn = sqlite3.connect(dest_path)
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
            return {"success": True, "file": "jarvis.db"}
        except Exception as e:
            # Fallback: simple copy
            try:
                shutil.copy2(db_path, dest_path)
                return {"success": True, "file": "jarvis.db"}
            except Exception as e2:
                return {"error": str(e2)}

    def _push_to_remote(self, archive_path: str, config: dict) -> dict:
        """Push backup archive to remote host via SCP."""
        host = config.get("remote_host", "")
        user = config.get("remote_user", "")
        remote_path = config.get("remote_path", "/tmp/jarvis_backups/")

        if not host:
            return {"error": "No remote host configured"}

        target = f"{user}@{host}:{remote_path}" if user else f"{host}:{remote_path}"
        filename = os.path.basename(archive_path)

        try:
            # Ensure remote directory exists
            mkdir_cmd = f"ssh {user + '@' if user else ''}{host} 'mkdir -p {remote_path}'"
            subprocess.run(mkdir_cmd, shell=True, timeout=10, capture_output=True)

            # SCP the file
            scp_cmd = f"scp {archive_path} {target}{filename}"
            result = subprocess.run(scp_cmd, shell=True, timeout=120, capture_output=True, text=True)

            if result.returncode == 0:
                self._record_backup("remote_copy", target, f"{remote_path}{filename}",
                                    os.path.getsize(archive_path), 0, "", "success")
                return {"success": True, "destination": f"{target}{filename}"}
            else:
                return {"error": result.stderr or "SCP failed"}
        except Exception as e:
            return {"error": str(e)}

    # ── Backup History & Monitoring ───────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get backup history."""
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                "SELECT * FROM backup_history ORDER BY created_at DESC LIMIT ?", (limit,))
            results = [{
                "id": r["id"], "type": r["backup_type"], "destination": r["destination"],
                "file_path": r["file_path"], "size_bytes": r["size_bytes"],
                "size_mb": round((r["size_bytes"] or 0) / 1024 / 1024, 2),
                "duration_seconds": r["duration_seconds"], "checksum": r["checksum"],
                "status": r["status"], "error": r["error_message"],
                "created_at": r["created_at"],
            } for r in cursor.fetchall()]
            conn.close()
            return results
        except Exception:
            return []

    def verify_backup(self, backup_id: int) -> dict:
        """Verify a backup's integrity by checking its checksum."""
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM backup_history WHERE id = ?", (backup_id,)).fetchone()
            conn.close()
            if not row:
                return {"error": "Backup not found"}

            file_path = row["file_path"]
            if not file_path or not os.path.exists(file_path):
                return {"valid": False, "error": "Backup file not found on disk"}

            current_checksum = self._sha256(file_path)
            stored_checksum = row["checksum"]

            return {
                "valid": current_checksum == stored_checksum,
                "stored_checksum": stored_checksum,
                "current_checksum": current_checksum,
                "file": file_path,
                "size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Heartbeat ─────────────────────────────────────────────────────────

    def update_heartbeat(self) -> None:
        """Update the heartbeat timestamp. Called periodically by the scheduler."""
        try:
            conn = self._get_conn()
            conn.execute(
                "UPDATE heartbeat SET last_beat = ? WHERE id = 1",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_heartbeat(self) -> dict:
        """Get current heartbeat status."""
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM heartbeat WHERE id = 1").fetchone()
            conn.close()
            if not row:
                return {"status": "unknown"}

            last_beat = datetime.strptime(row["last_beat"], "%Y-%m-%d %H:%M:%S")
            age_seconds = (datetime.now() - last_beat).total_seconds()

            return {
                "status": "healthy" if age_seconds < 120 else "warning" if age_seconds < 600 else "critical",
                "last_beat": row["last_beat"],
                "age_seconds": int(age_seconds),
                "instance_id": row["instance_id"],
                "version": row["version"],
            }
        except Exception:
            return {"status": "unknown"}

    # ── Disaster Recovery Plan ────────────────────────────────────────────

    def generate_dr_plan(self) -> dict:
        """Generate a disaster recovery plan based on current configuration."""
        config = self.get_config()
        history = self.get_history(limit=5)
        heartbeat = self.get_heartbeat()

        # Find latest successful backup
        latest_backup = None
        for h in history:
            if h["status"] == "success":
                latest_backup = h
                break

        # Calculate RPO (Recovery Point Objective)
        rpo_hours = None
        if latest_backup:
            backup_time = datetime.strptime(latest_backup["created_at"], "%Y-%m-%d %H:%M:%S")
            rpo_hours = round((datetime.now() - backup_time).total_seconds() / 3600, 1)

        plan = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "instance": heartbeat.get("instance_id", "unknown"),
            "health_status": heartbeat.get("status", "unknown"),
            "backup_status": {
                "enabled": config.get("enabled") == "true",
                "frequency": config.get("frequency", "daily"),
                "remote_enabled": config.get("remote_enabled") == "true",
                "latest_backup": latest_backup,
                "rpo_hours": rpo_hours,
            },
            "recovery_steps": [
                {
                    "step": 1,
                    "title": "Obtain Latest Backup",
                    "description": (
                        f"Latest backup: {latest_backup['file_path'] if latest_backup else 'NONE'}\n"
                        f"Remote copy: {'Yes — check ' + config.get('remote_host', 'N/A') + ':' + config.get('remote_path', '') if config.get('remote_enabled') == 'true' else 'Not configured'}"
                    ),
                },
                {
                    "step": 2,
                    "title": "Prepare New Instance",
                    "description": (
                        "1. Install Python 3.11+\n"
                        "2. Clone/copy the JarvisAssistant project directory\n"
                        "3. pip install -r requirements.txt\n"
                        "4. Extract backup ZIP into the project root"
                    ),
                },
                {
                    "step": 3,
                    "title": "Restore Database",
                    "description": (
                        "Copy jarvis.db from the backup to the project root.\n"
                        "This restores: conversations, notes, metrics, users, workflows, knowledge base."
                    ),
                },
                {
                    "step": 4,
                    "title": "Restore Configuration",
                    "description": (
                        "Copy .env from the backup. Verify API keys are still valid.\n"
                        "Update any host-specific settings (IP, paths)."
                    ),
                },
                {
                    "step": 5,
                    "title": "Restore Scripts & Plugins",
                    "description": "Copy scripts/ and plugins/ folders from the backup.",
                },
                {
                    "step": 6,
                    "title": "Start Jarvis",
                    "description": "Run: python run.py\nVerify: curl http://localhost:5000/health",
                },
                {
                    "step": 7,
                    "title": "Verify Recovery",
                    "description": (
                        "1. Log in with admin credentials\n"
                        "2. Check conversation history is intact\n"
                        "3. Verify notes and workflows are present\n"
                        "4. Test a chat message\n"
                        "5. Check knowledge base documents"
                    ),
                },
            ],
            "recommendations": [],
        }

        # Add recommendations based on current state
        if not latest_backup:
            plan["recommendations"].append("⚠️ CRITICAL: No backups found. Run a backup immediately.")
        elif rpo_hours and rpo_hours > 48:
            plan["recommendations"].append(f"⚠️ Last backup is {rpo_hours}h old. Consider increasing frequency.")
        if config.get("remote_enabled") != "true":
            plan["recommendations"].append("💡 Enable remote backups for off-site disaster recovery.")
        if config.get("include_uploads") != "true":
            plan["recommendations"].append("💡 Consider including uploads/ in backups for complete recovery.")

        return plan

    # ── Scheduled Check (called by scheduler) ─────────────────────────────

    def check_and_backup(self) -> None:
        """Check if a backup is due and run it. Called every minute by scheduler."""
        config = self.get_config()
        if config.get("enabled") != "true":
            return

        # Update heartbeat
        self.update_heartbeat()

        now = datetime.now()
        frequency = config.get("frequency", "daily")
        backup_time = config.get("time", "03:00")

        should_backup = False

        if frequency == "hourly" and now.minute == 0:
            should_backup = True
        elif frequency == "daily" and now.strftime("%H:%M") == backup_time:
            should_backup = True
        elif frequency == "weekly" and now.weekday() == 0 and now.strftime("%H:%M") == backup_time:
            should_backup = True

        if not should_backup:
            return

        # Check if already backed up this period
        history = self.get_history(limit=1)
        if history:
            last_time = datetime.strptime(history[0]["created_at"], "%Y-%m-%d %H:%M:%S")
            if frequency == "hourly" and (now - last_time).total_seconds() < 3500:
                return
            elif frequency == "daily" and (now - last_time).total_seconds() < 82800:
                return
            elif frequency == "weekly" and (now - last_time).total_seconds() < 600000:
                return

        # Run backup
        logger.info("Running scheduled %s backup", frequency)
        result = self.run_backup("full")
        if result.get("success"):
            logger.info("Backup completed: %s (%.2f MB)", result["archive"], result["size_mb"])
        else:
            logger.error("Backup failed: %s", result.get("error"))
            # Alert
            if self.scheduler:
                self.scheduler.notifications.append({
                    "message": f"⚠️ Scheduled backup failed: {result.get('error', 'Unknown error')}",
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "note_id": 0,
                    "type": "alert",
                })

    # ── Utilities ─────────────────────────────────────────────────────────

    def _record_backup(self, backup_type, destination, file_path, size, duration, checksum, status, error=None):
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO backup_history (backup_type, destination, file_path, size_bytes,
                                            duration_seconds, checksum, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (backup_type, destination, file_path, size, duration, checksum, status, error))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _cleanup_old_backups(self, retention_days: int):
        """Remove backup files older than retention period."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        try:
            for f in os.listdir(self.backup_dir):
                fpath = os.path.join(self.backup_dir, f)
                if os.path.isfile(fpath):
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                    if mtime < cutoff:
                        os.remove(fpath)
                        logger.info("Cleaned up old backup: %s", f)
        except Exception as e:
            logger.warning("Backup cleanup error: %s", e)

    @staticmethod
    def _sha256(file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
