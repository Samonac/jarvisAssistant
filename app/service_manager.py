"""Service Manager for Jarvis Assistant.

Manages Git repositories, Docker containers, and system services
on the local machine or remote hosts via SSH.

Features:
- Git: status, pull, log, branch management
- Docker: list containers, start/stop/restart, logs, stats
- Systemd/Services: list, start/stop/restart, status, enable/disable
- Remote execution via existing SSH integration
"""

import json
import logging
import platform
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages system services, Docker containers, and Git repos.

    Attributes:
        ssh_client: Optional SSH client for remote operations.
    """

    def __init__(self, ssh_client=None):
        self.ssh_client = ssh_client

    def _run(self, command: str, host: str = None, timeout: int = 30) -> dict:
        """Execute a command locally or remotely.

        Args:
            command: Shell command to run.
            host: If provided, run on remote host via SSH.
            timeout: Max execution time.

        Returns:
            Dict with stdout, stderr, exit_code.
        """
        if host and self.ssh_client:
            result = self.ssh_client.execute(host, command)
            return {
                "stdout": result.get("output", ""),
                "stderr": result.get("error", ""),
                "exit_code": result.get("exit_code", -1),
            }

        try:
            if platform.system() == "Windows":
                proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
            else:
                proc = subprocess.run(command, shell=True, executable="/bin/bash",
                                      capture_output=True, text=True, timeout=timeout)
            return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Command timed out", "exit_code": -1}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "exit_code": -1}

    # ── Git Operations ────────────────────────────────────────────────────

    def git_status(self, repo_path: str = ".", host: str = None) -> dict:
        """Get git status for a repository."""
        result = self._run(f"cd {repo_path} && git status --porcelain", host)
        if result["exit_code"] != 0:
            return {"error": result["stderr"] or "Not a git repository"}

        lines = result["stdout"].strip().split("\n") if result["stdout"].strip() else []
        branch_result = self._run(f"cd {repo_path} && git branch --show-current", host)
        branch = branch_result["stdout"].strip() if branch_result["exit_code"] == 0 else "unknown"

        return {
            "branch": branch,
            "clean": len(lines) == 0,
            "changes": len(lines),
            "files": [{"status": l[:2].strip(), "file": l[3:]} for l in lines[:20]],
        }

    def git_pull(self, repo_path: str = ".", host: str = None) -> dict:
        """Pull latest changes."""
        result = self._run(f"cd {repo_path} && git pull", host)
        return {
            "success": result["exit_code"] == 0,
            "output": result["stdout"] or result["stderr"],
        }

    def git_log(self, repo_path: str = ".", count: int = 10, host: str = None) -> dict:
        """Get recent git log."""
        fmt = "--pretty=format:%H|%an|%ar|%s"
        result = self._run(f"cd {repo_path} && git log {fmt} -n {count}", host)
        if result["exit_code"] != 0:
            return {"error": result["stderr"]}

        commits = []
        for line in result["stdout"].strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0][:8], "author": parts[1],
                        "time_ago": parts[2], "message": parts[3],
                    })
        return {"commits": commits}

    # ── Docker Operations ─────────────────────────────────────────────────

    def docker_list(self, all_containers: bool = False, host: str = None) -> dict:
        """List Docker containers."""
        flag = "-a" if all_containers else ""
        fmt = '{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}'
        result = self._run(f"docker ps {flag} --format '{fmt}'", host)
        if result["exit_code"] != 0:
            return {"error": result["stderr"] or "Docker not available"}

        containers = []
        for line in result["stdout"].strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 4)
                if len(parts) >= 4:
                    containers.append({
                        "id": parts[0], "name": parts[1], "image": parts[2],
                        "status": parts[3], "ports": parts[4] if len(parts) > 4 else "",
                    })
        return {"containers": containers, "count": len(containers)}

    def docker_action(self, container: str, action: str, host: str = None) -> dict:
        """Start/stop/restart a Docker container."""
        if action not in ("start", "stop", "restart", "pause", "unpause"):
            return {"error": f"Invalid action: {action}"}
        result = self._run(f"docker {action} {container}", host)
        return {
            "success": result["exit_code"] == 0,
            "output": result["stdout"] or result["stderr"],
        }

    def docker_logs(self, container: str, lines: int = 50, host: str = None) -> dict:
        """Get Docker container logs."""
        result = self._run(f"docker logs --tail {lines} {container}", host)
        return {
            "container": container,
            "logs": (result["stdout"] or result["stderr"])[:5000],
        }

    def docker_stats(self, host: str = None) -> dict:
        """Get Docker resource usage stats."""
        fmt = '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}'
        result = self._run(f"docker stats --no-stream --format '{fmt}'", host)
        if result["exit_code"] != 0:
            return {"error": result["stderr"]}

        stats = []
        for line in result["stdout"].strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 3)
                if len(parts) >= 3:
                    stats.append({
                        "name": parts[0], "cpu": parts[1],
                        "memory": parts[2], "network": parts[3] if len(parts) > 3 else "",
                    })
        return {"stats": stats}

    # ── System Services (systemd) ─────────────────────────────────────────

    def service_list(self, host: str = None) -> dict:
        """List system services (systemd on Linux, services on Windows)."""
        if platform.system() == "Windows" and not host:
            result = self._run("sc query type= service state= all", timeout=15)
            # Parse Windows service output (simplified)
            services = []
            current = {}
            for line in result["stdout"].split("\n"):
                if "SERVICE_NAME:" in line:
                    if current:
                        services.append(current)
                    current = {"name": line.split(":", 1)[1].strip(), "status": "unknown"}
                elif "STATE" in line and ":" in line:
                    state_part = line.split(":", 1)[1].strip()
                    current["status"] = "running" if "RUNNING" in state_part else "stopped"
            if current:
                services.append(current)
            return {"services": services[:50], "platform": "windows"}
        else:
            result = self._run(
                "systemctl list-units --type=service --no-pager --plain --no-legend | head -50",
                host
            )
            if result["exit_code"] != 0:
                return {"error": result["stderr"] or "systemctl not available"}

            services = []
            for line in result["stdout"].strip().split("\n"):
                parts = line.split(None, 4)
                if len(parts) >= 4:
                    services.append({
                        "name": parts[0].replace(".service", ""),
                        "load": parts[1], "active": parts[2],
                        "sub": parts[3], "description": parts[4] if len(parts) > 4 else "",
                    })
            return {"services": services, "platform": "linux"}

    def service_action(self, service_name: str, action: str, host: str = None) -> dict:
        """Start/stop/restart/enable/disable a system service."""
        if action not in ("start", "stop", "restart", "enable", "disable", "status"):
            return {"error": f"Invalid action: {action}"}

        if platform.system() == "Windows" and not host:
            win_action = {"start": "start", "stop": "stop", "restart": "stop && sc start"}.get(action, action)
            result = self._run(f"sc {win_action} {service_name}")
        else:
            result = self._run(f"sudo systemctl {action} {service_name}", host)

        return {
            "success": result["exit_code"] == 0,
            "service": service_name,
            "action": action,
            "output": result["stdout"] or result["stderr"],
        }

    def service_status(self, service_name: str, host: str = None) -> dict:
        """Get detailed status of a service."""
        if platform.system() == "Windows" and not host:
            result = self._run(f"sc query {service_name}")
        else:
            result = self._run(f"systemctl status {service_name} --no-pager", host)

        return {
            "service": service_name,
            "output": result["stdout"] or result["stderr"],
            "running": "running" in (result["stdout"] or "").lower() or "RUNNING" in (result["stdout"] or ""),
        }
