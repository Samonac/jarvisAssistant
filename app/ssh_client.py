"""SSH Client for Jarvis Assistant.

Executes commands on remote machines via SSH using paramiko.
Supports multiple named hosts configured via environment variables.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SSH_TIMEOUT = 30


class SSHClient:
    """Manages SSH connections to remote hosts.

    Hosts are configured as a list of dicts with:
    - name: friendly name (e.g., "server", "nas")
    - host: hostname or IP
    - port: SSH port (default 22)
    - username: SSH username
    - password: SSH password (optional if key_path is set)
    - key_path: path to private key file (optional)

    Attributes:
        hosts: Dict of configured hosts keyed by name.
    """

    def __init__(self, hosts: Optional[list[dict]] = None):
        self.hosts: dict[str, dict] = {}
        if hosts:
            for h in hosts:
                name = h.get("name", h.get("host", "unknown"))
                self.hosts[name.lower()] = h

    def is_configured(self) -> bool:
        return len(self.hosts) > 0

    def list_hosts(self) -> list[dict]:
        """List all configured SSH hosts."""
        return [
            {"name": name, "host": h["host"], "port": h.get("port", 22), "username": h.get("username", "")}
            for name, h in self.hosts.items()
        ]

    def execute(self, host_name: str, command: str) -> dict:
        """Execute a command on a remote host via SSH.

        Args:
            host_name: The friendly name of the host.
            command: The command to execute.

        Returns:
            Dict with 'stdout', 'stderr', 'return_code', or 'error'.
        """
        try:
            import paramiko
        except ImportError:
            return {"error": "SSH support requires paramiko: pip install paramiko"}

        host_name_lower = host_name.lower()
        if host_name_lower not in self.hosts:
            available = ", ".join(self.hosts.keys()) if self.hosts else "none configured"
            return {"error": f"Unknown host '{host_name}'. Available: {available}"}

        host_config = self.hosts[host_name_lower]
        hostname = host_config["host"]
        port = host_config.get("port", 22)
        username = host_config.get("username", "pi")
        password = host_config.get("password")
        key_path = host_config.get("key_path")

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": hostname,
                "port": port,
                "username": username,
                "timeout": SSH_TIMEOUT,
            }

            if key_path:
                connect_kwargs["key_filename"] = key_path
            elif password:
                connect_kwargs["password"] = password
            else:
                return {"error": f"No authentication method for host '{host_name}'. Set password or key_path."}

            client.connect(**connect_kwargs)

            stdin, stdout, stderr = client.exec_command(command, timeout=SSH_TIMEOUT)
            exit_code = stdout.channel.recv_exit_status()

            result = {
                "stdout": stdout.read().decode("utf-8", errors="replace"),
                "stderr": stderr.read().decode("utf-8", errors="replace"),
                "return_code": exit_code,
                "host": host_name,
            }

            client.close()
            return result

        except paramiko.AuthenticationException:
            return {"error": f"SSH authentication failed for '{host_name}'. Check credentials."}
        except paramiko.SSHException as e:
            return {"error": f"SSH error connecting to '{host_name}': {e}"}
        except TimeoutError:
            return {"error": f"SSH connection to '{host_name}' timed out."}
        except Exception as e:
            logger.error("SSH error for '%s': %s", host_name, e)
            return {"error": f"SSH error: {e}"}
