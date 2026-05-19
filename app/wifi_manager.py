"""WiFi Network Manager for Jarvis Assistant.

Monitors devices connected to the local WiFi network, tracks connection history,
and provides SSH connectivity to discovered devices.

Features:
- Scan network for connected devices (IP, MAC, hostname)
- Track connection history (first seen, last seen)
- Attempt SSH connections with default or custom credentials
- Maintain active SSH sessions for remote command execution
- Auto-try common credential combinations
"""

import logging
import platform
import socket
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Default credential combinations to try
DEFAULT_CREDENTIALS = [
    ("admin", "admin"),
    ("admin", "password"),
    ("admin", ""),
    ("root", "root"),
    ("root", "toor"),
    ("root", ""),
    ("pi", "raspberry"),
    ("pi", "pi"),
    ("ubuntu", "ubuntu"),
    ("user", "user"),
    ("user", "password"),
]


class WiFiManager:
    """Manages WiFi network device discovery and SSH connections.

    Attributes:
        db_manager: Database manager for persistent storage.
        network_scanner: Network scanner for device discovery.
        active_sessions: Dict of active SSH sessions keyed by IP.
    """

    def __init__(self, db_manager, network_scanner=None):
        self.db_manager = db_manager
        self.network_scanner = network_scanner
        self.active_sessions: dict = {}  # ip -> {connection, username, connected_at}
        self._lock = threading.Lock()
        self._init_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        try:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS wifi_devices (
                    mac TEXT PRIMARY KEY,
                    ip TEXT,
                    hostname TEXT,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ssh_username TEXT,
                    ssh_password TEXT,
                    ssh_port INTEGER DEFAULT 22,
                    ssh_verified INTEGER DEFAULT 0,
                    custom_name TEXT,
                    notes TEXT
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to init wifi tables: %s", e)

    def scan_network(self) -> list[dict]:
        """Scan the network and update device database.

        Returns:
            List of discovered devices with connection history.
        """
        devices = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use network scanner if available
        if self.network_scanner:
            try:
                raw_devices = self.network_scanner.scan()
                if raw_devices:
                    devices = raw_devices
            except Exception as e:
                logger.warning("Network scan failed: %s", e)

        # If network scanner returned nothing, use our own fallback (works on Windows)
        if not devices:
            devices = self._fallback_scan()

        # Update database with discovered devices
        conn = self._get_conn()
        for d in devices:
            mac = d.get("mac", "").lower()
            ip = d.get("ip", "")
            hostname = d.get("hostname", "unknown")
            if not mac or mac == "00:00:00:00:00:00" or mac == "00-00-00-00-00-01":
                continue

            # Normalize MAC to colon format
            mac = mac.replace("-", ":")

            existing = conn.execute("SELECT * FROM wifi_devices WHERE mac = ?", (mac,)).fetchone()
            if existing:
                conn.execute("""
                    UPDATE wifi_devices SET ip = ?, hostname = ?, last_seen = ? WHERE mac = ?
                """, (ip, hostname, now_str, mac))
            else:
                conn.execute("""
                    INSERT INTO wifi_devices (mac, ip, hostname, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?)
                """, (mac, ip, hostname, now_str, now_str))

        conn.commit()
        conn.close()

        return self.get_devices()

    def _fallback_scan(self) -> list[dict]:
        """Fallback network scan using ARP table. Works on Windows and Linux."""
        devices = []
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import re
                    for line in result.stdout.split("\n"):
                        # Match IP + MAC (handles both dash and colon separators)
                        # Windows format: "  10.70.14.8            4c-b0-4a-60-13-05     dynamique"
                        match = re.search(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2}[-:][0-9a-fA-F]{2})", line)
                        if match:
                            ip = match.group(1)
                            mac = match.group(2).replace("-", ":").lower()
                            # Skip multicast (224.x.x.x, 239.x.x.x), broadcast, and gateway placeholders
                            if ip.startswith("224.") or ip.startswith("239.") or ip.startswith("255."):
                                continue
                            if mac == "ff:ff:ff:ff:ff:ff" or mac.startswith("01:00:5e"):
                                continue
                            if mac == "00:00:00:00:00:01":  # Gateway placeholder
                                continue
                            # Skip hostname resolution during scan (too slow on large networks)
                            devices.append({"ip": ip, "mac": mac, "hostname": "unknown"})
            else:
                # Linux: try ip neigh first, then /proc/net/arp
                result = subprocess.run(["ip", "neigh"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import re
                    for line in result.stdout.split("\n"):
                        if "FAILED" in line:
                            continue
                        match = re.search(r"(\d+\.\d+\.\d+\.\d+).*lladdr\s+([0-9a-fA-F:]{17})", line)
                        if match:
                            ip = match.group(1)
                            mac = match.group(2).lower()
                            if mac == "00:00:00:00:00:00":
                                continue
                            devices.append({"ip": ip, "mac": mac, "hostname": "unknown"})
                elif os.path.exists("/proc/net/arp"):
                    with open("/proc/net/arp", "r") as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.split()
                            if len(parts) >= 4:
                                ip = parts[0]
                                mac = parts[3].lower()
                                if mac != "00:00:00:00:00:00":
                                    devices.append({"ip": ip, "mac": mac, "hostname": "unknown"})
        except Exception as e:
            logger.warning("Fallback scan failed: %s", e)

        # Resolve hostnames in background (non-blocking, best-effort)
        # Only resolve a few to keep it fast
        for d in devices[:10]:
            d["hostname"] = self._resolve_hostname(d["ip"])

        logger.info("Fallback scan found %d device(s)", len(devices))
        return devices

    def _resolve_hostname(self, ip: str) -> str:
        try:
            socket.setdefaulttimeout(1)  # 1 second timeout for DNS
            hostname, _, _ = socket.gethostbyaddr(ip)
            return hostname
        except (socket.herror, socket.gaierror, OSError, socket.timeout):
            return "unknown"
        finally:
            socket.setdefaulttimeout(None)

    def get_devices(self) -> list[dict]:
        """Get all known WiFi devices from the database."""
        try:
            conn = self._get_conn()
            cursor = conn.execute("SELECT * FROM wifi_devices ORDER BY last_seen DESC")
            devices = []
            for r in cursor.fetchall():
                ip = r["ip"]
                devices.append({
                    "mac": r["mac"],
                    "ip": ip,
                    "hostname": r["hostname"],
                    "custom_name": r["custom_name"],
                    "first_seen": r["first_seen"],
                    "last_seen": r["last_seen"],
                    "ssh_username": r["ssh_username"],
                    "ssh_port": r["ssh_port"] or 22,
                    "ssh_verified": bool(r["ssh_verified"]),
                    "has_active_session": ip in self.active_sessions,
                    "notes": r["notes"],
                })
            conn.close()
            return devices
        except Exception:
            return []

    def update_device(self, mac: str, updates: dict) -> dict:
        """Update device metadata (custom name, SSH credentials, notes)."""
        allowed = {"custom_name", "ssh_username", "ssh_password", "ssh_port", "notes"}
        try:
            conn = self._get_conn()
            clauses, values = [], []
            for k, v in updates.items():
                if k in allowed:
                    clauses.append(f"{k} = ?")
                    values.append(v)
            if not clauses:
                conn.close()
                return {"error": "No valid fields"}
            values.append(mac)
            conn.execute(f"UPDATE wifi_devices SET {', '.join(clauses)} WHERE mac = ?", values)
            conn.commit()
            conn.close()
            return {"message": "Device updated."}
        except Exception as e:
            return {"error": str(e)}

    # ── SSH Connection ────────────────────────────────────────────────────

    def ssh_connect(self, ip: str, username: str = None, password: str = None,
                    port: int = 22, auto_try: bool = True) -> dict:
        """Attempt SSH connection to a device.

        Args:
            ip: Target IP address.
            username: SSH username (if None, tries stored or defaults).
            password: SSH password (if None, tries stored or defaults).
            port: SSH port.
            auto_try: If True and no credentials work, try DEFAULT_CREDENTIALS.

        Returns:
            Dict with success status and session info.
        """
        try:
            import paramiko
        except ImportError:
            return {"error": "paramiko not installed. Run: pip install paramiko"}

        # Get stored credentials if not provided
        if not username or not password:
            stored = self._get_stored_credentials(ip)
            if stored:
                username = username or stored.get("username")
                password = password or stored.get("password")
                port = stored.get("port", port)

        # Build credential list to try
        creds_to_try = []
        if username and password:
            creds_to_try.append((username, password))
        if auto_try:
            creds_to_try.extend(DEFAULT_CREDENTIALS)

        # Try each credential pair
        for user, passwd in creds_to_try:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(ip, port=port, username=user, password=passwd, timeout=5)

                # Success! Store the working credentials
                self._save_credentials(ip, user, passwd, port)

                with self._lock:
                    self.active_sessions[ip] = {
                        "client": client,
                        "username": user,
                        "connected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "port": port,
                    }

                logger.info("SSH connected to %s@%s:%d", user, ip, port)
                return {
                    "success": True,
                    "ip": ip,
                    "username": user,
                    "port": port,
                    "message": f"Connected to {user}@{ip}:{port}",
                }
            except paramiko.AuthenticationException:
                continue
            except Exception as e:
                return {"error": f"Connection failed: {e}"}

        return {
            "error": "Authentication failed. All credential combinations exhausted.",
            "tried": len(creds_to_try),
            "needs_credentials": True,
        }

    def ssh_disconnect(self, ip: str) -> dict:
        """Disconnect an active SSH session."""
        with self._lock:
            session = self.active_sessions.pop(ip, None)
        if session:
            try:
                session["client"].close()
            except Exception:
                pass
            return {"message": f"Disconnected from {ip}"}
        return {"error": "No active session for this IP"}

    def ssh_execute(self, ip: str, command: str, timeout: int = 30) -> dict:
        """Execute a command on a connected device.

        Args:
            ip: Target device IP.
            command: Shell command to run.
            timeout: Command timeout in seconds.

        Returns:
            Dict with stdout, stderr, exit_code.
        """
        session = self.active_sessions.get(ip)
        if not session:
            return {"error": f"No active SSH session for {ip}. Connect first."}

        client = session["client"]
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            return {
                "stdout": stdout.read().decode("utf-8", errors="replace"),
                "stderr": stderr.read().decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "ip": ip,
            }
        except Exception as e:
            # Connection might have dropped
            with self._lock:
                self.active_sessions.pop(ip, None)
            return {"error": f"Command execution failed: {e}. Session closed."}

    def get_active_sessions(self) -> list[dict]:
        """Get all active SSH sessions."""
        sessions = []
        for ip, info in self.active_sessions.items():
            sessions.append({
                "ip": ip,
                "username": info["username"],
                "port": info["port"],
                "connected_at": info["connected_at"],
            })
        return sessions

    def _get_stored_credentials(self, ip: str) -> Optional[dict]:
        """Get stored SSH credentials for a device by IP."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT ssh_username, ssh_password, ssh_port FROM wifi_devices WHERE ip = ? AND ssh_verified = 1",
                (ip,)
            ).fetchone()
            conn.close()
            if row and row["ssh_username"]:
                return {"username": row["ssh_username"], "password": row["ssh_password"], "port": row["ssh_port"] or 22}
        except Exception:
            pass
        return None

    def _save_credentials(self, ip: str, username: str, password: str, port: int):
        """Save working SSH credentials for a device."""
        try:
            conn = self._get_conn()
            conn.execute("""
                UPDATE wifi_devices SET ssh_username = ?, ssh_password = ?, ssh_port = ?, ssh_verified = 1
                WHERE ip = ?
            """, (username, password, port, ip))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to save SSH credentials: %s", e)
