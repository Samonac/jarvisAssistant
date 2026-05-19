"""Network Scanner module for Jarvis Assistant.

Discovers devices on the local network using Linux networking tools.
Strategy: Try arp-scan first (requires sudo), fall back to 'ip neigh',
then parse /proc/net/arp as a last resort.
"""

import re
import socket
import subprocess


class NetworkScanner:
    """Scans the local network for connected devices using Linux tools.

    Uses a fallback strategy:
        1. sudo arp-scan --localnet (most comprehensive, requires sudo)
        2. ip neigh (shows known neighbors from kernel ARP cache)
        3. /proc/net/arp (reads the kernel ARP table directly)

    Args:
        timeout: Maximum time in seconds for the scan operation (default: 120).
    """

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    def scan(self) -> list[dict]:
        """Discover active devices on the local network.

        Returns:
            A list of dicts with keys:
                - ip (str): The device's IP address
                - mac (str): The device's MAC address
                - hostname (str): The resolved hostname or "unknown"

        Raises no exceptions; returns an empty list on failure with
        appropriate error information logged.
        """
        try:
            subnet = self._get_local_subnet()
        except (PermissionError, RuntimeError):
            subnet = None

        # Strategy 1: Try arp-scan (most comprehensive)
        try:
            devices = self._arp_scan(subnet)
            if devices:
                return self._resolve_hostnames(devices)
        except PermissionError:
            raise PermissionError(
                "Network scan requires elevated privileges. "
                "Please run with sudo or grant appropriate permissions."
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Network scan timed out after {self.timeout} seconds."
            )
        except (FileNotFoundError, RuntimeError):
            pass

        # Strategy 2: Fallback to ip neigh
        try:
            devices = self._ip_neigh_scan()
            if devices:
                return self._resolve_hostnames(devices)
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Network scan timed out after {self.timeout} seconds."
            )
        except (RuntimeError, OSError):
            pass

        # Strategy 3: Fallback to /proc/net/arp
        try:
            devices = self._proc_arp_scan()
            if devices:
                return self._resolve_hostnames(devices)
        except (RuntimeError, OSError):
            pass

        return []

    def _get_local_subnet(self) -> str:
        """Determine the local subnet from the WiFi interface (wlan0).

        Parses the output of 'ip addr show wlan0' to extract the
        IP address and subnet mask in CIDR notation.

        Returns:
            The subnet in CIDR notation (e.g., "192.168.1.0/24").

        Raises:
            RuntimeError: If the interface is not found or has no IP.
        """
        try:
            result = subprocess.run(
                ["ip", "addr", "show", "wlan0"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            raise RuntimeError("'ip' command not found")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timed out getting interface info")

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get wlan0 info: {result.stderr.strip()}"
            )

        # Parse inet line: e.g., "inet 192.168.1.100/24 brd 192.168.1.255 ..."
        match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/(\d+)", result.stdout)
        if not match:
            raise RuntimeError("No IPv4 address found on wlan0")

        ip_addr = match.group(1)
        prefix_len = int(match.group(2))

        # Calculate network address from IP and prefix length
        ip_parts = [int(p) for p in ip_addr.split(".")]
        ip_int = (ip_parts[0] << 24) | (ip_parts[1] << 16) | (ip_parts[2] << 8) | ip_parts[3]
        mask = (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF
        network_int = ip_int & mask

        network_addr = ".".join([
            str((network_int >> 24) & 0xFF),
            str((network_int >> 16) & 0xFF),
            str((network_int >> 8) & 0xFF),
            str(network_int & 0xFF),
        ])

        return f"{network_addr}/{prefix_len}"

    def _arp_scan(self, subnet: str | None) -> list[dict]:
        """Scan the network using sudo arp-scan --localnet.

        Args:
            subnet: The subnet to scan (unused; arp-scan uses --localnet).

        Returns:
            A list of dicts with 'ip' and 'mac' keys.

        Raises:
            PermissionError: If arp-scan fails due to insufficient permissions.
            FileNotFoundError: If arp-scan is not installed.
            subprocess.TimeoutExpired: If the scan exceeds the timeout.
            RuntimeError: If arp-scan fails for other reasons.
        """
        try:
            result = subprocess.run(
                ["sudo", "arp-scan", "--localnet"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise FileNotFoundError("arp-scan is not installed")

        # Check for permission errors
        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            if "permission" in stderr_lower or "operation not permitted" in stderr_lower:
                raise PermissionError(
                    "Network scan requires elevated privileges. "
                    "Please run with sudo or grant appropriate permissions."
                )
            raise RuntimeError(f"arp-scan failed: {result.stderr.strip()}")

        return self._parse_arp_scan_output(result.stdout)

    def _parse_arp_scan_output(self, output: str) -> list[dict]:
        """Parse arp-scan output into device list.

        arp-scan output format:
            192.168.1.1\t00:11:22:33:44:55\tVendor Name
            192.168.1.2\t66:77:88:99:aa:bb\tAnother Vendor

        Args:
            output: Raw stdout from arp-scan.

        Returns:
            A list of dicts with 'ip' and 'mac' keys.
        """
        devices = []
        # Match lines with IP and MAC address pattern
        pattern = re.compile(
            r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F:]{17})"
        )

        for line in output.splitlines():
            match = pattern.search(line)
            if match:
                devices.append({
                    "ip": match.group(1),
                    "mac": match.group(2).lower(),
                })

        return devices

    def _ip_neigh_scan(self) -> list[dict]:
        """Scan the network using 'ip neigh' (ARP neighbor table).

        Returns:
            A list of dicts with 'ip' and 'mac' keys.

        Raises:
            subprocess.TimeoutExpired: If the command exceeds the timeout.
            RuntimeError: If the command fails.
        """
        try:
            result = subprocess.run(
                ["ip", "neigh"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise RuntimeError("'ip' command not found")

        if result.returncode != 0:
            raise RuntimeError(f"ip neigh failed: {result.stderr.strip()}")

        return self._parse_ip_neigh_output(result.stdout)

    def _parse_ip_neigh_output(self, output: str) -> list[dict]:
        """Parse 'ip neigh' output into device list.

        ip neigh output format:
            192.168.1.1 dev wlan0 lladdr 00:11:22:33:44:55 REACHABLE
            192.168.1.2 dev wlan0 lladdr 66:77:88:99:aa:bb STALE

        Args:
            output: Raw stdout from 'ip neigh'.

        Returns:
            A list of dicts with 'ip' and 'mac' keys.
            Only includes entries with a valid MAC (lladdr) that are not FAILED.
        """
        devices = []
        pattern = re.compile(
            r"(\d+\.\d+\.\d+\.\d+)\s+.*?lladdr\s+([0-9a-fA-F:]{17})\s+(\w+)"
        )

        for line in output.splitlines():
            # Skip entries marked as FAILED (no valid ARP entry)
            if "FAILED" in line.upper():
                continue

            match = pattern.search(line)
            if match:
                devices.append({
                    "ip": match.group(1),
                    "mac": match.group(2).lower(),
                })

        return devices

    def _proc_arp_scan(self) -> list[dict]:
        """Read and parse /proc/net/arp for cached ARP entries.

        Returns:
            A list of dicts with 'ip' and 'mac' keys.

        Raises:
            OSError: If /proc/net/arp cannot be read.
        """
        try:
            with open("/proc/net/arp", "r") as f:
                content = f.read()
        except (FileNotFoundError, PermissionError) as e:
            raise OSError(f"Cannot read /proc/net/arp: {e}")

        return self._parse_proc_arp_output(content)

    def _parse_proc_arp_output(self, content: str) -> list[dict]:
        """Parse /proc/net/arp content into device list.

        /proc/net/arp format:
            IP address       HW type     Flags       HW address            Mask     Device
            192.168.1.1      0x1         0x2         00:11:22:33:44:55     *        wlan0

        Args:
            content: Raw content of /proc/net/arp.

        Returns:
            A list of dicts with 'ip' and 'mac' keys.
            Excludes entries with MAC 00:00:00:00:00:00 (incomplete).
        """
        devices = []
        lines = content.strip().splitlines()

        # Skip the header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                ip_addr = parts[0]
                mac_addr = parts[3].lower()

                # Validate IP format
                if not re.match(r"\d+\.\d+\.\d+\.\d+", ip_addr):
                    continue

                # Skip incomplete entries (all zeros MAC)
                if mac_addr == "00:00:00:00:00:00":
                    continue

                # Validate MAC format
                if re.match(r"[0-9a-f]{2}(:[0-9a-f]{2}){5}", mac_addr):
                    devices.append({
                        "ip": ip_addr,
                        "mac": mac_addr,
                    })

        return devices

    def _resolve_hostnames(self, devices: list[dict]) -> list[dict]:
        """Resolve hostnames for discovered devices with graceful fallback.

        Attempts reverse DNS lookup for each device. If resolution fails,
        sets hostname to "unknown".

        Args:
            devices: List of dicts with 'ip' and 'mac' keys.

        Returns:
            The same list with 'hostname' key added to each dict.
        """
        for device in devices:
            device["hostname"] = self._resolve_hostname(device["ip"])
        return devices

    def _resolve_hostname(self, ip: str) -> str:
        """Resolve a single IP address to a hostname.

        Args:
            ip: The IP address to resolve.

        Returns:
            The resolved hostname, or "unknown" if resolution fails.
        """
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            return hostname
        except (socket.herror, socket.gaierror, OSError):
            return "unknown"
