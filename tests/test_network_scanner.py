"""Tests for Network Scanner.

Property-based tests for result structure (Property 7) and
unit tests for error handling (permissions, no devices, timeout, missing arp-scan fallback).
"""

import socket
import subprocess
from unittest.mock import patch, MagicMock, mock_open

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.network_scanner import NetworkScanner


# --- Strategies ---

# Strategy for generating valid IPv4 addresses
ipv4_strategy = st.tuples(
    st.integers(min_value=1, max_value=254),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=1, max_value=254),
).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

# Strategy for generating valid MAC addresses (not all zeros)
mac_strategy = st.lists(
    st.integers(min_value=0, max_value=255),
    min_size=6,
    max_size=6,
).filter(
    lambda octets: any(o != 0 for o in octets)
).map(
    lambda octets: ":".join(f"{o:02x}" for o in octets)
)

# Strategy for generating a valid arp-scan output line
arp_scan_line_strategy = st.tuples(ipv4_strategy, mac_strategy).map(
    lambda t: f"{t[0]}\t{t[1]}\tSome Vendor"
)

# Strategy for generating valid arp-scan output with multiple devices
arp_scan_output_strategy = st.lists(
    arp_scan_line_strategy, min_size=1, max_size=10
).map(lambda lines: "\n".join(lines) + "\n")

# Strategy for generating a valid ip neigh output line
ip_neigh_line_strategy = st.tuples(ipv4_strategy, mac_strategy).map(
    lambda t: f"{t[0]} dev wlan0 lladdr {t[1]} REACHABLE"
)

# Strategy for generating valid ip neigh output
ip_neigh_output_strategy = st.lists(
    ip_neigh_line_strategy, min_size=1, max_size=10
).map(lambda lines: "\n".join(lines) + "\n")

# Strategy for generating a valid /proc/net/arp line
proc_arp_line_strategy = st.tuples(ipv4_strategy, mac_strategy).map(
    lambda t: f"{t[0]}      0x1         0x2         {t[1]}     *        wlan0"
)

# Strategy for generating valid /proc/net/arp content
proc_arp_output_strategy = st.lists(
    proc_arp_line_strategy, min_size=1, max_size=10
).map(
    lambda lines: "IP address       HW type     Flags       HW address            Mask     Device\n"
    + "\n".join(lines) + "\n"
)


class TestProperty7NetworkScanResultStructure:
    """Property 7: IP and MAC present in scan results.

    For any device discovered during a network scan, the result SHALL contain
    a valid `ip` address string and a `mac` address string.

    **Validates: Requirements 6.3**
    """

    @given(output=arp_scan_output_strategy)
    @settings(max_examples=200)
    def test_arp_scan_results_have_ip_and_mac(self, output):
        """Every device from arp-scan has valid ip and mac fields."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = output
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                devices = scanner._arp_scan(None)
                devices = scanner._resolve_hostnames(devices)

        for device in devices:
            assert "ip" in device, "Device missing 'ip' field"
            assert "mac" in device, "Device missing 'mac' field"
            assert isinstance(device["ip"], str) and device["ip"].strip() != ""
            assert isinstance(device["mac"], str) and device["mac"].strip() != ""
            # Validate IP format (four octets separated by dots)
            ip_parts = device["ip"].split(".")
            assert len(ip_parts) == 4
            for part in ip_parts:
                assert part.isdigit()
                assert 0 <= int(part) <= 255
            # Validate MAC format (six hex pairs separated by colons)
            mac_parts = device["mac"].split(":")
            assert len(mac_parts) == 6
            for part in mac_parts:
                assert len(part) == 2
                int(part, 16)  # Raises ValueError if not valid hex

    @given(output=ip_neigh_output_strategy)
    @settings(max_examples=200)
    def test_ip_neigh_results_have_ip_and_mac(self, output):
        """Every device from ip neigh has valid ip and mac fields."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = output
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                devices = scanner._ip_neigh_scan()
                devices = scanner._resolve_hostnames(devices)

        for device in devices:
            assert "ip" in device, "Device missing 'ip' field"
            assert "mac" in device, "Device missing 'mac' field"
            assert isinstance(device["ip"], str) and device["ip"].strip() != ""
            assert isinstance(device["mac"], str) and device["mac"].strip() != ""
            # Validate IP format
            ip_parts = device["ip"].split(".")
            assert len(ip_parts) == 4
            for part in ip_parts:
                assert part.isdigit()
                assert 0 <= int(part) <= 255
            # Validate MAC format
            mac_parts = device["mac"].split(":")
            assert len(mac_parts) == 6
            for part in mac_parts:
                assert len(part) == 2
                int(part, 16)

    @given(content=proc_arp_output_strategy)
    @settings(max_examples=200)
    def test_proc_arp_results_have_ip_and_mac(self, content):
        """Every device from /proc/net/arp has valid ip and mac fields."""
        scanner = NetworkScanner(timeout=120)

        with patch("builtins.open", mock_open(read_data=content)):
            with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                devices = scanner._proc_arp_scan()
                devices = scanner._resolve_hostnames(devices)

        for device in devices:
            assert "ip" in device, "Device missing 'ip' field"
            assert "mac" in device, "Device missing 'mac' field"
            assert isinstance(device["ip"], str) and device["ip"].strip() != ""
            assert isinstance(device["mac"], str) and device["mac"].strip() != ""
            # Validate IP format
            ip_parts = device["ip"].split(".")
            assert len(ip_parts) == 4
            for part in ip_parts:
                assert part.isdigit()
                assert 0 <= int(part) <= 255
            # Validate MAC format
            mac_parts = device["mac"].split(":")
            assert len(mac_parts) == 6
            for part in mac_parts:
                assert len(part) == 2
                int(part, 16)

    @given(output=arp_scan_output_strategy)
    @settings(max_examples=100)
    def test_scan_results_include_hostname_field(self, output):
        """Every device in scan results has a hostname field (may be 'unknown')."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = output
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                devices = scanner._arp_scan(None)
                devices = scanner._resolve_hostnames(devices)

        for device in devices:
            assert "hostname" in device, "Device missing 'hostname' field"
            assert isinstance(device["hostname"], str)


class TestErrorHandlingPermissions:
    """Unit tests for permission error handling."""

    def test_arp_scan_permission_error_raises(self):
        """When arp-scan fails due to permissions, PermissionError is raised."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ERROR: Operation not permitted"

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            with pytest.raises(PermissionError) as exc_info:
                scanner._arp_scan(None)

        assert "sudo" in str(exc_info.value).lower() or "privileges" in str(exc_info.value).lower()

    def test_scan_permission_error_suggests_sudo(self):
        """When scan fails due to permissions, the error message suggests sudo."""
        scanner = NetworkScanner(timeout=120)

        mock_arp_result = MagicMock()
        mock_arp_result.returncode = 1
        mock_arp_result.stdout = ""
        mock_arp_result.stderr = "ERROR: Operation not permitted"

        # Mock _get_local_subnet to return a valid subnet
        with patch.object(scanner, "_get_local_subnet", return_value="192.168.1.0/24"):
            with patch("app.network_scanner.subprocess.run", return_value=mock_arp_result):
                with pytest.raises(PermissionError) as exc_info:
                    scanner.scan()

        assert "sudo" in str(exc_info.value).lower()


class TestErrorHandlingNoDevices:
    """Unit tests for no devices found scenario."""

    def test_empty_arp_scan_output_returns_empty_list(self):
        """When arp-scan returns no devices, an empty list is returned."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Interface: wlan0, type: EN10MB\nEnding arp-scan.\n"
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            devices = scanner._arp_scan(None)

        assert devices == []

    def test_empty_ip_neigh_output_returns_empty_list(self):
        """When ip neigh returns no entries, an empty list is returned."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            devices = scanner._ip_neigh_scan()

        assert devices == []

    def test_proc_arp_header_only_returns_empty_list(self):
        """When /proc/net/arp has only the header, an empty list is returned."""
        scanner = NetworkScanner(timeout=120)

        content = "IP address       HW type     Flags       HW address            Mask     Device\n"

        with patch("builtins.open", mock_open(read_data=content)):
            devices = scanner._proc_arp_scan()

        assert devices == []

    def test_scan_all_methods_empty_returns_empty_list(self):
        """When all scan methods find no devices, scan() returns empty list."""
        scanner = NetworkScanner(timeout=120)

        # arp-scan not found
        with patch.object(scanner, "_get_local_subnet", return_value="192.168.1.0/24"):
            with patch.object(scanner, "_arp_scan", side_effect=FileNotFoundError("not installed")):
                with patch.object(scanner, "_ip_neigh_scan", return_value=[]):
                    with patch.object(scanner, "_proc_arp_scan", return_value=[]):
                        devices = scanner.scan()

        assert devices == []


class TestErrorHandlingTimeout:
    """Unit tests for timeout handling."""

    def test_arp_scan_timeout_raises_timeout_error(self):
        """When arp-scan exceeds timeout, TimeoutError is raised from scan()."""
        scanner = NetworkScanner(timeout=120)

        with patch.object(scanner, "_get_local_subnet", return_value="192.168.1.0/24"):
            with patch(
                "app.network_scanner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="arp-scan", timeout=120),
            ):
                with pytest.raises(TimeoutError) as exc_info:
                    scanner.scan()

        assert "120" in str(exc_info.value) or "timed out" in str(exc_info.value).lower()

    def test_ip_neigh_timeout_raises_timeout_error(self):
        """When ip neigh exceeds timeout, TimeoutError is raised from scan()."""
        scanner = NetworkScanner(timeout=120)

        def side_effect_fn(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "arp-scan" in str(cmd):
                raise FileNotFoundError("not installed")
            raise subprocess.TimeoutExpired(cmd="ip neigh", timeout=120)

        with patch.object(scanner, "_get_local_subnet", return_value="192.168.1.0/24"):
            with patch("app.network_scanner.subprocess.run", side_effect=side_effect_fn):
                with pytest.raises(TimeoutError) as exc_info:
                    scanner.scan()

        assert "timed out" in str(exc_info.value).lower()

    def test_timeout_value_is_configurable(self):
        """The timeout value passed to the constructor is used."""
        scanner = NetworkScanner(timeout=60)
        assert scanner.timeout == 60

        scanner2 = NetworkScanner(timeout=300)
        assert scanner2.timeout == 300


class TestErrorHandlingMissingArpScanFallback:
    """Unit tests for missing arp-scan with fallback behavior."""

    def test_missing_arp_scan_falls_back_to_ip_neigh(self):
        """When arp-scan is not installed, falls back to ip neigh."""
        scanner = NetworkScanner(timeout=120)

        ip_neigh_output = "192.168.1.1 dev wlan0 lladdr aa:bb:cc:dd:ee:ff REACHABLE\n"

        def run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "arp-scan" in cmd_str:
                raise FileNotFoundError("arp-scan not found")
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ip_neigh_output
            mock_result.stderr = ""
            return mock_result

        with patch.object(scanner, "_get_local_subnet", return_value="192.168.1.0/24"):
            with patch("app.network_scanner.subprocess.run", side_effect=run_side_effect):
                with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                    devices = scanner.scan()

        assert len(devices) == 1
        assert devices[0]["ip"] == "192.168.1.1"
        assert devices[0]["mac"] == "aa:bb:cc:dd:ee:ff"

    def test_missing_arp_scan_and_ip_neigh_falls_back_to_proc_arp(self):
        """When arp-scan and ip neigh fail, falls back to /proc/net/arp."""
        scanner = NetworkScanner(timeout=120)

        proc_content = (
            "IP address       HW type     Flags       HW address            Mask     Device\n"
            "192.168.1.1      0x1         0x2         aa:bb:cc:dd:ee:ff     *        wlan0\n"
        )

        with patch.object(scanner, "_get_local_subnet", return_value="192.168.1.0/24"):
            with patch.object(scanner, "_arp_scan", side_effect=FileNotFoundError("not installed")):
                with patch.object(scanner, "_ip_neigh_scan", return_value=[]):
                    with patch("builtins.open", mock_open(read_data=proc_content)):
                        with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                            devices = scanner.scan()

        assert len(devices) == 1
        assert devices[0]["ip"] == "192.168.1.1"
        assert devices[0]["mac"] == "aa:bb:cc:dd:ee:ff"

    def test_arp_scan_runtime_error_falls_back(self):
        """When arp-scan fails with RuntimeError, falls back to ip neigh."""
        scanner = NetworkScanner(timeout=120)

        ip_neigh_output = "10.0.0.5 dev eth0 lladdr 11:22:33:44:55:66 STALE\n"

        def run_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "arp-scan" in cmd_str:
                mock_result = MagicMock()
                mock_result.returncode = 2
                mock_result.stdout = ""
                mock_result.stderr = "Some unknown error"
                return mock_result
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ip_neigh_output
            mock_result.stderr = ""
            return mock_result

        with patch.object(scanner, "_get_local_subnet", return_value="10.0.0.0/24"):
            with patch("app.network_scanner.subprocess.run", side_effect=run_side_effect):
                with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
                    devices = scanner.scan()

        assert len(devices) == 1
        assert devices[0]["ip"] == "10.0.0.5"
        assert devices[0]["mac"] == "11:22:33:44:55:66"


class TestHostnameResolution:
    """Unit tests for hostname resolution."""

    def test_successful_hostname_resolution(self):
        """When gethostbyaddr succeeds, the hostname is included."""
        scanner = NetworkScanner(timeout=120)

        with patch("app.network_scanner.socket.gethostbyaddr") as mock_resolve:
            mock_resolve.return_value = ("myrouter.local", [], ["192.168.1.1"])
            hostname = scanner._resolve_hostname("192.168.1.1")

        assert hostname == "myrouter.local"

    def test_failed_hostname_resolution_returns_unknown(self):
        """When gethostbyaddr fails, 'unknown' is returned."""
        scanner = NetworkScanner(timeout=120)

        with patch("app.network_scanner.socket.gethostbyaddr") as mock_resolve:
            mock_resolve.side_effect = socket.herror("Host not found")
            hostname = scanner._resolve_hostname("192.168.1.99")

        assert hostname == "unknown"

    def test_resolve_hostnames_adds_hostname_to_all_devices(self):
        """_resolve_hostnames adds hostname field to every device."""
        scanner = NetworkScanner(timeout=120)

        devices = [
            {"ip": "192.168.1.1", "mac": "aa:bb:cc:dd:ee:ff"},
            {"ip": "192.168.1.2", "mac": "11:22:33:44:55:66"},
        ]

        with patch.object(scanner, "_resolve_hostname", return_value="unknown"):
            result = scanner._resolve_hostnames(devices)

        assert len(result) == 2
        for device in result:
            assert "hostname" in device
            assert device["hostname"] == "unknown"


class TestSubnetDetection:
    """Unit tests for subnet detection from wlan0."""

    def test_valid_wlan0_output_returns_subnet(self):
        """Parses valid ip addr show wlan0 output correctly."""
        scanner = NetworkScanner(timeout=120)

        ip_output = """3: wlan0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b8:27:eb:12:34:56 brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.100/24 brd 192.168.1.255 scope global dynamic noprefixroute wlan0
       valid_lft 86399sec preferred_lft 75599sec
"""

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ip_output
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            subnet = scanner._get_local_subnet()

        assert subnet == "192.168.1.0/24"

    def test_wlan0_not_found_raises_runtime_error(self):
        """When wlan0 interface doesn't exist, RuntimeError is raised."""
        scanner = NetworkScanner(timeout=120)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Device \"wlan0\" does not exist."

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError):
                scanner._get_local_subnet()

    def test_wlan0_no_ip_raises_runtime_error(self):
        """When wlan0 has no IPv4 address, RuntimeError is raised."""
        scanner = NetworkScanner(timeout=120)

        ip_output = """3: wlan0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500
    link/ether b8:27:eb:12:34:56 brd ff:ff:ff:ff:ff:ff
"""

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ip_output
        mock_result.stderr = ""

        with patch("app.network_scanner.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="No IPv4 address"):
                scanner._get_local_subnet()


class TestParsing:
    """Unit tests for output parsing methods."""

    def test_parse_arp_scan_typical_output(self):
        """Parses typical arp-scan output correctly."""
        scanner = NetworkScanner(timeout=120)

        output = """Interface: wlan0, type: EN10MB, MAC: b8:27:eb:12:34:56
Starting arp-scan 1.9.7 with 256 hosts
192.168.1.1\t00:11:22:33:44:55\tNetgear
192.168.1.50\taa:bb:cc:dd:ee:ff\tApple, Inc.
192.168.1.100\t11:22:33:44:55:66\tRaspberry Pi Foundation

3 packets received by filter, 0 packets dropped by kernel
Ending arp-scan 1.9.7: 256 hosts scanned in 2.5 seconds.
"""
        devices = scanner._parse_arp_scan_output(output)

        assert len(devices) == 3
        assert devices[0] == {"ip": "192.168.1.1", "mac": "00:11:22:33:44:55"}
        assert devices[1] == {"ip": "192.168.1.50", "mac": "aa:bb:cc:dd:ee:ff"}
        assert devices[2] == {"ip": "192.168.1.100", "mac": "11:22:33:44:55:66"}

    def test_parse_ip_neigh_typical_output(self):
        """Parses typical ip neigh output correctly."""
        scanner = NetworkScanner(timeout=120)

        output = """192.168.1.1 dev wlan0 lladdr 00:11:22:33:44:55 REACHABLE
192.168.1.50 dev wlan0 lladdr aa:bb:cc:dd:ee:ff STALE
192.168.1.200 dev wlan0  FAILED
fe80::1 dev wlan0 lladdr 00:11:22:33:44:55 router STALE
"""
        devices = scanner._parse_ip_neigh_output(output)

        # Should include REACHABLE and STALE, skip FAILED and IPv6
        assert len(devices) == 2
        assert devices[0] == {"ip": "192.168.1.1", "mac": "00:11:22:33:44:55"}
        assert devices[1] == {"ip": "192.168.1.50", "mac": "aa:bb:cc:dd:ee:ff"}

    def test_parse_proc_arp_typical_output(self):
        """Parses typical /proc/net/arp content correctly."""
        scanner = NetworkScanner(timeout=120)

        content = """IP address       HW type     Flags       HW address            Mask     Device
192.168.1.1      0x1         0x2         00:11:22:33:44:55     *        wlan0
192.168.1.50     0x1         0x2         aa:bb:cc:dd:ee:ff     *        wlan0
192.168.1.99     0x1         0x0         00:00:00:00:00:00     *        wlan0
"""
        devices = scanner._parse_proc_arp_output(content)

        # Should skip the incomplete entry (all zeros MAC)
        assert len(devices) == 2
        assert devices[0] == {"ip": "192.168.1.1", "mac": "00:11:22:33:44:55"}
        assert devices[1] == {"ip": "192.168.1.50", "mac": "aa:bb:cc:dd:ee:ff"}
