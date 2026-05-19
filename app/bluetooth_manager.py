"""Bluetooth Manager for Jarvis Assistant.

Manages Bluetooth device discovery, connections, and data streaming.
Uses bleak (cross-platform BLE library) for Bluetooth Low Energy devices.
Falls back to system commands (bluetoothctl on Linux, PowerShell on Windows) for classic Bluetooth.

Primary use case: IMU sensors sending gyroscope/accelerometer data via BLE.
"""

import asyncio
import logging
import platform
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BluetoothDevice:
    """Represents a discovered Bluetooth device."""
    address: str
    name: str = "Unknown"
    rssi: int = 0
    connected: bool = False
    device_type: str = "unknown"  # "ble", "classic", "imu", "phone"
    last_seen: str = ""
    services: list = field(default_factory=list)


@dataclass
class SensorData:
    """Real-time sensor data from a connected device."""
    device_address: str
    timestamp: str
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0


class BluetoothManager:
    """Manages Bluetooth device discovery and connections.

    Attributes:
        devices: Dict of discovered devices keyed by address.
        connected_devices: Set of currently connected device addresses.
        sensor_data: Rolling buffer of recent sensor readings per device.
        is_scanning: Whether a scan is currently in progress.
    """

    def __init__(self):
        self.devices: dict[str, BluetoothDevice] = {}
        self.connected_devices: set[str] = set()
        self.sensor_data: dict[str, deque] = {}  # address -> deque of SensorData
        self.is_scanning = False
        self._bleak_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if bleak is available for BLE operations."""
        try:
            import bleak
            self._bleak_available = True
        except ImportError:
            logger.warning("bleak not installed. BLE features limited. Install: pip install bleak")

    def scan(self, duration: int = 5) -> list[dict]:
        """Scan for nearby Bluetooth devices.

        Args:
            duration: Scan duration in seconds.

        Returns:
            List of discovered device dicts.
        """
        self.is_scanning = True
        try:
            if self._bleak_available:
                return self._scan_ble(duration)
            else:
                return self._scan_system(duration)
        finally:
            self.is_scanning = False

    def _scan_ble(self, duration: int) -> list[dict]:
        """Scan using bleak (BLE)."""
        try:
            import asyncio
            from bleak import BleakScanner

            async def do_scan():
                devices = await BleakScanner.discover(timeout=duration)
                return devices

            # Run in a new event loop (since we might be in a sync context)
            loop = asyncio.new_event_loop()
            try:
                discovered = loop.run_until_complete(do_scan())
            finally:
                loop.close()

            for d in discovered:
                addr = d.address
                self.devices[addr] = BluetoothDevice(
                    address=addr,
                    name=d.name or "Unknown",
                    rssi=d.rssi if hasattr(d, 'rssi') else 0,
                    connected=addr in self.connected_devices,
                    device_type="ble",
                    last_seen=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )

            return [self._device_to_dict(d) for d in self.devices.values()]

        except Exception as e:
            logger.error("BLE scan error: %s", e)
            return self._scan_system(duration)

    def _scan_system(self, duration: int) -> list[dict]:
        """Scan using system commands (fallback)."""
        try:
            if platform.system() == "Windows":
                # Use PowerShell to list paired devices
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-PnpDevice -Class Bluetooth | Select-Object Name, Status, InstanceId | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=duration + 5
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json
                    devices = json.loads(result.stdout)
                    if isinstance(devices, dict):
                        devices = [devices]
                    for d in devices:
                        name = d.get("Name", "Unknown")
                        instance_id = d.get("InstanceId", "")
                        addr = instance_id[-17:].replace("_", ":") if len(instance_id) > 17 else instance_id
                        self.devices[addr] = BluetoothDevice(
                            address=addr,
                            name=name,
                            connected=d.get("Status") == "OK",
                            device_type="classic",
                            last_seen=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        )
            else:
                # Linux: use bluetoothctl
                result = subprocess.run(
                    ["bluetoothctl", "devices"],
                    capture_output=True, text=True, timeout=duration + 5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line.startswith("Device "):
                            parts = line.split(" ", 2)
                            if len(parts) >= 3:
                                addr = parts[1]
                                name = parts[2]
                                self.devices[addr] = BluetoothDevice(
                                    address=addr,
                                    name=name,
                                    device_type="classic",
                                    last_seen=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                )

        except Exception as e:
            logger.error("System BT scan error: %s", e)

        return [self._device_to_dict(d) for d in self.devices.values()]

    def get_devices(self) -> list[dict]:
        """Get all known devices."""
        return [self._device_to_dict(d) for d in self.devices.values()]

    def get_device(self, address: str) -> Optional[dict]:
        """Get a specific device by address."""
        d = self.devices.get(address)
        return self._device_to_dict(d) if d else None

    def report_sensor_data(self, address: str, data: dict) -> bool:
        """Store incoming sensor data from a connected device.

        Args:
            address: Device address.
            data: Dict with sensor readings (accel_x/y/z, gyro_x/y/z, etc.)

        Returns:
            True if stored successfully.
        """
        if address not in self.sensor_data:
            self.sensor_data[address] = deque(maxlen=100)  # Keep last 100 readings

        reading = SensorData(
            device_address=address,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            accel_x=data.get("accel_x", 0.0),
            accel_y=data.get("accel_y", 0.0),
            accel_z=data.get("accel_z", 0.0),
            gyro_x=data.get("gyro_x", 0.0),
            gyro_y=data.get("gyro_y", 0.0),
            gyro_z=data.get("gyro_z", 0.0),
            mag_x=data.get("mag_x", 0.0),
            mag_y=data.get("mag_y", 0.0),
            mag_z=data.get("mag_z", 0.0),
        )
        self.sensor_data[address].append(reading)

        # Mark device as connected
        self.connected_devices.add(address)
        if address in self.devices:
            self.devices[address].connected = True
            self.devices[address].device_type = "imu"

        return True

    def get_sensor_data(self, address: str, last_n: int = 10) -> list[dict]:
        """Get recent sensor readings for a device.

        Args:
            address: Device address.
            last_n: Number of recent readings to return.

        Returns:
            List of sensor data dicts.
        """
        if address not in self.sensor_data:
            return []
        readings = list(self.sensor_data[address])[-last_n:]
        return [
            {
                "timestamp": r.timestamp,
                "accel": {"x": r.accel_x, "y": r.accel_y, "z": r.accel_z},
                "gyro": {"x": r.gyro_x, "y": r.gyro_y, "z": r.gyro_z},
                "mag": {"x": r.mag_x, "y": r.mag_y, "z": r.mag_z},
            }
            for r in readings
        ]

    def get_fused_context(self) -> dict:
        """Get fused sensor + location context for scripts.

        Combines IMU data with phone GPS to provide room-level positioning.
        Scripts can call this to know both motion state and location.
        """
        # Get latest IMU data from all connected sensors
        imu_data = {}
        for addr, readings in self.sensor_data.items():
            if readings:
                latest = readings[-1]
                imu_data[addr] = {
                    "accel": {"x": latest.accel_x, "y": latest.accel_y, "z": latest.accel_z},
                    "gyro": {"x": latest.gyro_x, "y": latest.gyro_y, "z": latest.gyro_z},
                    "timestamp": latest.timestamp,
                }

        return {
            "imu_devices": imu_data,
            "connected_count": len(self.connected_devices),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _device_to_dict(self, d: BluetoothDevice) -> dict:
        return {
            "address": d.address,
            "name": d.name,
            "rssi": d.rssi,
            "connected": d.connected,
            "device_type": d.device_type,
            "last_seen": d.last_seen,
            "has_sensor_data": d.address in self.sensor_data and len(self.sensor_data[d.address]) > 0,
        }
