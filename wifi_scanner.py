"""Background WiFi scanner for Raspberry Pi managed-mode interfaces.

This module continuously executes `iw dev <iface> scan`, parses the results,
and stores the latest network list in a thread-safe state object.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import config

LOGGER = logging.getLogger("wifi_scanner")

BSS_PATTERN = re.compile(r"^BSS\s+([0-9a-fA-F:]{17})")
SSID_PATTERN = re.compile(r"^SSID:\s*(.*)$")
SIGNAL_PATTERN = re.compile(r"^signal:\s*(-?\d+(?:\.\d+)?)\s*dBm$")
FREQ_PATTERN = re.compile(r"^freq:\s*(\d+(?:\.\d+)?)$")

IWLIST_CELL_PATTERN = re.compile(r"^Cell\s+\d+\s+-\s+Address:\s*([0-9a-fA-F:]{17})$")
IWLIST_ESSID_PATTERN = re.compile(r'^ESSID:"(.*)"$')
IWLIST_SIGNAL_PATTERN = re.compile(r"Signal\s+level\s*[=:-]\s*(-?\d+(?:\.\d+)?)\s*dBm", re.IGNORECASE)
IWLIST_FREQ_PATTERN = re.compile(r"Frequency\s*[:=]\s*(\d+(?:\.\d+)?)\s*GHz", re.IGNORECASE)


class WifiScanner:
    """Threaded WiFi scanner with locked state and safe read access."""

    def __init__(
        self,
        interface: str = config.WIFI_INTERFACE,
        scan_interval: float = config.SCAN_INTERVAL,
        command_timeout: int = 20,
    ) -> None:
        """Initialize scanner state.

        Args:
            interface: WiFi network interface, usually wlan0.
            scan_interval: Seconds between scans in background mode.
            command_timeout: Subprocess timeout for each scan command.
        """
        self.interface = interface
        self.scan_interval = scan_interval
        self.command_timeout = command_timeout

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: Dict[str, Any] = {
            "networks": [],
            "last_scan_epoch": 0.0,
            "last_scan_duration_ms": 0.0,
            "last_error": None,
        }

    def start_scanner(self) -> None:
        """Start the background scanner thread if it is not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._scan_loop, name="wifi-scan-thread", daemon=True)
            self._thread.start()

    def stop_scanner(self, join_timeout: float = 2.0) -> None:
        """Stop the background scanner thread.

        Args:
            join_timeout: Max seconds to wait for scanner thread exit.
        """
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout)

    def get_networks(self) -> List[Dict[str, Any]]:
        """Return a copy of the latest discovered networks list."""
        with self._lock:
            return [dict(item) for item in self._state["networks"]]

    def get_last_error(self) -> Optional[str]:
        """Return the latest scan error string if one exists."""
        with self._lock:
            error = self._state["last_error"]
        return str(error) if error else None

    def get_last_scan_epoch(self) -> float:
        """Return POSIX timestamp of the most recent scan completion."""
        with self._lock:
            return float(self._state["last_scan_epoch"])

    def get_last_scan_duration_ms(self) -> float:
        """Return duration of the most recent scan command in milliseconds."""
        with self._lock:
            return float(self._state["last_scan_duration_ms"])

    def scan_once(self) -> List[Dict[str, Any]]:
        """Perform one WiFi scan and update internal state.

        Returns:
            Parsed network list in the format:
            [{"ssid": str, "bssid": str, "dbm": float, "freq": int}, ...]
        """
        started = time.perf_counter()
        success, stdout, error_message = self._run_scan_command()
        duration_ms = (time.perf_counter() - started) * 1000.0
        now_epoch = time.time()

        if not success:
            with self._lock:
                # Keep last successful networks visible on transient scan failures.
                self._state["last_error"] = error_message
                self._state["last_scan_epoch"] = now_epoch
                self._state["last_scan_duration_ms"] = duration_ms
            LOGGER.warning("WiFi scan failed: %s", error_message)
            with self._lock:
                return [dict(item) for item in self._state["networks"]]

        networks = parse_scan_output(stdout)
        if not networks and "Cell" in stdout and "Address" in stdout:
            networks = parse_iwlist_output(stdout)

        with self._lock:
            self._state["networks"] = networks
            self._state["last_error"] = None
            self._state["last_scan_epoch"] = now_epoch
            self._state["last_scan_duration_ms"] = duration_ms

        if not networks:
            LOGGER.info("WiFi scan complete: no networks detected.")
        else:
            LOGGER.info("WiFi scan complete: %d network(s) detected.", len(networks))

        return [dict(item) for item in networks]

    def _scan_loop(self) -> None:
        """Background loop that repeatedly scans until stop is requested."""
        while not self._stop_event.is_set():
            self.scan_once()
            self._stop_event.wait(self.scan_interval)

    def _run_scan_command(self) -> tuple[bool, str, str]:
        """Execute iw scan command with non-blocking permission fallback.

        Returns:
            A tuple of:
            - success flag
            - stdout content (scan output if success)
            - error string if failed
        """
        commands: Sequence[Sequence[str]] = (
            ("iw", "dev", self.interface, "scan"),
            ("iw", "dev", self.interface, "scan", "ap-force"),
            ("sudo", "-n", "iw", "dev", self.interface, "scan"),
            ("sudo", "-n", "iw", "dev", self.interface, "scan", "ap-force"),
            ("iwlist", self.interface, "scanning"),
            ("sudo", "-n", "iwlist", self.interface, "scanning"),
        )

        last_error = "Unknown scan failure"

        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                last_error = f"Command not found: {exc}"
                continue
            except subprocess.TimeoutExpired:
                last_error = (
                    f"WiFi scan timeout after {self.command_timeout}s using: {' '.join(command)}"
                )
                continue

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            if result.returncode == 0 and stdout:
                return True, stdout, ""

            if result.returncode == 0 and not stdout:
                last_error = f"Scan command returned no output ({' '.join(command)})."
                continue

            lower_stderr = stderr.lower()
            if "operation not permitted" in lower_stderr or "permission denied" in lower_stderr:
                last_error = (
                    "Permission denied for WiFi scan. Run scanner with sudo privileges "
                    "or grant CAP_NET_ADMIN to iw."
                )
                continue
            if "sudo: a password is required" in lower_stderr:
                last_error = (
                    "sudo password is required for WiFi scan. Run the process with sudo, "
                    "or configure passwordless sudo for iw."
                )
                continue
            if "no such device" in lower_stderr:
                last_error = f"WiFi interface '{self.interface}' not found."
                continue
            if "network is down" in lower_stderr:
                last_error = (
                    f"WiFi interface '{self.interface}' appears down. Bring it up with "
                    f"'sudo ip link set {self.interface} up'."
                )
                continue
            if "device or resource busy" in lower_stderr:
                last_error = "WiFi interface busy during scan. Retrying with fallback command."
                continue

            if stderr:
                last_error = f"Scan command failed ({' '.join(command)}): {stderr}"
            else:
                last_error = (
                    f"Scan command failed ({' '.join(command)}) with return code {result.returncode}."
                )

        return False, "", last_error


def parse_scan_output(raw_output: str) -> List[Dict[str, Any]]:
    """Parse iw scan output into normalized network dictionaries.

    Expected block format:
        BSS aa:bb:cc:dd:ee:ff
        SSID: MyNetwork
        signal: -65.00 dBm
        freq: 2437

    Args:
        raw_output: Raw stdout text from `iw dev <iface> scan`.

    Returns:
        A list of dictionaries with keys ssid, bssid, dbm, and freq.
    """
    networks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        bss_match = BSS_PATTERN.match(line)
        if bss_match:
            if current is not None:
                networks.append(_finalize_network(current))
            current = {
                "ssid": "<hidden>",
                "bssid": bss_match.group(1).lower(),
                "dbm": -100.0,
                "freq": 0,
            }
            continue

        if current is None:
            continue

        ssid_match = SSID_PATTERN.match(line)
        if ssid_match:
            ssid_value = ssid_match.group(1)
            current["ssid"] = ssid_value if ssid_value else "<hidden>"
            continue

        signal_match = SIGNAL_PATTERN.match(line)
        if signal_match:
            try:
                current["dbm"] = float(signal_match.group(1))
            except ValueError:
                current["dbm"] = -100.0
            continue

        freq_match = FREQ_PATTERN.match(line)
        if freq_match:
            try:
                current["freq"] = int(float(freq_match.group(1)))
            except ValueError:
                current["freq"] = 0
            continue

    if current is not None:
        networks.append(_finalize_network(current))

    # Keep only networks with a parsed BSSID and sort by strongest signal first.
    networks = [item for item in networks if item.get("bssid")]
    networks.sort(key=lambda item: float(item.get("dbm", -100.0)), reverse=True)
    return networks


def parse_iwlist_output(raw_output: str) -> List[Dict[str, Any]]:
    """Parse iwlist scanning output into normalized network dictionaries."""
    networks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        cell_match = IWLIST_CELL_PATTERN.match(line)
        if cell_match:
            if current is not None:
                networks.append(_finalize_network(current))
            current = {
                "ssid": "<hidden>",
                "bssid": cell_match.group(1).lower(),
                "dbm": -100.0,
                "freq": 0,
            }
            continue

        if current is None:
            continue

        essid_match = IWLIST_ESSID_PATTERN.search(line)
        if essid_match:
            ssid_value = str(essid_match.group(1) or "").strip()
            current["ssid"] = ssid_value if ssid_value else "<hidden>"

        signal_match = IWLIST_SIGNAL_PATTERN.search(line)
        if signal_match:
            try:
                current["dbm"] = float(signal_match.group(1))
            except ValueError:
                current["dbm"] = -100.0

        freq_match = IWLIST_FREQ_PATTERN.search(line)
        if freq_match:
            try:
                ghz = float(freq_match.group(1))
                current["freq"] = int(round(ghz * 1000.0))
            except ValueError:
                current["freq"] = 0

    if current is not None:
        networks.append(_finalize_network(current))

    networks = [item for item in networks if item.get("bssid")]
    networks.sort(key=lambda item: float(item.get("dbm", -100.0)), reverse=True)
    return networks


def _finalize_network(network: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one parsed network entry before storage/return."""
    ssid = str(network.get("ssid", "<hidden>"))
    bssid = str(network.get("bssid", "")).lower()

    try:
        dbm = float(network.get("dbm", -100.0))
    except (TypeError, ValueError):
        dbm = -100.0

    try:
        freq = int(network.get("freq", 0))
    except (TypeError, ValueError):
        freq = 0

    return {
        "ssid": ssid,
        "bssid": bssid,
        "dbm": dbm,
        "freq": freq,
    }


_CONTEXT_LOCK = threading.Lock()
_CONTEXT_SCANNER: Optional[WifiScanner] = None


def start_scanner(
    interface: str = config.WIFI_INTERFACE,
    scan_interval: float = config.SCAN_INTERVAL,
    command_timeout: int = 20,
) -> WifiScanner:
    """Start module-level scanner and return scanner instance.

    This function satisfies Stage 1 API requirements while keeping internal
    state protected by a lock.
    """
    global _CONTEXT_SCANNER

    with _CONTEXT_LOCK:
        if _CONTEXT_SCANNER is None:
            _CONTEXT_SCANNER = WifiScanner(
                interface=interface,
                scan_interval=scan_interval,
                command_timeout=command_timeout,
            )
        _CONTEXT_SCANNER.start_scanner()
        return _CONTEXT_SCANNER


def get_networks() -> List[Dict[str, Any]]:
    """Return latest module-level scanner results.

    Returns an empty list if scanner has not been started yet.
    """
    with _CONTEXT_LOCK:
        scanner = _CONTEXT_SCANNER

    if scanner is None:
        return []
    return scanner.get_networks()


def _configure_logging(verbose: bool) -> None:
    """Configure module logger for direct script execution."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _format_network_line(network: Dict[str, Any]) -> str:
    """Format one network dictionary for human-readable logs."""
    ssid = network.get("ssid", "<hidden>")
    bssid = network.get("bssid", "")
    dbm = float(network.get("dbm", -100.0))
    freq = int(network.get("freq", 0))
    return f"SSID={ssid!r} BSSID={bssid} SIGNAL={dbm:.2f} dBm FREQ={freq} MHz"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for scanner standalone mode."""
    parser = argparse.ArgumentParser(description="Background WiFi scanner for Raspberry Pi")
    parser.add_argument("--interface", default=config.WIFI_INTERFACE, help="WiFi interface (default: wlan0)")
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=config.SCAN_INTERVAL,
        help="Background scan interval in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Subprocess timeout seconds for each scan (default: 20)",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=5.0,
        help="Seconds between status logs in standalone mode (default: 5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """Run scanner standalone and log discovered networks every few seconds."""
    args = _parse_args()
    _configure_logging(args.verbose)

    scanner = start_scanner(
        interface=args.interface,
        scan_interval=args.scan_interval,
        command_timeout=args.timeout,
    )

    LOGGER.info("Scanner started on interface=%s interval=%.2fs", args.interface, args.scan_interval)

    try:
        while True:
            time.sleep(args.print_interval)
            networks = scanner.get_networks()
            error = scanner.get_last_error()

            if error:
                LOGGER.warning("Latest scan error: %s", error)
                continue

            if not networks:
                LOGGER.info("No WiFi networks found in latest scan.")
                continue

            LOGGER.info("Discovered %d network(s):", len(networks))
            for item in networks:
                LOGGER.info("  %s", _format_network_line(item))

    except KeyboardInterrupt:
        LOGGER.info("Stopping scanner...")
        scanner.stop_scanner()
        LOGGER.info("Scanner stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
