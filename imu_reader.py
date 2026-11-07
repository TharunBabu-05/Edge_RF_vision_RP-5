"""MPU-9250 / MPU-6500 IMU Sensor Reader & Complementary Orientation Filter.

Reads accelerometer and gyroscope motion data (6-DOF)
over I2C on Raspberry Pi 5 (default bus 1, address 0x68 or 0x69).
Computes stable Euler orientation angles (Yaw, Pitch, Roll) using a fast
complementary filter. Heading uses gyro integration and will drift.

Features:
  - Auto hardware detection: checks if smbus2 and I2C address 0x68/0x69 are available.
  - Missing hardware is unavailable; simulation requires an explicit flag.
  - Threaded background reading loop (~60 Hz).
  - Clean Thread-safe API for fetching orientation snapshot: {yaw, pitch, roll, hardware_detected}.

Hardware Wiring (Raspberry Pi 5):
  VCC -> Pin 1 (3.3V)
  GND -> Pin 6 (Ground)
  SDA -> Pin 3 (GPIO 2 / I2C1_SDA)
  SCL -> Pin 5 (GPIO 3 / I2C1_SCL)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

LOGGER = logging.getLogger("imu_reader")

# MPU-9250 / MPU-6500 I2C Registers & Constants
_MPU6500_DEFAULT_ADDR = 0x68
_MPU6500_ALT_ADDR     = 0x69

_REG_PWR_MGMT_1   = 0x6B
_REG_SMPLRT_DIV   = 0x19
_REG_CONFIG       = 0x1A
_REG_GYRO_CONFIG  = 0x1B
_REG_ACCEL_CONFIG = 0x1C
_REG_ACCEL_XOUT_H = 0x3B
_REG_GYRO_XOUT_H  = 0x43
_REG_WHO_AM_I     = 0x75


@dataclass
class IMUOrientation:
    """Immutable snapshot of IMU orientation angles."""
    yaw: float          # Heading angle in degrees (-180 to 180)
    pitch: float        # Tilt up/down in degrees (-90 to 90)
    roll: float         # Bank left/right in degrees (-180 to 180)
    yaw_rad: float      # Yaw in radians
    hardware_detected: bool
    timestamp: float


class IMUReader:
    """Thread-safe MPU-9250 / MPU-6500 IMU Sensor Reader."""

    def __init__(self, bus_num: int = 1, sample_rate_hz: float = 60.0, alpha: float = 0.96, simulate: bool = False) -> None:
        self.simulate = simulate
        self._last_measurement = 0.0
        self._io_lock = threading.Lock()
        self.bus_num = bus_num
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.alpha = alpha  # Complementary filter weight (gyro weight)

        self._lock = threading.Lock()
        self._bus = None
        self._addr = _MPU6500_DEFAULT_ADDR
        self.hardware_detected = False

        # Calculated angles (degrees)
        self._yaw = 0.0
        self._pitch = 0.0
        self._roll = 0.0

        # Gyro biases (degrees/sec)
        self._gyro_bias_x = 0.0
        self._gyro_bias_y = 0.0
        self._gyro_bias_z = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Attempt to initialize hardware
        if not simulate:
            self._init_hardware()

    def _init_hardware(self) -> bool:
        """Attempt to open SMBus and initialize MPU-9250/6500 hardware."""
        try:
            import smbus2
            self._bus = smbus2.SMBus(self.bus_num)

            # Only initialize supported MPU-family chip IDs.
            for addr in (_MPU6500_DEFAULT_ADDR, _MPU6500_ALT_ADDR):
                try:
                    who = self._bus.read_byte_data(addr, _REG_WHO_AM_I)
                    if who not in (0x68, 0x70, 0x71, 0x73):
                        continue
                    # Device is present if it responded — log chip ID for info
                    LOGGER.info("IMU device found at I2C addr 0x%02X, WHO_AM_I=0x%02X", addr, who)
                    self._addr = addr
                    self.hardware_detected = True
                    break
                except Exception:
                    continue

            if not self.hardware_detected:
                LOGGER.warning("MPU sensor not detected on I2C bus %d; orientation unavailable.", self.bus_num)
                self._bus.close()
                self._bus = None
                return False

            # Reset & Wake Up MPU
            self._bus.write_byte_data(self._addr, _REG_PWR_MGMT_1, 0x00)
            time.sleep(0.05)
            # Sample rate divider = 0 (1kHz)
            self._bus.write_byte_data(self._addr, _REG_SMPLRT_DIV, 0x00)
            # DLPF Config = 3 (44Hz filtering)
            self._bus.write_byte_data(self._addr, _REG_CONFIG, 0x03)
            # Gyro scale: +- 500 deg/s (FS_SEL=1 => 65.5 LSB/deg/s)
            self._bus.write_byte_data(self._addr, _REG_GYRO_CONFIG, 0x08)
            # Accel scale: +- 4g (AFS_SEL=1 => 8192 LSB/g)
            self._bus.write_byte_data(self._addr, _REG_ACCEL_CONFIG, 0x08)

            LOGGER.info("Successfully connected to MPU-9250/6500 IMU at I2C address 0x%02X on bus %d", self._addr, self.bus_num)
            self._calibrate_gyro()
            return True

        except Exception as exc:
            LOGGER.info("IMU unavailable: %s", exc)
            self.hardware_detected = False
            return False

    def _calibrate_gyro(self, samples: int = 100) -> None:
        """Measure gyro offsets while stationary."""
        LOGGER.info("Calibrating IMU gyroscope biases (%d samples)...", samples)
        bx, by, bz = 0.0, 0.0, 0.0
        count = 0
        for _ in range(samples):
            raw = self._read_raw_data()
            if raw is not None:
                gx, gy, gz = raw[3], raw[4], raw[5]
                bx += gx
                by += gy
                bz += gz
                count += 1
            time.sleep(0.005)
        if count > 0:
            self._gyro_bias_x = bx / count
            self._gyro_bias_y = by / count
            self._gyro_bias_z = bz / count
            LOGGER.info("Gyro calibration complete: bias=(%.2f, %.2f, %.2f) deg/s",
                        self._gyro_bias_x, self._gyro_bias_y, self._gyro_bias_z)

    def _read_raw_data(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Read Accel (g) and Gyro (deg/s) from hardware."""
        if not self.hardware_detected or self._bus is None:
            return None
        try:
            # Read 14 bytes: Accel (6), Temp (2), Gyro (6)
            with self._io_lock:
                data = self._bus.read_i2c_block_data(self._addr, _REG_ACCEL_XOUT_H, 14)

            def _to_int16(h, l):
                val = (h << 8) | l
                return val - 65536 if val >= 32768 else val

            ax_raw = _to_int16(data[0], data[1])
            ay_raw = _to_int16(data[2], data[3])
            az_raw = _to_int16(data[4], data[5])

            gx_raw = _to_int16(data[8], data[9])
            gy_raw = _to_int16(data[10], data[11])
            gz_raw = _to_int16(data[12], data[13])

            # Convert: Accel (+-4g -> 8192 LSB/g), Gyro (+-500 dps -> 65.5 LSB/dps)
            ax = ax_raw / 8192.0
            ay = ay_raw / 8192.0
            az = az_raw / 8192.0

            gx = (gx_raw / 65.5) - self._gyro_bias_x
            gy = (gy_raw / 65.5) - self._gyro_bias_y
            gz = (gz_raw / 65.5) - self._gyro_bias_z

            return (ax, ay, az, gx, gy, gz)
        except Exception as exc:
            LOGGER.debug("IMU read error: %s", exc)
            return None

    def start(self) -> None:
        """Start background IMU updating thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=_loop_worker, args=(self,), daemon=True, name="IMUWorker")
            self._thread.start()
            LOGGER.info("IMU background reader started.")

    def stop(self) -> None:
        """Stop background thread."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._bus:
            try:
                self._bus.close()
            except Exception:
                pass

    def get_orientation(self) -> IMUOrientation:
        """Thread-safe fetch of current orientation."""
        with self._lock:
            y = float(self._yaw)
            p = float(self._pitch)
            r = float(self._roll)
            hw = self.hardware_detected
            timestamp = self._last_measurement

        # Wrap yaw to [-180, 180]
        y_wrapped = (y + 180.0) % 360.0 - 180.0
        y_rad = math.radians(y_wrapped)

        return IMUOrientation(
            yaw=round(y_wrapped, 2),
            pitch=round(p, 2),
            roll=round(r, 2),
            yaw_rad=round(y_rad, 4),
            hardware_detected=hw,
            timestamp=timestamp,
        )


def _loop_worker(imu: IMUReader) -> None:
    """Worker loop for complementary filtering."""
    sim_t = 0.0
    previous = time.perf_counter() - imu.dt
    while imu._running:
        t0 = time.perf_counter()
        dt = min(0.1, max(0.0001, t0 - previous))
        previous = t0

        raw = imu._read_raw_data()
        with imu._lock:
            if raw is not None:
                ax, ay, az, gx, gy, gz = raw

                # Calculate pitch & roll from accelerometer
                pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
                roll_acc  = math.degrees(math.atan2(ay, az))

                # Integrate gyroscope rates
                alpha = imu.alpha ** (dt / imu.dt)
                imu._pitch = alpha * (imu._pitch + gy * dt) + (1.0 - alpha) * pitch_acc
                imu._roll  = alpha * (imu._roll + gx * dt) + (1.0 - alpha) * roll_acc
                imu._yaw  += gz * dt
                imu._last_measurement = time.time()
            elif imu.simulate:
                # Simulation mode if hardware isn't attached
                sim_t += dt
                # Subtle simulated slow rotation for demonstration when offline
                imu._yaw = (sim_t * 8.0) % 360.0
                imu._pitch = math.sin(sim_t * 0.5) * 2.0
                imu._roll  = math.cos(sim_t * 0.5) * 1.5
                imu._last_measurement = time.time()

        # Sleep to maintain sampling frequency (~60 Hz)
        elapsed = time.perf_counter() - t0
        sleep_dur = max(0.001, imu.dt - elapsed)
        time.sleep(sleep_dur)


# Standalone self-test
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    reader = IMUReader()
    reader.start()
    LOGGER.info("Running IMU reader test for 5 seconds...")
    for _ in range(10):
        ori = reader.get_orientation()
        LOGGER.info("HW: %s | Yaw: %6.1f° | Pitch: %5.1f° | Roll: %5.1f°",
                    ori.hardware_detected, ori.yaw, ori.pitch, ori.roll)
        time.sleep(0.5)
    reader.stop()


if __name__ == "__main__":
    main()
