"""Explicit synthetic sources for UI development without Raspberry Pi hardware."""
import math
import time

import cv2
import numpy as np

from pose_estimator import PoseEstimate
from wifi_scanner import WifiScanner


class DemoScanner(WifiScanner):
    def _run_scan_command(self):
        t = time.monotonic() / 8
        rows = []
        for i, (name, base, freq) in enumerate((("Demo · Engineering", -48, 5180), ("Demo · Lab", -65, 2437), ("Demo · Guest", -78, 2462))):
            rows.append(f"BSS 02:00:00:00:00:{i:02x}\nSSID: {name}\nsignal: {base + 8 * math.sin(t + i):.2f} dBm\nfreq: {freq}")
        return True, "\n".join(rows), ""


class DemoCamera:
    def __init__(self, width, height, fps):
        self.width, self.height, self.fps = width, height, max(1, fps)

    def open(self):
        return True

    def backend_name(self):
        return "DEMO / synthetic"

    def release(self):
        pass

    def read(self):
        time.sleep(1 / self.fps)
        frame = np.full((self.height, self.width, 3), (29, 23, 16), np.uint8)
        for x in range(0, self.width, 60):
            cv2.line(frame, (x, 0), (x, self.height), (49, 41, 27), 1)
        for y in range(0, self.height, 60):
            cv2.line(frame, (0, y), (self.width, y), (49, 41, 27), 1)
        t = time.monotonic() / 8
        center = (int(self.width * (0.5 + .28 * math.cos(t))), int(self.height * (.5 + .25 * math.sin(t))))
        for radius in (18, 45, 80):
            cv2.circle(frame, center, radius, (178, 219, 54), 2, cv2.LINE_AA)
        cv2.putText(frame, "DEMO - SIMULATED SURVEY", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, .8, (190, 220, 225), 2)
        return frame


class DemoPoseEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = time.monotonic()
        self.frames = 0

    def update(self, frame):
        t = (time.monotonic() - self.started) / 8
        self.frames += 1
        return PoseEstimate(3 + 2.5 * math.cos(t), 2.5 + 2 * math.sin(t), 1.2, t, 1.0, self.frames, 0, "TRACKING")

    def get_tracked_keypoints(self):
        return None
