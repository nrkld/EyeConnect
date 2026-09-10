"""CSV-лог сессии: gaze 30Гц + dwell-события (дока 8.4)."""
import csv
import time
from pathlib import Path
from .. import config as C


class SessionLog:
    def __init__(self, name="session"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = C.PROFILE_DIR / f"{name}_{ts}.csv"
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "kind", "x", "y", "key", "text"])

    def gaze(self, x, y):
        self.w.writerow([time.time(), "gaze", f"{x:.1f}", f"{y:.1f}", "", ""])

    def event(self, key, text=""):
        self.w.writerow([time.time(), "dwell", "", "", key, text])

    def close(self):
        self.f.close()
        return self.path
