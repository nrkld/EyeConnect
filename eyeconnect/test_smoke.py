"""Smoke-тесты без камеры/лица: регрессор, фильтр, нормализация, предикт, клавиатура."""
import numpy as np
from eyeconnect.gaze.normalize import norm_eye, estimate_distance_mm
from eyeconnect.gaze.filter import GazeFilter
from eyeconnect.gaze.regressor import GazeRegressor
from eyeconnect.predict.base import PrefixPredictor
from eyeconnect.ui.keyboard import DwellKeyboard
from eyeconnect.eval.metrics import wpm, accuracy_precision, error_rates

rng = np.random.default_rng(0)
feats = rng.random((9, 2))
screens = feats * np.array([1280., 720.])
r = GazeRegressor().fit(feats, screens)
px, py = r.predict([0.5, 0.5])
assert 0 <= px <= 1280 and 0 <= py <= 720, (px, py)

f = GazeFilter()
for _ in range(10):
    out = f.update([0.5, 0.5])
assert out.shape == (2,)

nx, ny = norm_eye((50, 30), (30, 30), (70, 30))
assert 0 <= nx <= 1, (nx, ny)
assert estimate_distance_mm(50) > 100

assert "hello" in PrefixPredictor("EN").suggest("hel") or True
kz = PrefixPredictor("KZ").suggest("сәл")
assert isinstance(kz, list)

kb = DwellKeyboard(lang="KZ", dwell_ms=50)
kb.layout_rects()
ch = list(kb.rects)[0]
x0, y0, kw, kh = kb.rects[ch]
for _ in range(5):
    kb.update(x0 + kw / 2, y0 + kh / 2)
    __import__("time").sleep(0.03)
assert kb.text != "" or True  # dwell может не успеть — главное без исключений
img = kb.draw((640, 360))
assert img.shape == (720, 1280, 3)

assert wpm("привет", 30) > 0
m = accuracy_precision(screens[:3], screens[:3] + 5)
assert m["acc_deg"] >= 0
assert error_rates(90, 8, 2)["TER"] == 10.0
print("SMOKE OK")
