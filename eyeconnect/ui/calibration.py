"""Калибровка 9 точек 3x3 (дока 4.4, Прил.В): 3с фиксация, 0.4с усреднение."""
import time
import cv2
import numpy as np
from .. import config as C


def grid_points(w, h, n: int = C.CALIB_GRID, margin: float = 0.12):
    xs = np.linspace(w * margin, w * (1 - margin), n)
    ys = np.linspace(h * margin, h * (1 - margin), n)
    return [(float(x), float(y)) for y in ys for x in xs]


def run_calibration(cam, tracker, filt, screen_w=1280, screen_h=720):
    """Интерактив: смотри на красную точку 3с. Возвращает (feats, screens)."""
    pts = grid_points(screen_w, screen_h)
    feats, screens = [], []
    win = "EyeConnect калибровка (Q - отмена)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, screen_w // 2, screen_h // 2)
    try:
        for i, (sx, sy) in enumerate(pts):
            t_end = time.time() + C.CALIBDWELL_S
            buf = []
            t_avg_start = t_end - C.CALIB_AVG_S
            while time.time() < t_end:
                f = cam.read()
                if f is None:
                    continue
                feat, conf, _ = tracker.process(f)
                if feat is not None and conf >= C.MP_MIN_CONF:
                    sm = filt.update(feat)
                    if time.time() >= t_avg_start:
                        buf.append(np.asarray(feat))
                canvas = np.zeros((screen_h // 2, screen_w // 2, 3), np.uint8)
                cx, cy = int(sx / 2), int(sy / 2)
                cv2.circle(canvas, (cx, cy), 18, (0, 0, 255), -1)
                cv2.putText(canvas, f"{i+1}/9 смотри на точку", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    raise KeyboardInterrupt
            if buf:
                feats.append(np.mean(buf, axis=0))
                screens.append((sx, sy))
    except KeyboardInterrupt:
        print("Калибровка прервана")
    finally:
        cv2.destroyWindow(win)
    return np.array(feats), np.array(screens)
