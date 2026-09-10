"""Полноэкранный курсор взгляда: калибровка на весь экран + крестик за взглядом.

Плюс опционально двигает настоящий системный курсор Windows (ctypes),
чтобы было видно точку взгляда поверх любых окон после выхода (Esc сворачивает).
"""
import ctypes
import json
import time
import cv2
import numpy as np

from .. import config as C
from ..gaze.regressor import GazeRegressor
from .calibration import grid_points


def screen_size():
    try:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        pass
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        w, h = int(r.winfo_screenwidth()), int(r.winfo_screenheight())
        r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def meta_path():
    return C.PROFILE_DIR / "gaze_meta.json"


def fullscreen_calibrate(cam, tracker, filt):
    """Калибровка 9 точек на весь экран. Возвращает (reg, W, H)."""
    W, H = screen_size()
    pts = grid_points(W, H, margin=0.10)
    win = "EyeConnect: smotri na krasnuyu tochku"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    feats, screens = [], []
    try:
        for i, (sx, sy) in enumerate(pts):
            t_end = time.time() + C.CALIBDWELL_S
            t_avg = t_end - C.CALIB_AVG_S
            buf = []
            while time.time() < t_end:
                f = cam.read()
                if f is None:
                    continue
                feat, conf, _ = tracker.process(f)
                if feat is not None and conf >= C.MP_MIN_CONF:
                    filt.update(feat)
                    if time.time() >= t_avg:
                        buf.append(np.asarray(feat, dtype=float))
                canvas = np.zeros((H, W, 3), np.uint8)
                cv2.circle(canvas, (int(sx), int(sy)), 22, (0, 0, 255), -1)
                cv2.circle(canvas, (int(sx), int(sy)), 30, (255, 255, 255), 3)
                cv2.putText(canvas, f"{i+1}/9  smotri na tochku, ne dvigaysya",
                            (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt
            if buf:
                feats.append(np.mean(buf, axis=0))
                screens.append((sx, sy))
    except KeyboardInterrupt:
        print("Kalibrovka prervana (Esc)")
    finally:
        cv2.destroyWindow(win)
    if len(feats) < 4:
        raise RuntimeError("Malo tochek: lico teryalos. Syad blizhe (60sm) i povtori.")
    reg = GazeRegressor().fit(np.array(feats), np.array(screens))
    reg.save(C.PROFILE_DIR / "gaze.npz")
    meta_path().write_text(json.dumps({"screen_w": W, "screen_h": H}), encoding="utf-8")
    print(f"Kalibrovka OK: {len(feats)}/9 tochek, ekran {W}x{H}")
    return reg, W, H


def load_or_calibrate(cam, tracker, filt, force=False):
    prof = C.PROFILE_DIR / "gaze.npz"
    W, H = screen_size()
    if not force and prof.exists() and meta_path().exists():
        try:
            meta = json.loads(meta_path().read_text(encoding="utf-8"))
            if meta.get("screen_w") == W and meta.get("screen_h") == H:
                return GazeRegressor().load(prof), W, H
            print(f"Razreshenie smenilos ({meta} -> {W}x{H}), perekabilrovka")
        except Exception as e:
            print(f"Profil bityy ({e}), perekalibrivka")
    return fullscreen_calibrate(cam, tracker, filt)


def set_sys_cursor(x, y):
    try:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        return True
    except Exception:
        return False


def run_cursor(cam, tracker, filt, reg, W, H, syscursor=True):
    win = "EyeConnect vzglyad (Q/Esc - vyhod)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Kursor sledit za vzglyadom. Q/Esc — vyhod.")
    try:
        while True:
            f = cam.read()
            if f is None:
                continue
            feat, conf, _ = tracker.process(f)
            img = np.zeros((H, W, 3), np.uint8)
            if feat is None or conf < C.MP_LOST_CONF:
                cv2.putText(img, "Lico ne naydeno — syad pered kameroy",
                            (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 165, 255), 3)
            else:
                sm = filt.update(feat)
                gx, gy = reg.predict(sm)
                gx = float(np.clip(gx, 0, W - 1))
                gy = float(np.clip(gy, 0, H - 1))
                if syscursor:
                    set_sys_cursor(gx, gy)
                # крестик + кольца
                x, y = int(gx), int(gy)
                cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 48, 3)
                cv2.circle(img, (x, y), 26, (0, 255, 0), 3)
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(img, f"x={x} y={y} fps={cam.fps:.0f}",
                            (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), ord("Q"), 27):
                break
    finally:
        cv2.destroyWindow(win)
