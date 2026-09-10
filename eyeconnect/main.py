"""EyeConnect MVP: preview | calibrate | keyboard.

Примеры:
  python -m eyeconnect.main --mode preview
  python -m eyeconnect.main --mode calibrate
  python -m eyeconnect.main --mode keyboard --lang KZ
  python -m eyeconnect.main --mode keyboard --lang EN --mouse  (отладка UI без камеры)
"""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from . import config as C
from .gaze.capture import Camera
from .gaze.facemesh import FaceTracker
from .gaze.filter import GazeFilter
from .gaze.regressor import GazeRegressor
from .ui.calibration import run_calibration
from .ui.cursor import load_or_calibrate, run_cursor
from .ui.keyboard import DwellKeyboard
from .predict.base import PrefixPredictor
from .eval.logger import SessionLog
from .eval.metrics import accuracy_precision


def profile_path(name="gaze.npz"):
    return C.PROFILE_DIR / name


def cmd_preview(args):
    cam = Camera(args.camera)
    tr = FaceTracker()
    flt = GazeFilter()
    print("Q — выход")
    while True:
        f = cam.read()
        if f is None:
            continue
        feat, conf, dbg = tr.process(f)
        if feat is not None:
            sm = flt.update(feat)
            txt = f"feat {feat[0]:.2f},{feat[1]:.2f} conf {conf:.2f} fps {cam.fps:.1f}"
        else:
            txt = f"no face fps {cam.fps:.1f}"
        cv2.putText(f, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for k in ("l_iris", "r_iris"):
            if k in dbg:
                x, y = dbg[k]
                sx = int(x / C.FRAME_W * f.shape[1])
                sy = int(y / C.FRAME_H * f.shape[0])
                cv2.circle(f, (sx, sy), 5, (0, 0, 255), -1)
        cv2.imshow("preview", f)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    tr.close()
    cam.release()
    cv2.destroyAllWindows()


def cmd_calibrate(args):
    cam = Camera(args.camera)
    tr = FaceTracker()
    flt = GazeFilter()
    feats, screens = run_calibration(cam, tr, flt)
    tr.close()
    cam.release()
    if len(feats) < 4:
        print("Мало точек — калибровка не сохранена")
        return
    reg = GazeRegressor().fit(feats, screens)
    # самопроверка на тех же точках (валидация; честная — повторным прогоном)
    preds = [reg.predict(f) for f in feats]
    m = accuracy_precision(screens, preds)
    print(f"train-fit: acc {m['acc_px']:.1f}px / {m['acc_deg']:.2f}° (цель прод ≤2.0°, лаб ≤1.5°)")
    reg.save(profile_path())
    print(f"Сохранено: {profile_path()}")


def cmd_keyboard(args):
    reg = GazeRegressor()
    pp = profile_path()
    use_gaze = pp.exists() and not args.mouse
    if use_gaze:
        reg.load(pp)
        print(f"Профиль: {pp}")
    else:
        print("Режим мыши (отладка UI) — калибровка не требуется" if args.mouse
              else "Нет профиля — сначала --mode calibrate")
        if not args.mouse:
            return
    kb = DwellKeyboard(lang=args.lang)
    pred = PrefixPredictor(lang=args.lang)
    log = SessionLog(f"kbd_{args.lang.lower()}")
    cam = Camera(args.camera) if not args.mouse else None
    tr = FaceTracker() if cam else None
    flt = GazeFilter()
    W, H = 1280, 720
    mouse = [W // 2, H // 2]

    def on_mouse(e, x, y, f, p):
        mouse[0], mouse[1] = x, y

    win = f"EyeConnect [{args.lang}] Q-выход"
    cv2.namedWindow(win)
    if args.mouse:
        cv2.setMouseCallback(win, on_mouse)
    print("Смотри/веди на клавишу 1.0с. LANG — смена EN<->KZ")
    try:
        while True:
            if cam:
                f = cam.read()
                if f is None:
                    continue
                feat, conf, _ = tr.process(f)
                if feat is None or conf < C.MP_LOST_CONF:
                    img = kb.draw()
                else:
                    sm = flt.update(feat)
                    gx, gy = reg.predict(sm) if reg.trained else (W / 2, H / 2)
                    gx = float(np.clip(gx, 0, W - 1))
                    gy = float(np.clip(gy, 0, H - 1))
                    log.gaze(gx, gy)
                    ev = kb.update(gx, gy)
                    if ev:
                        log.event(ev, kb.text)
                        if ev == "LANG":
                            pred = PrefixPredictor(lang=kb.lang)
                    kb.pred = pred.suggest(kb.text)
                    img = kb.draw((gx, gy))
            else:
                ev = kb.update(*mouse)
                if ev:
                    log.event(ev, kb.text)
                    if ev == "LANG":
                        pred = PrefixPredictor(lang=kb.lang)
                kb.pred = pred.suggest(kb.text)
                img = kb.draw(mouse)
            cv2.imshow(win, img)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
    finally:
        print("Текст:", kb.text)
        print("Лог:", log.close())
        if tr:
            tr.close()
        if cam:
            cam.release()
        cv2.destroyAllWindows()


def cmd_cursor(args):
    cam = Camera(args.camera)
    tr = FaceTracker()
    flt = GazeFilter()
    try:
        reg, W, H = load_or_calibrate(cam, tr, flt, force=args.recalib)
        run_cursor(cam, tr, flt, reg, W, H, syscursor=not args.no_syscursor)
    finally:
        tr.close()
        cam.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preview", "calibrate", "keyboard", "cursor"], default="preview")
    ap.add_argument("--lang", choices=["KZ", "EN"], default="KZ")
    ap.add_argument("--camera", type=int, default=C.CAM_INDEX)
    ap.add_argument("--mouse", action="store_true", help="отладка клавиатуры мышью")
    ap.add_argument("--recalib", action="store_true", help="перекалибровать заново")
    ap.add_argument("--no-syscursor", action="store_true", help="не двигать системный курсор")
    args = ap.parse_args()
    {"preview": cmd_preview, "calibrate": cmd_calibrate,
     "keyboard": cmd_keyboard, "cursor": cmd_cursor}[args.mode](args)


if __name__ == "__main__":
    main()
