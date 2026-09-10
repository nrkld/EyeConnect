"""Виртуальная dwell-клавиатура OpenCV: EN + KZ-кириллица (дока 4.6, 5.5).

Раскладка крупная: центр 150px / периферия 180px логики учтены размером окна 1280x720.
Dwell MVP 1000мс с кольцом-прогрессом. Зоны: SOS (грубый выбор без калибровки),
LANG (EN<->KZ), SPACE, DEL, пауза взглядом в левый-верхний угол 3с.
"""
import time
import cv2
import numpy as np
from .. import config as C

LAYOUTS = {
    "EN": ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM,."],
    "KZ": ["ӘІҢҒҮҰҚӨҺ", "АБВГДЕЖЗИ", "ЙКЛМНОПРСТ", "УФХЦЧШЩЪЫ", "ЬЭЮЯ,.?"],
}
SPECIAL = ["LANG", "SPACE", "DEL", "SOS"]


class DwellKeyboard:
    def __init__(self, lang="KZ", dwell_ms: int = C.DWELL_MS, w=1280, h=720):
        assert lang in LAYOUTS
        self.lang = lang
        self.dwell = dwell_ms / 1000
        self.w, self.h = w, h
        self.text = ""
        self._focus = None
        self._t0 = None
        self._pause_until = 0
        self.pred = []
        self.rects = {}

    def toggle_lang(self):
        self.lang = "EN" if self.lang == "KZ" else "KZ"

    def layout_rects(self):
        rows = LAYOUTS[self.lang]
        self.rects = {}
        top = 150  # верх отдаём под текст + предикт
        rh = (self.h - top - 70) / (len(rows) + 1)
        for r, row in enumerate(rows):
            n = len(row)
            kw = min(C.KEY_CENTER, (self.w - 40) / max(n, 1) - C.KEY_GAP)
            y0 = top + r * rh
            x_start = (self.w - n * (kw + C.KEY_GAP)) / 2
            for i, ch in enumerate(row):
                x0 = x_start + i * (kw + C.KEY_GAP)
                self.rects[ch] = (x0, y0, kw, rh - C.KEY_GAP)
        # нижний ряд спецкнопок
        y0 = self.h - rh - 10
        for i, s in enumerate(SPECIAL):
            x0 = 20 + i * ((self.w - 40) / len(SPECIAL))
            self.rects[s] = (x0, y0, (self.w - 40) / len(SPECIAL) - C.KEY_GAP, rh - 10)
        return self.rects

    def update(self, gx, gy):
        """gx,gy — координаты взгляда в px окна. Возвращает событие-строку или None."""
        now = time.time()
        if now < self._pause_until:
            return None
        # пауза: взгляд в угол 3с — упрощённо: если в углу, ставим паузу 3с
        if gx < 80 and gy < 80:
            if self._focus != "PAUSE":
                self._focus, self._t0 = "PAUSE", now
            elif now - self._t0 > 3.0:
                self._pause_until = now + 3.0
                self._focus = None
                return "PAUSE"
            return None
        key = None
        for k, (x0, y0, kw, kh) in self.layout_rects().items():
            if x0 <= gx <= x0 + kw and y0 <= gy <= y0 + kh:
                key = k
                break
        if key != self._focus:
            self._focus, self._t0 = key, now
            return None
        if key is not None and (now - self._t0) >= self.dwell:
            self._focus, self._t0 = None, now
            return self.press(key)
        return None

    def press(self, key):
        if key == "LANG":
            self.toggle_lang()
        elif key == "SPACE":
            self.text += " "
        elif key == "DEL":
            self.text = self.text[:-1]
        elif key == "SOS":
            self.text += " [SOS] "
        elif key == "PAUSE":
            pass
        else:
            self.text += key
        return key

    def progress(self):
        if self._focus is None or self._t0 is None:
            return 0.0
        return min(1.0, (time.time() - self._t0) / self.dwell)

    def draw(self, gaze=None):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        cv2.rectangle(img, (0, 0), (self.w, 150), (30, 30, 30), -1)
        cv2.putText(img, self.text[-48:], (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, " | ".join(self.pred[:3]) + f"   [{self.lang}] dwell {self.dwell:.1f}s",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for k, (x0, y0, kw, kh) in self.layout_rects().items():
            col = (70, 70, 70) if k not in SPECIAL else (50, 90, 140)
            if k == self._focus:
                col = (0, 160, 0)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + kw), int(y0 + kh)), col, -1)
            cv2.putText(img, k, (int(x0) + 12, int(y0 + kh / 2 + 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if k == self._focus:
                p = self.progress()
                cv2.ellipse(img, (int(x0 + kw / 2), int(y0 + kh / 2)),
                            (30, 30), 0, 0, int(360 * p), (0, 255, 0), 4)
        if gaze is not None:
            cv2.circle(img, (int(gaze[0]), int(gaze[1])), 10, (0, 0, 255), 2)
        return img
