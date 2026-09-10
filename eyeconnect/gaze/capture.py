"""Захват камеры через OpenCV (дока 4.1, 6.3)."""
import os
import time
import cv2
from .. import config as C


class Camera:
    def __init__(self, index: int = C.CAM_INDEX, w: int = C.FRAME_W, h: int = C.FRAME_H):
        if os.name == "nt":
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(index)
        # CAP_DSHOW быстрее на Windows; на Linux флаг игнорируется
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, C.FPS_TARGET)
        self._t0 = time.time()
        self._n = 0
        self.fps = 0.0
        if not self.cap.isOpened():
            raise RuntimeError(f"Камера {index} не открылась. Проверь подключение / индекс.")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)  # зеркалим как в selfie-режиме
        self._n += 1
        dt = time.time() - self._t0
        if dt >= 1.0:
            self.fps = self._n / dt
            self._n = 0
            self._t0 = time.time()
        return frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    cam = Camera()
    print("Нажми Q для выхода")
    while True:
        f = cam.read()
        if f is None:
            continue
        cv2.putText(f, f"FPS {cam.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("EyeConnect capture (Q - выход)", f)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
    cam.release()
    cv2.destroyAllWindows()
