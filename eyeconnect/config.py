"""EyeConnect shared config (дока разд. 4-6)."""
from pathlib import Path

# Видео
FRAME_W, FRAME_H = 640, 480
CAM_INDEX = 0
FPS_TARGET = 30

# MediaPipe
MP_MIN_CONF = 0.5
MP_LOST_CONF = 0.3

# Калибровка: сетка 3x3, 3с фиксация, 0.4с усреднение, веса углов 1.5x
CALIB_GRID = 3
CALIBDWELL_S = 3.0
CALIB_AVG_S = 0.4
EDGE_WEIGHT = 1.5

# Сглаживание: EMA alpha=0.4 по 5 кадрам + Калман
EMA_ALPHA = 0.4
EMA_N = 5

# Dwell MVP: фиксированный 1000мс (адаптивный 800-1500 — фаза 2)
DWELL_MS = 1000

# UI: клавиши 150px центр / 180px периферия, зазор 14px
KEY_CENTER = 150
KEY_EDGE = 180
KEY_GAP = 14

# Геометрия для перевода px -> градусы (ноутбук по умолчанию, переопределить под монитор)
SCREEN_W_PX, SCREEN_H_PX = 1920, 1080
SCREEN_W_MM = 531.0  # 24" 16:9, померять линейкой и поправить
VIEW_DIST_MM = 600.0  # 50-70см по доке

PROFILE_DIR = Path.home() / ".eyeconnect"
PROFILE_DIR.mkdir(exist_ok=True)
