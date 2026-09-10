"""Геометрическая нормализация взгляда (дока 4.3).

Вход: координаты ирисов и уголков глаз в пикселях кадра.
Выход: нормализованный вектор [0..1] + оценка дистанции по ирису.
Инвариантно к расстоянию 50-70см и размеру лица.
"""
import numpy as np

IRIS_REAL_MM = 11.7  # средний диаметр ириса, Chelak & Yuan


def norm_eye(iris_xy, outer_xy, inner_xy, top_xy=None, bottom_xy=None):
    """Нормализация одного глаза -> (nx, ny) в [0..1]."""
    w = float(np.linalg.norm(np.asarray(inner_xy) - np.asarray(outer_xy))) + 1e-6
    nx = (float(iris_xy[0]) - float(outer_xy[0])) / w
    if top_xy is not None and bottom_xy is not None:
        h = float(np.linalg.norm(np.asarray(bottom_xy) - np.asarray(top_xy))) + 1e-6
        ny = (float(iris_xy[1]) - float(top_xy[1])) / h
    else:
        ny = (float(iris_xy[1]) - float(outer_xy[1])) / w
    return float(np.clip(nx, -0.5, 1.5)), float(np.clip(ny, -0.5, 1.5))


def fuse_eyes(left_n, right_n):
    return ((left_n[0] + right_n[0]) / 2.0, (left_n[1] + right_n[1]) / 2.0)


def estimate_distance_mm(iris_px, focal_px=650.0):
    """d = f * D_real / D_px. focal_px калибруется один раз на камеру."""
    iris_px = max(float(iris_px), 1.0)
    return float(focal_px * IRIS_REAL_MM / iris_px)


def features_from_eyes(l_iris, l_outer, l_inner, r_iris, r_outer, r_inner):
    ln = norm_eye(l_iris, l_outer, l_inner)
    rn = norm_eye(r_iris, r_outer, r_inner)
    fx, fy = fuse_eyes(ln, rn)
    return np.array([fx, fy], dtype=np.float64)
