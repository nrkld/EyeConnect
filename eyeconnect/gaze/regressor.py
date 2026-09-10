"""Полиномиальная Ridge-регрессия 2-й степени, 9 точек, веса углов 1.5x (дока 4.4)."""
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from .. import config as C


class GazeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.mx = Ridge(alpha=alpha)
        self.my = Ridge(alpha=alpha)
        self.trained = False

    def fit(self, feats_xy, screens_xy, edge_weight: float = C.EDGE_WEIGHT):
        X = np.asarray(feats_xy, dtype=float)
        Y = np.asarray(screens_xy, dtype=float)
        # вес 1.5x точкам у краёв экрана (периферия сложнее)
        mx, my = X[:, 0].mean(), X[:, 1].mean()
        dist = np.abs(X[:, 0] - mx) + np.abs(X[:, 1] - my)
        thr = np.quantile(dist, 0.6)
        w = np.where(dist >= thr, edge_weight, 1.0)
        Xp = self.poly.fit_transform(X)
        self.mx.fit(Xp, Y[:, 0], sample_weight=w)
        self.my.fit(Xp, Y[:, 1], sample_weight=w)
        self.trained = True
        return self

    def predict(self, feat_xy):
        assert self.trained, "Сначала fit() или load()"
        Xp = self.poly.transform(np.asarray(feat_xy, dtype=float).reshape(1, -1))
        return float(self.mx.predict(Xp)[0]), float(self.my.predict(Xp)[0])

    def save(self, path: Path):
        path = Path(path)
        np.savez(path, coef_x=self.mx.coef_, b_x=self.mx.intercept_,
                 coef_y=self.my.coef_, b_y=self.my.intercept_,
                 powers=self.poly.powers_)

    def load(self, path: Path):
        d = np.load(Path(path))
        # восстанавливаем полином по сохранённым степеням
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        # fit на dummy чтобы задать powers_, затем подменяем
        self.poly.fit(np.zeros((1, 2)))
        self.poly.powers_ = d["powers"]
        self.poly.n_features_in_ = 2
        self.poly.n_output_features_ = d["coef_x"].shape[0]
        self.mx.coef_ = d["coef_x"]
        self.mx.intercept_ = d["b_x"]
        self.my.coef_ = d["coef_y"]
        self.my.intercept_ = d["b_y"]
        for m in (self.mx, self.my):
            m.n_features_in_ = self.poly.n_output_features_
        self.trained = True
        return self


if __name__ == "__main__":
    # синтетический smoke-тест без камеры
    rng = np.random.default_rng(0)
    feats = rng.random((9, 2))
    screens = feats * np.array([1920, 1080]) + rng.normal(0, 10, (9, 2))
    r = GazeRegressor().fit(feats, screens)
    print("pred:", r.predict([0.5, 0.5]))
