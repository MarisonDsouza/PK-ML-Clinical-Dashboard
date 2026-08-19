"""pk_core/models.py — extra NumPy learners for the grey-box engine
(kernel-SVR, gradient boosting, stacking). Self-contained; reuses the CART in
tiny_forest.py. No scikit-learn."""
from __future__ import annotations
import numpy as np
from .tiny_forest import _build, _pred_tree, RandomForestRegressor  # noqa: F401


class RidgeNP:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd == 0] = 1
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        self.w = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ y); return self
    def predict(self, X):
        X = np.asarray(X, float); return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.w


class KernelRidgeRBF:
    """Analytic RBF kernel-ridge — stands in for the RBF-SVR family."""
    def __init__(self, alpha=1.0, gamma=None): self.alpha = alpha; self.gamma = gamma
    def _k(self, A, B):
        a2 = (A ** 2).sum(1)[:, None]; b2 = (B ** 2).sum(1)[None, :]
        return np.exp(-self.gamma * np.maximum(a2 + b2 - 2 * A @ B.T, 0))
    def fit(self, X, y):
        X = np.asarray(X, float); self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd == 0] = 1
        self.Xtr = (X - self.mu) / self.sd
        if self.gamma is None: self.gamma = 1.0 / self.Xtr.shape[1]
        K = self._k(self.Xtr, self.Xtr)
        self.a = np.linalg.solve(K + self.alpha * np.eye(len(K)), np.asarray(y, float)); return self
    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd; return self._k(Z, self.Xtr) @ self.a


class GBM:
    """Gradient-boosted regression trees (squared loss) on tiny_forest stumps."""
    def __init__(self, n=400, lr=0.04, depth=3, subsample=0.8, colsample=None, min_leaf=5, seed=0):
        self.n = n; self.lr = lr; self.depth = depth; self.sub = subsample
        self.cs = colsample; self.min_leaf = min_leaf; self.seed = seed
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed); n, p = X.shape
        mf = p if self.cs is None else max(1, int(self.cs * p))
        self.init = float(np.median(y)); F = np.full(n, self.init); self.trees = []
        for m in range(self.n):
            g = y - F
            idx = rng.choice(n, max(2, int(self.sub * n)), replace=False)
            tr = _build(X[idx], g[idx], 0, self.depth, self.min_leaf, mf,
                        np.random.default_rng(self.seed + m + 1))
            F = F + self.lr * _pred_tree(tr, X); self.trees.append(tr)
        return self
    def predict(self, X):
        X = np.asarray(X, float); out = np.full(len(X), self.init)
        for tr in self.trees: out += self.lr * _pred_tree(tr, X)
        return out


class Stack:
    """Base learners (RF + GBM + kernel-SVR) -> ridge meta-learner."""
    def __init__(self, seed=0): self.seed = seed
    @staticmethod
    def _bases():
        return [RandomForestRegressor(80, 12, 3, random_state=7),
                GBM(150, .05, 4, seed=3), KernelRidgeRBF(1.0)]
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.b = self._bases()
        Z = np.column_stack([bb.fit(X, y).predict(X) for bb in self.b])
        self.meta = RidgeNP(1.0).fit(Z, y); return self
    def predict(self, X):
        return self.meta.predict(np.column_stack([bb.predict(X) for bb in self.b]))
