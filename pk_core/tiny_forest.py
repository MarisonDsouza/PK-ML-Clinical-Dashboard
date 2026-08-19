#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency-free (NumPy-only) random-forest regressor.

Exists because the sandbox used to build this project cannot install
scikit-learn / XGBoost. The API mirrors the small subset we need:
    RandomForestRegressor(n_estimators, max_depth, min_leaf, max_features)
        .fit(X, y).predict(X)
It is not meant to beat XGBoost — it is a faithful, inspectable tree ensemble
so the grey-box vs legacy comparison is a real trained model, not a mock.
"""
from __future__ import annotations
import numpy as np


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "val")

    def __init__(self):
        self.feat = -1; self.thr = 0.0
        self.left = None; self.right = None; self.val = None


def _build(X, y, depth, max_depth, min_leaf, max_features, rng):
    node = _Node()
    n = len(y)
    if depth >= max_depth or n < 2 * min_leaf or np.all(y == y[0]):
        node.val = float(y.mean()); return node

    n_feat = X.shape[1]
    m = max(1, int(max_features)) if max_features else n_feat
    feats = rng.choice(n_feat, size=min(m, n_feat), replace=False)

    best_sse = np.inf; best_f = -1; best_thr = 0.0
    parent_sum = y.sum(); parent_sq = (y * y).sum()
    for f in feats:
        xf = X[:, f]
        order = np.argsort(xf, kind="quicksort")
        xs = xf[order]; ys = y[order]
        csum = np.cumsum(ys); ctot = csum[-1]
        # candidate split after position i (0..n-2), left has i+1 points
        left_n = np.arange(1, n)
        left_sum = csum[:-1]
        right_n = n - left_n
        right_sum = ctot - left_sum
        # SSE = sum_sq - sum^2/n  -> only need the sum^2/n reduction terms
        left_mean_sq = left_sum ** 2 / left_n
        right_mean_sq = right_sum ** 2 / right_n
        gain = left_mean_sq + right_mean_sq            # larger = better
        # invalidate splits where x equal on both sides, or leaf too small
        valid = (xs[1:] != xs[:-1]) & (left_n >= min_leaf) & (right_n >= min_leaf)
        if not valid.any():
            continue
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        sse = parent_sq - gain[i]                       # equivalent objective
        if sse < best_sse:
            best_sse = sse; best_f = f
            best_thr = 0.5 * (xs[i] + xs[i + 1])

    if best_f < 0:
        node.val = float(y.mean()); return node

    mask = X[:, best_f] <= best_thr
    if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
        node.val = float(y.mean()); return node

    node.feat = best_f; node.thr = best_thr
    node.left = _build(X[mask], y[mask], depth + 1, max_depth, min_leaf, max_features, rng)
    node.right = _build(X[~mask], y[~mask], depth + 1, max_depth, min_leaf, max_features, rng)
    return node


def _pred_tree(node, X):
    out = np.empty(len(X))
    for i in range(len(X)):
        nd = node
        while nd.val is None:
            nd = nd.left if X[i, nd.feat] <= nd.thr else nd.right
        out[i] = nd.val
    return out


class RandomForestRegressor:
    def __init__(self, n_estimators=40, max_depth=9, min_leaf=4,
                 max_features="sqrt", random_state=42):
        self.n = n_estimators; self.max_depth = max_depth
        self.min_leaf = min_leaf; self.max_features = max_features
        self.random_state = random_state; self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        rng = np.random.default_rng(self.random_state)
        mf = (int(np.sqrt(X.shape[1])) if self.max_features == "sqrt"
              else X.shape[1] if self.max_features in (None, "all")
              else int(self.max_features))
        self.trees = []
        for t in range(self.n):
            idx = rng.integers(0, len(y), len(y))          # bootstrap
            tr = _build(X[idx], y[idx], 0, self.max_depth, self.min_leaf, mf,
                        np.random.default_rng(self.random_state + t + 1))
            self.trees.append(tr)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        acc = np.zeros(len(X))
        for tr in self.trees:
            acc += _pred_tree(tr, X)
        return acc / len(self.trees)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(600, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.1, 600)
    tr, te = slice(0, 450), slice(450, 600)
    rf = RandomForestRegressor(n_estimators=40, max_depth=9).fit(X[tr], y[tr])
    p = rf.predict(X[te])
    ss = 1 - np.sum((y[te]-p)**2)/np.sum((y[te]-y[te].mean())**2)
    print(f"self-test R^2 on held-out = {ss:.3f}  (expect > 0.9)")
