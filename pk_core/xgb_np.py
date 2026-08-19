#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoostNP — dependency-free NumPy re-implementation of XGBoost's regularized
gradient-boosted trees, used as the drop-in 8th-model replacement when the
xgboost / scikit-learn wheels cannot be installed.

Faithful to XGBoost's objective (Chen & Guestrin 2016):
  leaf weight  w* = -G / (H + lambda)
  split gain   = 1/2 [ G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ) ] - gamma
with eta shrinkage, row subsample, column subsample, and min_child_weight
(minimum child Hessian sum). Squared loss => g_i = F-y, h_i = 1, so the leaf
solution and gain match XGBoost's regression path exactly.
"""
import numpy as np


class _Node:
    __slots__ = ("f", "t", "l", "r", "w")


def _build(X, g, depth, max_depth, mcw, lam, gamma, mf, rng):
    nd = _Node(); nd.f = -1; nd.l = nd.r = None
    n = len(g); G = float(g.sum())
    nd.w = G / (n + lam)                      # leaf weight (residual fit, h=1)
    if depth >= max_depth or n < 2 * mcw:
        return nd
    p = X.shape[1]; feats = rng.choice(p, min(mf, p), replace=False)
    parent = G * G / (n + lam)
    best = -np.inf; bf = -1; bt = 0.0
    for f in feats:
        xf = X[:, f]; order = np.argsort(xf, kind="quicksort")
        xs = xf[order]; gs = g[order]
        cs = np.cumsum(gs); GL = cs[:-1]; nL = np.arange(1, n); GR = G - GL; nR = n - nL
        gain = GL * GL / (nL + lam) + GR * GR / (nR + lam) - parent
        valid = (xs[1:] != xs[:-1]) & (nL >= mcw) & (nR >= mcw)
        if not valid.any():
            continue
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > best:
            best = gain[i]; bf = f; bt = 0.5 * (xs[i] + xs[i + 1])
    if bf < 0 or 0.5 * best <= gamma:          # gain must clear gamma
        nd.f = -1; return nd
    m = X[:, bf] <= bt
    if m.sum() < mcw or (~m).sum() < mcw:
        nd.f = -1; return nd
    nd.f = bf; nd.t = bt
    nd.l = _build(X[m], g[m], depth + 1, max_depth, mcw, lam, gamma, mf, rng)
    nd.r = _build(X[~m], g[~m], depth + 1, max_depth, mcw, lam, gamma, mf, rng)
    return nd


def _pred(nd, X):
    out = np.empty(len(X))

    def rec(node, idx):
        if node.f < 0:
            out[idx] = node.w; return
        m = X[idx, node.f] <= node.t
        rec(node.l, idx[m]); rec(node.r, idx[~m])
    rec(nd, np.arange(len(X)))
    return out


class XGBoostNP:
    def __init__(self, n_estimators=300, eta=0.05, max_depth=4, lam=1.0, gamma=0.0,
                 subsample=0.8, colsample=0.7, min_child_weight=5, seed=0):
        self.n = n_estimators; self.eta = eta; self.max_depth = max_depth
        self.lam = lam; self.gamma = gamma; self.subsample = subsample
        self.colsample = colsample; self.mcw = min_child_weight; self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed); n, p = X.shape
        mf = max(1, int(self.colsample * p))
        self.base = float(np.median(y)); F = np.full(n, self.base); self.trees = []
        for m in range(self.n):
            resid = y - F
            idx = rng.choice(n, max(2 * self.mcw, int(self.subsample * n)), replace=False)
            tr = _build(X[idx], resid[idx], 0, self.max_depth, self.mcw,
                        self.lam, self.gamma, mf, np.random.default_rng(self.seed + m + 1))
            F = F + self.eta * _pred(tr, X)
            self.trees.append(tr)
        return self

    def predict(self, X):
        X = np.asarray(X, float); out = np.full(len(X), self.base)
        for tr in self.trees:
            out += self.eta * _pred(tr, X)
        return out
