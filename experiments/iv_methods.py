
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import softmax
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def calibration_curve(m, h, e, z):
    """Mean uncentered EIF: m[n,K,T], h[n,T], e[n,K], z[n]."""
    m, h, e, z = map(np.asarray, (m, h, e, z))
    rows = np.arange(len(z))
    return 1 - np.mean(m.sum(axis=1) +
                       (h - m[rows, z]) / e[rows, z, None], axis=0)


def last_crossing(grid, curve, tail, alpha):
    """Supremum of the piecewise-linear curve's alpha superlevel set.

    Infinite tail equality is treated conservatively as an infinite set.
    Include the two sides of jumps in grid when the curve is discontinuous.
    """
    grid, curve = np.asarray(grid), np.asarray(curve)
    if tail >= alpha - 1e-12:
        return np.inf
    passing = np.flatnonzero(curve >= alpha)
    if not len(passing):
        return 0.0
    j = passing[-1]
    if j == len(grid) - 1:
        # A truncated grid cannot justify claiming a finite prediction set.
        return np.inf
    frac = (curve[j] - alpha) / (curve[j] - curve[j + 1])
    return float(grid[j] + frac * (grid[j + 1] - grid[j]))


def conformal_radius(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(scores) + 1) * (1 - alpha) - 1e-12))
    return float(scores[k - 1]) if 0 < k <= len(scores) else np.inf


def weighted_radii(scores, wcal, wtest, alpha):
    """Weighted split conformal with a separate +infinity test atom."""
    order = np.argsort(scores)
    scores = np.asarray(scores)[order]
    cumulative = np.cumsum(np.asarray(wcal)[order])
    targets = (1 - alpha) * (cumulative[-1] + np.asarray(wtest))
    ids = np.searchsorted(cumulative, targets, side="left")
    return np.r_[scores, np.inf][ids]


@dataclass
class Prepared:
    grid: np.ndarray
    m_soft: np.ndarray
    m_policy: np.ndarray
    e: np.ndarray
    response: np.ndarray
    center: np.ndarray
    gamma_plugin: np.ndarray


class IVPredictor:
    def __init__(self, seed=20260918, beta=50., n_trees=120, min_leaf=20,
                 propensity_floor=.02, known_e=None, grid_size=801):
        self.seed, self.beta = seed, beta
        self.n_trees, self.min_leaf = n_trees, min_leaf
        self.propensity_floor, self.known_e = propensity_floor, known_e
        self.grid_size = grid_size

    def _rf(self, offset=0):
        return RandomForestRegressor(n_estimators=self.n_trees,
            min_samples_leaf=self.min_leaf, max_features=1.,
            random_state=self.seed + offset, n_jobs=1)

    def fit(self, x, z, r, y):
        x, z, r = np.asarray(x), np.asarray(z, int), np.asarray(r, bool)
        # Inaccessible outcomes cannot affect any subsequent computation.
        yobs = np.asarray(y, float)[r]
        if not np.isfinite(yobs).all():
            raise ValueError("Observed outcomes must be finite")
        self.k = int(z.max()) + 1
        if set(np.unique(z)) != set(range(self.k)):
            raise ValueError("Instrument labels must be consecutive from zero")
        self.location = float(yobs.mean())
        self.scale = max(float(yobs.std(ddof=1)), 1e-8)
        ys = (yobs - self.location) / self.scale
        self.x_scaler = StandardScaler().fit(x)
        self.score = self._rf(1).fit(x[r], ys)
        if self.known_e is None:
            self.e_model = LogisticRegression(C=1., max_iter=2000).fit(
                self.x_scaler.transform(x), z)
        self.response_model = self._rf(2).fit(x, r.astype(float))
        self.pi, self.moments = [], []
        for j in range(self.k):
            iz, io = z == j, (z == j) & r
            if io.sum() < 10:
                raise ValueError(f"Too few observed training outcomes in arm {j}")
            self.pi.append(self._rf(10+j).fit(x[iz], r[iz].astype(float)))
            yj = (np.asarray(y, float)[io] - self.location) / self.scale
            self.moments.append(self._rf(30+j).fit(x[io], np.c_[yj, yj**2]))
        # Manuscript policy-choice paragraph: pseudo-label RF classifier.
        # q = pi*f (the paragraph's e*f is a notation error).
        q = self.subdensities(x[r], ys[:, None])[:, 0, :]
        labels = q.argmax(axis=1)
        self.policy = RandomForestClassifier(n_estimators=self.n_trees,
            min_samples_leaf=max(5, self.min_leaf//2), random_state=self.seed+90,
            n_jobs=1).fit(np.c_[x[r], ys], labels)
        return self

    def parameters(self, x):
        pi = np.column_stack([p.predict(x) for p in self.pi])
        mom = np.stack([m.predict(x) for m in self.moments], axis=1)
        sd = np.sqrt(np.maximum(mom[:, :, 1] - mom[:, :, 0]**2, .05**2))
        return np.clip(pi, 0., 1.), mom[:, :, 0], sd

    def subdensities(self, x, ys, parameters=None):
        pi, mu, sd = self.parameters(x) if parameters is None else parameters
        return pi[:, None, :] * norm.pdf(
            (ys[:, :, None] - mu[:, None, :]) / sd[:, None, :]) / sd[:, None, :]

    def e_predict(self, x):
        if self.known_e is not None:
            return np.tile(np.asarray(self.known_e), (len(x), 1))
        p = self.e_model.predict_proba(self.x_scaler.transform(x))
        # Mix with the uniform distribution: normalized and >= floor.
        floor = min(self.propensity_floor, .5/self.k)
        return (1 - self.k*floor) * p + floor

    def weights_at_observed(self, x, y, r, z):
        """Return observed-arm soft/hard weights; never evaluate missing Y."""
        r, z = np.asarray(r, bool), np.asarray(z, int)
        ws, wp = np.zeros(len(x)), np.zeros(len(x))
        if r.any():
            ys = (np.asarray(y)[r] - self.location) / self.scale
            q = self.subdensities(x[r], ys[:, None])[:, 0, :]
            ws[r] = softmax(self.beta*q, axis=1)[np.arange(r.sum()), z[r]]
            wp[r] = (self.policy.predict(np.c_[x[r], ys]) == z[r]).astype(float)
        return ws, wp

    def prepare(self, x, grid=None, progress=None):
        x = np.asarray(x)
        center = self.score.predict(x)
        pars = self.parameters(x)
        if grid is None:
            end = np.max(np.abs(pars[1] - center[:, None]) + 9*pars[2])
            grid = np.linspace(0., max(float(end), 1.), self.grid_size)
        n, t = len(x), len(grid)
        ms = np.zeros((n, self.k, t), dtype=np.float32)
        mp = np.zeros_like(ms)
        gamma = np.zeros(n)
        for start in range(0, n, 64):
            stop = min(start + 64, n)
            xc, cc = x[start:stop], center[start:stop]
            pp = tuple(a[start:stop] for a in pars)
            ints, intp, intmax = [], [], []
            for sign in (-1, 1):
                yy = cc[:, None] + sign * grid[None, :]
                q = self.subdensities(xc, yy, pp)
                lam = softmax(self.beta * q, axis=2)
                features = np.c_[np.repeat(xc, t, axis=0), yy.ravel()]
                chosen = self.policy.predict(features).reshape(len(xc), t)
                hard = (chosen[:, :, None] == np.arange(self.k)[None, None, :])
                ints.append(lam*q)
                intp.append(hard*q)
                intmax.append(q.max(axis=2))
            ms[start:stop] = cumulative_trapezoid(ints[0]+ints[1], grid,
                axis=1, initial=0).transpose(0, 2, 1)
            mp[start:stop] = cumulative_trapezoid(intp[0]+intp[1], grid,
                axis=1, initial=0).transpose(0, 2, 1)
            gamma[start:stop] = np.trapz(intmax[0]+intmax[1], grid, axis=1)
            if progress is not None and (start == 0 or stop == n or start % 640 == 0):
                progress(f"integrated {stop}/{n} covariates")
        # Individual conditional moments are bounded; their sum need not be.
        ms, mp = np.clip(ms, 0., 1.), np.clip(mp, 0., 1.)
        return Prepared(grid, ms, mp, self.e_predict(x),
            np.clip(self.response_model.predict(x), .02, .98), center, gamma)

    def calibrate(self, prepared, ids, z, r, y, x, alphas):
        ids, z, r = np.asarray(ids), np.asarray(z, int), np.asarray(r, bool)
        n = len(ids)
        e = prepared.e[ids]
        actual_e = e[np.arange(n), z]
        scores = np.full(n, np.inf)
        scores[r] = np.abs((np.asarray(y)[r]-self.location)/self.scale -
                           prepared.center[ids][r])
        ws, wp = self.weights_at_observed(x, y, r, z)
        result, diagnostics = {}, {}
        for name, moments, obsweight in [("Softmax IV", prepared.m_soft, ws),
                                         ("Policy IV", prepared.m_policy, wp)]:
            m = moments[ids].astype(float)
            smooth = 1 - np.mean(m.sum(axis=1)-m[np.arange(n), z]/actual_e[:, None], axis=0)
            order = np.argsort(scores[r])
            s = scores[r][order]
            jumps = np.cumsum((obsweight[r]/actual_e[r])[order])/n
            # Both sides of every empirical jump, and every integration knot.
            knots = np.unique(np.r_[prepared.grid, s, np.nextafter(s, -np.inf)])
            knots = knots[knots >= 0]
            pos = np.searchsorted(s, knots, side="right")
            curve = np.interp(knots, prepared.grid, smooth) - np.r_[0., jumps][pos]
            tail = smooth[-1] - (jumps[-1] if len(jumps) else 0.)
            result[name] = np.array([last_crossing(knots, curve, tail, a) for a in alphas])
            diagnostics[name] = {"identified_mass_eif": float(1-tail),
                "upward_increments": int(np.sum(np.diff(curve) > 1e-8))}
        result["Naive CP"] = np.array([conformal_radius(scores[r], a) for a in alphas])
        diagnostics["calibration"] = {"observed": int(r.sum()),
            "min_e": float(actual_e.min()), "max_inverse_e": float(1/actual_e.min())}
        return result, diagnostics, scores
