"""The gradient-boosted trees (scikit-learn's histogram gradient boosting) and the linear model on the same features: the target minus the weekday climatology, from the inputs at the cell, their 3 x 3, 9 x 9 and 27 x 27 means, upwind NOx, calendar features and the climatology."""
import math
import numpy as np
from baselines import common
SCALES = (3, 9, 27)
BOX_FIELDS = ('NOX', 'E-4', 'M-8', 'M-3', 'M-4', 'M-5', 'M-2')
NOX_PARTS = ('E-1', 'E-2', 'E-3', 'P-1')
RINGS = ((0.5, 3.5), (3.5, 9.5))
HALF_SECTOR = math.pi / 8.0
CELLS_PER_DAY = 3400
GBT_OPTIONS = {'loss': 'squared_error', 'learning_rate': 0.1, 'max_iter': 500, 'max_leaf_nodes': 255, 'min_samples_leaf': 200, 'l2_regularization': 1.0, 'max_bins': 255, 'early_stopping': False}
MODEL_ID = 5

def box_mean_clipped(f, k):
    h, w = f.shape
    r = k // 2
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = f.cumsum(0).cumsum(1)
    one = np.zeros((h + 1, w + 1), dtype=np.float64)
    one[1:, 1:] = np.ones((h, w)).cumsum(0).cumsum(1)
    rr = np.arange(h)
    cc = np.arange(w)
    r0 = np.clip(rr - r, 0, h)
    r1 = np.clip(rr + r + 1, 0, h)
    c0 = np.clip(cc - r, 0, w)
    c1 = np.clip(cc + r + 1, 0, w)

    def rect(a):
        return a[r1][:, c1] - a[r0][:, c1] - a[r1][:, c0] + a[r0][:, c0]
    return rect(ii) / rect(one)

def ring_offsets():
    out = []
    rmax = int(math.ceil(RINGS[-1][1]))
    for lo, hi in RINGS:
        dr, dc, ang = ([], [], [])
        for a in range(-rmax, rmax + 1):
            for b in range(-rmax, rmax + 1):
                d = math.hypot(a, b)
                if lo <= d < hi:
                    dr.append(a)
                    dc.append(b)
                    ang.append(math.atan2(a, b))
        out.append((np.array(dr), np.array(dc), np.array(ang)))
    return out
RING_OFFSETS = ring_offsets()

def sector_means(nox, u, v, rows, cols, chunk=4096):
    h, w = nox.shape
    up_all = np.arctan2(-v, -u)
    out = np.empty((len(RING_OFFSETS), len(rows)), dtype=np.float64)
    for a in range(0, len(rows), chunk):
        r_, c_, up = (rows[a:a + chunk], cols[a:a + chunk], up_all[a:a + chunk])
        for j, (dr, dc, ang) in enumerate(RING_OFFSETS):
            rr = r_[:, None] + dr[None, :]
            cc = c_[:, None] + dc[None, :]
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            diff = np.abs((ang[None, :] - up[:, None] + math.pi) % (2.0 * math.pi) - math.pi)
            sel = inside & (diff <= HALF_SECTOR + 1e-12)
            vals = nox[np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)]
            n = sel.sum(axis=1)
            s = np.where(sel, vals, 0.0).sum(axis=1)
            out[j, a:a + chunk] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out

class Features:

    def __init__(self, fold, clim):
        self.fold = fold
        self.clim = clim
        self.adv = fold.configs.channel_set('advanced')
        need = ('E-1', 'E-2', 'E-3', 'E-4', 'M-2', 'M-3', 'M-4', 'M-5', 'M-8', 'X-3', 'X-4')
        miss = [n for n in need if n not in self.adv]
        if miss:
            raise ValueError('GBT: the advanced set lacks %s' % miss)
        fold.check_built(self.adv)
        self.nox_parts = [n for n in NOX_PARTS if n in self.adv]
        self.names = list(self.adv) + ['box%d_%s' % (k, f) for f in BOX_FIELDS for k in SCALES] + ['upwind_day_1to3', 'upwind_day_4to9', 'upwind_prev_1to3', 'upwind_prev_4to9', 'ln(M-3 x M-8)', 'sin_doy', 'cos_doy', 'weekday', 'CLIM-WD']
        self.weekday_index = self.names.index('weekday')
        h, w = fold.grid
        self.rows_all = fold.cells // w
        self.cols_all = fold.cells % w

    def day(self, y, i, d, mode, pick=None):
        f = self.fold
        g = f.channels_grid(y, i, self.adv, mode)
        k = {n: j for j, n in enumerate(self.adv)}
        idx = f.cells if pick is None else f.cells[pick]
        rows = self.rows_all if pick is None else self.rows_all[pick]
        cols = self.cols_all if pick is None else self.cols_all[pick]
        flat = g.reshape(len(self.adv), -1)
        cols_out = [flat[:, idx].T]
        nox = sum((g[k[n]] for n in self.nox_parts))
        fields = {'NOX': nox}
        for n in BOX_FIELDS[1:]:
            fields[n] = g[k[n]]
        box = []
        for fn in BOX_FIELDS:
            for s in SCALES:
                box.append(box_mean_clipped(fields[fn], s).ravel()[idx])
        cols_out.append(np.stack(box, axis=1))
        sd = sector_means(nox, g[k['M-4']].ravel()[idx], g[k['M-5']].ravel()[idx], rows, cols)
        sp = sector_means(nox, g[k['X-3']].ravel()[idx], g[k['X-4']].ravel()[idx], rows, cols)
        with np.errstate(divide='ignore', invalid='ignore'):
            vent = np.log(g[k['M-3']].ravel()[idx] * g[k['M-8']].ravel()[idx])
        doy = common.day_of_year(d)
        ang = 2.0 * math.pi * doy / 366.0
        kk = len(idx)
        cw = self.clim.clim_wd(d)
        cw = cw if pick is None else cw[pick]
        rest = np.stack([sd[0], sd[1], sp[0], sp[1], vent, np.full(kk, math.sin(ang)), np.full(kk, math.cos(ang)), np.full(kk, float(common.weekday(d))), cw], axis=1)
        cols_out.append(rest)
        return (np.concatenate(cols_out, axis=1), cw)

def gbt_seed(year):
    from train import cfg
    return cfg.SEED_BASE + 100 * cfg.fold_index(year) + MODEL_ID

class NonFiniteFeature(Exception):
    pass

class GBTFit:

    def __init__(self, fold, clim, note=None, cells_per_day=CELLS_PER_DAY, options=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self.fold = fold
        self.feat = Features(fold, clim)
        nf = len(self.feat.names)
        ncell = len(fold.cells)
        per = min(int(cells_per_day), ncell)
        n_rows = per * len(fold.train_days)
        common.memory_check(n_rows * (nf + 2) * 8 * 1.6 / 1000000000.0 + 0.5, 'GBT fit (%d rows)' % n_rows)
        self.seed = gbt_seed(fold.year)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))
        x = np.empty((n_rows, nf), dtype=np.float64)
        t = np.empty(n_rows, dtype=np.float64)
        a = 0
        for kd, (y, i, d) in enumerate(fold.train_days):
            pick = np.sort(rng.choice(ncell, size=per, replace=False))
            fx, cw = self.feat.day(y, i, d, 'train', pick)
            x[a:a + per] = fx
            t[a:a + per] = fold.target_cells(y, i)[pick] - cw
            a += per
            if note and (kd + 1) % 200 == 0:
                note('GBT fold %d: features of %d of %d training days' % (fold.year, kd + 1, len(fold.train_days)))
        bad = int((~np.isfinite(x)).sum() + (~np.isfinite(t)).sum())
        if bad:
            raise NonFiniteFeature('GBT fold %d: %d non-finite feature or target values; nothing substituted' % (fold.year, bad))
        self.n_rows = n_rows
        opts = dict(GBT_OPTIONS)
        opts.update(options or {})
        self.gbt = HistGradientBoostingRegressor(random_state=self.seed, categorical_features=np.array([self.feat.weekday_index], dtype=np.int64), **opts)
        self.gbt.fit(x, t)
        self.categorical = [int(j) for j in np.flatnonzero(np.asarray(self.gbt.is_categorical_))]
        self.scaler = StandardScaler(copy=False)
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        self.ridge = Ridge(alpha=1.0, copy_X=False)
        self.ridge.fit(x, t)
        del x, t

    def predict(self, y, i, d):
        fx, cw = self.feat.day(y, i, d, 'predict')
        if not np.all(np.isfinite(fx)):
            raise NonFiniteFeature('GBT fold %d: non-finite features on a held-out day' % self.fold.year)
        g = cw + self.gbt.predict(fx)
        r = cw + self.ridge.predict(self.scaler.transform(fx.copy()))
        return (g, r)

    def describe(self):
        return {'rows': self.n_rows, 'features': list(self.feat.names), 'n_features': len(self.feat.names), 'weekday_index': self.feat.weekday_index, 'is_categorical_true_at': self.categorical, 'random_state': self.seed, 'options': dict(GBT_OPTIONS), 'nox_parts': self.feat.nox_parts, 'n_iter': int(self.gbt.n_iter_)}

def predict_gbt(fold, fit, work_dir):
    wg = common.PredWriter(work_dir, 'GBT', fold)
    wr = common.PredWriter(work_dir, 'GBT-RIDGE', fold)
    for k, (y, i, d) in enumerate(fold.held_days):
        g, r = fit.predict(y, i, d)
        wg.write(k, g)
        wr.write(k, r)
    extra = {'fit': fit.describe(), 'design': 'the study design'}
    return (wg.close(extra), wr.close(extra))
