"""MET-REG, the weather regression: for each cell a ridge regression of ln(y + 0.1) - ln(C + 0.1) on 14 standardized meteorological predictors, with C the weekday climatology, returned to ppb with a smearing factor and clipped at 0."""
import numpy as np
from baselines import common
CHANNELS = ('M-8', 'M-3', 'M-4', 'M-5', 'M-2', 'M-6', 'M-7', 'M-9', 'X-1', 'X-2', 'X-3', 'X-4', 'X-5', 'X-6')
PREDICTORS = ('ln(M-8)', 'ln(M-3 + 0.5)', 'M-4', 'M-5', 'M-2', 'M-6', 'M-7', 'ln(1 + M-9 / 0.01)', 'ln(X-1)', 'ln(X-2 + 0.5)', 'X-3', 'X-4', 'X-5', 'ln(X-6)')
PENALTY = 1.0

class NonFinitePredictor(Exception):
    pass

def predictors(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.empty_like(v)
        x[0] = np.log(v[0])
        x[1] = np.log(v[1] + 0.5)
        x[2:7] = v[2:7]
        x[7] = np.log(1.0 + v[7] / 0.01)
        x[8] = np.log(v[8])
        x[9] = np.log(v[9] + 0.5)
        x[10:13] = v[10:13]
        x[13] = np.log(v[13])
    return x

def _z(y, clim_wd):
    return np.log(y + 0.1) - np.log(clim_wd + 0.1)

class MetRegFit:

    def __init__(self, fold, clim, note=None):
        nc = len(fold.cells)
        p = len(CHANNELS)
        common.memory_check(8.0 * nc * (p * p + 4 * p + 8) / 1000000000.0 + 0.3, 'MET-REG fit')
        su = np.zeros((p, nc))
        suu = np.zeros((nc, p, p))
        sz = np.zeros(nc)
        suz = np.zeros((p, nc))
        n = 0
        shift = None
        bad = 0
        for k, (y, i, d) in enumerate(fold.train_days):
            x = predictors(fold.channels_cells(y, i, CHANNELS, 'train'))
            z = _z(fold.target_cells(y, i), clim.clim_wd(d))
            nb = int((~np.isfinite(x)).sum() + (~np.isfinite(z)).sum())
            if nb:
                bad += nb
                continue
            if shift is None:
                shift = x.copy()
            u = x - shift
            su += u
            suu += np.einsum('ic,jc->cij', u, u)
            sz += z
            suz += u * z
            n += 1
            if note and (k + 1) % 200 == 0:
                note('MET-REG fold %d: %d of %d training days' % (fold.year, k + 1, len(fold.train_days)))
        if bad:
            raise NonFinitePredictor('MET-REG fold %d: %d non-finite predictor or z values on the training days; nothing substituted' % (fold.year, bad))
        self.n = n
        mu_u = su / n
        cov = suu / n - np.einsum('ic,jc->cij', mu_u, mu_u)
        sd = np.sqrt(np.maximum(np.einsum('cii->ic', cov), 0.0))
        self.n_const = int((sd == 0).sum())
        sd = np.where(sd > 0, sd, 1.0)
        zbar = sz / n
        cuz = suz / n - mu_u * zbar
        g = n * cov / (sd.T[:, :, None] * sd.T[:, None, :])
        c = n * (cuz / sd).T
        a = g + PENALTY * np.eye(p)[None]
        self.beta = np.linalg.solve(a, c[:, :, None])[:, :, 0].T
        self.shift = shift
        self.mu = mu_u + shift
        self.sd = sd
        self.zbar = zbar
        se = np.zeros(nc)
        for y, i, d in fold.train_days:
            x = predictors(fold.channels_cells(y, i, CHANNELS, 'train'))
            z = _z(fold.target_cells(y, i), clim.clim_wd(d))
            se += np.exp(z - self.zhat(x))
        self.s = se / n

    def zhat(self, x):
        return self.zbar + np.sum(self.beta * (x - self.mu) / self.sd, axis=0)

    def predict(self, x, clim_wd):
        return (clim_wd + 0.1) * np.exp(self.zhat(x)) * self.s - 0.1

    def describe(self):
        return {'training_days': self.n, 'predictors': list(PREDICTORS), 'channels': list(CHANNELS), 'penalty': PENALTY, 'constant_predictor_cells': self.n_const, 'standardization': 'per cell, population sd; sd 0 -> 1'}

def predict_metreg(fold, clim, fit, work_dir):
    w = common.PredWriter(work_dir, 'MET-REG', fold)
    bad = 0
    for k, (y, i, d) in enumerate(fold.held_days):
        x = predictors(fold.channels_cells(y, i, CHANNELS, 'predict'))
        bad += int((~np.isfinite(x)).sum())
        if bad:
            raise NonFinitePredictor('MET-REG fold %d: non-finite predictors on a held-out day; nothing substituted' % fold.year)
        w.write(k, fit.predict(x, clim.clim_wd(d)))
    return w.close({'fit': fit.describe(), 'design': 'the study design'})
