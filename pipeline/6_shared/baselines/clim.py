"""CLIM-C (each cell's mean target over the training days) and CLIM-WD, the weekday climatology: a day-of-year mean with a 61-day triangular weight times a per-cell weekday factor, every sum in float64, clipped at 0."""
import numpy as np
from baselines import common
N_POS = 366
HALF = 31

def circ_dist(p, q):
    d = abs(int(p) - int(q)) % N_POS
    return min(d, N_POS - d)

def tri_weight(dd):
    return max(0.0, 1.0 - abs(float(dd)) / HALF)

class ClimFit:

    def __init__(self, fold, note=None):
        common.memory_check(8.0 * len(fold.cells) * (N_POS + 20) / 1000000000.0, 'CLIM fit')
        nc = len(fold.cells)
        s = np.zeros((N_POS, nc), dtype=np.float64)
        wsum = np.zeros(N_POS, dtype=np.float64)
        tot = np.zeros(nc, dtype=np.float64)
        n = 0
        pos_of = []
        wd_of = []
        for k, (y, i, d) in enumerate(fold.train_days):
            yt = fold.target_cells(y, i)
            p = common.calendar_position(d)
            pos_of.append(p)
            wd_of.append(common.weekday(d))
            tot += yt
            n += 1
            for off in range(-(HALF - 1), HALF):
                w = tri_weight(off)
                q = (p - 1 + off) % N_POS
                s[q] += w * yt
                wsum[q] += w
            if note and (k + 1) % 200 == 0:
                note('CLIM fold %d: %d of %d training days' % (fold.year, k + 1, len(fold.train_days)))
        if n == 0:
            raise ValueError('no training days')
        self.clim_c = tot / n
        self.n_pos_empty = int((wsum <= 0).sum())
        safe = np.where(wsum > 0, wsum, 1.0)
        self.doy = np.where((wsum > 0)[:, None], s / safe[:, None], np.nan)
        num = np.zeros((7, nc), dtype=np.float64)
        den = np.zeros((7, nc), dtype=np.float64)
        for (y, i, d), p, wd in zip(fold.train_days, pos_of, wd_of):
            num[wd] += fold.target_cells(y, i)
            den[wd] += self.doy[p - 1]
        self.r = np.where(den < 1e-06, 1.0, num / np.where(den < 1e-06, 1.0, den))
        self.n_r_one = int((den < 1e-06).sum())
        self.n_train = n

    def clim_wd(self, date):
        p = common.calendar_position(date)
        return self.doy[p - 1] * self.r[common.weekday(date)]

    def describe(self):
        return {'training_days': self.n_train, 'weekday_factor_set_to_1': self.n_r_one, 'calendar_positions_without_training_day': self.n_pos_empty, 'calendar_positions': N_POS, 'triangle_half_width_days': HALF}

def predict_clim(fold, fit, work_dir, which):
    w = common.PredWriter(work_dir, which, fold)
    for k, (y, i, d) in enumerate(fold.held_days):
        fold.split.guard(y, i, 'predict')
        row = fit.clim_c if which == 'CLIM-C' else fit.clim_wd(d)
        w.write(k, row)
    return w.close({'fit': fit.describe(), 'design': 'the study design'})
