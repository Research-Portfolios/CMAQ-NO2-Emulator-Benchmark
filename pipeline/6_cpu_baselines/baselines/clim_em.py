"""CLIM-EM, the emission-scaled climatology: the weekday climatology times the ratio of the day's mean NOx emission over the 9 x 9 cells around each cell to its climatological value fitted on the training days."""
import numpy as np
from baselines import clim, common, gbt
B_EXP = 1.0
NOX_FORM = 'gbt'
E9_CLIM_FORM = 'clim_wd'
BOX = 9

def nox_parts(fold):
    adv = fold.configs.channel_set('advanced')
    parts = gbt.NOX_PARTS if NOX_FORM == 'gbt' else ('E-1', 'E-2', 'E-3')
    got = [n for n in parts if n in adv]
    if not got:
        raise ValueError('CLIM-EM: no NOx channel in the advanced set')
    return got

def e9_cells(fold, y, i, mode, parts):
    g = fold.channels_grid(y, i, parts, mode)
    nox = sum((g[k] for k in range(len(parts))))
    return gbt.box_mean_clipped(nox, BOX).ravel()[fold.cells]

class E9View:

    def __init__(self, fold, parts):
        self.fold = fold
        self.cells = fold.cells
        self.train_days = fold.train_days
        self.year = fold.year
        self.parts = parts

    def target_cells(self, y, i):
        return e9_cells(self.fold, y, i, 'train', self.parts)

class ClimEM:

    def __init__(self, fold, clim_all, note=None):
        self.fold = fold
        self.clim_all = clim_all
        self.parts = nox_parts(fold)
        self.e9 = clim.ClimFit(E9View(fold, self.parts), note=note)
        self.ratio_one = 0

    def e9_clim(self, d):
        if E9_CLIM_FORM == 'clim_wd':
            return self.e9.clim_wd(d)
        if E9_CLIM_FORM == 'clim_doy':
            return self.e9.doy[common.calendar_position(d) - 1]
        if E9_CLIM_FORM == 'mean':
            return self.e9.clim_c
        raise ValueError('E9_CLIM_FORM %r' % (E9_CLIM_FORM,))

    def predict(self, d, e9_day):
        den = np.asarray(self.e9_clim(d), dtype=np.float64)
        ok = np.isfinite(den) & (den > 0)
        self.ratio_one += int((~ok).sum())
        ratio = np.where(ok, e9_day / np.where(ok, den, 1.0), 1.0)
        return self.clim_all.clim_wd(d) * ratio ** B_EXP

    def describe(self):
        return {'form': 'CLIM-WD_all x (E9_d / E9_clim)^%g' % B_EXP, 'nox_form': NOX_FORM, 'nox_parts': self.parts, 'e9': 'the %d x %d box mean of the NOx sum, cells inside the grid (gbt.box_mean_clipped)' % (BOX, BOX), 'e9_clim_form': E9_CLIM_FORM, 'e9_fit': self.e9.describe(), 'ratio_set_to_1': self.ratio_one, 'decided': 'NOX_FORM gbt and E9_CLIM_FORM clim_wd, the ratio 1 where E9_clim is not above 0, counted', 'sources': 'the study design'}
