"""AN-02: for each training and fold, the per-cell range of each emission input over the training days, and the share of test cell-days with an emission input outside its cell's range, by held-out year and urban class."""
import numpy as np
from train import cfg
EMISSION = tuple(cfg.EMISSION_CHANNELS)
MODEL_SETS = {'P': 'baseline', 'A1': 'baseline'}
DEFAULT_SET = 'advanced'
C7_MODELS = ('M-C7',)
CHUNK = 64

def channels_of(sources, model):
    names = sources.channel_set(MODEL_SETS.get(model, DEFAULT_SET))
    ch = [n for n in names if n in EMISSION]
    return ('c7' if model in C7_MODELS else 'main', ch)

def _rows(sources, fold, days, name, purpose, source, opening):
    if source == 'c7':
        order, _key = sources.f6a()
        if name in order:
            return sources.c7_rows(fold, days, name, purpose, opening)
    return sources.channel_rows(fold, days, name, purpose, opening)

def ranges(sources, fold, source, channels):
    days = sources.days(fold, 'TRAIN')
    n = len(sources.cells)
    mn = np.full((len(channels), n), np.inf, dtype=np.float32)
    mx = np.full((len(channels), n), -np.inf, dtype=np.float32)
    for j, name in enumerate(channels):
        for a in range(0, len(days), CHUNK):
            v = _rows(sources, fold, days[a:a + CHUNK], name, 'stat', source, None)
            if not np.all(np.isfinite(v)):
                raise ValueError('AN-02: %s has non-finite stored values on TRAIN days of fold %d' % (name, fold))
            np.minimum(mn[j], v.min(axis=0), out=mn[j])
            np.maximum(mx[j], v.max(axis=0), out=mx[j])
    return (mn, mx, len(days))

def flags(sources, fold, days, purpose, source, channels, mn, mx, opening=None, counts=None):
    out = np.zeros((len(days), len(sources.cells)), dtype=bool)
    for j, name in enumerate(channels):
        v = _rows(sources, fold, days, name, purpose, source, opening)
        if not np.all(np.isfinite(v)):
            raise ValueError('AN-02: %s has non-finite stored values on %s days of fold %d' % (name, purpose, fold))
        o = (v < mn[j][None, :]) | (v > mx[j][None, :])
        if counts is not None:
            counts[name] = int(o.sum())
        out |= o
    return out

def share(fl, urban, urban_names):
    n = int(fl.size)
    out = {'n_cell_days': n, 'n_outside': int(fl.sum()), 'share': float(fl.sum()) / n if n else None, 'by_urban': {}}
    for k, name in enumerate(urban_names):
        s = fl[:, urban == k]
        out['by_urban'][name] = {'n_cell_days': int(s.size), 'n_outside': int(s.sum()), 'share': float(s.sum()) / s.size if s.size else None}
    return out

def mc7_check(fl, dates, cells):
    if not fl.any():
        return {'status': 'PASS', 'rule': 'share 0 by construction'}
    d, c = np.nonzero(fl)
    return {'status': 'FAIL', 'rule': 'share 0 by construction: a nonzero value is a FAIL of the use of the emission climatology, examined before any reading', 'n_cell_days': int(d.size), 'cell_days': [[str(dates[int(i)]), int(cells[int(j)])] for i, j in zip(d, c)]}
