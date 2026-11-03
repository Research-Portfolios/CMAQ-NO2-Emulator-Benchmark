"""Finds, for each model and fold, the test days with any input outside its training range (the per-channel minimum and maximum over all cells of the training days), for the error on those days."""
import numpy as np
from eval import an02
from eval import io as EIO

def channels_of(sources, model):
    names = sources.channel_set(an02.MODEL_SETS.get(model, an02.DEFAULT_SET))
    src = 'c7' if model in an02.C7_MODELS else 'main'
    order = list(sources.f6a()[0]) if src == 'c7' else []
    held = [n for n in names if n in sources.by_name or n in order]
    missing = [n for n in names if n not in held]
    return (src, held, missing)

def ranges(sources, fold, source, channels):
    days = sources.days(fold, 'TRAIN')
    lo = np.empty(len(channels))
    hi = np.empty(len(channels))
    for j, name in enumerate(channels):
        mn, mx = sources.grid_minmax(fold, days, name, 'stat', source)
        lo[j] = mn.min()
        hi[j] = mx.max()
    return (lo, hi, len(days))

def flags(sources, fold, days, purpose, source, channels, lo, hi, opening=None, counts=None):
    out = np.zeros((len(days), len(sources.cells)), dtype=bool)
    for j, name in enumerate(channels):
        v = an02._rows(sources, fold, days, name, purpose, source, opening)
        if not np.all(np.isfinite(v)):
            raise EIO.Refused('D8: %s has non-finite stored values on %s days of fold %d' % (name, purpose, fold))
        o = (v < lo[j]) | (v > hi[j])
        if counts is not None:
            counts[name] = int(o.sum())
        out |= o
    return out
