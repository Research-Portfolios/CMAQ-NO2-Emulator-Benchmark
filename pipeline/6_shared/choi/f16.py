"""Float16 storage of the arrays: maps pressure from Pa to [0, 1] by (x - 50,000) / 60,000 before the cast and back, casts other channels plainly, and tests a channel's round trip, keeping it in float32 when it has non-finite values or a 99th-percentile relative error above 0.001."""
import warnings
import numpy as np
PRES_OFFSET = 50000.0
PRES_SCALE = 60000.0
REL_ERR_THRESHOLD = 0.001

def pres_to_store(x):
    return ((np.asarray(x, dtype=np.float64) - PRES_OFFSET) / PRES_SCALE).astype(np.float16)

def pres_from_store(y):
    return np.asarray(y, dtype=np.float64) * PRES_SCALE + PRES_OFFSET

def f16_store(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.asarray(x, dtype=np.float64).astype(np.float16)

def f16_load(y):
    return np.asarray(y, dtype=np.float64)

def roundtrip_error(x, store, load, threshold=REL_ERR_THRESHOLD):
    x = np.asarray(x, dtype=np.float64).ravel()
    fin = np.isfinite(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        back = np.asarray(load(store(x)), dtype=np.float64).ravel()
    n_nf_rt = int((~np.isfinite(back) & fin).sum())
    pos = x[fin & (x > 0)]
    out = {'n': int(x.size), 'n_nonfinite_input': int((~fin).sum()), 'n_nonfinite_roundtrip': n_nf_rt, 'n_positive': int(pos.size), 'p1_positive': None, 'n_used': 0, 'p99_rel_error': None, 'max_rel_error': None}
    if pos.size:
        p1 = float(np.percentile(pos, 1))
        use = fin & (x > p1) & np.isfinite(back)
        out['p1_positive'] = p1
        out['n_used'] = int(use.sum())
        if use.any():
            rel = np.abs(back[use] - x[use]) / np.abs(x[use])
            out['p99_rel_error'] = float(np.percentile(rel, 99))
            out['max_rel_error'] = float(rel.max())
    bad = n_nf_rt > 0 or out['p99_rel_error'] is None or out['p99_rel_error'] > threshold
    out['verdict'] = 'float32' if bad else 'float16'
    return out
