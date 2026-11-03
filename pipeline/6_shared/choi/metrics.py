"""The evaluation metrics R, IOA, MAE, mean error, MSE, RMSE, NMB, NME and R2 over paired emulated and CMAQ values, and the skill score against a reference, every sum in float64 with an optional mask; a non-finite pair raises an error."""
import numpy as np

def _pairs(e, o, mask=None):
    e = np.asarray(e, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    if e.shape != o.shape:
        raise ValueError('E and O shapes differ: %s, %s' % (e.shape, o.shape))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != e.shape:
            raise ValueError('mask shape %s differs from %s' % (m.shape, e.shape))
        e = e[m]
        o = o[m]
    else:
        e = e.ravel()
        o = o.ravel()
    if e.size == 0:
        raise ValueError('no pairs')
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(o))):
        raise ValueError('non-finite values among the pairs')
    return (e, o)

def r(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    if np.all(e == e[0]) or np.all(o == o[0]):
        return float('nan')
    de = e - e.mean()
    do = o - o.mean()
    den = np.sqrt(np.sum(de * de)) * np.sqrt(np.sum(do * do))
    if den == 0:
        return float('nan')
    return float(np.sum(de * do) / den)

def ioa(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    obar = o.mean()
    num = np.sum((e - o) ** 2)
    den = np.sum((np.abs(e - obar) + np.abs(o - obar)) ** 2)
    if den == 0:
        return float('nan')
    return float(1.0 - num / den)

def mae(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    return float(np.sum(np.abs(e - o)) / e.size)

def me(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    return float(np.sum(e - o) / e.size)

def mse(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    return float(np.sum((e - o) ** 2) / e.size)

def rmse(e, o, mask=None):
    return float(np.sqrt(mse(e, o, mask)))

def ss(e, o, ref, mask=None):
    m_model = mse(e, o, mask)
    m_ref = mse(ref, o, mask)
    return ss_from_mse(m_model, m_ref)

def ss_from_mse(mse_model, mse_ref):
    if mse_ref == 0:
        return float('nan')
    return float(1.0 - float(mse_model) / float(mse_ref))

def nmb(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    so = np.sum(o)
    if so == 0:
        return float('nan')
    return float(np.sum(e - o) / so)

def nme(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    so = np.sum(o)
    if so == 0:
        return float('nan')
    return float(np.sum(np.abs(e - o)) / so)

def r2(e, o, mask=None):
    e, o = _pairs(e, o, mask)
    d = o - o.mean()
    sst = np.sum(d * d)
    if sst == 0:
        return float('nan')
    return float(1.0 - np.sum((e - o) ** 2) / sst)

def all_metrics(e, o, mask=None):
    ee, oo = _pairs(e, o, mask)
    return {'n': int(ee.size), 'R': r(ee, oo), 'IOA': ioa(ee, oo), 'MAE': mae(ee, oo), 'ME': me(ee, oo), 'RMSE': rmse(ee, oo), 'NMB': nmb(ee, oo), 'NME': nme(ee, oo), 'R2': r2(ee, oo)}
