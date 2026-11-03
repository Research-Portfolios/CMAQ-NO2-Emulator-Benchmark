"""The metrics, strata and diagnostics of the evaluation (n, MB, MAE, RMSE, NMB, NME, R, R2, IOA, the skill score, the anomaly correlation, the map RMSE, deciles, exceedance scores, the spread ratio, the spatial spectrum, the weekday contrast and others), every sum in float64; its docstring defines each one."""
import math
import numpy as np
from baselines import common as BC
from choi import metrics as G
SEASONS = ('winter', 'spring', 'summer', 'fall')
URBAN = ('urban', 'intermediate', 'rural')
COAST = ('edge', 'coast', 'border', 'interior')
EDGE_CELLS = 32
BORDER_CELLS = 5
URBAN_HI = 0.5
URBAN_LO = 0.1
LEVELS = (13.3, 26.6, 63.83)
MIN_EVENTS = 100
ABOVE = 30.0
NEAR_LO = 1.0
NEAR_HI = 10.0
WINDOW = 64
STRIDE = 32
CELL_KM = 12.0
BANDS_KM = ((24.0, 48.0), (48.0, 96.0), (96.0, 192.0), (192.0, 768.0))
MIN_WINDOWS = 4
ACF_LAGS = tuple(range(1, 15))
ACF_BELOW = 0.2
EXCEED_BINS = ((0.0, 0.1), (0.1, 0.25), (0.25, None))
INTERIOR_FOLDS = (2016, 2017, 2018)
FORWARD_FOLD = 2019

def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None

def as64(a):
    return np.asarray(a, dtype=np.float64)

def basic(e, o, c=None):
    m = G.all_metrics(e, o)
    r = m['R']
    out = {'n': int(m['n']), 'MB': _f(m['ME']), 'MAE': _f(m['MAE']), 'RMSE': _f(m['RMSE']), 'NMB': _f(m['NMB']), 'NME': _f(m['NME']), 'R': _f(r), 'R2': _f(m['R2']), 'r_squared': _f(r * r) if math.isfinite(r) else None, 'IOA': _f(m['IOA'])}
    if c is not None:
        out['SS'] = _f(G.ss(e, o, c))
    return out

def sse_per_day(e, o):
    d = as64(e) - as64(o)
    return np.einsum('ij,ij->i', d, d)

def top_decile_mask(o):
    o = as64(o)
    edges = np.percentile(o.ravel(), np.arange(10, 100, 10))
    return o >= edges[-1]

def class_day_sums(e, o, urban, coast, top=None):
    d = as64(e) - as64(o)
    np.multiply(d, d, out=d)
    nd = d.shape[0]
    out = {}
    for kind, cls, names in (('urban', urban, URBAN), ('coast', coast, COAST)):
        cls = np.asarray(cls)
        for k, name in enumerate(names):
            sel = np.flatnonzero(cls == k)
            out[kind, name] = (d[:, sel].sum(axis=1), np.full(nd, float(sel.size)))
    if top is not None:
        t = np.asarray(top, dtype=bool)
        out['tail', 'CMAQ top decile'] = (np.where(t, d, 0.0).sum(axis=1), t.sum(axis=1).astype(np.float64))
    return out

def pearson_cells(a, b):
    a = as64(a)
    b = as64(b)
    da = a - a.mean(axis=0)
    db = b - b.mean(axis=0)
    num = np.einsum('ij,ij->j', da, db)
    den = np.sqrt(np.einsum('ij,ij->j', da, da) * np.einsum('ij,ij->j', db, db))
    out = np.full(a.shape[1], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out

def acc_cells(e, o, c):
    c = as64(c)
    return pearson_cells(as64(e) - c, as64(o) - c)

def r_time_cells(e, o):
    return pearson_cells(e, o)

def summary(v, pct=(10, 50, 90)):
    v = as64(v)
    ok = np.isfinite(v)
    out = {'n_cells': int(v.size), 'n_undefined': int((~ok).sum())}
    names = {10: 'p10', 25: 'p25', 50: 'median', 75: 'p75', 90: 'p90'}
    if ok.any():
        q = np.percentile(v[ok], pct)
        for p, x in zip(pct, q):
            out[names[p]] = _f(x)
    else:
        for p in pct:
            out[names[p]] = None
    return out

def r_space(e, o):
    return _f(G.r(as64(e).mean(axis=0), as64(o).mean(axis=0)))

def maprmse(e, o):
    d = as64(e).mean(axis=0) - as64(o).mean(axis=0)
    return float(np.sqrt(np.mean(d * d)))

def skill_cells(e, o, c):
    o = as64(o)
    de = as64(e) - o
    dc = as64(c) - o
    num = np.einsum('ij,ij->j', de, de)
    den = np.einsum('ij,ij->j', dc, dc)
    out = np.full(o.shape[1], np.nan)
    ok = den > 0
    out[ok] = 1.0 - num[ok] / den[ok]
    return out

def share_le0(v):
    v = as64(v)
    ok = np.isfinite(v)
    n = int(ok.sum())
    return {'share': _f(np.sum(v[ok] <= 0) / n) if n else None, 'n_cells': n, 'n_undefined': int((~ok).sum())}

def murphy_terms(e, o, c):
    f = as64(e).ravel()
    x = as64(o).ravel()
    mu = as64(c).ravel()
    sx = x.std()

    def terms(g):
        sg = g.std()
        if not (sx > 0 and sg > 0):
            return None
        r = float(np.mean((g - g.mean()) * (x - x.mean())) / (sg * sx))
        return {'A_r2': r * r, 'B_conditional_bias': (r - sg / sx) ** 2, 'C_unconditional_bias': ((g.mean() - x.mean()) / sx) ** 2, 'r': r, 's_ratio': float(sg / sx), 'mean_diff_over_sx': float((g.mean() - x.mean()) / sx)}
    tf, tm = (terms(f), terms(mu))
    out = {'model': tf, 'climatology': tm, 'form': 'Murphy 1988 eq. 15 (Case IV), population moments over the pooled cell-days'}
    mse_f = float(np.mean((f - x) ** 2))
    mse_m = float(np.mean((mu - x) ** 2))
    out['SS_direct'] = _f(G.ss_from_mse(mse_f, mse_m))
    if tf is None or tm is None:
        out['SS_eq15'] = None
        out['identity_gap'] = None
        return out
    den = 1.0 - tm['A_r2'] + tm['B_conditional_bias'] + tm['C_unconditional_bias']
    num = tf['A_r2'] - tf['B_conditional_bias'] - tf['C_unconditional_bias'] - (tm['A_r2'] - tm['B_conditional_bias'] - tm['C_unconditional_bias'])
    out['SS_eq15'] = _f(num / den) if den > 0 else None
    out['identity_gap'] = _f(abs(out['SS_eq15'] - out['SS_direct'])) if out['SS_eq15'] is not None and out['SS_direct'] is not None else None
    return out

def _errs(e, o):
    n = int(e.size)
    if n == 0:
        return {'n': 0, 'MB': None, 'MAE': None, 'RMSE': None}
    d = e - o
    return {'n': n, 'MB': _f(d.mean()), 'MAE': _f(np.abs(d).mean()), 'RMSE': _f(np.sqrt(np.mean(d * d)))}

def deciles(e, o):
    e = as64(e).ravel()
    o = as64(o).ravel()
    edges = np.percentile(o, np.arange(10, 100, 10))
    k = np.searchsorted(edges, o, side='right')
    rows = []
    for j in range(10):
        s = k == j
        r = _errs(e[s], o[s])
        r.update({'decile': j + 1, 'lower': None if j == 0 else _f(edges[j - 1]), 'upper': None if j == 9 else _f(edges[j])})
        rows.append(r)
    return {'rows': rows, 'rule': 'edges: numpy linear percentiles 10 to 90 of CMAQ over the pooled cell-days; decile k holds edges[k-1] <= O < edges[k]', 'expected': "MB below 0 in decile 10 for every model; M's R above 30 ppb exceeds P's (the study design D1)"}

def own_top_decile(e, o):
    e = as64(e).ravel()
    o = as64(o).ravel()
    thr = float(np.percentile(e, 90))
    s = e >= thr
    r = _errs(e[s], o[s])
    r['overall_MB'] = _f((e - o).mean())
    r['rule'] = "cell-days with the model's value at or above its own 90th percentile (numpy linear) over the pooled cell-days"
    return r

def above_level(e, o, level=ABOVE):
    e = as64(e).ravel()
    o = as64(o).ravel()
    s = o > level
    n = int(s.sum())
    out = {'level_ppb': level, 'n': n, 'R': None, 'IOA': None, 'expected': "M's R above 30 ppb exceeds P's (the study design D1)"}
    if n >= 2:
        out['R'] = _f(G.r(e[s], o[s]))
        out['IOA'] = _f(G.ioa(e[s], o[s]))
    return out

def exceedance(e, o, level, min_events=MIN_EVENTS):
    eo = as64(o).ravel() > level
    ee = as64(e).ravel() > level
    a = int(np.sum(ee & eo))
    b = int(np.sum(ee & ~eo))
    c = int(np.sum(~ee & eo))
    n = int(eo.size)
    out = {'level_ppb': level, 'hits_a': a, 'false_alarms_b': b, 'misses_c': c, 'n': n, 'cmaq_events': a + c, 'model_events': a + b, 'scored': a + c >= min_events, 'POD': None, 'FAR': None, 'CSI': None, 'ETS': None}
    if not out['scored']:
        return out
    out['POD'] = _f(a / (a + c))
    out['FAR'] = _f(b / (a + b)) if a + b else None
    out['CSI'] = _f(a / (a + b + c))
    alpha = (a + b) * (a + c) / float(n)
    den = a + b + c - alpha
    out['ETS'] = _f((a - alpha) / den) if den != 0 else None
    return out

def urban_class(purb):
    p = np.asarray(purb).astype(np.float32)
    hi = np.float32(np.float16(URBAN_HI))
    lo = np.float32(np.float16(URBAN_LO))
    if not np.all(np.isfinite(p)):
        raise ValueError('L-3 has non-finite values at masked cells')
    out = np.full(p.shape, 2, dtype=np.int8)
    out[p >= lo] = 1
    out[p >= hi] = 0
    return out

def coast_class(mask, lwmask, edge=EDGE_CELLS, border=BORDER_CELLS, tests=None):
    m = np.asarray(mask, dtype=bool)
    lw = np.asarray(lwmask).astype(np.float32)
    if lw.shape != m.shape:
        raise ValueError('LWMASK %s, mask %s' % (lw.shape, m.shape))
    h, w = m.shape
    water = lw < 0.5
    land_out = ~water & ~m
    pw = np.pad(water, 1, constant_values=False)
    nw = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nw |= pw[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    rb = int(border)
    pl = np.pad(land_out, rb, constant_values=False)
    nl = np.zeros((h, w), dtype=bool)
    for dr in range(-rb, rb + 1):
        for dc in range(-rb, rb + 1):
            if dr * dr + dc * dc <= rb * rb:
                nl |= pl[rb + dr:rb + dr + h, rb + dc:rb + dc + w]
    rr, cc = np.indices((h, w))
    ed = (rr < edge) | (rr >= h - edge) | (cc < edge) | (cc >= w - edge)
    cls = np.full((h, w), 3, dtype=np.int8)
    cls[nl] = 2
    cls[nw] = 1
    cls[ed] = 0
    out = cls.ravel()[np.flatnonzero(m.ravel())]
    if tests is not None:
        for name, t in (('edge', ed), ('coast', nw), ('border', nl)):
            tests[name] = int((t & m).sum())
        for k, name in enumerate(COAST):
            tests['class_' + name] = int((out == k).sum())
    return out

def season_index(dates):
    return np.array([SEASONS.index(BC.season(d)) for d in dates], dtype=np.int8)

def weekdays(dates):
    return np.array([BC.weekday(d) for d in dates], dtype=np.int8)

def strata_rows(e, o, c, dates, urban, coast):
    e = as64(e)
    o = as64(o)
    c = None if c is None else as64(c)
    si = season_index(dates)
    rows = []

    def add(kind, name, rs, cs):
        if not rs.any() or not cs.any():
            rows.append({'stratum_kind': kind, 'stratum': name, 'n': 0, 'cells': int(cs.sum()), 'status': 'NOT RUN', 'reason': 'no cell-day'})
            return
        sub = np.ix_(np.flatnonzero(rs), np.flatnonzero(cs))
        r = basic(e[sub], o[sub], None if c is None else c[sub])
        r.update({'stratum_kind': kind, 'stratum': name, 'status': 'RUN', 'cells': int(cs.sum())})
        rows.append(r)
    alld = np.ones(e.shape[0], dtype=bool)
    allc = np.ones(e.shape[1], dtype=bool)
    add('all', 'all', alld, allc)
    for k, s in enumerate(SEASONS):
        add('season', s, si == k, allc)
    for k, u in enumerate(URBAN):
        add('urban', u, alld, urban == k)
    for k, s in enumerate(COAST):
        add('coast', s, alld, coast == k)
    for k, s in enumerate(SEASONS):
        for j, u in enumerate(URBAN):
            add('season_x_urban', '%s|%s' % (s, u), si == k, urban == j)
    return rows

def _err_rel(e, o):
    r = _errs(e, o)
    so = float(o.sum()) if o.size else 0.0
    r['NME'] = _f(np.abs(e - o).sum() / so) if so else None
    return r

def d_rows(e, o, dates, urban, coast, pblh=None):
    e = as64(e)
    o = as64(o)
    si = season_index(dates)
    out = {'D4': [], 'D5': [], 'D6': [], 'D7': []}
    for k, s in enumerate(SEASONS):
        for j, u in enumerate(URBAN):
            sub = np.ix_(np.flatnonzero(si == k), np.flatnonzero(urban == j))
            ee, oo = (e[sub].ravel(), o[sub].ravel())
            r = _err_rel(ee, oo)
            r.update({'season': s, 'urban': u})
            out['D4'].append(r)
            out['D7'].append({'season': s, 'urban': u, 'n': int(ee.size), 'mean_residual_cmaq_minus_model': _f((oo - ee).mean()) if ee.size else None})
    for k, s in enumerate(COAST):
        ee, oo = (e[:, coast == k].ravel(), o[:, coast == k].ravel())
        r = _err_rel(ee, oo)
        r['class'] = s
        r['cells'] = int(np.sum(coast == k))
        out['D5'].append(r)
    if pblh is None:
        out['D6'] = {'status': 'NOT RUN', 'reason': 'M-8 not read'}
    else:
        p = as64(pblh).ravel()
        if not np.all(np.isfinite(p)):
            raise ValueError('M-8 has non-finite values on test cell-days')
        edges = np.percentile(p, np.arange(10, 100, 10))
        k = np.searchsorted(edges, p, side='right')
        ee, oo = (e.ravel(), o.ravel())
        for j in range(10):
            s = k == j
            r = _errs(ee[s], oo[s])
            r['pblh_decile'] = j + 1
            out['D6'].append(r)
    out['expected'] = {'D4': "largest relative error in rural winter cells; M's rural winter error below P's", 'D5': 'larger at coast and edge than interior for every model', 'D6': 'largest at the lowest PBLH decile', 'D7': 'positive in rural summer cells'}
    return out

def d7_maps(e, o, dates):
    e = as64(e)
    o = as64(o)
    si = season_index(dates)
    out = np.full((len(SEASONS), e.shape[1]), np.nan, dtype=np.float32)
    for k in range(len(SEASONS)):
        s = si == k
        if s.any():
            out[k] = (o[s] - e[s]).mean(axis=0).astype(np.float32)
    return out

def d8_rows(e, o, tmax_cell, tmax_domain, flagged=None, flagged_norm=None, norm_info=None):
    e = as64(e)
    o = as64(o)
    tm = np.broadcast_to(as64(tmax_cell)[None, :], o.shape)
    over = o > tm
    pos = tm > 0
    rel = np.full(o.shape, np.inf)
    rel[pos] = (o[pos] - tm[pos]) / tm[pos]
    out = {'cell_max': [], 'domain_max': None, 'inputs_outside': None, 'inputs_outside_norm': None, 'rule': "exceedance (O - max) / max of the cell's TRAIN-day maximum; a maximum not above 0 gives an infinite exceedance (the last bin); bins (0, 0.10], (0.10, 0.25], above 0.25"}
    for lo, hi in EXCEED_BINS:
        s = over & (rel > lo) & (rel <= (np.inf if hi is None else hi))
        ee, oo = (e[s], o[s])
        r = _errs(ee, oo)
        r['ratio_sum_e_over_o'] = _f(ee.sum() / oo.sum()) if oo.size and oo.sum() > 0 else None
        r['bin'] = '%g-%s' % (100 * lo, '%g percent' % (100 * hi) if hi is not None else 'up')
        out['cell_max'].append(r)
    s = over
    r = _errs(e[s], o[s])
    r['bin'] = 'all'
    r['ratio_sum_e_over_o'] = _f(e[s].sum() / o[s].sum()) if s.any() and o[s].sum() > 0 else None
    out['cell_max'].append(r)
    sd = o > float(tmax_domain)
    r = _errs(e[sd], o[sd])
    r['ratio_sum_e_over_o'] = _f(e[sd].sum() / o[sd].sum()) if sd.any() and o[sd].sum() > 0 else None
    out['domain_max'] = r
    if flagged_norm is not None:
        f = np.asarray(flagged_norm, dtype=bool)
        r = _errs(e[f], o[f])
        r['n_days'] = int(f.any(axis=1).sum())
        r['inputs'] = "each input channel of the model's set held in the arrays, outside its [min, max] over all cells of the fold's TRAIN days (PROP 4.3; READING)"
        r.update(norm_info or {})
        out['inputs_outside_norm'] = r
    if flagged is not None:
        f = np.asarray(flagged, dtype=bool)
        out['inputs_outside'] = _errs(e[f], o[f])
        out['inputs_outside']['n_days'] = int(f.any(axis=1).sum())
        out['inputs_outside']['inputs'] = 'emission channels, per-cell range (AN-02 (i))'
    out['expected'] = 'under-prediction there (the study design D8)'
    return out

def nearzero(e, o, coast):
    e = as64(e)
    o = as64(o)
    s = (e < NEAR_LO) & (o > NEAR_HI)
    per_cell = s.sum(axis=0).astype(np.int32)
    by = {name: int(per_cell[coast == k].sum()) for k, name in enumerate(COAST)}
    return ({'count': int(s.sum()), 'by_class': by, 'rule': 'prediction below %g ppb where CMAQ is above %g ppb' % (NEAR_LO, NEAR_HI), 'expected': 'near none for every model'}, per_cell)

def spread(e, o):
    e = as64(e)
    o = as64(o)
    se = e.std(axis=0)
    so = o.std(axis=0)
    ratio = np.full(e.shape[1], np.nan)
    ok = so > 0
    ratio[ok] = se[ok] / so[ok]
    out = summary(ratio, (25, 50, 75))
    out['median_r_time_same_cells'] = summary(np.where(ok, r_time_cells(e, o), np.nan), (50,))['median']
    out['expected'] = 'below 1 for all; lower for the MSE networks than for P at equal network (A1 against P) (the study design D3)'
    return out

def windows(mask, size=WINDOW, stride=STRIDE):
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    pos = np.full(h * w, -1, dtype=np.int64)
    flat = np.flatnonzero(m.ravel())
    pos[flat] = np.arange(flat.size)
    pos = pos.reshape(h, w)
    corners, idx = ([], [])
    for r0 in range(0, h - size + 1, stride):
        for c0 in range(0, w - size + 1, stride):
            if m[r0:r0 + size, c0:c0 + size].all():
                corners.append((r0, c0))
                idx.append(pos[r0:r0 + size, c0:c0 + size].ravel())
    return (corners, np.array(idx, dtype=np.int64) if idx else np.zeros((0, size * size), dtype=np.int64))

def _radial(size=WINDOW):
    k = np.fft.fftfreq(size) * size
    rad = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    return np.rint(rad).astype(np.int64)

def spectrum_power(x, widx, size=WINDOW):
    x = as64(x)
    rad = _radial(size).ravel()
    nb = int(rad.max()) + 1
    taper = np.outer(np.hanning(size), np.hanning(size))
    ps = np.zeros(nb)
    cnt = np.zeros(nb)
    nw = widx.shape[0]
    for row in x:
        win = row[widx].reshape(nw, size, size)
        win = win - win.mean(axis=(1, 2), keepdims=True)
        p = np.abs(np.fft.fft2(win * taper, axes=(1, 2))) ** 2
        ps += np.bincount(rad, weights=p.reshape(nw, -1).sum(axis=0), minlength=nb)
        cnt += np.bincount(rad, minlength=nb) * nw
    return (ps, cnt)

def band_of(m, size=WINDOW):
    if m < 1:
        return None
    lam = size * CELL_KM / m
    for k, (lo, hi) in enumerate(BANDS_KM):
        last = k == len(BANDS_KM) - 1
        if lo <= lam < hi or (last and lam == hi):
            return k
    return None

def band_power(ps, cnt, size=WINDOW):
    out = [0.0] * len(BANDS_KM)
    for m in range(len(ps)):
        k = band_of(m, size)
        if k is not None and cnt[m] > 0:
            out[k] += ps[m] / cnt[m]
    return out

def band_ratios(model_bands, cmaq_bands):
    rows = []
    for (lo, hi), a, b in zip(BANDS_KM, model_bands, cmaq_bands):
        rows.append({'band_km': '%g-%g' % (lo, hi), 'ratio_model_over_cmaq': _f(a / b) if b > 0 else None})
    return rows

def residual_acf(e, o, dates, sel=None, lags=ACF_LAGS):
    import datetime
    r = as64(e) - as64(o)
    t = np.array([datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])).toordinal() for d in dates])
    if sel is not None:
        sel = np.asarray(sel, dtype=bool)
        r = r[sel]
        t = t[sel]
    out = {'n_days': int(r.shape[0]), 'lags': [], 'first_below_0_2': None}
    if r.shape[0] < 3:
        out['status'] = 'NOT RUN'
        return out
    rc = r - r.mean(axis=0)
    var = np.mean(rc * rc, axis=0)
    where = {int(x): k for k, x in enumerate(t)}
    for lag in lags:
        a = [k for k, x in enumerate(t) if int(x) + lag in where]
        if not a:
            out['lags'].append({'lag': lag, 'pairs': 0, 'median': None})
            continue
        b = [where[int(t[k]) + lag] for k in a]
        num = np.mean(rc[a] * rc[b], axis=0)
        acf = np.full(r.shape[1], np.nan)
        ok = var > 0
        acf[ok] = num[ok] / var[ok]
        med = summary(acf, (50,))['median']
        out['lags'].append({'lag': lag, 'pairs': len(a), 'median': med})
        if out['first_below_0_2'] is None and med is not None and (med < ACF_BELOW):
            out['first_below_0_2'] = lag
    out['status'] = 'RUN'
    return out

def f12_masks(dates):
    wd = weekdays(dates)
    return ((wd >= 1) & (wd <= 4), wd == 6)

def f12_map(x, dates):
    tf, su = f12_masks(dates)
    if not tf.any() or not su.any():
        raise ValueError('F-12: no Tuesday-to-Friday or no Sunday day')
    x = as64(x)
    return x[tf].mean(axis=0) - x[su].mean(axis=0)

def agreement(cm, co):
    cm = as64(cm)
    co = as64(co)
    dco = co - co.mean()
    var = float(np.sum(dco * dco))
    slope = float(np.sum(dco * (cm - cm.mean())) / var) if var > 0 else None
    return {'n_cells': int(cm.size), 'R': _f(G.r(cm, co)), 'slope': _f(slope), 'RMSE': _f(np.sqrt(np.mean((cm - co) ** 2)))}

def rel_change(rmse_by_fold):
    if FORWARD_FOLD not in rmse_by_fold or not all((y in rmse_by_fold for y in INTERIOR_FOLDS)):
        return None
    base = sum((rmse_by_fold[y] for y in INTERIOR_FOLDS)) / len(INTERIOR_FOLDS)
    return _f((rmse_by_fold[FORWARD_FOLD] - base) / base) if base else None

def still_improving(result):
    if not isinstance(result, dict) or 'steps_run' not in result or 'best_step' not in result:
        return None
    last = int(result['steps_run'])
    return bool(not result.get('early_stopped') and int(result['best_step']) == last)

def hist2d(e, o, top=100.0, bins=200):
    e = as64(e).ravel()
    o = as64(o).ravel()
    h, _, _ = np.histogram2d(o, e, bins=bins, range=[[0, top], [0, top]])
    inside = int(h.sum())
    return (h.astype(np.int64), int(e.size) - inside)

def city_series(e, o, block_flat, cells):
    e = as64(e)
    o = as64(o)
    pos = {int(c): k for k, c in enumerate(cells)}
    out, series = ({}, {})
    for city, flat in sorted(block_flat.items()):
        ks = [pos[int(f)] for f in flat if int(f) in pos]
        if not ks:
            out[city] = {'status': 'NOT RUN', 'reason': 'no masked cell'}
            continue
        se = e[:, ks].mean(axis=1)
        so = o[:, ks].mean(axis=1)
        r = basic(se, so)
        r.update({'status': 'RUN', 'cells': len(ks)})
        out[city] = r
        series[city] = (se, so)
    return (out, series)
