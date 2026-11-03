"""The four comparisons fixed in advance, C-1 to C-4: a seeded paired moving-block bootstrap of whole test days within each year, statistics from per-day sums, percentile intervals, Holm's step-down procedure with the margins, the seed rule, the 14-day sensitivity, the exploratory intervals and the held-back-day replication."""
import math
import numpy as np
SEED = 20261016
B_FAMILY = 10000
B_EXPLORATORY = 2000
BLOCK = 7
BLOCK_SENSITIVITY = 14
ALPHA = 0.05
FAMILY = ('C-1', 'C-2', 'C-3', 'C-4')
MARGINS = {'C-1': 0.0, 'C-2': 0.05, 'C-3': 0.05, 'C-4': 0.05}
EXPLORATORY_MARGIN = 0.05
HALF_WIDTH_LIMIT = 0.05
PARTS = {'cv': 0, 'lockbox': 1}
DEFINITIONS = {'C-1': 'SS of M against CLIM-WD (1 - MSE_M / MSE_CLIM-WD)', 'C-2': 'RMSE(GBT) - RMSE(M), ppb', 'C-3': 'RMSE(P) - RMSE(M), ppb', 'C-4': 'MAPRMSE(CLIM-WD) - MAPRMSE(M), ppb'}
REFERENCE = {'C-1': 'CLIM-WD', 'C-2': 'GBT', 'C-3': 'P', 'C-4': 'CLIM-WD'}
COVERS = "the sampling of test days given the trained models and the fixed inputs; not training randomness (the seed rule reads it), input error, or CMAQ's error against the atmosphere (outside the claims)"
NOT_SEPARATED = 'not separated from training randomness'
SEPARATED = 'separated from training randomness'

def rng_for(year, block, n_res, part='cv'):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, int(year), int(block), int(n_res), PARTS[part]])))

def block_starts(rng, n_days, block, n_res):
    if n_days < 1:
        raise ValueError('no day to resample')
    L = min(int(block), int(n_days))
    k = int(math.ceil(n_days / float(L)))
    starts = rng.integers(0, n_days - L + 1, size=(n_res, k))
    return (starts, L, k)

def block_weights(starts, n_days, L):
    n_res, k = starts.shape
    w = np.zeros((n_res, n_days), dtype=np.float64)
    rows = np.arange(n_res)
    for j in range(k):
        length = L if j < k - 1 else n_days - (k - 1) * L
        for t in range(length):
            w[rows, starts[:, j] + t] += 1.0
    return w

def concatenated_days(starts_row, n_days, L):
    k = len(starts_row)
    out = []
    for j, s in enumerate(starts_row):
        length = L if j < k - 1 else n_days - (k - 1) * L
        out.extend(range(int(s), int(s) + length))
    return out

def weights_for(year, n_days, block, n_res, part='cv'):
    rng = rng_for(year, block, n_res, part)
    starts, L, k = block_starts(rng, n_days, block, n_res)
    return (block_weights(starts, n_days, L), {'seed': SEED, 'stream': 'PCG64(SeedSequence([%d, %d, %d, %d, %d]))' % (SEED, int(year), int(block), int(n_res), PARTS[part]), 'block': int(block), 'block_used': L, 'blocks_per_resample': k, 'n_days': int(n_days), 'n_resamples': int(n_res)})

def rmse_w(w, sse_d, n_d):
    return np.sqrt(w @ sse_d / (w @ n_d))

def ss_w(w, sse_model, sse_ref):
    den = w @ sse_ref
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, 1.0 - w @ sse_model / den, np.nan)

def maprmse_w(w, diff, chunk=250):
    diff = np.asarray(diff, dtype=np.float64)
    out = np.empty(w.shape[0])
    for a in range(0, w.shape[0], chunk):
        wc = w[a:a + chunk]
        m = wc @ diff / wc.sum(axis=1)[:, None]
        out[a:a + chunk] = np.sqrt(np.mean(m * m, axis=1))
    return out

class YearSums:

    def __init__(self, year, n_cells, n_days):
        self.year = int(year)
        self.n_days = int(n_days)
        self.n = np.full(n_days, float(n_cells))
        self.sse = {}
        self.diff = {}

    def add(self, model, e, o, keep_diff=False):
        d = np.asarray(e, dtype=np.float64) - np.asarray(o, dtype=np.float64)
        if d.shape[0] != self.n_days:
            raise ValueError('%s: %d days, the year has %d' % (model, d.shape[0], self.n_days))
        self.sse[model] = np.einsum('ij,ij->i', d, d)
        if keep_diff:
            self.diff[model] = d

    def has(self, models):
        return all((m in self.sse for m in models))
FAMILY_MODELS = ('M', 'P', 'GBT', 'CLIM-WD')

def family_stats(ys, w):
    s = ys.sse
    rm = rmse_w(w, s['M'], ys.n)
    return {'C-1': ss_w(w, s['M'], s['CLIM-WD']), 'C-2': rmse_w(w, s['GBT'], ys.n) - rm, 'C-3': rmse_w(w, s['P'], ys.n) - rm, 'C-4': maprmse_w(w, ys.diff['CLIM-WD']) - maprmse_w(w, ys.diff['M'])}

def family_point(ys):
    w = np.ones((1, ys.n_days))
    return {k: float(v[0]) for k, v in family_stats(ys, w).items()}

def family_reps(years, block, n_res, part='cv'):
    pts, reps, draws = ({}, {}, [])
    for ys in years:
        w, rec = weights_for(ys.year, ys.n_days, block, n_res, part)
        draws.append(dict(rec, year=ys.year))
        st = family_stats(ys, w)
        pts[ys.year] = family_point(ys)
        for c in FAMILY:
            reps.setdefault(c, []).append(st[c])
    out = {'by_year': pts, 'draws': draws, 'point': {c: float(np.mean([pts[y][c] for y in pts])) for c in FAMILY}, 'reps': {c: np.mean(np.vstack(reps[c]), axis=0) for c in FAMILY}}
    return out

def interval(rep, level):
    a = 1.0 - level
    q = np.quantile(np.asarray(rep, dtype=np.float64), [a / 2, 1 - a / 2])
    return (float(q[0]), float(q[1]))

def p_beyond(rep, margin):
    rep = np.asarray(rep, dtype=np.float64)
    return float(min(1.0, 2.0 * min(np.mean(rep <= margin), np.mean(rep >= -margin))))

def reading(lo, hi, margin, closer=('M closer to CMAQ', 'the reference closer')):
    if lo > margin:
        return closer[0]
    if hi < -margin:
        return closer[1]
    if lo >= -margin and hi <= margin:
        return 'the interval excludes differences larger than %g in either direction' % margin
    return 'inconclusive at this precision'

def holm(reps, points, margins=MARGINS, alpha=ALPHA, names=FAMILY):
    m = len(names)
    p = {c: p_beyond(reps[c], margins[c]) for c in names}
    order = sorted(names, key=lambda c: (p[c], names.index(c)))
    out = {}
    failed_at = None
    for k, c in enumerate(order):
        a = alpha / (m - k) if failed_at is None else failed_at
        lo, hi = interval(reps[c], 1.0 - a)
        passes = failed_at is None and (lo > margins[c] or hi < -margins[c])
        if failed_at is None and (not passes):
            failed_at = a
        out[c] = {'contrast': c, 'definition': DEFINITIONS.get(c, c), 'point': points[c], 'holm_rank': k + 1, 'p_beyond': p[c], 'level': 1.0 - a, 'lo': lo, 'hi': hi, 'half_width': (hi - lo) / 2.0, 'margin': margins[c], 'passes_holm_step': bool(passes), 'reading': reading(lo, hi, margins[c]), 'n_resamples': int(len(reps[c]))}
        if c != 'C-1' and (hi - lo) / 2.0 > HALF_WIDTH_LIMIT:
            out[c]['resolution'] = 'unable to resolve the margin'
        elif c != 'C-1':
            out[c]['resolution'] = 'resolves the margin'
    return out

def c1_holds(row, acc_median):
    return bool(row['lo'] > 0 and acc_median is not None and (acc_median > 0))

def sensitivity(main, other):
    return {c: 'same reading' if main[c]['reading'] == other[c]['reading'] and main[c]['passes_holm_step'] == other[c]['passes_holm_step'] else 'sensitive to the block length' for c in main}

def exploratory(rep, point, name, definition, margin=0.0, closer=None):
    rep = np.asarray(rep, dtype=np.float64)
    ok = np.isfinite(rep)
    out = {'contrast': name, 'definition': definition, 'point': point, 'level': 1.0 - ALPHA, 'margin': margin, 'label': 'exploratory', 'n_resamples': int(len(rep)), 'finite_resamples': int(ok.sum())}
    if ok.sum() < 2:
        out.update({'lo': None, 'hi': None, 'half_width': None, 'reading': 'no interval (fewer than two finite resamples)'})
        return out
    lo, hi = interval(rep if ok.all() else rep[ok], 1.0 - ALPHA)
    out.update({'lo': lo, 'hi': hi, 'half_width': (hi - lo) / 2.0, 'reading': reading(lo, hi, margin, closer or ('positive', 'negative'))})
    return out

def table(rows, kind):
    n = sum((1 for r in rows if r.get('lo') is not None))
    return {'kind': kind, 'count_of_intervals_read': n, 'rows': rows, 'covers': COVERS}

def rmse_contrast(w, sse_a, sse_b, n_d):
    return rmse_w(w, sse_a, n_d) - rmse_w(w, sse_b, n_d)

def rmse_contrast_sub(w, sse_a, sse_b, n_d, day_sel=None):
    ww = w if day_sel is None else w * np.asarray(day_sel, dtype=np.float64)[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(ww @ sse_a / (ww @ n_d)) - np.sqrt(ww @ sse_b / (ww @ n_d))

def _row_stats(cm, co):
    dco = co - co.mean(axis=1, keepdims=True)
    dcm = cm - cm.mean(axis=1, keepdims=True)
    sxy = np.einsum('ij,ij->i', dco, dcm)
    sxx = np.einsum('ij,ij->i', dco, dco)
    syy = np.einsum('ij,ij->i', dcm, dcm)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.where(sxx > 0, sxy / sxx, np.nan)
        r = np.where((sxx > 0) & (syy > 0), sxy / np.sqrt(sxx * syy), np.nan)
    d = cm - co
    return {'R': r, 'slope': slope, 'RMSE': np.sqrt(np.einsum('ij,ij->i', d, d) / d.shape[1])}

def f12_boot(w, e, o, tf, su, groups=None, chunk=100):
    e = np.asarray(e, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    wt = w * np.asarray(tf, dtype=np.float64)[None, :]
    ws = w * np.asarray(su, dtype=np.float64)[None, :]
    nt = wt.sum(axis=1)
    ns = ws.sum(axis=1)
    groups = dict(groups or {})
    keys = ['overall'] + sorted(groups)
    out = {k: {s: np.empty(w.shape[0]) for s in ('R', 'slope', 'RMSE')} for k in keys}
    for a in range(0, w.shape[0], chunk):
        sl = slice(a, a + chunk)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = wt[sl] @ e / nt[sl][:, None] - ws[sl] @ e / ns[sl][:, None]
            co = wt[sl] @ o / nt[sl][:, None] - ws[sl] @ o / ns[sl][:, None]
        for k in keys:
            sel = slice(None) if k == 'overall' else groups[k]
            st = _row_stats(cm[:, sel], co[:, sel])
            for s in st:
                out[k][s][sl] = st[s]
    return out

def interval_nan(rep, level=1.0 - ALPHA):
    rep = np.asarray(rep, dtype=np.float64)
    ok = np.isfinite(rep)
    if ok.sum() < 2:
        return (None, None, int(ok.sum()))
    lo, hi = interval(rep[ok], level)
    return (lo, hi, int(ok.sum()))

def replication(lo, hi):
    if lo > 0:
        return 'replicated'
    if hi < 0:
        return 'contradicted'
    return 'not contradicted'

def s6(word_2019, word_lockbox):
    ok = ('replicated', 'not contradicted')
    if 'contradicted' in (word_2019, word_lockbox):
        return 'withdrawn (contradicted on 2019 or the lockbox)'
    if word_2019 in ok and word_lockbox in ok:
        return 'supported (replicated or not contradicted on both)'
    return 'not read (a reading missing)'

def replication_rows(reps, points):
    rows = {}
    for c in FAMILY:
        if c not in reps:
            continue
        lo, hi = interval(reps[c], 1.0 - ALPHA)
        rows[c] = {'contrast': c, 'point': points[c], 'lo': lo, 'hi': hi, 'level': 1.0 - ALPHA, 'n_resamples': int(len(reps[c])), 'word': replication(lo, hi)}
    return rows

def seed_range(values):
    v = [float(x) for x in values]
    if len(v) < 2:
        return None
    return {'min': min(v), 'max': max(v), 'range': max(v) - min(v), 'n_seeds': len(v)}

def seed_rule(point, ranges):
    rs = {k: v for k, v in ranges.items() if v is not None}
    if not rs:
        return {'status': 'NOT RUN', 'reason': 'no model with two or more seeds on this fold', 'ranges': ranges}
    inside = any((abs(point) <= v['range'] for v in rs.values()))
    return {'status': 'RUN', 'point': point, 'ranges': rs, 'reading': NOT_SEPARATED if inside else SEPARATED, 'rule': "|point| <= max - min of either model's seed values on the fold (READING of PROP)"}

def chi2_cdf(x, df):
    if x <= 0:
        return 0.0
    df = int(df)
    if df < 1:
        raise ValueError('df %r' % df)
    h = x / 2.0
    if df % 2 == 0:
        term, s = (1.0, 1.0)
        for k in range(1, df // 2):
            term *= h / k
            s += term
        return 1.0 - math.exp(-h) * s
    s = math.erf(math.sqrt(h))
    term = math.sqrt(h) * math.exp(-h) * 2.0 / math.sqrt(math.pi)
    for k in range(1, (df - 1) // 2 + 1):
        s -= term
        term *= h / (k + 0.5)
    return s

def chi2_ppf(q, df):
    lo, hi = (0.0, 1.0)
    while chi2_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if chi2_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def sd_interval(values, level=0.95):
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    n = v.size
    if n < 2:
        return None
    s = float(v.std(ddof=1))
    df = n - 1
    a = 1.0 - level
    return {'sd': s, 'n_seeds': int(n), 'level': level, 'lo': s * math.sqrt(df / chi2_ppf(1 - a / 2, df)), 'hi': s * math.sqrt(df / chi2_ppf(a / 2, df))}
