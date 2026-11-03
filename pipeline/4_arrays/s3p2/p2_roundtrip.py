"""Tests the storage type of every channel on every stored day: takes each channel through its default type and records the 99th-percentile and largest relative errors and the values made zero or subnormal; a float16 channel that fails in any year is stored as float32 for every year."""
import datetime
import math
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from choi import f16
from s3 import s3_assemble as A
from s3 import s3_build as B
from s3 import s3common as S
from s3p2 import p2_emisdays as ED
from s3p2 import p2common as P
K = 520000
MET = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'X6')
THRESH = f16.REL_ERR_THRESHOLD
TINY = {'float16': float(np.finfo(np.float16).smallest_normal), 'float16_affine': float(np.finfo(np.float16).smallest_normal), 'float32': float(np.finfo(np.float32).smallest_normal)}

class SmallK64:

    def __init__(self, k=None):
        self.k = int(K if k is None else k)
        self.buf = np.empty(0, np.float64)
        self.pend, self.npend, self.n = ([], 0, 0)
        self.thr = np.inf

    def add(self, x):
        x = np.asarray(x, np.float64).ravel()
        self.n += int(x.size)
        if self.buf.size >= self.k:
            x = x[x <= self.thr]
        if x.size:
            self.pend.append(x)
            self.npend += x.size
        if self.buf.size + self.npend > 2 * self.k:
            self._compact()

    def _compact(self):
        allv = np.concatenate([self.buf] + self.pend) if self.pend else self.buf
        self.pend, self.npend = ([], 0)
        if allv.size > self.k:
            allv = np.partition(allv, self.k - 1)[:self.k]
        self.buf = allv
        if self.buf.size >= self.k:
            self.thr = float(self.buf.max())

    def result(self):
        self._compact()
        return (np.sort(self.buf), self.n)

class TopK64:

    def __init__(self, k=None):
        self.k = int(K if k is None else k)
        self.buf = np.empty(0, np.float64)
        self.pend, self.npend, self.n = ([], 0, 0)
        self.thr = -np.inf

    def add(self, x):
        x = np.asarray(x, np.float64).ravel()
        self.n += int(x.size)
        if self.buf.size >= self.k:
            x = x[x >= self.thr]
        if x.size:
            self.pend.append(x)
            self.npend += x.size
        if self.buf.size + self.npend > 2 * self.k:
            self._compact()

    def _compact(self):
        allv = np.concatenate([self.buf] + self.pend) if self.pend else self.buf
        self.pend, self.npend = ([], 0)
        if allv.size > self.k:
            allv = np.partition(allv, allv.size - self.k)[-self.k:]
        self.buf = allv
        if self.buf.size >= self.k:
            self.thr = float(self.buf.min())

    def result(self):
        self._compact()
        return (np.sort(self.buf), self.n)

def lerp(a, b, t):
    d = b - a
    r = a + d * t
    if t >= 0.5:
        r = b - d * (1 - t)
    return r

def quantile_low(sorted_small, n, q):
    vi = (n - 1) * q
    lo = int(math.floor(vi))
    hi = min(lo + 1, n - 1)
    if hi >= sorted_small.size:
        raise ValueError('rank %d not kept (K %d)' % (hi, sorted_small.size))
    return float(lerp(float(sorted_small[lo]), float(sorted_small[hi]), vi - lo))

def quantile_high(sorted_top, n, q):
    vi = (n - 1) * q
    lo = int(math.floor(vi))
    hi = min(lo + 1, n - 1)
    off = n - sorted_top.size
    if lo < off:
        raise ValueError('rank %d not kept (K %d of %d)' % (lo, sorted_top.size, n))
    return float(lerp(float(sorted_top[lo - off]), float(sorted_top[hi - off]), vi - lo))

class ChanYear:

    def __init__(self, kind):
        self.kind = kind
        self.store = A.store_fn(kind)
        self.load = A.load_fn(kind)
        self.n = 0
        self.n_nf_in = 0
        self.n_nf_rt = 0
        self.n_fin_stored = 0
        self.made_zero = 0
        self.subnormal = 0
        self.max_abs = 0.0
        self.small = SmallK64()
        self.p1 = None
        self.top = None
        self.n_used = 0

    def pass_a(self, x):
        x = np.asarray(x, np.float64).ravel()
        fin = np.isfinite(x)
        st = self.store(x)
        back = np.asarray(self.load(st), np.float64).ravel()
        stf = np.asarray(st, np.float64).ravel()
        self.n += int(x.size)
        self.n_nf_in += int((~fin).sum())
        bf = np.isfinite(back)
        self.n_nf_rt += int((~bf & fin).sum())
        self.n_fin_stored += int(bf.sum())
        self.made_zero += int((fin & (x != 0) & (stf == 0)).sum())
        self.subnormal += int((fin & (stf != 0) & (np.abs(stf) < TINY[self.kind])).sum())
        ok = fin & bf
        if ok.any():
            self.max_abs = max(self.max_abs, float(np.max(np.abs(back[ok] - x[ok]))))
        self.small.add(x[fin & (x > 0)])

    def end_a(self):
        v, npos = self.small.result()
        self.npos = npos
        self.p1 = quantile_low(v, npos, 0.01) if npos else None
        self.small = None
        self.top = TopK64()

    def pass_b(self, x):
        if self.p1 is None:
            return
        x = np.asarray(x, np.float64).ravel()
        back = np.asarray(self.load(self.store(x)), np.float64).ravel()
        use = np.isfinite(x) & (x > self.p1) & np.isfinite(back)
        if use.any():
            rel = np.abs(back[use] - x[use]) / np.abs(x[use])
            self.top.add(rel)
        self.n_used += int(use.sum())

    def result(self):
        p99 = mx = None
        if self.p1 is not None and self.n_used:
            v, n = self.top.result()
            if n != self.n_used:
                raise ValueError('counts differ')
            p99 = quantile_high(v, n, 0.99)
            mx = float(v[-1])
        bad = self.n_nf_rt > 0 or p99 is None or p99 > THRESH
        return {'test': self.kind, 'n': self.n, 'n_nonfinite_input': self.n_nf_in, 'n_nonfinite_roundtrip': self.n_nf_rt, 'n_finite_stored': self.n_fin_stored, 'n_positive': self.npos, 'p1_positive': self.p1, 'n_used': self.n_used, 'p99_rel_error': p99, 'max_rel_error': mx, 'max_abs_error': self.max_abs, 'n_made_zero': self.made_zero, 'n_made_subnormal': self.subnormal, 'subnormal_below': TINY[self.kind], 'verdict': 'float32' if bad else 'float16'}

def year_channels(y):
    if y in P.EMIS_YEARS:
        return list(P.CH_NAMES)
    return [c for c in P.CH_NAMES if c not in P.EMIS]

def month_blocks(y, m, reader, prev, static, veg, emis):
    md = {ser: reader.read_month(ser, y, m) for ser in MET}
    days = [d for d in P.days_of_year(y) if d.month == m]
    want = [S.yyyyddd(d) for d in days]
    for ser, (_v, dates) in md.items():
        if dates != want:
            raise ValueError('%s %d-%02d dates' % (ser, y, m))
    k1 = len(days)
    out = {}
    for ch, ser in P.MET_SERIES.items():
        out[ch] = np.asarray(md[ser][0], np.float64)
    for x, mch in P.LAG_OF.items():
        cur = out[mch]
        out[x] = np.concatenate([np.asarray(prev[mch], np.float64)[None], cur[:-1]], axis=0)
    for ch in ('L-1', 'L-2', 'L-3', 'L-6'):
        out[ch] = np.broadcast_to(static[ch], (k1,) + P.G)
    out['L-5'] = np.broadcast_to(veg[m], (k1,) + P.G)
    if emis is not None:
        fl = emis.month(m)
        keys = [P.ymd(d) for d in days]
        stack = np.stack([fl[k] for k in keys])
        for j, c in enumerate(P.EMIS):
            out[c] = stack[:, j]
    nprev = {c: np.asarray(md[P.MET_SERIES[c]][0][-1]).copy() for c in B.LAGGED}
    return (out, nprev, days)

def run_year(y, reader, static, veg, log):
    chans = year_channels(y)
    st = {c: ChanYear(P.DEFAULT_STORAGE[c]) for c in chans}
    rec16 = {c: ChanYear('float16') for c in chans if P.DEFAULT_STORAGE[c] == 'float32'}
    t0 = time.time()
    nchk = 0
    for p in ('a', 'b'):
        emis = ED.Fields(y, check_stored=p == 'a') if y in P.EMIS_YEARS else None
        prev = B.prev_day_sources(reader, y)
        for m in range(1, 13):
            blocks, prev, days = month_blocks(y, m, reader, prev, static, veg, emis)
            for c in chans:
                x = blocks[c]
                if p == 'a':
                    st[c].pass_a(x)
                    if c in rec16:
                        rec16[c].pass_a(x)
                else:
                    st[c].pass_b(x)
                    if c in rec16:
                        rec16[c].pass_b(x)
            del blocks
        if p == 'a':
            for c in chans:
                st[c].end_a()
                if c in rec16:
                    rec16[c].end_a()
            if emis is not None:
                log('%d: emis_true days checked against their float64 fields: %d' % (y, emis.checked))
                nchk = emis.checked
        log('%d pass %s done, %.0f s, memory %.2f GB' % (y, p, time.time() - t0, P.mem_gb()))
    res = {}
    for c in chans:
        r = st[c].result()
        if c in rec16:
            r['float16_plain_cast_for_the_record'] = rec16[c].result()
        res[c] = r
    return (res, round(time.time() - t0, 1), nchk)

def selftest():
    global K
    rng = np.random.default_rng(20261020)
    k_save = K
    ok = True
    for kind, gen in (('float16', lambda n: np.round(rng.lognormal(0, 3, n), 2) * rng.choice([-1, 0, 1, 1, 1], n)), ('float16_affine', lambda n: rng.uniform(60000, 105000, n)), ('float32', lambda n: rng.lognormal(-10, 4, n)), ('float16', lambda n: rng.choice([0.0, 1.0], n)), ('float16', lambda n: np.where(rng.random(n) < 0.001, np.nan, rng.lognormal(0, 6, n)))):
        x = gen(200000)
        K = 3000
        cy = ChanYear(kind)
        chunks = np.array_split(x, 17)
        for c in chunks:
            cy.pass_a(c)
        cy.end_a()
        for c in chunks:
            cy.pass_b(c)
        r = cy.result()
        ref = f16.roundtrip_error(x, A.store_fn(kind), A.load_fn(kind))
        same = all((r[k] == ref[k] for k in ('n', 'n_nonfinite_input', 'n_nonfinite_roundtrip', 'n_positive', 'p1_positive', 'n_used', 'p99_rel_error', 'max_rel_error', 'verdict')))
        ok = ok and same
        print('selftest %-15s equal to roundtrip_error: %s (p1 %r, p99 %r)' % (kind, same, r['p1_positive'], r['p99_rel_error']))
    K = k_save
    return ok

def main():
    a = sys.argv[1:]
    S.common.set_temp()
    P.ensure_dirs()
    if '--selftest' in a:
        ok = selftest()
        print('SELFTEST %s' % ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)
    years = [int(x) for x in a[a.index('--years') + 1:]] if '--years' in a else list(P.YEARS)
    t0 = time.time()
    start = P.now_local()
    inputs = S.inventory_inputs()
    reader = A.MonthReader(inputs)
    static = A.load_static(inputs)
    veg, veg_used = A.load_veg(inputs)

    def log(msg):
        print('%s %s' % (P.now_local(), msg), flush=True)
    out = {'start': start, 'rule': 'the storage rule: every stored day of every year; per year; float32 for every year if any year fails', 'threshold': THRESH, 'k': K, 'veg_records_used': {str(k): v for k, v in veg_used.items()}, 'years': {}, 'seconds': {}, 'emis_days_checked': {}}
    for y in years:
        res, sec, nchk = run_year(y, reader, static, veg, log)
        out['years'][str(y)] = res
        out['seconds'][str(y)] = sec
        out['emis_days_checked'][str(y)] = nchk
    storage, why = ({}, {})
    for c in P.CH_NAMES:
        d = P.DEFAULT_STORAGE[c]
        if d == 'float32':
            storage[c] = 'float32'
            why[c] = 'the storage rule: M-9, E-1 to E-4 and P-1 float32'
            continue
        fails = []
        for y in years:
            r = out['years'][str(y)].get(c)
            if r is None:
                continue
            if r['n_nonfinite_roundtrip'] > 0:
                fails.append('%d non-finite after the float16 round trip' % y)
            elif r['p99_rel_error'] is None:
                fails.append('%d no value above its 1st percentile of positive values, so no error is defined (the safe side)' % y)
            elif r['p99_rel_error'] > THRESH:
                fails.append('%d 99th percentile of relative error above 1e-3' % y)
        if fails:
            storage[c] = 'float32'
            why[c] = 'moved by the storage rule: ' + '; '.join(fails)
        else:
            storage[c] = d
            why[c] = 'round trip within the storage rule in every year tested'
    out['storage'] = storage
    out['why'] = why
    out['moved_to_float32'] = [c for c in P.CH_NAMES if P.DEFAULT_STORAGE[c] != 'float32' and storage[c] == 'float32']
    out['years_tested'] = years
    out['waiting_for_stage_8'] = {str(y): list(P.EMIS) for y in years if y in P.C7A_YEARS}
    out['end'] = P.now_local()
    out['seconds_all'] = round(time.time() - t0, 1)
    info = P.write_json(os.path.join(P.WORK_P2, 'roundtrip_p2.json'), out)
    log('written %s sha16 %s; moved to float32: %s' % (info['path'], info['sha256'][:16], out['moved_to_float32']))
if __name__ == '__main__':
    P.run_main(main)
