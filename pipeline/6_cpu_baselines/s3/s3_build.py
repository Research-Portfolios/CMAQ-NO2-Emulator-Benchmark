"""Builds the yearly arrays for 2015 to 2019 month by month: the float16 and float32 input channels and the NO2 target in ppb (day x 299 x 459), with each day's channel minima and maxima and the smallest positive values used for the statistics; it pauses when free memory falls below 1.5 GB."""
import datetime
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from numpy.lib.format import open_memmap
from s3 import s3_assemble as A
from s3 import s3common as S
K_SMALL = 2000000
MET = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'X6')
LAGGED = ('M-2', 'M-3', 'M-4', 'M-5', 'M-8')

class SmallK:

    def __init__(self, k):
        self.k = int(k)
        self.buf = np.empty(0, np.float32)
        self.pend = []
        self.npend = 0
        self.thr = np.float32(np.inf)
        self.npos = 0

    def add(self, x):
        x = np.asarray(x, np.float32).ravel()
        pos = x[x > 0]
        self.npos += int(pos.size)
        if self.buf.size >= self.k:
            pos = pos[pos <= self.thr]
        if pos.size:
            self.pend.append(pos)
            self.npend += pos.size
        if self.buf.size + self.npend > 2 * self.k:
            self._compact()

    def _compact(self):
        if self.pend:
            allv = np.concatenate([self.buf] + self.pend)
            self.pend, self.npend = ([], 0)
        else:
            allv = self.buf
        if allv.size > self.k:
            allv = np.partition(allv, self.k - 1)[:self.k]
        self.buf = allv
        if self.buf.size >= self.k:
            self.thr = self.buf.max()

    def result(self):
        self._compact()
        return (np.sort(self.buf), self.npos)

def wait_memory(floor=1.5, cap_s=3600):
    t0 = time.time()
    g = S.mem_gb()
    while g < floor:
        if time.time() - t0 > cap_s:
            raise MemoryError('free memory under %.1f GB for %d s' % (floor, cap_s))
        print('%s pause: memory %.2f GB' % (S.now_local(), g), flush=True)
        time.sleep(30)
        g = S.mem_gb()
    return g

def storage_plan():
    import json
    with open(os.path.join(S.WORK_S3, 'roundtrip.json'), 'r', encoding='ascii') as f:
        rt = json.load(f)
    st = rt['storage']
    f16c = [c for c in S.CH_NAMES if st[c] != 'float32']
    f32c = [c for c in S.CH_NAMES if st[c] == 'float32']
    return (st, f16c, f32c)

def prev_day_sources(reader, y):
    if y == 2015:
        return A.read_lag_answers(reader)
    out = {}
    for ch in LAGGED:
        ser = S.MET_SERIES[ch]
        v, dates = reader.read_month(ser, y - 1, 12)
        if dates[-1] != S.yyyyddd(datetime.date(y - 1, 12, 31)):
            raise ValueError('%s %d-12 ends %s' % (ser, y - 1, dates[-1]))
        out[ch] = v[-1].copy()
    return out

def build_year(y, reader, static, veg, clim, mask, st, f16c, f32c):
    t0 = time.time()
    start = S.now_local()
    mem0 = wait_memory()
    days = S.days_of_year(y)
    n = len(days)
    names = {'f16': 'inputs_f16_%d.npy' % y, 'f32': 'inputs_f32_%d.npy' % y, 'target': 'target_%d.npy' % y}
    paths = {k: os.path.join(S.ARRAYS, v) for k, v in names.items()}
    for p in paths.values():
        if os.path.exists(p):
            raise FileExistsError(p)
        if os.path.exists(p + '.partial'):
            os.remove(p + '.partial')
    G = (S.NROWS, S.NCOLS)
    shapes = {'f16': ((n, len(f16c)) + G, np.float16), 'f32': ((n, len(f32c)) + G, np.float32), 'target': ((n,) + G, np.float32)}
    for k, (shp, dt) in shapes.items():
        mk = open_memmap(paths[k] + '.partial', mode='w+', dtype=dt, shape=shp)
        mk.flush()
        del mk
    store = {c: A.store_fn(st[c]) for c in S.CH_NAMES}
    daymin = np.full((n, len(S.CH_NAMES)), np.nan)
    daymax = np.full((n, len(S.CH_NAMES)), np.nan)
    tmin = np.full(n, np.nan)
    tmax = np.full(n, np.nan)
    isval = np.array([S.is_validation(d) for d in days])
    small = {c: SmallK(K_SMALL) for c in S.LOG_CHANNELS}
    prev = prev_day_sources(reader, y)
    lag_ok = 0
    i0 = 0
    reads = 0
    mem_min = mem0
    for m in range(1, 13):
        mem_min = min(mem_min, wait_memory())
        md = {}
        for ser in ('T1',) + MET:
            md[ser] = reader.read_month(ser, y, m)
            reads += 1
        mdays = [d for d in days if d.month == m]
        want = [S.yyyyddd(d) for d in mdays]
        for ser, (_v, dates) in md.items():
            if dates != want:
                raise ValueError('%s %d-%02d dates' % (ser, y, m))
        k1 = len(mdays)
        b16 = np.empty((k1, len(f16c)) + G, np.float16)
        b32 = np.empty((k1, len(f32c)) + G, np.float32)
        bt = np.empty((k1,) + G, np.float32)
        for k, d in enumerate(mdays):
            met = A.met_of_day(md, k)
            src = A.day_sources(d, met, prev, static, veg, clim)
            i = i0 + k
            for j, c in enumerate(f16c):
                b16[k, j] = store[c](src[c])
            for j, c in enumerate(f32c):
                b32[k, j] = store[c](src[c])
            t = md['T1'][0][k]
            if t.dtype != np.float32:
                raise ValueError('target dtype %s' % t.dtype)
            bt[k] = t
            for j, c in enumerate(f16c):
                a = b16[k, j]
                daymin[i, S.CH_INDEX[c]] = float(a.min())
                daymax[i, S.CH_INDEX[c]] = float(a.max())
            for j, c in enumerate(f32c):
                a = b32[k, j]
                daymin[i, S.CH_INDEX[c]] = float(a.min())
                daymax[i, S.CH_INDEX[c]] = float(a.max())
                if c in small and (not isval[i]):
                    small[c].add(a)
            tm = t[mask]
            tmin[i] = float(tm.min())
            tmax[i] = float(tm.max())
            prev = {c: np.asarray(met[c]).copy() for c in LAGGED}
        for x, mch in S.LAG_OF.items():
            if st[x] == st[mch] and st[x] != 'float32' and (k1 > 1):
                jx, jm = (f16c.index(x), f16c.index(mch))
                if not np.array_equal(b16[1:, jx], b16[:-1, jm]):
                    raise ValueError('lag %s differs from %s' % (x, mch))
                lag_ok += 1
        for key, blk in (('f16', b16), ('f32', b32), ('target', bt)):
            mk = np.load(paths[key] + '.partial', mmap_mode='r+')
            if mk.shape != shapes[key][0] or mk.dtype != shapes[key][1]:
                raise ValueError('%s: shape %s dtype %s' % (key, mk.shape, mk.dtype))
            mk[i0:i0 + k1] = blk
            mk.flush()
            del mk
        i0 += k1
        del md, b16, b32, bt, blk
        print('%s %d-%02d written (%d days), %.0f s, memory %.2f GB' % (S.now_local(), y, m, k1, time.time() - t0, S.mem_gb()), flush=True)
    if i0 != n:
        raise ValueError('days written %d of %d' % (i0, n))
    for k, p in paths.items():
        os.replace(p + '.partial', p)
    files = {k: S.file_info(p) for k, p in paths.items()}
    dates = [S.ymd(d) for d in days]
    np.savez(os.path.join(S.WORK_S3, 'daystats_%d.npz' % y), daymin=daymin, daymax=daymax, tmin=tmin, tmax=tmax, isval=isval, dates=np.array(dates))
    sk = {}
    npos = {}
    for c, s in small.items():
        v, npc = s.result()
        sk[c.replace('-', '')] = v
        npos[c] = npc
    np.savez(os.path.join(S.WORK_S3, 'smallk_%d.npz' % y), **sk)
    info = {'year': y, 'start': start, 'end': S.now_local(), 'seconds': round(time.time() - t0, 1), 'days': n, 'dates_first_last': [dates[0], dates[-1]], 'validation_days': int(isval.sum()), 'answers_read': reads, 'memory_gb_start': round(mem0, 2), 'memory_gb_min_at_month_starts': round(mem_min, 2), 'memory_gb_end': round(S.mem_gb(), 2), 'lag_blocks_checked': lag_ok, 'files': files, 'f16_channels': f16c, 'f32_channels': f32c, 'smallk_npos_nonvalidation': npos, 'k_small': K_SMALL, 'daystats': S.file_info(os.path.join(S.WORK_S3, 'daystats_%d.npz' % y)), 'smallk': S.file_info(os.path.join(S.WORK_S3, 'smallk_%d.npz' % y)), 'disk_free_gb_end': round(S.disk_free_gb(), 2)}
    S.write_json(os.path.join(S.WORK_S3, 'build_%d.json' % y), info)
    return info

def main():
    a = sys.argv[1:]
    years = [int(x) for x in a[a.index('--years') + 1:]] if '--years' in a else list(S.YEARS)
    S.common.set_temp()
    inputs = S.inventory_inputs()
    reader = A.MonthReader(inputs)
    static = A.load_static(inputs)
    veg, _used = A.load_veg(inputs)
    clim = np.load(os.path.join(S.ARRAYS, 'emis_clim_2017_2019.npy'), mmap_mode='r')
    mask = np.load(os.path.join(S.ARRAYS, 'mask_conus_land.npy'))
    st, f16c, f32c = storage_plan()
    todo = [y for y in years if not os.path.exists(os.path.join(S.WORK_S3, 'build_%d.json' % y))]
    need = sum((len(S.days_of_year(y)) * S.NROWS * S.NCOLS * (2 * len(f16c) + 4 * len(f32c) + 4) for y in todo)) / 1000000000.0
    need += 1.5
    free = S.disk_free_gb()
    print('%s start; f16 %s; f32 %s; disk free %.2f GB; need %.2f GB, asked %.2f GB' % (S.now_local(), f16c, f32c, free, need, need + 30.0), flush=True)
    if free < need + 30.0:
        raise SystemExit('STOP: disk under the disk rule')
    for y in years:
        done = os.path.join(S.WORK_S3, 'build_%d.json' % y)
        if os.path.exists(done):
            print('%d: done earlier (%s), skipped' % (y, done), flush=True)
            continue
        info = build_year(y, reader, static, veg, clim, mask, st, f16c, f32c)
        print('%s %d done in %.0f s; disk free %.2f GB' % (S.now_local(), y, info['seconds'], info['disk_free_gb_end']), flush=True)
    print('%s all done' % S.now_local())
if __name__ == '__main__':
    main()
