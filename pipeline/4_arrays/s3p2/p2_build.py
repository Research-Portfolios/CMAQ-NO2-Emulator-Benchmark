"""Builds the yearly input and target arrays of the second pass for 2015 to 2019 month by month (float16 and float32 channels and the NO2 target in ppb), leaves the emission slots of 2015, 2016 and 2018 as NaN for the later fill, and keeps the per-day extremes and smallest positive values for the statistics."""
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from numpy.lib.format import open_memmap
from s3 import s3_assemble as A
from s3 import s3_build as B
from s3 import s3common as S
from s3p2 import p2_disk as DK
from s3p2 import p2_roundtrip as RT
from s3p2 import p2common as P

def storage_plan():
    rt = P.jload(os.path.join(P.WORK_P2, 'roundtrip_p2.json'))
    if sorted((int(y) for y in rt['years_tested'])) != list(P.YEARS):
        raise SystemExit('the round trip did not test every year')
    st = rt['storage']
    f16c = [c for c in P.CH_NAMES if st[c] != 'float32']
    f32c = [c for c in P.CH_NAMES if st[c] == 'float32']
    return (st, f16c, f32c, rt)

def build_year(y, reader, static, veg, mask, st, f16c, f32c):
    t0 = time.time()
    start = P.now_local()
    mem0 = B.wait_memory()
    days = P.days_of_year(y)
    n = len(days)
    names = {'f16': 'inputs_f16_%d.npy' % y, 'f32': 'inputs_f32_%d.npy' % y, 'target': 'target_%d.npy' % y}
    paths = {k: os.path.join(P.ARRAYS_P2, v) for k, v in names.items()}
    for p in paths.values():
        if os.path.exists(p):
            raise FileExistsError(p)
        if os.path.exists(p + '.partial'):
            os.remove(p + '.partial')
    G = P.G
    shapes = {'f16': ((n, len(f16c)) + G, np.float16), 'f32': ((n, len(f32c)) + G, np.float32), 'target': ((n,) + G, np.float32)}
    for k, (shp, dt) in shapes.items():
        mk = open_memmap(paths[k] + '.partial', mode='w+', dtype=dt, shape=shp)
        mk.flush()
        del mk
    store = {c: A.store_fn(st[c]) for c in P.CH_NAMES}
    built = RT.year_channels(y)
    unfilled = [c for c in f32c if c not in built]
    emis = None
    if y in P.EMIS_YEARS:
        emis = np.load(os.path.join(P.ARRAYS_P2, 'emis_true_%d.npy' % y), mmap_mode='r')
    daymin = np.full((n, len(P.CH_NAMES)), np.nan)
    daymax = np.full((n, len(P.CH_NAMES)), np.nan)
    tmin = np.full(n, np.nan)
    tmax = np.full(n, np.nan)
    isval = np.array([S.is_validation(d) for d in days])
    small = {c: B.SmallK(B.K_SMALL) for c in P.LOG_CHANNELS if c in built}
    prev = B.prev_day_sources(reader, y)
    lag_ok = 0
    i0 = 0
    reads = 0
    mem_min = mem0
    for m in range(1, 13):
        mem_min = min(mem_min, B.wait_memory())
        blocks, prev, mdays = RT.month_blocks(y, m, reader, prev, static, veg, None)
        reads += len(RT.MET)
        tv, tdates = reader.read_month('T1', y, m)
        reads += 1
        if tdates != [S.yyyyddd(d) for d in mdays]:
            raise ValueError('T1 %d-%02d dates' % (y, m))
        if tv.dtype != np.float32:
            raise ValueError('target dtype %s' % tv.dtype)
        k1 = len(mdays)
        if emis is not None:
            eb = np.asarray(emis[i0:i0 + k1])
            for j, c in enumerate(P.EMIS):
                blocks[c] = eb[:, j]
        b16 = np.empty((k1, len(f16c)) + G, np.float16)
        b32 = np.empty((k1, len(f32c)) + G, np.float32)
        for j, c in enumerate(f16c):
            b16[:, j] = store[c](blocks[c])
        for j, c in enumerate(f32c):
            if c in unfilled:
                b32[:, j] = np.nan
            else:
                b32[:, j] = store[c](blocks[c])
        bt = tv
        if bt.shape != (k1,) + G:
            raise ValueError('target shape %s' % (bt.shape,))
        for k in range(k1):
            i = i0 + k
            for j, c in enumerate(f16c):
                a = b16[k, j]
                daymin[i, P.CH_INDEX[c]] = float(a.min())
                daymax[i, P.CH_INDEX[c]] = float(a.max())
            for j, c in enumerate(f32c):
                if c in unfilled:
                    continue
                a = b32[k, j]
                daymin[i, P.CH_INDEX[c]] = float(a.min())
                daymax[i, P.CH_INDEX[c]] = float(a.max())
                if c in small and (not isval[i]):
                    small[c].add(a)
            tm = bt[k][mask]
            tmin[i] = float(tm.min())
            tmax[i] = float(tm.max())
        for x, mch in P.LAG_OF.items():
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
        del blocks, b16, b32, bt, blk, tv
        print('%s %d-%02d written (%d days), %.0f s, memory %.2f GB' % (P.now_local(), y, m, k1, time.time() - t0, P.mem_gb()), flush=True)
    if i0 != n:
        raise ValueError('days written %d of %d' % (i0, n))
    del emis
    for k, p in paths.items():
        os.replace(p + '.partial', p)
    files = {k: P.file_info(p) for k, p in paths.items()}
    dates = [P.ymd(d) for d in days]
    np.savez(os.path.join(P.WORK_P2, 'daystats_%d.npz' % y), daymin=daymin, daymax=daymax, tmin=tmin, tmax=tmax, isval=isval, dates=np.array(dates), channels=np.array(P.CH_NAMES))
    sk, npos = ({}, {})
    for c, s in small.items():
        v, npc = s.result()
        sk[c.replace('-', '')] = v
        npos[c] = npc
    np.savez(os.path.join(P.WORK_P2, 'smallk_%d.npz' % y), **sk)
    info = {'year': y, 'start': start, 'end': P.now_local(), 'seconds': round(time.time() - t0, 1), 'days': n, 'dates_first_last': [dates[0], dates[-1]], 'validation_days': int(isval.sum()), 'answers_read': reads, 'memory_gb_start': round(mem0, 2), 'memory_gb_min_at_month_starts': round(mem_min, 2), 'memory_gb_end': round(P.mem_gb(), 2), 'lag_blocks_checked': lag_ok, 'files': files, 'f16_channels': f16c, 'f32_channels': f32c, 'built_channels': built, 'unfilled_channels': unfilled, 'unfilled_fill': 'float32 NaN in inputs_f32 at the indices of these channels, every day; stage (8) fills them' if unfilled else None, 'emission_source': "W\\arrays_p2\\emis_true_%d.npy (each day's own E-1 to E-4 and P-1)" % y if y in P.EMIS_YEARS else 'not built: stage (8), the C-7a extracts', 'smallk_npos_nonvalidation': npos, 'k_small': B.K_SMALL, 'daystats': P.file_info(os.path.join(P.WORK_P2, 'daystats_%d.npz' % y)), 'smallk': P.file_info(os.path.join(P.WORK_P2, 'smallk_%d.npz' % y)), 'disk_free_gb_end': round(P.disk_free_gb(), 2)}
    P.write_json(os.path.join(P.WORK_P2, 'build_%d.json' % y), info)
    return info

def main():
    a = sys.argv[1:]
    years = [int(x) for x in a[a.index('--years') + 1:]] if '--years' in a else list(P.YEARS)
    S.common.set_temp()
    P.ensure_dirs()
    st, f16c, f32c, _rt = storage_plan()
    dk = DK.compute(len(f16c), len(f32c), skip_done=True)
    P.write_json(os.path.join(P.TMP_P2, 'disk_build.json'), dk)
    print('%s start; f16 %s; f32 %s; disk free %.2f GB; need %.2f GB, asked %.2f GB: %s' % (P.now_local(), f16c, f32c, dk['free_gb'], dk['need_gb'], dk['asked_gb'], dk['verdict']), flush=True)
    if dk['verdict'] != 'MET':
        raise SystemExit('STOP: disk under the disk rule')
    inputs = S.inventory_inputs()
    reader = A.MonthReader(inputs)
    static = A.load_static(inputs)
    veg, _used = A.load_veg(inputs)
    mask = np.load(os.path.join(P.ARRAYS_P2, 'mask_conus_land.npy'))
    for y in years:
        done = os.path.join(P.WORK_P2, 'build_%d.json' % y)
        if os.path.exists(done):
            print('%d: done earlier, skipped' % y, flush=True)
            continue
        info = build_year(y, reader, static, veg, mask, st, f16c, f32c)
        print('%s %d done in %.0f s; disk free %.2f GB' % (P.now_local(), y, info['seconds'], info['disk_free_gb_end']), flush=True)
    print('%s all done' % P.now_local())
if __name__ == '__main__':
    P.run_main(main)
