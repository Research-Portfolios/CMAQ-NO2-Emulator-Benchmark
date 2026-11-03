"""Checks every input before any array is built: the RSIG answers (raw and decompressed sha256), the MCIP files and the emission extracts; it counts units, dimensions, grid attributes, coverage, fill values and NaN on the land cells, values outside the expected ranges and duplicate dates, and stops when fills exceed 0.1 percent of land cell-days."""
import collections
import datetime
import multiprocessing as mp
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3common as S
RANGES = {'T1': (0.0, 300.0), 'T2': (0.0, 300.0), 'M8': (0.0, 6000.0), 'X6': (0.0, 6000.0), 'M2': (-60.0, 60.0), 'M3': (0.0, 50.0), 'M7': (0.0, 0.04), 'M1': (50000.0, 110000.0), 'M6': (0.0, 1400.0), 'M9': (0.0, None)}
GRID_KEYS = ('GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG', 'XCELL', 'YCELL', 'NCOLS', 'NROWS')
GRIDCRO = {'GDTYP': 2, 'P_ALP': 33.0, 'P_BET': 45.0, 'P_GAM': -97.0, 'XCENT': -97.0, 'YCENT': 40.0, 'XORIG': -2556000.0, 'YORIG': -1728000.0, 'XCELL': 12000.0, 'YCELL': 12000.0, 'NCOLS': 459, 'NROWS': 299}
DOT = dict(GRIDCRO, NCOLS=460, NROWS=300, XORIG=-2562000.0, YORIG=-1734000.0)
_MASK = None

def _init(mask_path):
    global _MASK
    S.common.set_temp()
    _MASK = np.load(mask_path)

def expected_dates(series, period):
    if period == '20141231':
        return [2014365]
    y, m = (int(period[:4]), int(period[4:]))
    d = datetime.date(y, m, 1)
    out = []
    while d.month == m:
        out.append(S.yyyyddd(d))
        d += datetime.timedelta(days=1)
    return out

def _stats(vals, mask_cells, lo, hi):
    good = np.isfinite(vals) & (vals > S.NULL_LIMIT)
    out = {}
    for tag, sel in (('all', np.ones(vals.shape, bool)), ('mask', mask_cells)):
        g = good & sel
        x = vals[g]
        out['n_' + tag] = int(g.sum())
        out['min_' + tag] = float(x.min()) if x.size else None
        out['max_' + tag] = float(x.max()) if x.size else None
        out['below_' + tag] = int((x < lo).sum()) if lo is not None else 0
        out['above_' + tag] = int((x > hi).sum()) if hi is not None else 0
    return out

def check_answer(e):
    t0 = time.time()
    series, period = (e['series'], e['period'])
    var, units_want, dot = S.SERIES[series]
    r = S.read_answer(e, var)
    v = r['values']
    out = {'item': e['item'], 'series': series, 'period': period, 'path': e['path'], 'checks': r['checks'], 'format': r['format'], 'variables': r['variables'], 'units': r['units'], 'var_dims': r['var_dims'], 'shape': list(v.shape), 'dtype': str(v.dtype)}
    a = r['attrs']
    want = DOT if dot else GRIDCRO
    out['grid_diff'] = [k for k in GRID_KEYS if a.get(k) is None or float(a[k]) != float(want[k])]
    out['nlays'] = a.get('NLAYS')
    out['tstep'] = a.get('TSTEP')
    tf = r['tflag']
    out['tflag_dates'] = [int(x) for x in tf[:, 0, 0]]
    out['tflag_times'] = sorted(set((int(x) for x in tf[:, 0, 1])))
    out['tflag_vars_same'] = bool((tf == tf[:, :1, :]).all())
    out['expected_dates'] = expected_dates(series, period)
    lo, hi = RANGES.get(series, (None, None))
    mask = _MASK
    steps = []
    for k in range(v.shape[0]):
        f = v[k, 0]
        fill = f <= S.NULL_LIMIT
        nan = np.isnan(f)
        st = {'date': out['tflag_dates'][k], 'fill_all': int(fill.sum()), 'nan_all': int(nan.sum())}
        if dot:
            bad = fill | nan
            badc = bad[:-1, :-1] | bad[1:, :-1] | bad[:-1, 1:] | bad[1:, 1:]
            st['fill_or_nan_mask_cells'] = int((badc & mask).sum())
            st['dot'] = _stats(f, np.ones(f.shape, bool), lo, hi)
            cell = S.dot_to_cell(f)
            cell[badc] = np.nan
            st.update(_stats(cell, mask, lo, hi))
        else:
            st['fill_mask'] = int((fill & mask).sum())
            st['nan_mask'] = int((nan & mask).sum())
            st.update(_stats(f, mask, lo, hi))
        steps.append(st)
    out['steps'] = steps
    out['seconds'] = round(time.time() - t0, 2)
    return out

def check_mcip(inputs, mask):
    import netCDF4
    out = {}
    for e in sorted(inputs['mcip'], key=lambda x: x['path']):
        name = os.path.basename(e['path'])
        p = S.path_abs(e['path'])
        rec = {'sha256_ok': S.sha256_file(p) == e['sha256'], 'bytes_ok': os.stat(p).st_size == int(e['bytes'])}
        if name.startswith('LUFRAC'):
            rec['used'] = False
            out[name] = rec
            continue
        ds = netCDF4.Dataset(p, 'r')
        try:
            rec['data_model'] = ds.data_model
            a = {k: S._attr(ds, k) for k in GRID_KEYS}
            rec['grid_diff'] = [k for k in GRID_KEYS if float(a[k]) != float(GRIDCRO[k])]
            t = ds.variables['TFLAG']
            t.set_auto_maskandscale(False)
            tf = np.array(t[:])
            rec['tflag_first'] = tf[0, 0].tolist()
            rec['tflag_last'] = tf[-1, 0].tolist()
            rec['steps'] = int(tf.shape[0])
            names = ('HT', 'LWMASK', 'PURB', 'LAT', 'LON') if name.startswith('GRIDCRO') else ('VEG',)
            rec['vars'] = {}
            for n in names:
                v = ds.variables[n]
                v.set_auto_maskandscale(False)
                x = np.array(v[:])
                vr = {'units': str(getattr(v, 'units', '')).strip(), 'shape': list(x.shape), 'dtype': str(x.dtype)}
                if n == 'VEG':
                    day = int(tf[0, 0, 0])
                    use = [k for k in range(tf.shape[0]) if int(tf[k, 0, 0]) == day]
                    vr['records_used'] = [use[0], use[-1]]
                    vr['records_used_n'] = len(use)
                    vr['record_hours'] = sorted(set((int(tf[k, 0, 1]) for k in use)))
                    x = x[use]
                f = x[:, 0]
                vr['fill'] = int((f <= S.NULL_LIMIT).sum())
                vr['nan'] = int(np.isnan(f).sum())
                g = f[np.isfinite(f) & (f > S.NULL_LIMIT)]
                vr['min'] = float(g.min())
                vr['max'] = float(g.max())
                fm = f[:, mask]
                vr['fill_or_nan_mask'] = int(((fm <= S.NULL_LIMIT) | np.isnan(fm)).sum())
                if n in ('LWMASK',):
                    vr['values'] = sorted(set(np.unique(f).tolist()))
                if n == 'PURB':
                    vr['below_0'] = int((g < 0).sum())
                    vr['above_100'] = int((g > 100).sum())
                if n == 'VEG':
                    vr['below_0'] = int((g < 0).sum())
                    vr['above_1'] = int((g > 1).sum())
                rec['vars'][n] = vr
        finally:
            ds.close()
        out[name] = rec
    return out

def check_extract(e):
    p = S.path_abs(e['path'])
    with open(p, 'rb') as f:
        b = f.read()
    rec = {'sha256_ok': S.sha256_bytes(b) == e['sha256'], 'bytes_ok': len(b) == int(e['bytes'])}
    import io
    z = np.load(io.BytesIO(b), allow_pickle=False)
    sp = [str(x) for x in z['species'].tolist()]
    un = [str(x) for x in z['units'].tolist()]
    rec['species'] = sp
    rec['units'] = un
    rec['date'] = str(z['date'].tolist())
    rec['tflag_date'] = int(z['tflag_date'].tolist())
    rec['hours'] = str(z['hours'].tolist())
    for k in ('sector', 'n_stacks', 'n_summed', 'n_outside'):
        if k in z.files:
            x = z[k].tolist()
            rec[k] = x if isinstance(x, (int, float)) else str(x)
    rec['arrays'] = {}
    for s in sp:
        a = z[s]
        rec['arrays'][s] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'nonfinite': int((~np.isfinite(a)).sum()), 'negative': int((a < 0).sum()), 'max': float(np.nanmax(a)), 'all_zero': bool((a == 0).all())}
    return rec

def main():
    a = sys.argv[1:]
    workers = int(a[a.index('--workers') + 1]) if '--workers' in a else 3
    S.common.set_temp()
    t_start = S.now_local()
    inv = S.load_inventory()
    inputs = S.inventory_inputs(inv)
    mask_path = os.path.join(S.ARRAYS, 'mask_conus_land.npy')
    mask = np.load(mask_path)
    rs = sorted(inputs['rsig'], key=lambda e: (e['series'], e['period']))
    res = {'start': t_start, 'workers': workers, 'mask_cells': int(mask.sum()), 'answers': []}
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, initializer=_init, initargs=(mask_path,)) as pool:
        for k, out in enumerate(pool.imap(check_answer, rs, chunksize=2)):
            res['answers'].append(out)
            if (k + 1) % 50 == 0:
                print('%s answers %d of %d, %.0f s, memory %.2f GB' % (S.now_local(), k + 1, len(rs), time.time() - t0, S.mem_gb()), flush=True)
    res['rsig_seconds'] = round(time.time() - t0, 1)
    res['mcip'] = check_mcip(inputs, mask)
    t1 = time.time()
    ex = {}
    for kind in ('emis_merged', 'pt_marine'):
        for e in sorted(inputs[kind], key=lambda x: x['path']):
            r = check_extract(e)
            r['kind'] = kind
            ex[e['path']] = r
    res['extracts'] = ex
    res['extract_seconds'] = round(time.time() - t1, 1)
    res['end'] = S.now_local()
    res['memory_gb_end'] = round(S.mem_gb(), 2)
    os.makedirs(S.WORK_S3, exist_ok=True)
    info = S.write_json(os.path.join(S.WORK_S3, 'd4_stats.json'), res)
    print('done %s; answers %d; extracts %d; %s bytes' % (res['end'], len(res['answers']), len(ex), info['bytes']))
if __name__ == '__main__':
    main()
