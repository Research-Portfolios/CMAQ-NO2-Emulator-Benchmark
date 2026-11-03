"""Forms the daily emission fields of 2017 and 2019 (NO, NO2 and HONO in mol/s from the merged and sector extracts, with weekday-class means for the representative-day sectors and holidays left out; VOC in g/s), writes them as arrays, and builds the monthly weekday emission climatology and the yearly NOx totals."""
import collections
import datetime
import io
import json
import os
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from numpy.lib.format import open_memmap
from s3 import s3common as S
SPECIES = ('NO', 'NO2', 'HONO', 'VOC_INV')
PT_SPECIES = ('NO', 'NO2', 'HONO')
G = (S.NROWS, S.NCOLS)

def load_npz(e, species):
    p = S.path_abs(e['path'])
    with open(p, 'rb') as f:
        b = f.read()
    if S.sha256_bytes(b) != e['sha256'] or len(b) != int(e['bytes']):
        raise ValueError('extract differs from the inventory: ' + e['path'])
    z = np.load(io.BytesIO(b), allow_pickle=False)
    out = {}
    for s in species:
        a = z[s]
        if a.shape != G or a.dtype != np.float32:
            raise ValueError('%s %s: %s %s' % (e['path'], s, a.shape, a.dtype))
        out[s] = a
    out['_date'] = str(z['date'].tolist())
    return out

def holiday_dates(plan):
    out = {}
    for y, lst in plan['pt_marine']['federal_holidays'].items():
        s = set()
        for h in lst:
            s.add(h['legal'])
            s.add(h['observed'])
        out[int(y)] = s
    return out

def new_memmap(path, shape):
    if os.path.exists(path):
        raise FileExistsError(path)
    return open_memmap(path, mode='w+', dtype=np.float32, shape=shape)

def main():
    S.common.set_temp()
    t0 = time.time()
    start = S.now_local()
    inputs = S.inventory_inputs()
    with open(os.path.join(S.CFG, 'S2A_PLAN_S3.json'), 'r', encoding='ascii') as f:
        plan = json.load(f)
    kept = S.load_inventory()['sectors']['kept']
    dropped = S.load_inventory()['sectors']['dropped']
    daily = [s for s in S.DAILY_SECTORS if s in kept]
    rep = [s for s in S.REP_SECTORS if s in kept]
    hol = holiday_dates(plan)
    mask = np.load(os.path.join(S.ARRAYS, 'mask_conus_land.npy'))
    merged = {e['date']: e for e in inputs['emis_merged']}
    pt = collections.defaultdict(dict)
    for e in inputs['pt_marine']:
        pt[e['sector']][e['date']] = e
    det = {'start': start, 'sectors_kept': kept, 'sectors_dropped': dropped, 'daily_sectors': daily, 'rep_sectors': rep, 'holidays': {str(y): sorted(v) for y, v in hol.items()}, 'weekday_index': 'Python weekday: Monday 0 .. Sunday 6', 'classes': {}, 'fallbacks': [], 'holiday_members_left_out': [], 'counts_mw': {}, 'days': {}}
    os.makedirs(S.ARRAYS, exist_ok=True)
    days = {y: S.days_of_year(y) for y in S.EMIS_YEARS}
    for y in S.EMIS_YEARS:
        miss = [S.ymd(d) for d in days[y] if S.ymd(d) not in merged]
        if miss:
            raise ValueError('merged days missing in %d: %s' % (y, miss))
        for s in daily:
            miss = [S.ymd(d) for d in days[y] if S.ymd(d) not in pt[s]]
            if miss:
                raise ValueError('%s days missing in %d: %s' % (s, y, miss))
    true_mm = {y: new_memmap(os.path.join(S.ARRAYS, 'emis_true_%d.npy' % y), (len(days[y]), 4) + G) for y in S.EMIS_YEARS}
    clim_mm = {k: new_memmap(os.path.join(S.ARRAYS, 'emis_clim_%s.npy' % k), (12, 7, 4) + G) for k in ('2017_2019', '2017', '2019')}
    parts = ['merged'] + daily + rep
    nox = {y: {p: {'mask': 0.0, 'grid': 0.0} for p in parts + ['total']} for y in S.EMIS_YEARS}
    for m in range(1, 13):
        sums = {}
        cnts = {}
        for y in S.EMIS_YEARS:
            cmean = {}
            for s in rep:
                mem = sorted((d for d in pt[s] if int(d[:4]) == y and int(d[4:6]) == m))
                by = collections.defaultdict(list)
                for d in mem:
                    if d in hol[y]:
                        det['holiday_members_left_out'].append('%s %s' % (s, d))
                        continue
                    dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
                    by[S.CLASS_OF_WEEKDAY[dd.weekday()]].append(d)
                det['classes']['%s_%d_%02d' % (s, y, m)] = {c: by.get(c, []) for c in ('Mon', 'TueFri', 'Sat', 'Sun')}
                cmean[s] = {}
                for c in ('Mon', 'TueFri', 'Sat', 'Sun'):
                    use = by.get(c) or by.get('TueFri')
                    if not by.get(c):
                        det['fallbacks'].append('%s %d-%02d %s -> TueFri' % (s, y, m, c))
                    if not use:
                        raise ValueError('%s %d-%02d: no TueFri member' % (s, y, m))
                    acc = np.zeros((3,) + G, np.float64)
                    for d in use:
                        z = load_npz(pt[s][d], PT_SPECIES)
                        for k, sp in enumerate(PT_SPECIES):
                            acc[k] += z[sp]
                    cmean[s][c] = acc / len(use)
            sm = np.zeros((7, 4) + G, np.float64)
            cn = np.zeros(7, np.int64)
            for i, d in enumerate(days[y]):
                if d.month != m:
                    continue
                w = d.weekday()
                key = S.ymd(d)
                z = load_npz(merged[key], SPECIES)
                if z['_date'] != key:
                    raise ValueError('merged date %s != %s' % (z['_date'], key))
                f = np.zeros((4,) + G, np.float64)
                for k, sp in enumerate(SPECIES):
                    f[k] = z[sp]
                part_nox = {'merged': f[0] + f[1] + f[2]}
                for s in daily:
                    zp = load_npz(pt[s][key], PT_SPECIES)
                    if zp['_date'] != key:
                        raise ValueError('%s date' % s)
                    add = np.zeros((3,) + G, np.float64)
                    for k, sp in enumerate(PT_SPECIES):
                        add[k] = zp[sp]
                    f[:3] += add
                    part_nox[s] = add[0] + add[1] + add[2]
                for s in rep:
                    add = cmean[s][S.CLASS_OF_WEEKDAY[w]]
                    f[:3] += add
                    part_nox[s] = add[0] + add[1] + add[2]
                tot = f[0] + f[1] + f[2]
                part_nox['total'] = tot
                for p, a in part_nox.items():
                    nox[y][p]['mask'] += float(a[mask].sum())
                    nox[y][p]['grid'] += float(a.sum())
                true_mm[y][i] = f.astype(np.float32)
                sm[w] += f
                cn[w] += 1
            sums[y] = sm
            cnts[y] = cn
            for w in range(7):
                clim_mm[str(y)][m - 1, w] = (sm[w] / cn[w]).astype(np.float32)
        both = sums[2017] + sums[2019]
        nb = cnts[2017] + cnts[2019]
        for w in range(7):
            clim_mm['2017_2019'][m - 1, w] = (both[w] / nb[w]).astype(np.float32)
        det['counts_mw']['%02d' % m] = {'2017': cnts[2017].tolist(), '2019': cnts[2019].tolist()}
        print('%s month %d done, %.0f s, memory %.2f GB' % (S.now_local(), m, time.time() - t0, S.mem_gb()), flush=True)
    for mm in list(true_mm.values()) + list(clim_mm.values()):
        mm.flush()
    del true_mm, clim_mm
    files = {}
    for y in S.EMIS_YEARS:
        dl = [S.ymd(d) for d in days[y]]
        det['days'][str(y)] = len(dl)
        p = os.path.join(S.ARRAYS, 'emis_true_%d_days.json' % y)
        files['emis_true_%d_days' % y] = S.write_json(p, dl)
        files['emis_true_%d' % y] = S.file_info(os.path.join(S.ARRAYS, 'emis_true_%d.npy' % y))
    for k in ('2017_2019', '2017', '2019'):
        files['emis_clim_%s' % k] = S.file_info(os.path.join(S.ARRAYS, 'emis_clim_%s.npy' % k))
    d5 = {'run': 'emul', 'stage': 'S3 step (4)', 'made': S.now_local(), 'design': "the study design's emission fields and emission climatology", 'units': "mol/s (NO + NO2 + HONO), summed over cells; the mean over each year's extracted days", 'mask_cells': int(mask.sum()), 'grid_cells': int(mask.size), 'sectors_kept': kept, 'sectors_dropped': dropped, 'years': {}}
    for y in S.EMIS_YEARS:
        n = len(days[y])
        yd = {'extracted_days': n, 'nox_mean_mask': nox[y]['total']['mask'] / n, 'nox_mean_grid': nox[y]['total']['grid'] / n, 'parts': {}}
        for p in parts:
            yd['parts'][p] = {'nox_mean_mask': nox[y][p]['mask'] / n, 'nox_mean_grid': nox[y][p]['grid'] / n, 'share_mask': nox[y][p]['mask'] / nox[y]['total']['mask'], 'share_grid': nox[y][p]['grid'] / nox[y]['total']['grid']}
        pm = [p for p in parts if p != 'merged']
        yd['point_and_marine_share_mask'] = sum((yd['parts'][p]['share_mask'] for p in pm))
        yd['point_and_marine_share_grid'] = sum((yd['parts'][p]['share_grid'] for p in pm))
        d5['years'][str(y)] = yd
    d5['change_2019_minus_2017_mask'] = d5['years']['2019']['nox_mean_mask'] - d5['years']['2017']['nox_mean_mask']
    d5['ratio_2019_to_2017_mask'] = d5['years']['2019']['nox_mean_mask'] / d5['years']['2017']['nox_mean_mask']
    d5['files'] = files
    files['S3_D5_NOX'] = S.write_json(os.path.join(S.CFG, 'S3_D5_NOX.json'), d5)
    det['files'] = files
    det['end'] = S.now_local()
    det['seconds'] = round(time.time() - t0, 1)
    S.write_json(os.path.join(S.WORK_S3, 'emis_detail.json'), det)
    print('done %s in %.0f s' % (det['end'], time.time() - t0))
if __name__ == '__main__':
    main()
