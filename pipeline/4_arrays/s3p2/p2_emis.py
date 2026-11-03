"""Forms the daily emission fields of 2017 and 2019 for the second pass: gridded NO, NO2, HONO and VOC from the merged extracts, and the point and marine NOx channel as the sum of NO, NO2 and HONO over six sectors with weekday-class means for the representative-day sectors; it also builds the monthly weekday emission climatology."""
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
from s3 import s3_emis as E1
from s3 import s3common as S
from s3p2 import p2common as P
SPECIES = ('NO', 'NO2', 'HONO', 'VOC_INV')
PT_SPECIES = ('NO', 'NO2', 'HONO')
G = P.G
NCH = 5

def p1_of(z):
    return z['NO'].astype(np.float64) + z['NO2'].astype(np.float64) + z['HONO'].astype(np.float64)

def new_partial(path, shape):
    if os.path.exists(path):
        raise FileExistsError(path)
    part = path + '.partial'
    if os.path.exists(part):
        os.remove(part)
    return open_memmap(part, mode='w+', dtype=np.float32, shape=shape)

def main():
    S.common.set_temp()
    P.ensure_dirs()
    t0 = time.time()
    start = P.now_local()
    inputs = S.inventory_inputs()
    with open(os.path.join(S.CFG, 'S2A_PLAN_S3.json'), 'r', encoding='ascii') as f:
        plan = json.load(f)
    inv = S.load_inventory()
    kept = inv['sectors']['kept']
    dropped = inv['sectors']['dropped']
    daily = [s for s in S.DAILY_SECTORS if s in kept]
    rep = [s for s in S.REP_SECTORS if s in kept]
    hol = E1.holiday_dates(plan)
    mask = np.load(os.path.join(P.ARRAYS_P2, 'mask_conus_land.npy'))
    merged = {e['date']: e for e in inputs['emis_merged']}
    pt = collections.defaultdict(dict)
    for e in inputs['pt_marine']:
        pt[e['sector']][e['date']] = e
    det = {'start': start, 'sectors_kept': kept, 'sectors_dropped': dropped, 'daily_sectors': daily, 'rep_sectors': rep, 'holidays': {str(y): sorted(v) for y, v in hol.items()}, 'weekday_index': 'Python weekday: Monday 0 .. Sunday 6', 'channel_order': ['E-1 NO', 'E-2 NO2', 'E-3 HONO', 'E-4 VOC_INV', 'P-1 NOx'], 'classes': {}, 'fallbacks': [], 'holiday_members_left_out': [], 'counts_mw': {}, 'days': {}, 'p1_nonfinite': 0, 'p1_negative': 0, 'e_nonfinite': 0, 'e_negative': 0}
    days = {y: P.days_of_year(y) for y in P.EMIS_YEARS}
    for y in P.EMIS_YEARS:
        miss = [P.ymd(d) for d in days[y] if P.ymd(d) not in merged]
        if miss:
            raise ValueError('merged days missing in %d: %s' % (y, miss))
        for s in daily:
            miss = [P.ymd(d) for d in days[y] if P.ymd(d) not in pt[s]]
            if miss:
                raise ValueError('%s days missing in %d: %s' % (s, y, miss))
    names = {}
    for y in P.EMIS_YEARS:
        names['true_%d' % y] = (os.path.join(P.ARRAYS_P2, 'emis_true_%d.npy' % y), (len(days[y]), NCH) + G)
    for k in ('2017_2019', '2017', '2019'):
        names['clim_' + k] = (os.path.join(P.ARRAYS_P2, 'emis_clim_%s.npy' % k), (12, 7, NCH) + G)
    mm = {k: new_partial(p, shp) for k, (p, shp) in names.items()}
    parts = ['merged'] + daily + rep
    nox = {y: {p: {'mask': 0.0, 'grid': 0.0} for p in parts + ['total']} for y in P.EMIS_YEARS}
    for m in range(1, 13):
        sums, cnts = ({}, {})
        for y in P.EMIS_YEARS:
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
                    acc = np.zeros(G, np.float64)
                    for d in use:
                        z = E1.load_npz(pt[s][d], PT_SPECIES)
                        acc += p1_of(z)
                    cmean[s][c] = acc / len(use)
            sm = np.zeros((7, NCH) + G, np.float64)
            cn = np.zeros(7, np.int64)
            for i, d in enumerate(days[y]):
                if d.month != m:
                    continue
                w = d.weekday()
                key = P.ymd(d)
                z = E1.load_npz(merged[key], SPECIES)
                if z['_date'] != key:
                    raise ValueError('merged date %s != %s' % (z['_date'], key))
                f = np.zeros((NCH,) + G, np.float64)
                for k, sp in enumerate(SPECIES):
                    f[k] = z[sp]
                part_nox = {'merged': f[0] + f[1] + f[2]}
                p1 = np.zeros(G, np.float64)
                for s in daily:
                    zp = E1.load_npz(pt[s][key], PT_SPECIES)
                    if zp['_date'] != key:
                        raise ValueError('%s date' % s)
                    add = p1_of(zp)
                    p1 += add
                    part_nox[s] = add
                for s in rep:
                    add = cmean[s][S.CLASS_OF_WEEKDAY[w]]
                    p1 += add
                    part_nox[s] = add
                f[4] = p1
                det['p1_nonfinite'] += int((~np.isfinite(p1)).sum())
                det['p1_negative'] += int((p1 < 0).sum())
                det['e_nonfinite'] += int((~np.isfinite(f[:4])).sum())
                det['e_negative'] += int((f[:4] < 0).sum())
                part_nox['total'] = part_nox['merged'] + p1
                for p, a in part_nox.items():
                    nox[y][p]['mask'] += float(a[mask].sum())
                    nox[y][p]['grid'] += float(a.sum())
                mm['true_%d' % y][i] = f.astype(np.float32)
                sm[w] += f
                cn[w] += 1
            sums[y] = sm
            cnts[y] = cn
            for w in range(7):
                mm['clim_%d' % y][m - 1, w] = (sm[w] / cn[w]).astype(np.float32)
        both = sums[2017] + sums[2019]
        nb = cnts[2017] + cnts[2019]
        for w in range(7):
            mm['clim_2017_2019'][m - 1, w] = (both[w] / nb[w]).astype(np.float32)
        det['counts_mw']['%02d' % m] = {'2017': cnts[2017].tolist(), '2019': cnts[2019].tolist()}
        print('%s month %d done, %.0f s, memory %.2f GB' % (P.now_local(), m, time.time() - t0, P.mem_gb()), flush=True)
    for a in mm.values():
        a.flush()
    del mm, a
    files = {}
    for k, (p, _shp) in names.items():
        os.replace(p + '.partial', p)
        files[os.path.basename(p)] = P.npy_info(p)
    for y in P.EMIS_YEARS:
        dl = [P.ymd(d) for d in days[y]]
        det['days'][str(y)] = len(dl)
        p = os.path.join(P.ARRAYS_P2, 'emis_true_%d_days.json' % y)
        files['emis_true_%d_days.json' % y] = P.write_json(p, dl)
    d5 = {'run': 'emul', 'stage': 'the second pass, step (4)', 'made': P.now_local(), 'design': "the study design's emission fields (E-1 + E-2 + E-3 + P-1) and emission climatology", 'units': "mol/s (NO + NO2 + HONO), summed over cells; the mean over each year's extracted days", 'mask': P.file_info(os.path.join(P.ARRAYS_P2, 'mask_conus_land.npy')), 'mask_cells': int(mask.sum()), 'grid_cells': int(mask.size), 'sectors_kept': kept, 'sectors_dropped': dropped, 'years': {}}
    for y in P.EMIS_YEARS:
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
    files['S3_D5_NOX.json'] = P.write_json(os.path.join(P.CFG_P2, 'S3_D5_NOX.json'), d5)
    det['files'] = files
    det['end'] = P.now_local()
    det['seconds'] = round(time.time() - t0, 1)
    P.write_json(os.path.join(P.WORK_P2, 'emis_detail.json'), det)
    print('done %s in %.0f s; P-1 non-finite %d, negative %d; E non-finite %d, negative %d' % (det['end'], time.time() - t0, det['p1_nonfinite'], det['p1_negative'], det['e_nonfinite'], det['e_negative']))
if __name__ == '__main__':
    P.run_main(main)
