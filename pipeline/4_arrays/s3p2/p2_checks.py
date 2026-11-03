"""Runs the second pass's own checks after the build: a second reading of the target and two input channels from the raw answers, the lagged channels, finiteness and the unfilled NaN slots, and the new channels and arrays checked by separate code against their sources."""
import collections
import datetime
import json
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3common as S
from s3p2 import p2common as P

def patch():
    S.CFG = P.CFG_P2
    S.ARRAYS = P.ARRAYS_P2
    S.WORK_S3 = P.WORK_P2

def check_finite():
    out = {}
    for y in P.YEARS:
        z = np.load(os.path.join(P.WORK_P2, 'daystats_%d.npz' % y))
        b = P.jload(os.path.join(P.WORK_P2, 'build_%d.json' % y))
        built = [c for c in P.CH_NAMES if c in b['built_channels']]
        unf = b['unfilled_channels']
        jb = [P.CH_INDEX[c] for c in built]
        ju = [P.CH_INDEX[c] for c in unf]
        bad = int((~np.isfinite(z['daymin'][:, jb])).sum() + (~np.isfinite(z['daymax'][:, jb])).sum() + (~np.isfinite(z['tmin'])).sum() + (~np.isfinite(z['tmax'])).sum())
        nan_unf = bool(np.isnan(z['daymin'][:, ju]).all() and np.isnan(z['daymax'][:, ju]).all()) if ju else None
        out[str(y)] = {'days': int(z['daymin'].shape[0]), 'built_channels': len(built), 'nonfinite_extremes_built': bad, 'unfilled': unf, 'unfilled_extremes_all_nan': nan_unf, 'min': {c: float(np.min(z['daymin'][:, P.CH_INDEX[c]])) for c in built}, 'max': {c: float(np.max(z['daymax'][:, P.CH_INDEX[c]])) for c in built}}
    return out

def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}

def check_new():
    import h5py
    ch = P.jload(os.path.join(P.CFG_P2, 'S3_CHANNELS.json'))
    cm = {c['name']: c for c in ch['channels']}
    inv = S.load_inventory()
    files = [e for e in inv['files'] if e.get('s3_input') is True]
    g = [e for e in files if os.path.basename(e['path']) == 'GRIDCRO2D_20160701.nc4'][0]
    with h5py.File(os.path.join(P.ROOT, g['path']), 'r') as f:
        dl = np.array(f['DLUSE'][0, 0], dtype=np.float32)
    want16 = dl.astype(np.float16)
    res = {'L6': {}, 'E': {}, 'c7': {}, 'unfilled': {}}
    j6 = cm['L-6']['index']
    if cm['L-6']['array'] != 'inputs_f16':
        raise ValueError('L-6 is not in inputs_f16')
    for y in P.YEARS:
        a = np.load(os.path.join(P.ARRAYS_P2, 'inputs_f16_%d.npy' % y), mmap_mode='r')
        bad = sum((0 if np.array_equal(np.asarray(a[i, j6]), want16) else 1 for i in range(a.shape[0])))
        res['L6'][str(y)] = {'days': int(a.shape[0]), 'days_differing': bad}
        del a
    merged = {e['date']: e for e in files if e['kind'] == 'emis_merged'}
    pt = collections.defaultdict(dict)
    for e in files:
        if e['kind'] == 'pt_marine':
            pt[e['sector']][e['date']] = e
    with open(os.path.join(P.CFG_B, 'S2A_PLAN_S3.json'), 'r', encoding='ascii') as f:
        plan = json.load(f)
    hol = {}
    for yy, lst in plan['pt_marine']['federal_holidays'].items():
        hol[int(yy)] = set((h['legal'] for h in lst)) | set((h['observed'] for h in lst))
    cls = {0: 'Mon', 1: 'TF', 2: 'TF', 3: 'TF', 4: 'TF', 5: 'Sat', 6: 'Sun'}
    jE = {c: cm[c]['index'] for c in P.EMIS}
    for c in P.EMIS:
        if cm[c]['array'] != 'inputs_f32':
            raise ValueError('%s is not in inputs_f32' % c)
    sp3 = ('NO', 'NO2', 'HONO')
    for y in P.EMIS_YEARS:
        a32 = np.load(os.path.join(P.ARRAYS_P2, 'inputs_f32_%d.npy' % y), mmap_mode='r')
        rec = {'days': 0, 'E_exact': 0, 'P1_within_4_steps': 0, 'P1_max_steps': 0, 'P1_max_rel': 0.0}
        for m in range(1, 13):
            d = datetime.date(y, m, 15)
            i = (d - datetime.date(y, 1, 1)).days
            key = d.strftime('%Y%m%d')
            z = load_npz(os.path.join(P.ROOT, merged[key]['path']), ('NO', 'NO2', 'HONO', 'VOC_INV'))
            okE = all((np.array_equal(np.asarray(a32[i, jE[c]]), z[s]) for c, s in zip(('E-1', 'E-2', 'E-3', 'E-4'), ('NO', 'NO2', 'HONO', 'VOC_INV'))))
            tot = {s: np.zeros(P.G, np.float64) for s in sp3}
            for s in ('othpt', 'pt_oilgas', 'ptnonipm'):
                mem = [dd for dd in pt[s] if dd[:6] == key[:6] and dd not in hol[y]]
                byc = collections.defaultdict(list)
                for dd in mem:
                    wd = datetime.date(int(dd[:4]), int(dd[4:6]), int(dd[6:])).weekday()
                    byc[cls[wd]].append(dd)
                use = byc.get(cls[d.weekday()]) or byc.get('TF')
                for sp in sp3:
                    acc = np.zeros(P.G, np.float64)
                    for dd in use:
                        acc += load_npz(os.path.join(P.ROOT, pt[s][dd]['path']), (sp,))[sp]
                    tot[sp] += acc / len(use)
            for s in ('cmv_c3_12', 'cmv_c1c2_12', 'ptegu'):
                zz = load_npz(os.path.join(P.ROOT, pt[s][key]['path']), sp3)
                for sp in sp3:
                    tot[sp] += zz[sp]
            p1 = tot['HONO'] + tot['NO2'] + tot['NO']
            st = np.asarray(a32[i, jE['P-1']])
            ai = st.view(np.int32).astype(np.int64)
            bi = p1.astype(np.float32).view(np.int32).astype(np.int64)
            steps = np.abs(ai - bi)
            nz = p1 != 0
            rel = np.zeros(P.G)
            rel[nz] = np.abs(st[nz] - p1[nz]) / np.abs(p1[nz])
            rec['days'] += 1
            rec['E_exact'] += int(okE)
            rec['P1_within_4_steps'] += int(steps.max() <= 4)
            rec['P1_max_steps'] = max(rec['P1_max_steps'], int(steps.max()))
            rec['P1_max_rel'] = max(rec['P1_max_rel'], float(rel.max()))
        res['E'][str(y)] = rec
        del a32
    clim = np.load(os.path.join(P.ARRAYS_P2, 'emis_clim_2017_2019.npy'), mmap_mode='r')
    for y in P.YEARS:
        c7 = np.load(os.path.join(P.ARRAYS_P2, 'inputs_c7_%d.npy' % y), mmap_mode='r')
        a32 = np.load(os.path.join(P.ARRAYS_P2, 'inputs_f32_%d.npy' % y), mmap_mode='r')
        ok = 0
        nan_ok = 0
        fin_ok = 0
        others = [c for c in ch['f32_channels'] if c not in P.EMIS]
        for m in range(1, 13):
            d = datetime.date(y, m, 15)
            i = (d - datetime.date(y, 1, 1)).days
            ok += int(np.array_equal(np.asarray(c7[i]), np.asarray(clim[m - 1, d.weekday()])))
            if y in P.C7A_YEARS:
                nan_ok += int(all((np.isnan(np.asarray(a32[i, jE[c]])).all() for c in P.EMIS)))
            fin_ok += int(all((np.isfinite(np.asarray(a32[i, cm[c]['index']])).all() for c in others)))
        res['c7'][str(y)] = {'days': 12, 'equal_to_clim': ok}
        res['unfilled'][str(y)] = {'days': 12, 'emission_slots_all_nan': nan_ok if y in P.C7A_YEARS else None, 'other_float32_channels_finite': fin_ok, 'other_float32_channels': others}
        del c7, a32
    return res

def main():
    a = sys.argv[1:]
    only = [int(x) for x in a[a.index('--only') + 1:]] if '--only' in a else [1, 2, 3, 4]
    S.common.set_temp()
    patch()
    out = {'start': P.now_local()}
    if 1 in only:
        from s3 import s3_second
        s3_second.main()
        sr = P.jload(os.path.join(P.WORK_P2, 'second_reading.json'))
        out['second_reading'] = sr['summary']
        print('(1) second reading: %s' % sr['summary'], flush=True)
    if 2 in only:
        from s3 import s3_selfcheck_lags
        s3_selfcheck_lags.main()
        out['lags'] = "see the tool output (the earlier version's script prints only)"
    if 3 in only:
        out['finite'] = check_finite()
        print('(3) finite: %s' % {y: (v['nonfinite_extremes_built'], v['unfilled_extremes_all_nan']) for y, v in out['finite'].items()}, flush=True)
    if 4 in only:
        out['new'] = check_new()
        print('(4) new: %s' % json.dumps(out['new'], sort_keys=True), flush=True)
    out['end'] = P.now_local()
    name = 'checks_p2.json' if only == [1, 2, 3, 4] else 'checks_p2_%s.json' % ''.join((str(x) for x in only))
    info = P.write_json(os.path.join(P.WORK_P2, name), out)
    print('written %s sha16 %s' % (info['path'], info['sha256'][:16]))
if __name__ == '__main__':
    P.run_main(main)
