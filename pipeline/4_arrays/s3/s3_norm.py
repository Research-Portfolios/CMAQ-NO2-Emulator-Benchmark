"""Computes the scaling statistics of each fold and of each two-year emission test on the training days only: each input channel's minimum and maximum, for precipitation and the emission channels the first percentile c of the positive values and the bounds of log10(x + c), and the target's minimum and maximum over the land cells."""
import datetime
import json
import math
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3_build as B
from s3 import s3common as S
P = 1.0

def jload(p):
    with open(p, 'r', encoding='ascii') as f:
        return json.load(f)

def percentile_from_parts(parts, npos, k):
    index = (npos - 1) * (P / 100.0)
    lo = int(math.floor(index))
    frac = index - lo
    full = [p for p, n in parts if p.size >= k]
    if full and lo + 2 > k:
        raise ValueError('rank %d needs more than K = %d' % (lo + 2, k))
    allv = np.concatenate([p for p, _n in parts])
    if allv.size < npos and lo + 2 > allv.size:
        raise ValueError('not enough values kept')
    need = [lo, min(lo + 1, npos - 1)]
    part = np.partition(allv, need)
    a = float(part[need[0]])
    b = float(part[need[1]])
    c = a + frac * (b - a)
    return (c, {'npos': int(npos), 'index': index, 'lower_rank': lo, 'value_lower': a, 'value_upper': b, 'fraction': frac, 'k': int(k)})

def chan_stats(names, mins, maxs, cvals, cinfo):
    out = {}
    for c in names:
        r = {'min': mins[c], 'max': maxs[c]}
        if c in cvals:
            cc = cvals[c]
            r['c'] = cc
            r['log10_min'] = math.log10(mins[c] + cc)
            r['log10_max'] = math.log10(maxs[c] + cc)
            r['c_detail'] = cinfo[c]
        out[c] = r
    return out

def main():
    S.common.set_temp()
    t0 = S.now_local()
    sp = jload(os.path.join(S.CFG, 'SPLITS.json'))
    ch = jload(os.path.join(S.CFG, 'S3_CHANNELS.json'))
    arrays = os.path.join(S.CFG, 'S3_ARRAYS.json')
    arrays_sha = S.sha256_file(arrays)
    ds, sk = ({}, {})
    for y in S.YEARS:
        z = np.load(os.path.join(S.WORK_S3, 'daystats_%d.npz' % y))
        ds[y] = {k: z[k] for k in z.files}
        b = jload(os.path.join(S.WORK_S3, 'build_%d.json' % y))
        z2 = np.load(os.path.join(S.WORK_S3, 'smallk_%d.npz' % y))
        sk[y] = {c: (z2[c.replace('-', '')], b['smallk_npos_nonvalidation'][c]) for c in S.LOG_CHANNELS}
        kk = b['k_small']
    names = S.CH_NAMES
    common_note = {'stored_values': 'statistics of the stored values: M-1 as (x - 50000) / 60000 in float16; float16 and float32 channels as stored', 'no_clipping': 'held-out values outside the training range are kept', 'c_rule': "c: the 1st percentile (numpy's linear method) of the channel's positive values (x > 0) on the training days, all grid cells; log10 values in float64", 'target': 'T-1, ppb, minimum and maximum over the masked cells (W\\arrays\\mask_conus_land.npy) of the training days', 'S3_ARRAYS_sha256': arrays_sha, 'channels_file': 'run\\config\\S3_CHANNELS.json'}
    index = {'start': t0, 'files': {}}

    def train_sel(spl):
        return {int(y): np.array(v, dtype=np.int64) for y, v in spl['train']['index_by_year'].items()}

    def base_stats(sel):
        mins = {c: min((float(np.min(ds[y]['daymin'][ix, S.CH_INDEX[c]])) for y, ix in sel.items())) for c in names}
        maxs = {c: max((float(np.max(ds[y]['daymax'][ix, S.CH_INDEX[c]])) for y, ix in sel.items())) for c in names}
        tmin = min((float(np.min(ds[y]['tmin'][ix])) for y, ix in sel.items()))
        tmax = max((float(np.max(ds[y]['tmax'][ix])) for y, ix in sel.items()))
        return (mins, maxs, tmin, tmax)

    def check_sel(sel):
        for y, ix in sel.items():
            isval = ds[y]['isval']
            if isval[ix].any() or len(ix) != int((~isval).sum()):
                raise ValueError("training days of %d are not the year's non-validation days" % y)
    written = []
    for yh, f in sp['folds'].items():
        sel = train_sel(f)
        check_sel(sel)
        if int(yh) in sel:
            raise ValueError('held-out year in training')
        mins, maxs, tmin, tmax = base_stats(sel)
        cvals, cinfo = ({}, {})
        for c in S.LOG_CHANNELS:
            parts = [sk[y][c] for y in sel]
            npos = sum((n for _p, n in parts))
            cvals[c], cinfo[c] = percentile_from_parts(parts, npos, kk)
        name = 'NORM_fold%s' % yh
        doc = {'run': 'emul', 'stage': 'S3 step (8)', 'made': S.now_local(), 'name': name, 'kind': 'fold', 'held_out_year': int(yh), 'training_years': sorted(sel), 'training_days': int(sum((len(v) for v in sel.values()))), 'emission_source': 'inputs_f32 E-1 to E-4 (the emission climatology of 2017 and 2019, emis_clim_2017_2019.npy)', 'notes': common_note, 'channels': chan_stats(names, mins, maxs, cvals, cinfo), 'target': {'min': tmin, 'max': tmax}}
        p = os.path.join(S.CFG, name + '.json')
        index['files'][name] = S.write_json(p, doc)
        written.append(name)
        print('%s %s written' % (S.now_local(), name), flush=True)
    for cfg, f in sp['f6'].items():
        ytr, yte = (f['train_year'], f['test_year'])
        sel = train_sel(f)
        check_sel(sel)
        if list(sel) != [ytr]:
            raise ValueError('two-year emission test, training years %s' % list(sel))
        ix = sel[ytr]
        mins, maxs, tmin, tmax = base_stats(sel)
        days = S.days_of_year(ytr)
        true_mm = np.load(os.path.join(S.ARRAYS, 'emis_true_%d.npy' % ytr), mmap_mode='r')
        own = np.load(os.path.join(S.ARRAYS, 'emis_clim_%d.npy' % ytr), mmap_mode='r')
        other = np.load(os.path.join(S.ARRAYS, 'emis_clim_%d.npy' % yte), mmap_mode='r')
        for arm in ('TRUE', 'OTHER', 'OWN'):
            emin = {c: np.inf for c in S.EMIS}
            emax = {c: -np.inf for c in S.EMIS}
            small = {c: B.SmallK(kk) for c in S.EMIS}
            for i in ix:
                d = days[int(i)]
                if arm == 'TRUE':
                    fld = true_mm[int(i)]
                elif arm == 'OWN':
                    fld = own[d.month - 1, d.weekday()]
                else:
                    fld = other[d.month - 1, d.weekday()]
                for k, c in enumerate(S.EMIS):
                    a = np.asarray(fld[k])
                    emin[c] = min(emin[c], float(a.min()))
                    emax[c] = max(emax[c], float(a.max()))
                    small[c].add(a)
            mins2, maxs2 = (dict(mins), dict(maxs))
            cvals, cinfo = ({}, {})
            for c in S.EMIS:
                mins2[c], maxs2[c] = (emin[c], emax[c])
                v, npos = small[c].result()
                cvals[c], cinfo[c] = percentile_from_parts([(v, npos)], npos, kk)
            c = 'M-9'
            cvals[c], cinfo[c] = percentile_from_parts([sk[ytr][c]], sk[ytr][c][1], kk)
            name = 'NORM_f6_train%d_test%d_%s' % (ytr, yte, arm)
            src = {'TRUE': 'emis_true_%d.npy' % ytr, 'OWN': "emis_clim_%d.npy (the training year's own emission climatology)" % ytr, 'OTHER': "emis_clim_%d.npy (the other year's emission climatology)" % yte}[arm]
            doc = {'run': 'emul', 'stage': 'S3 step (8)', 'made': S.now_local(), 'name': name, 'kind': 'f6', 'train_year': ytr, 'test_year': yte, 'arm': arm, 'training_days': int(len(ix)), 'emission_source': 'E-1 to E-4 from W\\arrays\\' + src + ' on the training days', 'notes': common_note, 'channels': chan_stats(names, mins2, maxs2, cvals, cinfo), 'target': {'min': tmin, 'max': tmax}}
            p = os.path.join(S.CFG, name + '.json')
            index['files'][name] = S.write_json(p, doc)
            written.append(name)
            print('%s %s written' % (S.now_local(), name), flush=True)
        del true_mm, own, other
    index['end'] = S.now_local()
    index['written'] = written
    S.write_json(os.path.join(S.WORK_S3, 'norm_index.json'), index)
    print('done: %d files' % len(written))
if __name__ == '__main__':
    main()
