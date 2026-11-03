"""Computes the scaling statistics of the second pass for the folds whose channels were all built (the two-year emission tests and the climatological-emission folds): channel minima and maxima on the training days, the constant c and log bounds of the log-scaled channels, and the target's range over the land cells."""
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3_build as B
from s3 import s3_norm as N
from s3 import s3common as S
from s3p2 import p2common as P
LOGC = P.LOG_CHANNELS
EMIS = P.EMIS

def main():
    S.common.set_temp()
    t0 = P.now_local()
    sp = P.jload(os.path.join(P.CFG_P2, 'SPLITS.json'))
    arrays_sha = S.sha256_file(os.path.join(P.CFG_P2, 'S3_ARRAYS.json'))
    ds, sk, ds7, sk7 = ({}, {}, {}, {})
    kk = B.K_SMALL
    for y in P.YEARS:
        z = np.load(os.path.join(P.WORK_P2, 'daystats_%d.npz' % y))
        ds[y] = {k: z[k] for k in z.files}
        if list(ds[y]['channels']) != list(P.CH_NAMES):
            raise ValueError('daystats channels differ in %d' % y)
        b = P.jload(os.path.join(P.WORK_P2, 'build_%d.json' % y))
        z2 = np.load(os.path.join(P.WORK_P2, 'smallk_%d.npz' % y))
        sk[y] = {c: (z2[c.replace('-', '')], b['smallk_npos_nonvalidation'][c]) for c in LOGC if c in b['smallk_npos_nonvalidation']}
        if b['k_small'] != kk:
            raise ValueError('K differs')
        z = np.load(os.path.join(P.WORK_P2, 'daystats_c7_%d.npz' % y))
        ds7[y] = {k: z[k] for k in z.files}
        b7 = P.jload(os.path.join(P.WORK_P2, 'build_c7_%d.json' % y))
        z3 = np.load(os.path.join(P.WORK_P2, 'smallk_c7_%d.npz' % y))
        sk7[y] = {c: (z3[c.replace('-', '')], b7['smallk_npos_nonvalidation'][c]) for c in EMIS}
    names = P.CH_NAMES
    common_note = {'stored_values': 'statistics of the stored values: M-1 as (x - 50000) / 60000 in float16; float16 and float32 channels as stored', 'no_clipping': 'held-out values outside the training range are kept', 'c_rule': "c: the 1st percentile (numpy's linear method) of the channel's positive values (x > 0) on the training days, all grid cells; log10 values in float64", 'target': 'T-1, ppb, minimum and maximum over the masked cells (W\\arrays_p2\\mask_conus_land.npy) of the training days', 'S3_ARRAYS_sha256': arrays_sha, 'channels_file': 'run\\config\\p2\\S3_CHANNELS.json'}
    index = {'start': t0, 'files': {}, 'waiting': {}}

    def train_sel(spl):
        return {int(y): np.array(v, dtype=np.int64) for y, v in spl['train']['index_by_year'].items()}

    def check_sel(sel):
        for y, ix in sel.items():
            isval = ds[y]['isval']
            if isval[ix].any() or len(ix) != int((~isval).sum()):
                raise ValueError("training days of %d are not the year's non-validation days" % y)

    def base_stats(sel, chans):
        mins = {c: min((float(np.min(ds[y]['daymin'][ix, P.CH_INDEX[c]])) for y, ix in sel.items())) for c in chans}
        maxs = {c: max((float(np.max(ds[y]['daymax'][ix, P.CH_INDEX[c]])) for y, ix in sel.items())) for c in chans}
        tmin = min((float(np.min(ds[y]['tmin'][ix])) for y, ix in sel.items()))
        tmax = max((float(np.max(ds[y]['tmax'][ix])) for y, ix in sel.items()))
        for c in chans:
            if not (np.isfinite(mins[c]) and np.isfinite(maxs[c])):
                raise ValueError('%s: a non-finite extreme' % c)
        return (mins, maxs, tmin, tmax)
    written = []
    for yh in sp['folds']:
        tyears = sp['folds'][yh]['training_years']
        wait = [y for y in tyears if y in P.C7A_YEARS]
        index['waiting']['NORM_fold%s' % yh] = 'waits for stage (8): its training years %s include %s, whose E-1 to E-4 and P-1 are not built yet' % (tyears, wait)
    non_emis = [c for c in names if c not in EMIS]
    for yh, f in sp['f6a'].items():
        sel = train_sel(f)
        check_sel(sel)
        if int(yh) in sel:
            raise ValueError('held-out year in training')
        mins, maxs, tmin, tmax = base_stats(sel, non_emis)
        for j, c in enumerate(EMIS):
            mins[c] = min((float(np.min(ds7[y]['daymin'][ix, j])) for y, ix in sel.items()))
            maxs[c] = max((float(np.max(ds7[y]['daymax'][ix, j])) for y, ix in sel.items()))
        cvals, cinfo = ({}, {})
        parts = [sk[y]['M-9'] for y in sel]
        cvals['M-9'], cinfo['M-9'] = N.percentile_from_parts(parts, sum((n for _p, n in parts)), kk)
        for c in EMIS:
            parts = [sk7[y][c] for y in sel]
            cvals[c], cinfo[c] = N.percentile_from_parts(parts, sum((n for _p, n in parts)), kk)
        name = 'NORM_f6a_fold%s' % yh
        doc = {'run': 'emul', 'stage': 'the second pass', 'made': P.now_local(), 'name': name, 'kind': 'f6a', 'held_out_year': int(yh), 'training_years': sorted(sel), 'training_days': int(sum((len(v) for v in sel.values()))), 'emission_source': 'E-1 to E-4 and P-1 from W\\arrays_p2\\inputs_c7_<y>.npy (the emission climatology of 2017 and 2019) on the training days; every other channel from inputs_f16 and inputs_f32', 'notes': common_note, 'channels': N.chan_stats(names, mins, maxs, cvals, cinfo), 'target': {'min': tmin, 'max': tmax}}
        p = os.path.join(P.CFG_P2, name + '.json')
        index['files'][name] = P.write_json(p, doc)
        written.append(name)
        print('%s %s written' % (P.now_local(), name), flush=True)
    for cfg, f in sp['f6'].items():
        ytr, yte = (f['train_year'], f['test_year'])
        sel = train_sel(f)
        check_sel(sel)
        if list(sel) != [ytr]:
            raise ValueError('two-year emission test, training years %s' % list(sel))
        ix = sel[ytr]
        mins, maxs, tmin, tmax = base_stats(sel, names)
        days = P.days_of_year(ytr)
        true_mm = np.load(os.path.join(P.ARRAYS_P2, 'emis_true_%d.npy' % ytr), mmap_mode='r')
        own = np.load(os.path.join(P.ARRAYS_P2, 'emis_clim_%d.npy' % ytr), mmap_mode='r')
        other = np.load(os.path.join(P.ARRAYS_P2, 'emis_clim_%d.npy' % yte), mmap_mode='r')
        for arm in ('TRUE', 'OTHER', 'OWN'):
            emin = {c: np.inf for c in EMIS}
            emax = {c: -np.inf for c in EMIS}
            small = {c: B.SmallK(kk) for c in EMIS}
            for i in ix:
                d = days[int(i)]
                if arm == 'TRUE':
                    fld = np.asarray(true_mm[int(i)])
                elif arm == 'OWN':
                    fld = np.asarray(own[d.month - 1, d.weekday()])
                else:
                    fld = np.asarray(other[d.month - 1, d.weekday()])
                for k, c in enumerate(EMIS):
                    a = fld[k]
                    emin[c] = min(emin[c], float(a.min()))
                    emax[c] = max(emax[c], float(a.max()))
                    small[c].add(a)
            mins2, maxs2 = (dict(mins), dict(maxs))
            cvals, cinfo = ({}, {})
            for c in EMIS:
                mins2[c], maxs2[c] = (emin[c], emax[c])
                v, npos = small[c].result()
                cvals[c], cinfo[c] = N.percentile_from_parts([(v, npos)], npos, kk)
            c = 'M-9'
            cvals[c], cinfo[c] = N.percentile_from_parts([sk[ytr][c]], sk[ytr][c][1], kk)
            name = 'NORM_f6_train%d_test%d_%s' % (ytr, yte, arm)
            src = {'TRUE': 'emis_true_%d.npy' % ytr, 'OWN': "emis_clim_%d.npy (the training year's own emission climatology)" % ytr, 'OTHER': "emis_clim_%d.npy (the other year's emission climatology)" % yte}[arm]
            doc = {'run': 'emul', 'stage': 'the second pass', 'made': P.now_local(), 'name': name, 'kind': 'f6', 'train_year': ytr, 'test_year': yte, 'arm': arm, 'training_days': int(len(ix)), 'emission_source': 'E-1 to E-4 and P-1 from W\\arrays_p2\\' + src + ' on the training days', 'notes': common_note, 'channels': N.chan_stats(names, mins2, maxs2, cvals, cinfo), 'target': {'min': tmin, 'max': tmax}}
            p = os.path.join(P.CFG_P2, name + '.json')
            index['files'][name] = P.write_json(p, doc)
            written.append(name)
            print('%s %s written' % (P.now_local(), name), flush=True)
        del true_mm, own, other
    index['end'] = P.now_local()
    index['written'] = written
    P.write_json(os.path.join(P.WORK_P2, 'norm_index.json'), index)
    print('done: %d files; waiting: %s' % (len(written), sorted(index['waiting'])))
if __name__ == '__main__':
    P.run_main(main)
