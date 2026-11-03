"""Tests the day selection of the main folds' statistics on small synthetic arrays with traps (extreme validation days, scaled years, NaN slots), compares each result with a brute-force computation, and checks that every wrong input stops the computation."""
import json
import os
import shutil
import sys
import tempfile
CODE = '/content/emul/run/code'
for _p in (CODE, CODE + '/s8'):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from s3 import s3_build as B
from s3 import s3common as S
from s3p2 import p2common as P
import s8_d7 as D
YEARS = (2015, 2016, 2017, 2018, 2019)
C7A = (2015, 2016, 2018)
NDAY = 12
GR = (3, 4)
NAMES = list(P.CH_NAMES)
LOGC = list(P.LOG_CHANNELS)
EMIS = list(P.EMIS)
F32 = ['M-9', 'L-2'] + EMIS
RESULTS = []

def isval_of():
    return np.array([i % 4 == 3 for i in range(NDAY)])

def info(path):
    return {'bytes': os.stat(path).st_size, 'sha256': S.sha256_file(path)}

def wjson(path, obj):
    with open(path, 'w', encoding='ascii', newline='\n') as f:
        json.dump(obj, f, indent=1)

def truth():
    rng = np.random.default_rng(20261024)
    isval = isval_of()
    t = {}
    for y in YEARS:
        s = 1.0 + (y - 2015)
        g = {}
        for c in LOGC + ['L-2']:
            a = rng.uniform(0.0, 10.0, (NDAY,) + GR) * s
            a[:, 0, 0] = 0.0
            for i in range(NDAY):
                if isval[i]:
                    a[i, 1, 1] = 1000000.0
                    a[i, 2, 3] = 1e-09
            g[c] = a.astype(np.float32)
        dmin = np.zeros((NDAY, len(NAMES)))
        dmax = np.zeros((NDAY, len(NAMES)))
        for k, c in enumerate(NAMES):
            if c in g:
                dmin[:, k] = g[c].reshape(NDAY, -1).min(axis=1)
                dmax[:, k] = g[c].reshape(NDAY, -1).max(axis=1)
            else:
                lo = rng.uniform(-5.0, 0.0, NDAY) * s
                dmin[:, k] = lo
                dmax[:, k] = lo + rng.uniform(1.0, 5.0, NDAY) * s
        tmin = rng.uniform(0.0, 2.0, NDAY) * s
        tmax = tmin + rng.uniform(10.0, 50.0, NDAY) * s
        for i in range(NDAY):
            if isval[i]:
                dmin[i, :] = -1000000.0
                dmax[i, :] = 1000000.0
                tmin[i], tmax[i] = (-1000000.0, 1000000.0)
        for c in g:
            k = P.CH_INDEX[c] if c in P.CH_INDEX else None
            if k is not None:
                dmin[:, k] = np.where(isval, -1000000.0, g[c].reshape(NDAY, -1).min(axis=1))
                dmax[:, k] = np.where(isval, 1000000.0, g[c].reshape(NDAY, -1).max(axis=1))
        t[y] = {'g': g, 'dmin': dmin, 'dmax': dmax, 'tmin': tmin, 'tmax': tmax}
    return t

def build_tree(root, t):
    pl = {'work': os.path.join(root, 'work'), 'cfg': os.path.join(root, 'cfg'), 'arrays': os.path.join(root, 'arrays'), 'rule': None, 'years': YEARS, 'c7a': C7A}
    for k in ('work', 'cfg', 'arrays'):
        os.makedirs(pl[k], exist_ok=True)
    isval = isval_of()
    dates = np.array(['%d01%02d' % (2015, i + 1) for i in range(NDAY)])
    arrays_cfg = {'years': {}}
    for y in YEARS:
        w = pl['work']
        dmin, dmax = (t[y]['dmin'].copy(), t[y]['dmax'].copy())
        if y in C7A:
            for c in EMIS:
                dmin[:, P.CH_INDEX[c]] = np.nan
                dmax[:, P.CH_INDEX[c]] = np.nan
        dp = os.path.join(w, 'daystats_%d.npz' % y)
        np.savez(dp, daymin=dmin, daymax=dmax, tmin=t[y]['tmin'], tmax=t[y]['tmax'], isval=isval, dates=dates, channels=np.array(NAMES))
        sk, npos = ({}, {})
        for c in LOGC:
            if y in C7A and c in EMIS:
                continue
            v = t[y]['g'][c][~isval].ravel()
            v = np.sort(v[v > 0])
            sk[c.replace('-', '')] = v
            npos[c] = int(v.size)
        kp = os.path.join(w, 'smallk_%d.npz' % y)
        np.savez(kp, **sk)
        wjson(os.path.join(w, 'build_%d.json' % y), {'k_small': B.K_SMALL, 'smallk_npos_nonvalidation': npos, 'daystats': info(dp), 'smallk': info(kp)})
        arr = np.zeros((NDAY, len(F32)) + GR, np.float32)
        for j, c in enumerate(F32):
            arr[:, j] = t[y]['g'][c]
        ap = os.path.join(pl['arrays'], 'inputs_f32_%d.npy' % y)
        np.save(ap, arr)
        f32 = {'sha256': S.sha256_file(ap)}
        if y in C7A:
            f32['partial'] = False
        arrays_cfg['years'][str(y)] = {'files': {'inputs_f32': f32}}
        if y in C7A:
            dp8 = os.path.join(w, 'daystats_s8_%d.npz' % y)
            ks = [P.CH_INDEX[c] for c in EMIS]
            np.savez(dp8, daymin=t[y]['dmin'][:, ks], daymax=t[y]['dmax'][:, ks], isval=isval, dates=dates, channels=np.array(EMIS))
            sk8, n8 = ({}, {})
            for c in EMIS:
                v = t[y]['g'][c][~isval].ravel()
                v = np.sort(v[v > 0])
                sk8[c.replace('-', '')] = v
                n8[c] = int(v.size)
            kp8 = os.path.join(w, 'smallk_s8_%d.npz' % y)
            np.savez(kp8, **sk8)
            wjson(os.path.join(w, 'build_s8_%d.json' % y), {'k_small': B.K_SMALL, 'smallk_npos_nonvalidation': n8, 'daystats': info(dp8), 'smallk': info(kp8), 'file': {'sha256': S.sha256_file(ap)}})
            wjson(os.path.join(w, 'verify_s8_%d.json' % y), {'verdict': 'PASS'})
    wjson(os.path.join(pl['cfg'], 'S3_ARRAYS.json'), arrays_cfg)
    chans = [{'name': c, 'array': 'inputs_f32', 'index': j} for j, c in enumerate(F32)]
    chans += [{'name': c, 'array': 'inputs_f16', 'index': j} for j, c in enumerate((c for c in NAMES if c not in F32))]
    wjson(os.path.join(pl['cfg'], 'S3_CHANNELS.json'), {'channels': chans})
    folds = {}
    for yh in YEARS:
        ty = [y for y in YEARS if y != yh]
        folds[str(yh)] = {'training_years': ty, 'held_out_year': yh, 'train': {'index_by_year': {str(y): [i for i in range(NDAY) if not isval[i]] for y in ty}}}
    wjson(os.path.join(pl['cfg'], 'SPLITS.json'), {'folds': folds})
    wjson(os.path.join(pl['work'], 'norm_index.json'), {'start': 'x', 'files': {}, 'written': [], 'waiting': {'NORM_fold%d' % y: 'waits for stage (8)' for y in YEARS}})
    return pl

def expected(t, yh, lock=None):
    isval = isval_of()
    ty = [y for y in YEARS if y != yh]
    days = {}
    for y in ty:
        ix = [i for i in range(NDAY) if not isval[i]]
        if lock and y in lock:
            ix = [i for i in ix if i not in lock[y]]
        days[y] = ix
    mins = {c: min((float(t[y]['dmin'][i, P.CH_INDEX[c]]) for y in ty for i in days[y])) for c in NAMES}
    maxs = {c: max((float(t[y]['dmax'][i, P.CH_INDEX[c]]) for y in ty for i in days[y])) for c in NAMES}
    cv = {}
    for c in LOGC:
        v = np.concatenate([t[y]['g'][c][days[y]].ravel() for y in ty])
        v = v[v > 0].astype(np.float64)
        cv[c] = float(np.percentile(v, 1.0))
    tmin = min((float(t[y]['tmin'][i]) for y in ty for i in days[y]))
    tmax = max((float(t[y]['tmax'][i]) for y in ty for i in days[y]))
    return (mins, maxs, cv, tmin, tmax, sum((len(v) for v in days.values())))

def same_doc(doc, exp):
    mins, maxs, cv, tmin, tmax, nd = exp
    bad = []
    for c in NAMES:
        r = doc['channels'][c]
        if r['min'] != mins[c] or r['max'] != maxs[c]:
            bad.append('%s min/max' % c)
        if c in LOGC and abs(r['c'] - cv[c]) > 1e-09 * max(1.0, abs(cv[c])):
            bad.append('%s c' % c)
    if doc['target']['min'] != tmin or doc['target']['max'] != tmax:
        bad.append('target')
    if doc['training_days'] != nd:
        bad.append('training_days')
    if doc['held_out_year'] in doc['training_years']:
        bad.append('held-out year in training_years')
    return bad

def record(name, ok, why=''):
    RESULTS.append((name, ok))
    print('%s %s%s' % (name, 'PASS' if ok else 'FAIL', ': ' + why if why else ''), flush=True)

def expect_stop(name, pl, word):
    try:
        D.run(pl, write=False)
    except ValueError as e:
        record(name, word in str(e), str(e)[:90])
        return
    record(name, False, 'no stop')

def main():
    S.common.set_temp()
    t = truth()
    top = tempfile.mkdtemp(prefix='s8_d7_selftest_')
    try:
        pl = build_tree(os.path.join(top, 't1'), t)
        res = D.run(pl, write=True)
        bad = []
        if sorted(res['written']) != ['NORM_fold%d' % y for y in YEARS]:
            bad.append('written %s' % res['written'])
        for yh in YEARS:
            p = os.path.join(pl['cfg'], 'NORM_fold%d.json' % yh)
            doc = P.jload(p)
            bad += ['%d %s' % (yh, b) for b in same_doc(doc, expected(t, yh))]
            if doc.get('lockbox') != 'not decided' or doc.get('kind') != 'fold':
                bad.append('%d lockbox or kind' % yh)
        idx = P.jload(os.path.join(pl['work'], 'norm_index.json'))
        if idx['waiting']:
            bad.append('waiting not emptied')
        if not os.path.isfile(os.path.join(pl['work'], 'norm_index_Backup_PriorToS8_D7.json')):
            bad.append('no norm_index backup')
        lst = P.jload(os.path.join(pl['cfg'], 'S3C_LIST_8_NORM.json'))
        for e in lst['files']:
            fp = os.path.join(pl['work'] if e['name'] == 'norm_index.json' else pl['cfg'], e['name'] if e['name'] == 'norm_index.json' else e['name'] + '.json')
            if S.sha256_file(fp) != e['sha256']:
                bad.append('list row %s' % e['name'])
        if lst['n_files'] != 6:
            bad.append('list rows %d' % lst['n_files'])
        record('T1 five folds by brute force', not bad, '; '.join(bad[:6]))
        lock = {2016: [0, 5], 2019: [2]}
        for y, dl in lock.items():
            for i in dl:
                for c in LOGC + ['L-2']:
                    t[y]['g'][c][i, 0, 1] = np.float32(10000000.0)
                    t[y]['g'][c][i, 0, 2] = np.float32(1e-08)
                    k = P.CH_INDEX[c]
                    t[y]['dmax'][i, k] = float(t[y]['g'][c][i].max())
                    t[y]['dmin'][i, k] = float(t[y]['g'][c][i].min())
                for c in NAMES:
                    if c not in t[y]['g']:
                        t[y]['dmin'][i, P.CH_INDEX[c]] = -10000000.0
                t[y]['tmax'][i] = 10000000.0
        pl = build_tree(os.path.join(top, 't2'), t)
        wjson(os.path.join(pl['cfg'], 'LOCKBOX_DAYS.json'), {'decision': 'self-test', 'days': {str(y): v for y, v in lock.items()}})
        res = D.run(pl, write=False)
        bad = []
        for yh in YEARS:
            doc = res['docs']['NORM_fold%d' % yh]
            bad += ['%d %s' % (yh, b) for b in same_doc(doc, expected(t, yh, lock))]
            want = {str(y): len(v) for y, v in lock.items() if y != yh}
            if doc['lockbox']['days_left_out_by_year'] != want:
                bad.append('%d left out %s' % (yh, doc['lockbox']['days_left_out_by_year']))
        record('T2 lockbox days leave', not bad, '; '.join(bad[:6]))
        pl = build_tree(os.path.join(top, 't3'), t)
        sp = P.jload(os.path.join(pl['cfg'], 'SPLITS.json'))
        sp['folds']['2015']['train']['index_by_year']['2016'].append(3)
        wjson(os.path.join(pl['cfg'], 'SPLITS.json'), sp)
        expect_stop('T3 a validation day in training stops', pl, 'non-validation')
        pl = build_tree(os.path.join(top, 't4'), t)
        sp = P.jload(os.path.join(pl['cfg'], 'SPLITS.json'))
        sp['folds']['2015']['train']['index_by_year']['2015'] = [i for i in range(NDAY) if not isval_of()[i]]
        wjson(os.path.join(pl['cfg'], 'SPLITS.json'), sp)
        expect_stop('T4 the held-out year in training stops', pl, 'held-out')
        pl = build_tree(os.path.join(top, 't5'), t)
        dp = os.path.join(pl['work'], 'daystats_2016.npz')
        z = dict(np.load(dp))
        z['daymin'][0, P.CH_INDEX['E-1']] = 1.0
        np.savez(dp, **z)
        b = P.jload(os.path.join(pl['work'], 'build_2016.json'))
        b['daystats'] = info(dp)
        wjson(os.path.join(pl['work'], 'build_2016.json'), b)
        expect_stop('T5 a finite stage (1) E-1 of 2016 stops', pl, 'NaN')
        pl = build_tree(os.path.join(top, 't6'), t)
        kp = os.path.join(pl['work'], 'smallk_s8_2018.npz')
        z = dict(np.load(kp))
        z['E1'] = z['E1'][1:]
        np.savez(kp, **z)
        expect_stop('T6 a changed smallk_s8 stops', pl, 'build record')
        pl = build_tree(os.path.join(top, 't7'), t)
        os.remove(os.path.join(pl['work'], 'build_s8_2018.json'))
        res = D.run(pl, write=True)
        ok = res['written'] == ['NORM_fold2018'] and sorted(res['waiting']) == ['NORM_fold%d' % y for y in (2015, 2016, 2017, 2019)]
        idx = P.jload(os.path.join(pl['work'], 'norm_index.json'))
        ok = ok and sorted(idx['waiting']) == sorted(res['waiting'])
        ok = ok and (not same_doc(P.jload(os.path.join(pl['cfg'], 'NORM_fold2018.json')), expected(t, 2018)))
        record('T7 a C-7a year not completed: its folds wait', ok, 'written %s' % res['written'])
        pl = build_tree(os.path.join(top, 't8'), t)
        wjson(os.path.join(pl['cfg'], 'LOCKBOX_DAYS.json'), {'2016': {'rows': [1, 2]}})
        expect_stop('T8 a lockbox by place stops', pl, "decision's form")
        lock9 = {2017: [1], 2019: [2]}
        for y, dl in lock9.items():
            for i in dl:
                for c in LOGC + ['L-2']:
                    t[y]['g'][c][i, 0, 1] = np.float32(10000000.0)
                    t[y]['g'][c][i, 0, 2] = np.float32(1e-08)
                    k = P.CH_INDEX[c]
                    t[y]['dmax'][i, k] = float(t[y]['g'][c][i].max())
                    t[y]['dmin'][i, k] = float(t[y]['g'][c][i].min())
                t[y]['tmax'][i] = 10000000.0

        def lock_tree(name):
            p9 = build_tree(os.path.join(top, name), t)
            wjson(os.path.join(p9['cfg'], 'LOCKBOX_DAYS.json'), {'decision': 'self-test', 'days': {str(y): v for y, v in lock9.items()}})
            return p9
        pl = lock_tree('t9')
        a9 = P.jload(os.path.join(pl['cfg'], 'S3_ARRAYS.json'))
        bad = ['%d has the key' % y for y in lock9 if 'partial' in a9['years'][str(y)]['files']['inputs_f32']]
        res = D.run(pl, write=False)
        for yh in YEARS:
            doc = res['docs']['NORM_fold%d' % yh]
            bad += ['%d %s' % (yh, b) for b in same_doc(doc, expected(t, yh, lock9))]
            want = sorted((y for y in lock9 if y != yh))
            if doc['lockbox']['smallk_taken_again_for'] != want:
                bad.append('%d taken again %s' % (yh, doc['lockbox']['smallk_taken_again_for']))
        record('T9 a lockbox year without the partial key passes', not bad, '; '.join(bad[:6]))
        pl = lock_tree('t10')
        ap = os.path.join(pl['cfg'], 'S3_ARRAYS.json')
        a10 = P.jload(ap)
        a10['years']['2017']['files']['inputs_f32']['partial'] = True
        wjson(ap, a10)
        expect_stop('T10 a lockbox year with partial true stops', pl, 'differs from S3_ARRAYS')
        pl = lock_tree('t11')
        fp = os.path.join(pl['arrays'], 'inputs_f32_2019.npy')
        arr = np.load(fp)
        arr[2, 0, 1, 1] = np.float32(5.0)
        np.save(fp, arr)
        expect_stop('T11 a lockbox year whose array differs stops', pl, 'differs from S3_ARRAYS')
    finally:
        shutil.rmtree(top, ignore_errors=True)
    n_ok = sum((1 for _n, ok in RESULTS if ok))
    print('SELF-TEST s8_d7: %d of %d PASS' % (n_ok, len(RESULTS)), flush=True)
    sys.exit(0 if n_ok == len(RESULTS) == 11 else 1)
if __name__ == '__main__':
    S.run_main(main)
