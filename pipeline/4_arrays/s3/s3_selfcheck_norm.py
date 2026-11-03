"""Recomputes a few scaling statistics of the fold with 2019 held out directly from the stored arrays over its training days and compares them with the fold's statistics file (read only)."""
import json
import os
import sys
sys.path.insert(0, '.\\run\\code')
import numpy as np
from s3 import s3common as S

def main():
    S.common.set_temp()
    with open(os.path.join(S.CFG, 'SPLITS.json'), 'r', encoding='ascii') as f:
        sp = json.load(f)
    with open(os.path.join(S.CFG, 'NORM_fold2019.json'), 'r', encoding='ascii') as f:
        nm = json.load(f)
    with open(os.path.join(S.CFG, 'S3_CHANNELS.json'), 'r', encoding='ascii') as f:
        ch = {c['name']: c for c in json.load(f)['channels']}
    mask = np.load(os.path.join(S.ARRAYS, 'mask_conus_land.npy'))
    sel = sp['folds']['2019']['train']['index_by_year']
    res = {}
    acc = {'M-1': [np.inf, -np.inf], 'X-6': [np.inf, -np.inf], 'T': [np.inf, -np.inf]}
    pos = {'E-1': [], 'M-9': []}
    for y, ix in sel.items():
        y = int(y)
        a16 = np.load(os.path.join(S.ARRAYS, 'inputs_f16_%d.npy' % y), mmap_mode='r')
        a32 = np.load(os.path.join(S.ARRAYS, 'inputs_f32_%d.npy' % y), mmap_mode='r')
        tg = np.load(os.path.join(S.ARRAYS, 'target_%d.npy' % y), mmap_mode='r')
        for i in ix:
            for c in ('M-1', 'X-6'):
                v = np.asarray(a16[i, ch[c]['index']])
                acc[c][0] = min(acc[c][0], float(v.min()))
                acc[c][1] = max(acc[c][1], float(v.max()))
            t = np.asarray(tg[i])[mask]
            acc['T'][0] = min(acc['T'][0], float(t.min()))
            acc['T'][1] = max(acc['T'][1], float(t.max()))
            for c in pos:
                v = np.asarray(a32[i, ch[c]['index']]).ravel()
                pos[c].append(v[v > 0])
        del a16, a32, tg
    for c in ('M-1', 'X-6'):
        res[c] = {'recomputed': acc[c], 'norm': [nm['channels'][c]['min'], nm['channels'][c]['max']]}
    res['target'] = {'recomputed': acc['T'], 'norm': [nm['target']['min'], nm['target']['max']]}
    for c in pos:
        allv = np.concatenate(pos[c])
        pos[c] = None
        p = float(np.percentile(allv, 1))
        res[c] = {'npos': int(allv.size), 'c_np_percentile': p, 'c_norm': nm['channels'][c]['c'], 'rel_diff': abs(p - nm['channels'][c]['c']) / nm['channels'][c]['c']}
        del allv
    z = np.load(os.path.join(S.WORK_S3, 'daystats_2018.npz'))
    j = S.CH_INDEX['X-6']
    k = int(np.argmax(z['daymax'][:, j]))
    res['x6_2018_max'] = {'value': float(z['daymax'][k, j]), 'date': str(z['dates'][k]), 'validation_day': bool(z['isval'][k])}
    print(json.dumps(res, indent=1))
if __name__ == '__main__':
    main()
