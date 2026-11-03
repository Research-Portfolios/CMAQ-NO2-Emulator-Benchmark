"""Computes the scaling statistics of the five main folds (2015 to 2019 held out) by a rule fixed in advance, on each fold's training days only, with the filled emission channels for 2015, 2016 and 2018, and writes the five statistics files."""
import datetime
import json
import math
import os
import shutil
import sys
CODE = '/content/emul/run/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np
from s3 import s3_build as B
from s3 import s3_norm as N
from s3 import s3common as S
from s3p2 import p2common as P
RULE_SHA256 = '7889ddfee6303734164b974f4084be93be0ac038847702cd577bd943c131ced0'
LOGC = P.LOG_CHANNELS
EMIS = P.EMIS
NAMES = P.CH_NAMES
FOLD_KEYS = ('2015', '2016', '2017', '2018', '2019')

def vm_places():
    return {'work': P.WORK_P2, 'cfg': P.CFG_P2, 'arrays': P.ARRAYS_P2, 'rule': os.path.join(P.RUN, 'ref', 'D7_RULE_ASCII.txt'), 'years': P.YEARS, 'c7a': P.C7A_YEARS}

def say(msg):
    print('%s %s' % (P.now_local(), S.common.mask(msg)), flush=True)

def check_file(path, rec, what):
    if not os.path.isfile(path):
        raise ValueError('%s absent: %s' % (what, os.path.basename(path)))
    got = S.sha256_file(path)
    n = os.stat(path).st_size
    if got != rec.get('sha256') or n != int(rec.get('bytes', -1)):
        raise ValueError('%s differs from its build record: %s' % (what, os.path.basename(path)))
    return got

def npz_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}

def backup_beside(path, reason):
    stem, ext = os.path.splitext(path)
    base = '%s_Backup_PriorTo%s' % (stem, reason)
    dest, k = (base + ext, 1)
    while os.path.exists(dest):
        k += 1
        dest = '%s_%d%s' % (base, k, ext)
    shutil.copyfile(path, dest)
    if os.stat(dest).st_size == 0 or S.sha256_file(dest) != S.sha256_file(path):
        raise ValueError('backup failed: ' + os.path.basename(dest))
    return dest

def built_c7a(pl, y, arrays_cfg):
    w = pl['work']
    bp = os.path.join(w, 'build_s8_%d.json' % y)
    vp = os.path.join(w, 'verify_s8_%d.json' % y)
    if not (os.path.isfile(bp) and os.path.isfile(vp)):
        return (False, 'stage (8) has not written build_s8 and verify_s8')
    ver = P.jload(vp)
    if ver.get('verdict') != 'PASS':
        return (False, 'verify_s8_%d.json verdict %s' % (y, ver.get('verdict')))
    b8 = P.jload(bp)
    f32 = arrays_cfg['years'][str(y)]['files']['inputs_f32']
    if f32.get('partial') is not False:
        return (False, 'S3_ARRAYS.json still calls inputs_f32_%d partial' % y)
    if f32.get('sha256') != b8.get('file', {}).get('sha256'):
        return (False, "S3_ARRAYS.json's inputs_f32_%d sha256 differs from build_s8_%d.json's" % (y, y))
    return (True, 'completed by stage (8)')

def load_sources(pl, arrays_cfg):
    w = pl['work']
    kk = B.K_SMALL
    ds, sk, built, src = ({}, {}, {}, {})
    for y in pl['years']:
        b = P.jload(os.path.join(w, 'build_%d.json' % y))
        if b['k_small'] != kk:
            raise ValueError('K differs in build_%d.json' % y)
        dp = os.path.join(w, 'daystats_%d.npz' % y)
        check_file(dp, b['daystats'], 'daystats_%d.npz' % y)
        d = npz_dict(dp)
        if list(d['channels']) != list(NAMES):
            raise ValueError('daystats channels differ in %d' % y)
        kp = os.path.join(w, 'smallk_%d.npz' % y)
        check_file(kp, b['smallk'], 'smallk_%d.npz' % y)
        z2 = npz_dict(kp)
        npos = b['smallk_npos_nonvalidation']
        sk[y] = {c: (z2[c.replace('-', '')], npos[c]) for c in LOGC if c in npos}
        src[y] = {'all channels but the five': 'stage (1): daystats_%d.npz, smallk_%d.npz, build_%d.json' % (y, y, y)}
        built[y] = True
        if y in pl['c7a']:
            ok, why = built_c7a(pl, y, arrays_cfg)
            built[y] = ok
            if not ok:
                src[y]['E-1 to E-4, P-1'] = 'not built: ' + why
                ds[y] = d
                continue
            for c in EMIS:
                k = P.CH_INDEX[c]
                if np.isfinite(d['daymin'][:, k]).any() or np.isfinite(d['daymax'][:, k]).any():
                    raise ValueError("stage (1)'s %s extremes of %d are not NaN" % (c, y))
                if c in sk[y]:
                    raise ValueError("stage (1)'s smallk of %d holds %s" % (y, c))
            b8 = P.jload(os.path.join(w, 'build_s8_%d.json' % y))
            if b8['k_small'] != kk:
                raise ValueError('K differs in build_s8_%d.json' % y)
            dp8 = os.path.join(w, 'daystats_s8_%d.npz' % y)
            check_file(dp8, b8['daystats'], 'daystats_s8_%d.npz' % y)
            d8 = npz_dict(dp8)
            if list(d8['channels']) != list(EMIS):
                raise ValueError('daystats_s8 channels differ in %d' % y)
            if list(d8['dates']) != list(d['dates']) or not np.array_equal(d8['isval'], d['isval']):
                raise ValueError("daystats_s8_%d dates or isval differ from stage (1)'s" % y)
            for j, c in enumerate(EMIS):
                k = P.CH_INDEX[c]
                d['daymin'][:, k] = d8['daymin'][:, j]
                d['daymax'][:, k] = d8['daymax'][:, j]
            kp8 = os.path.join(w, 'smallk_s8_%d.npz' % y)
            check_file(kp8, b8['smallk'], 'smallk_s8_%d.npz' % y)
            z8 = npz_dict(kp8)
            n8 = b8['smallk_npos_nonvalidation']
            for c in EMIS:
                sk[y][c] = (z8[c.replace('-', '')], n8[c])
            src[y]['E-1 to E-4, P-1'] = 'stage (8): daystats_s8_%d.npz, smallk_s8_%d.npz, build_s8_%d.json' % (y, y, y)
        ds[y] = d
    return (ds, sk, built, src)

def read_lockbox(pl):
    p = os.path.join(pl['cfg'], 'LOCKBOX_DAYS.json')
    if not os.path.isfile(p):
        return None
    doc = P.jload(p)
    days = doc.get('days', doc) if isinstance(doc, dict) else None
    if not isinstance(days, dict):
        raise ValueError('LOCKBOX_DAYS.json is not year -> day indices')
    out = {}
    for k, v in days.items():
        if not (isinstance(k, str) and len(k) == 4 and k.isdigit()):
            continue
        if not isinstance(v, list) or not all((isinstance(i, int) and (not isinstance(i, bool)) for i in v)):
            raise ValueError("LOCKBOX_DAYS.json %s: not a list of day indices (a lockbox by place or period waits for its decision's form)" % k)
        out[int(k)] = sorted(set(v))
    if not out:
        raise ValueError("LOCKBOX_DAYS.json names no year (a lockbox by place or period waits for its decision's form)")
    return (out, S.sha256_file(p))

def f32_layout(pl):
    ch = P.jload(os.path.join(pl['cfg'], 'S3_CHANNELS.json'))
    idx = {}
    for e in ch['channels']:
        if e['name'] in LOGC:
            if e['array'] != 'inputs_f32':
                raise ValueError('%s is not in inputs_f32' % e['name'])
            idx[e['name']] = int(e['index'])
    if sorted(idx) != sorted(LOGC):
        raise ValueError('S3_CHANNELS.json lacks a channel of %s' % (LOGC,))
    return idx

def smallk_again(pl, y, ix, idx, arrays_cfg):
    kk = B.K_SMALL
    path = os.path.join(pl['arrays'], 'inputs_f32_%d.npy' % y)
    f32 = arrays_cfg['years'][str(y)]['files']['inputs_f32']
    if f32.get('partial') is True or S.sha256_file(path) != f32.get('sha256'):
        raise ValueError('inputs_f32_%d.npy differs from S3_ARRAYS.json' % y)
    arr = np.load(path, mmap_mode='r')
    small = {c: B.SmallK(kk) for c in LOGC}
    for i in ix:
        for c in LOGC:
            small[c].add(np.asarray(arr[int(i), idx[c]]))
    del arr
    return {c: small[c].result() for c in LOGC}

def run(pl, write=True):
    t0 = P.now_local()
    rule_sha = None
    if pl.get('rule'):
        rule_sha = S.sha256_file(pl['rule'])
        if rule_sha != RULE_SHA256:
            raise ValueError("the d7 rule's copy differs from the frozen sha256")
    sp = P.jload(os.path.join(pl['cfg'], 'SPLITS.json'))
    arrays_path = os.path.join(pl['cfg'], 'S3_ARRAYS.json')
    arrays_cfg = P.jload(arrays_path)
    arrays_sha = S.sha256_file(arrays_path)
    ds, sk, built, src = load_sources(pl, arrays_cfg)
    kk = B.K_SMALL
    lb = read_lockbox(pl)
    if lb is not None:
        for y, v in lb[0].items():
            if y not in ds or any((i < 0 or i >= len(ds[y]['isval']) for i in v)):
                raise ValueError('LOCKBOX_DAYS.json %d: a day index outside the year' % y)
    idx = f32_layout(pl) if lb is not None else None
    common_note = {'stored_values': 'statistics of the stored values: M-1 as (x - 50000) / 60000 in float16; float16 and float32 channels as stored', 'no_clipping': 'held-out values outside the training range are kept', 'c_rule': "c: the 1st percentile (numpy's linear method) of the channel's positive values (x > 0) on the training days, all grid cells; log10 values in float64", 'target': 'T-1, ppb, minimum and maximum over the masked cells (W\\arrays_p2\\mask_conus_land.npy) of the training days', 'S3_ARRAYS_sha256': arrays_sha, 'channels_file': 'run\\config\\p2\\S3_CHANNELS.json'}

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
    if sorted(sp['folds']) != list(FOLD_KEYS):
        raise ValueError('SPLITS.json folds are %s' % sorted(sp['folds']))
    docs, waiting, again = ({}, {}, {})
    for yh in FOLD_KEYS:
        f = sp['folds'][yh]
        name = 'NORM_fold%s' % yh
        sel = train_sel(f)
        check_sel(sel)
        if int(yh) in sel:
            raise ValueError('held-out year in training')
        if 'training_years' in f and sorted((int(y) for y in f['training_years'])) != sorted(sel):
            raise ValueError("%s: training_years differ from the days' years" % name)
        wait = [y for y in sorted(sel) if not built.get(y)]
        if wait:
            waiting[name] = 'waits for stage (8): its training years %s include %s, not completed (%s)' % (sorted(sel), wait, '; '.join((src[y]['E-1 to E-4, P-1'] for y in wait)))
            continue
        removed = {}
        if lb is not None:
            days, _lsha = lb
            for y in sorted(sel):
                drop = np.isin(sel[y], np.array(days.get(y, []), dtype=np.int64))
                if drop.any():
                    removed[y] = int(drop.sum())
                    sel[y] = sel[y][~drop]
        mins, maxs, tmin, tmax = base_stats(sel, NAMES)
        parts_of = {}
        for y in sorted(sel):
            if y in removed:
                if y not in again or not np.array_equal(again[y][0], sel[y]):
                    again[y] = (sel[y].copy(), smallk_again(pl, y, sel[y], idx, arrays_cfg))
                parts_of[y] = again[y][1]
            else:
                parts_of[y] = sk[y]
        cvals, cinfo = ({}, {})
        for c in LOGC:
            parts = [parts_of[y][c] for y in sorted(sel)]
            cvals[c], cinfo[c] = N.percentile_from_parts(parts, sum((n for _p, n in parts)), kk)
        lock = 'not decided'
        if lb is not None:
            lock = {'file': 'run\\config\\p2\\LOCKBOX_DAYS.json', 'sha256': lb[1], 'days_left_out_by_year': {str(y): n for y, n in sorted(removed.items())}, 'smallk_taken_again_for': sorted(removed)}
        docs[name] = {'run': 'emul', 'stage': 'the second pass, the statistics: the main folds, the fills stage, mode d7', 'made': P.now_local(), 'name': name, 'kind': 'fold', 'held_out_year': int(yh), 'training_years': sorted(sel), 'training_days': int(sum((len(v) for v in sel.values()))), 'emission_source': "E-1 to E-4 and P-1 from W\\arrays_p2\\inputs_f32_<y>.npy (each year's own days: 2017 and 2019 by stage (1), 2015, 2016 and 2018 by stage (8)) on the training days; every other channel from inputs_f16 and inputs_f32", 'sources': {str(y): src[y] for y in sorted(sel)}, 'rule': {'file': 'RUN/ref/D7_RULE_ASCII.txt', 'sha256': RULE_SHA256, 'checked': rule_sha is not None}, 'lockbox': lock, 'notes': common_note, 'channels': N.chan_stats(NAMES, mins, maxs, cvals, cinfo), 'target': {'min': tmin, 'max': tmax}}
    summary = {'start': t0, 'written': [], 'waiting': waiting, 'lockbox': 'not decided' if lb is None else 'applied'}
    if not write:
        summary['docs'] = docs
        return summary
    ip = os.path.join(pl['work'], 'norm_index.json')
    if not os.path.isfile(ip):
        raise ValueError('norm_index.json absent: stage (1) has not run')
    infos = {}
    for name in sorted(docs):
        p = os.path.join(pl['cfg'], name + '.json')
        if os.path.exists(p):
            backup_beside(p, 'S8_D7')
        infos[name] = P.write_json(p, docs[name])
        summary['written'].append(name)
        say('%s written (%d bytes, sha16 %s)' % (name, infos[name]['bytes'], infos[name]['sha256'][:16]))
    bk = backup_beside(ip, 'S8_D7')
    index = P.jload(ip)
    index.setdefault('waiting', {})
    index.setdefault('files', {})
    index.setdefault('written', [])
    for name in sorted(infos):
        index['waiting'].pop(name, None)
        index['files'][name] = infos[name]
        if name not in index['written']:
            index['written'].append(name)
    for name, why in waiting.items():
        index['waiting'][name] = why
    index['d7'] = {'made': P.now_local(), 'written': sorted(infos), 'waiting': sorted(waiting), 'rule_sha256': RULE_SHA256, 'lockbox': summary['lockbox'], 'backup_before': os.path.basename(bk)}
    ii = P.write_json(ip, index)
    files = [dict(infos[n], kind='norm', name=n) for n in sorted(infos)]
    files.append(dict(ii, kind='norm_index', name='norm_index.json'))
    lst = {'run': 'emul', 'stage': 'the fills stage, mode d7', 'made': P.now_local(), 'design': 'the frozen d7 rule, sha256 %s' % RULE_SHA256, 'files': files, 'n_files': len(files), 'bytes': int(sum((e['bytes'] for e in files))), 'waiting': waiting}
    li = P.write_json(os.path.join(pl['cfg'], 'S3C_LIST_8_NORM.json'), lst)
    say('written norm_index.json (backup %s) and S3C_LIST_8_NORM.json (%d bytes, sha16 %s)' % (os.path.basename(bk), li['bytes'], li['sha256'][:16]))
    read_back(pl, infos)
    summary['index'] = ii
    summary['list'] = li
    return summary

def read_back(pl, infos):
    for name, info in infos.items():
        p = os.path.join(pl['cfg'], name + '.json')
        if S.sha256_file(p) != info['sha256']:
            raise ValueError('%s changed after its write' % name)
        d = P.jload(p)
        if d.get('kind') != 'fold' or d.get('name') != name or list(d['channels']) != list(NAMES):
            raise ValueError('%s: form' % name)
        if int(d['held_out_year']) in d['training_years']:
            raise ValueError('%s: held-out year in training' % name)
        for c, r in d['channels'].items():
            vals = [r['min'], r['max']]
            if c in LOGC:
                if not r['c'] > 0:
                    raise ValueError('%s %s: c not positive' % (name, c))
                vals += [r['c'], r['log10_min'], r['log10_max']]
            if not all((math.isfinite(float(v)) for v in vals)):
                raise ValueError('%s %s: a non-finite value' % (name, c))
        if not (math.isfinite(d['target']['min']) and math.isfinite(d['target']['max'])):
            raise ValueError('%s: target not finite' % name)
    say('read back: %d files, form and values PASS' % len(infos))

def main():
    S.common.set_temp()
    pl = vm_places()
    say('d7 start; rule sha256 %s' % RULE_SHA256[:16])
    res = run(pl)
    say('d7 end: written %d (%s); waiting %d (%s); lockbox %s' % (len(res['written']), ', '.join(res['written']) or 'none', len(res['waiting']), ', '.join(sorted(res['waiting'])) or 'none', res['lockbox']))
    out = {'made': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'), 'written': res['written'], 'waiting': res['waiting'], 'lockbox': res['lockbox']}
    with open(os.path.join(P.LOGS, 'S8_D7_RESULT.json'), 'w', encoding='ascii', newline='\n') as f:
        json.dump(out, f, indent=1, sort_keys=True)
    sys.exit(0 if len(res['written']) == len(FOLD_KEYS) else 3)
if __name__ == '__main__':
    S.run_main(main)
