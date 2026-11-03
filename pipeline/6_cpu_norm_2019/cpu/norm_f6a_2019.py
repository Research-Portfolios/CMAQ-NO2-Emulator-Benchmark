"""Rewrites the scaling statistics of the climatological-emission fold with 2019 held out, with the held-back and buffer days left out of the training days, by the same code as the earlier rewrite."""
import ast
import json
import math
import os
import shutil
import sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common as _dlc
from choi import f16 as _f16
import numpy as np
from s3 import s3_build as B
from s3 import s3_norm as N
from s3 import s3common as S
from s3p2 import p2common as P
from train import guard as GD
ADD2_SHA256 = '25375e98ec39cb4ceadd4f02ad7cdc93715037ff3bc8dc4ac898fef7e5d16ae1'
ADD2_FILE = 'the rule of the 2019 rewrite'
ADD7_SHA256 = '2a6e8c926224a1c5febab6101589650c60019af8c7c9333157c68637770e69ef'
ADD7_FILE = 'the rule of the rewrite'
RULED_BY = 'R-01'
LOGC = P.LOG_CHANNELS
EMIS = P.EMIS
NAMES = P.CH_NAMES
F6A_FOLDS = ('2019',)
F6_KEYS = ()
ARMS = ('TRUE', 'OTHER', 'OWN')
INNER = ('train_sel', 'check_sel', 'base_stats')
P2_NORM = os.path.join(CODE, 's3p2', 'p2_norm.py')
LIST_NAME = 'S3C_LIST_9_NORM_F6A_2019.json'
BACKUP_REASON = 'NormF6aT2'
INDEX_KEY = 'norm_f6a_2019'
INDEX_7C_KEY = 'norm_f6'

def file_names():
    out = ['NORM_f6a_fold%s' % y for y in F6A_FOLDS]
    for k in F6_KEYS:
        for arm in ARMS:
            out.append('NORM_f6_%s_%s' % (k, arm))
    return out

def vm_places():
    return {'work': P.WORK_P2, 'cfg': P.CFG_P2, 'arrays': P.ARRAYS_P2, 'years': P.YEARS}

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

def _inner_nodes(path, outer):
    tree = ast.parse(open(path, 'rb').read().decode('ascii'), path)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == outer]
    if len(fn) != 1:
        raise ValueError('%s: no single %s()' % (os.path.basename(path), outer))
    return {n.name: ast.dump(n) for n in fn[0].body if isinstance(n, ast.FunctionDef) and n.name in INNER}

def inner_check(p2_norm_path=P2_NORM):
    mine = _inner_nodes(os.path.abspath(__file__), 'run')
    theirs = _inner_nodes(p2_norm_path, 'main')
    if sorted(mine) != sorted(INNER) or sorted(theirs) != sorted(INNER):
        raise ValueError('the nested functions %s are not all found' % (INNER,))
    bad = [n for n in INNER if mine[n] != theirs[n]]
    if bad:
        raise ValueError("the copies of %s differ from p2_norm.py's" % bad)
    return {'p2_norm_sha256': S.sha256_file(p2_norm_path), 'functions': list(INNER), 'equal': True}

def read_lockbox(cfg_dir):
    p = os.path.join(cfg_dir, 'LOCKBOX_DAYS.json')
    if not os.path.isfile(p):
        raise ValueError('config\\p2\\LOCKBOX_DAYS.json is absent: the rewrite needs the decided lockbox')
    doc = P.jload(p)
    days, _lock = GD.lockbox_maps(doc)
    return (doc, {y: sorted(s) for y, s in days.items()}, S.sha256_file(p))

class Arrays:

    def __init__(self, pl, arrays_cfg):
        self.pl = pl
        self.cfg = arrays_cfg
        self.tested = {}

    def entry(self, kind, y=None):
        if kind in ('inputs_f32', 'inputs_c7'):
            return ('%s_%d.npy' % (kind, y), self.cfg['years'][str(y)]['files'][kind])
        return (kind, self.cfg['other'][kind])

    def open(self, kind, y=None):
        name, ent = self.entry(kind, y)
        path = os.path.join(self.pl['arrays'], name)
        if name not in self.tested:
            if not os.path.isfile(path):
                raise ValueError('%s absent' % name)
            if kind == 'inputs_f32' and ent.get('partial') is True:
                raise ValueError('S3_ARRAYS.json still calls %s partial' % name)
            got = S.sha256_file(path)
            if got != ent.get('sha256'):
                raise ValueError('%s differs from S3_ARRAYS.json (sha256)' % name)
            self.tested[name] = got
        return np.load(path, mmap_mode='r')

def smallk_days(arr, ix, pick):
    sm = B.SmallK(B.K_SMALL)
    for i in ix:
        sm.add(np.asarray(arr[int(i)][pick]))
    return sm.result()

def run(pl, write=True):
    t0 = P.now_local()
    inner = inner_check()
    sp = P.jload(os.path.join(pl['cfg'], 'SPLITS.json'))
    arrays_path = os.path.join(pl['cfg'], 'S3_ARRAYS.json')
    arrays_cfg = P.jload(arrays_path)
    arrays_sha = S.sha256_file(arrays_path)
    chans = P.jload(os.path.join(pl['cfg'], 'S3_CHANNELS.json'))
    m9 = [int(e['index']) for e in chans['channels'] if e['name'] == 'M-9' and e['array'] == 'inputs_f32']
    if len(m9) != 1:
        raise ValueError('S3_CHANNELS.json: M-9 is not one inputs_f32 channel')
    m9 = m9[0]
    c7_order = [str(s).split()[0] for s in chans.get('f6a_emission_arrays', {}).get('channel_order', [])]
    f6_order = [str(s).split()[0] for s in chans.get('f6_emission_arrays', {}).get('channel_order', [])]
    if tuple(c7_order) != EMIS or tuple(f6_order) != EMIS:
        raise ValueError("the emission arrays' channel order is not %s" % (EMIS,))
    lb_doc, lb_days, lb_sha = read_lockbox(pl['cfg'])
    ip = os.path.join(pl['work'], 'norm_index.json')
    if not os.path.isfile(ip):
        raise ValueError('norm_index.json absent: stage (1) has not run')
    index0 = P.jload(ip)
    d7 = index0.get('d7') or {}
    if d7.get('lockbox') != 'applied' or len(d7.get('written', [])) != 5:
        raise ValueError('norm_index.json shows no d7 with the lockbox applied and five files written: the rewrite runs after d7')
    if INDEX_7C_KEY not in index0 or INDEX_KEY in index0:
        raise ValueError("norm_index.json is not the CPU stage's updated index (its %r entry absent, or %r there already): copy it from RUN/config/p2_rev/ again" % (INDEX_7C_KEY, INDEX_KEY))
    stage1 = {}
    for name in file_names():
        p = os.path.join(pl['cfg'], name + '.json')
        rec = (index0.get('files') or {}).get(name) or {}
        if not os.path.isfile(p):
            raise ValueError('%s.json absent from config\\p2' % name)
        got = S.sha256_file(p)
        if got != rec.get('sha256') or 'lockbox' in P.jload(p):
            raise ValueError("%s.json is not stage (1)'s file (its sha256 differs from norm_index.json's, or it holds a lockbox key): copy config\\p2 from the Drive again" % name)
        sp1 = os.path.join(pl['cfg'], name + '_STAGE1.json')
        if os.path.exists(sp1) and S.sha256_file(sp1) != got:
            raise ValueError("%s_STAGE1.json differs from stage (1)'s file" % name)
        stage1[name] = {'path': p, 'sha256': got}
    kk = B.K_SMALL
    ds, ds7 = ({}, {})
    for y in pl['years']:
        b = P.jload(os.path.join(pl['work'], 'build_%d.json' % y))
        if b['k_small'] != kk:
            raise ValueError('K differs in build_%d.json' % y)
        dp = os.path.join(pl['work'], 'daystats_%d.npz' % y)
        check_file(dp, b['daystats'], 'daystats_%d.npz' % y)
        ds[y] = npz_dict(dp)
        if list(ds[y]['channels']) != list(NAMES):
            raise ValueError('daystats channels differ in %d' % y)
        b7 = P.jload(os.path.join(pl['work'], 'build_c7_%d.json' % y))
        if b7['k_small'] != kk:
            raise ValueError('K differs in build_c7_%d.json' % y)
        dp7 = os.path.join(pl['work'], 'daystats_c7_%d.npz' % y)
        check_file(dp7, b7['daystats'], 'daystats_c7_%d.npz' % y)
        ds7[y] = npz_dict(dp7)
        if list(ds7[y]['channels']) != list(EMIS) or list(ds7[y]['dates']) != list(ds[y]['dates']) or (not np.array_equal(ds7[y]['isval'], ds[y]['isval'])):
            raise ValueError('daystats_c7_%d channels, dates or isval differ' % y)
    for y, v in lb_days.items():
        if y not in ds or any((i < 0 or i >= len(ds[y]['isval']) for i in v)):
            raise ValueError('LOCKBOX_DAYS.json %d: a day index outside the year' % y)
    arrs = Arrays(pl, arrays_cfg)
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

    def lockbox_out(sel, spl, what):
        removed = {}
        for y in sorted(sel):
            drop = np.isin(sel[y], np.array(lb_days.get(y, []), dtype=np.int64))
            removed[y] = int(drop.sum())
            sel[y] = sel[y][~drop]
        GD.roles_guard([(y, int(i)) for y in sorted(sel) for i in sel[y]], 'stat', GD.roles_of(spl, lb_doc))
        if not all((len(v) for v in sel.values())):
            raise ValueError('%s: a training year without days' % what)
        return removed

    def lock_key(removed):
        return {'file': 'LOCKBOX_DAYS.json', 'sha256': lb_sha, 'decided_by': RULED_BY, 'days_removed': {str(y): n for y, n in sorted(removed.items())}}
    again = {}

    def parts_of(kind, y, ix, pick, key):
        k = (kind, y, key, tuple((int(i) for i in ix)))
        if k not in again:
            arr = arrs.open(kind, y)
            again[k] = smallk_days(arr, ix, pick)
            del arr
        return again[k]
    stage = 'the second pass; written again with the lockbox and buffer days absent by %s (sha256 %s), which applies %s (sha256 %s) to this file (the 2019 normalization stage, code\\cpu\\norm_f6a_2019.py)' % (ADD2_FILE, ADD2_SHA256[:16], ADD7_FILE, ADD7_SHA256[:16])
    docs = {}
    names = NAMES
    non_emis = [c for c in names if c not in EMIS]
    for yh in F6A_FOLDS:
        f = sp['f6a'][yh]
        sel = train_sel(f)
        check_sel(sel)
        if int(yh) in sel:
            raise ValueError('held-out year in training')
        name = 'NORM_f6a_fold%s' % yh
        removed = lockbox_out(sel, f, name)
        mins, maxs, tmin, tmax = base_stats(sel, non_emis)
        for j, c in enumerate(EMIS):
            mins[c] = min((float(np.min(ds7[y]['daymin'][ix, j])) for y, ix in sel.items()))
            maxs[c] = max((float(np.max(ds7[y]['daymax'][ix, j])) for y, ix in sel.items()))
        cvals, cinfo = ({}, {})
        parts = [parts_of('inputs_f32', y, sel[y], m9, 'M-9') for y in sel]
        cvals['M-9'], cinfo['M-9'] = N.percentile_from_parts(parts, sum((n for _p, n in parts)), kk)
        for j, c in enumerate(EMIS):
            parts = [parts_of('inputs_c7', y, sel[y], j, c) for y in sel]
            cvals[c], cinfo[c] = N.percentile_from_parts(parts, sum((n for _p, n in parts)), kk)
        docs[name] = {'run': 'emul', 'stage': stage, 'made': P.now_local(), 'name': name, 'kind': 'f6a', 'held_out_year': int(yh), 'training_years': sorted(sel), 'training_days': int(sum((len(v) for v in sel.values()))), 'emission_source': 'E-1 to E-4 and P-1 from W\\arrays_p2\\inputs_c7_<y>.npy (the emission climatology of 2017 and 2019) on the training days; every other channel from inputs_f16 and inputs_f32', 'notes': common_note, 'channels': N.chan_stats(names, mins, maxs, cvals, cinfo), 'target': {'min': tmin, 'max': tmax}, 'lockbox': lock_key(removed)}
    if sorted(docs) != sorted(file_names()):
        raise ValueError('the files made are not the one file')
    summary = {'start': t0, 'written': [], 'lockbox_sha256': lb_sha, 'inner': inner, 'arrays_tested': dict(arrs.tested)}
    if not write:
        summary['docs'] = docs
        return summary
    infos, kept = ({}, {})
    for name in file_names():
        sp1 = os.path.join(pl['cfg'], name + '_STAGE1.json')
        if not os.path.exists(sp1):
            shutil.copyfile(stage1[name]['path'], sp1)
        if S.sha256_file(sp1) != stage1[name]['sha256']:
            raise ValueError("%s_STAGE1.json is not stage (1)'s file" % name)
        kept[name] = P.file_info(sp1)
        infos[name] = P.write_json(stage1[name]['path'], docs[name])
        summary['written'].append(name)
        say("%s written (%d bytes, sha16 %s); stage (1)'s kept as %s" % (name, infos[name]['bytes'], infos[name]['sha256'][:16], os.path.basename(sp1)))
    bk = backup_beside(ip, BACKUP_REASON)
    index = P.jload(ip)
    index.setdefault('files', {})
    for name in file_names():
        index['files'][name] = infos[name]
    index[INDEX_KEY] = {'made': P.now_local(), 'written': file_names(), 'rule': ADD2_FILE, 'rule_sha256': ADD2_SHA256, 'applies': ADD7_FILE, 'applies_sha256': ADD7_SHA256, 'lockbox_sha256': lb_sha, 'stage1_copies': {n: os.path.basename(kept[n]['path']) for n in file_names()}, 'backup_before': os.path.basename(bk)}
    ii = P.write_json(ip, index)
    files = [dict(infos[n], kind='norm', name=n) for n in file_names()]
    files += [dict(kept[n], kind='norm_stage1', name=n + '_STAGE1') for n in file_names()]
    index_files = [dict(ii, kind='norm_index', name='norm_index.json'), dict(P.file_info(bk), kind='norm_index_backup', name=os.path.basename(bk))]
    lst = {'run': 'emul', 'stage': 'NORM_f6a_fold2019 written again with the lockbox and buffer days absent, for T2 (the 2019 normalization stage)', 'made': P.now_local(), 'design': '%s sha256 %s, which applies %s sha256 %s (B) to (E)' % (ADD2_FILE, ADD2_SHA256, ADD7_FILE, ADD7_SHA256), 'lockbox': {'file': 'LOCKBOX_DAYS.json', 'sha256': lb_sha, 'decided_by': RULED_BY}, 'files': files, 'n_files': len(files), 'bytes': int(sum((e['bytes'] for e in files))), 'index_files': index_files, 'index_files_place': 'RUN/config/p2_rev2/ on the Drive (not under "files"'}
    li = P.write_json(os.path.join(pl['cfg'], LIST_NAME), lst)
    say('written norm_index.json (backup %s) and %s (%d bytes, sha16 %s)' % (os.path.basename(bk), LIST_NAME, li['bytes'], li['sha256'][:16]))
    read_back(pl, infos, lb_sha)
    summary['index'] = ii
    summary['backup'] = P.file_info(bk)
    summary['list'] = li
    summary['stage1'] = kept
    summary['infos'] = infos
    return summary

def read_back(pl, infos, lb_sha):
    for name, info in infos.items():
        p = os.path.join(pl['cfg'], name + '.json')
        if S.sha256_file(p) != info['sha256']:
            raise ValueError('%s changed after its write' % name)
        d = P.jload(p)
        kind = 'f6a' if name.startswith('NORM_f6a_') else 'f6'
        lk = d.get('lockbox') or {}
        if d.get('kind') != kind or d.get('name') != name or list(d['channels']) != list(NAMES) or (lk.get('sha256') != lb_sha) or (lk.get('decided_by') != RULED_BY) or (lk.get('file') != 'LOCKBOX_DAYS.json'):
            raise ValueError('%s: form' % name)
        if kind == 'f6a' and int(d['held_out_year']) in d['training_years']:
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
    a = sys.argv[1:]
    result = os.path.join(P.LOGS, 'NORM_F6A_2019_RESULT.json')
    for opt, key in (('--config-dir', 'cfg'), ('--work-dir', 'work'), ('--arrays-dir', 'arrays'), ('--result', 'result')):
        if opt in a:
            if key == 'result':
                result = a[a.index(opt) + 1]
            else:
                pl[key] = a[a.index(opt) + 1]
    say('norm_f6a_2019 start; rule sha256 %s (applies %s)' % (ADD2_SHA256[:16], ADD7_SHA256[:16]))
    res = run(pl)
    say('norm_f6a_2019 end: written %d (%s); lockbox sha16 %s' % (len(res['written']), ', '.join(res['written']), res['lockbox_sha256'][:16]))
    out = {'made': P.now_local(), 'written': res['written'], 'list': res['list'], 'index': res['index'], 'backup': res['backup'], 'rule_sha256': ADD2_SHA256, 'applies_sha256': ADD7_SHA256, 'lockbox_sha256': res['lockbox_sha256'], 'inner': res['inner'], 'arrays_tested': res['arrays_tested']}
    os.makedirs(os.path.dirname(os.path.abspath(result)), exist_ok=True)
    with open(result, 'w', encoding='ascii', newline='\n') as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write('\n')
    sys.exit(0 if len(res['written']) == len(file_names()) else 3)
if __name__ == '__main__':
    S.run_main(main)
