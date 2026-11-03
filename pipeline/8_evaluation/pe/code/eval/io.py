"""The readers of the sealed predictions of the networks and baselines, each read only after its done marker and sha256 verify; the target read through the roles guard; the held-back days read only under an opening record; and the sealed writer of JSON, CSV and done files with sha256."""
import csv
import datetime
import hashlib
import json
import os
import re
import time
import numpy as np
from train import cfg
from train import data as TD
from train import guard as GD
PLACES = {'pred_sealed': 'W_P2/results_sealed/TIER_A/pred', 'ckpt': 'TRAIN/ckpt', 'baselines_sealed': 'W_P2/results_sealed/TIER_A/baselines', 'lockbox_sealed': 'W_P2/results_sealed/TIER_A/lockbox', 'cpu_config': 'RUN/config/p2_rev', 'eval_sealed': 'W_P2/results_sealed/TIER_A/eval', 'train_logs': 'TRAIN/logs/train', 'train_logs_lane1': 'TRAIN/logs/tier_a2/lane1/train', 'train_logs_lane2': 'TRAIN/logs/tier_a2/lane2/train'}
TRAIN_LOG_PLACES = ('train_logs', 'train_logs_lane1', 'train_logs_lane2')
GUARD_SHA256 = 'bceda7455a5a3fe45dddce3f73e2d56a41b5a5de194aa85f8b557f1fb2d52c23'
PD17_NAME = 'PD17_RECORD.json'
OPENING_SCHEMA = 'tier-a-lockbox-opening-1'
OPENED_MARKER = 'LOCKBOX_OPENED_TIER_A.json'
DONE_NAME = 'EVAL_DONE.json'
FOLDS = tuple(cfg.FOLD_YEARS)
EXIT_REFUSED = 3
EXIT_PD17 = 4
EXIT_LOCKBOX = 6
RESULT_KEYS = ('steps_run', 'steps_planned', 'early_stopped', 'best_step', 'last_step')

class Refused(Exception):
    code = EXIT_REFUSED

class PD17Refused(Refused):
    code = EXIT_PD17

class LockboxRefused(Refused):
    code = EXIT_LOCKBOX

def say(text):
    print('%s %s' % (time.strftime('%H:%M:%S'), text), flush=True)

def sha256_file(path):
    return cfg.sha256_file(path)

def read_json(path):
    if not os.path.isfile(path):
        raise Refused('absent: %s' % path)
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        return json.loads(raw.decode('ascii'))
    except (UnicodeDecodeError, ValueError):
        raise Refused('not ASCII JSON: %s' % path)

def read_json_sha(path):
    if not os.path.isfile(path):
        raise Refused('absent: %s' % path)
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        d = json.loads(raw.decode('ascii'))
    except (UnicodeDecodeError, ValueError):
        raise Refused('not ASCII JSON: %s' % path)
    return (d, len(raw), hashlib.sha256(raw).hexdigest())

def guard_record():
    sha = sha256_file(GD.__file__)
    return {'file': 'train/guard.py', 'sha256': sha, 'expected_sha256': GUARD_SHA256, 'as_expected': sha == GUARD_SHA256}

def iso(d):
    s = str(d)
    return '%s-%s-%s' % (s[:4], s[4:6], s[6:8]) if len(s) == 8 else s

def dates_sha256(dates):
    b = ''.join((x + '\n' for x in sorted((iso(d) for d in dates))))
    return hashlib.sha256(b.encode('ascii')).hexdigest()

def load_config(path):
    conf = read_json(path)
    if conf.get('sealed') is not True:
        raise Refused('the config is not sealed ("sealed" true needed)')
    lb = conf.get('lockbox') or {}
    if lb.get('file') != 'LOCKBOX_DAYS.json' or not lb.get('sha256') or (not lb.get('decided_by')):
        raise Refused('the config names no decided lockbox')
    return conf

def places(conf):
    out = dict(PLACES)
    for k, v in (conf.get('places') or {}).items():
        if k in out and isinstance(v, str) and v:
            out[k] = v
    for k, v in out.items():
        if '..' in v or v.startswith('/'):
            raise Refused('place %s is not below EMUL: %s' % (k, v))
    return out

class _Paths:

    def __init__(self, config_dir, arrays_dir):
        self.config_dir = os.path.abspath(config_dir)
        self.arrays_dir = os.path.abspath(arrays_dir)
        self.root = cfg.ROOT

    def config(self, name):
        return os.path.join(self.config_dir, name)

    def array(self, path_from_root):
        name = str(path_from_root).replace('\\', '/').rsplit('/', 1)[-1]
        return os.path.join(self.arrays_dir, name)

class Sources:
    channel_set = cfg.Configs.channel_set
    f6a = cfg.Configs.f6a
    city_cells = cfg.Configs.city_cells

    def __init__(self, config_dir, arrays_dir, conf):
        self.paths = _Paths(config_dir, arrays_dir)
        self.sha256 = {}
        self.arrays = self._load('S3_ARRAYS.json')
        self.channels = self._load('S3_CHANNELS.json')
        self.splits = self._load('SPLITS.json')
        self.lockbox = self._load('LOCKBOX_DAYS.json')
        if self.sha256['LOCKBOX_DAYS.json'] != conf['lockbox']['sha256']:
            raise Refused("LOCKBOX_DAYS.json's sha256 differs from the config's lockbox sha256")
        self.mask = TD.load_mask(self, self.paths)
        self.cells = TD.mask_cells(self.mask)
        self.mask_sha256 = self.channels['mask']['sha256']
        self.grid = tuple(self.mask.shape)
        self._roles = {}
        self._ya = {}
        self._c7arr = {}
        self._gmm = {}
        self.by_name = {c['name']: c for c in self.channels['channels']}

    def _load(self, name):
        p = self.paths.config(name)
        d = read_json(p)
        self.sha256[name] = sha256_file(p)
        return d

    def dates(self, year):
        return list(self.arrays['years'][str(year)]['dates'])

    def roles(self, fold):
        if fold not in self._roles:
            self._roles[fold] = GD.roles_of(self.splits['folds'][str(fold)], self.lockbox)
        return self._roles[fold]

    def days(self, fold, role, year=None):
        out = []
        for y, i in GD.days_with(self.roles(fold), role, year):
            out.append((y, i, self.dates(y)[i]))
        return out

    def test_cv(self, fold):
        return self.days(fold, 'TEST_CV', fold)

    def lockbox_days(self, fold, opening):
        need_opening(opening)
        return self.days(fold, 'LOCKBOX', fold)

    def year_arrays(self, year):
        if year not in self._ya:
            self._ya[year] = TD.YearArrays(self, self.paths, year, with_target=True)
        return self._ya[year]

    def target(self, fold, days, purpose, opening=None):
        if purpose == 'predict_lockbox':
            need_opening(opening)
        try:
            GD.roles_guard(days, purpose, self.roles(fold))
        except GD.GuardError as e:
            raise Refused('the target: %s' % e)
        out = np.empty((len(days), len(self.cells)), dtype=np.float64)
        for k, d in enumerate(days):
            ya = self.year_arrays(int(d[0]))
            out[k] = np.asarray(ya.tgt[int(d[1])], dtype=np.float64).ravel()[self.cells]
        if not np.all(np.isfinite(out)):
            raise Refused('the target has non-finite values at masked cells (fold %d, %s)' % (fold, purpose))
        return out

    def target_max(self, fold):
        days = self.days(fold, 'TRAIN')
        GD.roles_guard(days, 'stat', self.roles(fold))
        mx = np.full(len(self.cells), -np.inf)
        for y, i, _d in days:
            row = np.asarray(self.year_arrays(y).tgt[i], dtype=np.float64).ravel()[self.cells]
            np.maximum(mx, row, out=mx)
        return (mx, float(mx.max()), len(days))

    def channel_rows(self, fold, days, name, purpose, opening=None):
        if purpose == 'predict_lockbox':
            need_opening(opening)
        GD.roles_guard(days, purpose, self.roles(fold))
        ent = self.by_name.get(name)
        if ent is None:
            raise Refused('S3_CHANNELS.json has no channel %s' % name)
        arr, ix = (ent['array'], int(ent['index']))
        out = np.empty((len(days), len(self.cells)), dtype=np.float32)
        for k, (y, i, _d) in enumerate(days):
            ya = self.year_arrays(int(y))
            a = ya.f16 if arr == 'inputs_f16' else ya.f32
            out[k] = np.asarray(a[int(i), ix]).astype(np.float32).ravel()[self.cells]
        return out

    def c7_rows(self, fold, days, name, purpose, opening=None):
        if purpose == 'predict_lockbox':
            need_opening(opening)
        GD.roles_guard(days, purpose, self.roles(fold))
        order, key = self.f6a()
        if name not in order:
            raise Refused('the emission-climatology arrays hold no channel %s' % name)
        j = order.index(name)
        out = np.empty((len(days), len(self.cells)), dtype=np.float32)
        cache = {}
        for k, (y, i, _d) in enumerate(days):
            y = int(y)
            if y not in cache:
                e = self.arrays['years'][str(y)]['files'][key]
                cache[y] = TD._open_npy(self.paths.array(e['path']), e['shape'], e['dtype'], e.get('bytes'))
            out[k] = np.asarray(cache[y][int(i), j]).astype(np.float32).ravel()[self.cells]
        return out

    def grid_minmax(self, fold, days, name, purpose, source='main', opening=None):
        if purpose == 'predict_lockbox':
            need_opening(opening)
        GD.roles_guard(days, purpose, self.roles(fold))
        order = self.f6a()[0] if source == 'c7' else []
        use_c7 = name in order
        if not use_c7:
            ent = self.by_name.get(name)
            if ent is None:
                raise Refused('S3_CHANNELS.json has no channel %s' % name)
            arr, ix = (ent['array'], int(ent['index']))
        mn = np.empty(len(days))
        mx = np.empty(len(days))
        for k, (y, i, _d) in enumerate(days):
            ck = ('c7' if use_c7 else 'main', name, int(y), int(i))
            if ck not in self._gmm:
                if use_c7:
                    a = self._c7_year(int(y))[int(i), order.index(name)]
                else:
                    ya = self.year_arrays(int(y))
                    a = (ya.f16 if arr == 'inputs_f16' else ya.f32)[int(i), ix]
                v = np.asarray(a, dtype=np.float64)
                if not np.all(np.isfinite(v)):
                    raise Refused('%s has non-finite stored values (year %d, day index %d)' % (name, int(y), int(i)))
                self._gmm[ck] = (float(v.min()), float(v.max()))
            mn[k], mx[k] = self._gmm[ck]
        return (mn, mx)

    def _c7_year(self, y):
        if y not in self._c7arr:
            _order, key = self.f6a()
            e = self.arrays['years'][str(y)]['files'][key]
            self._c7arr[y] = TD._open_npy(self.paths.array(e['path']), e['shape'], e['dtype'], e.get('bytes'))
        return self._c7arr[y]

    def static_grid(self, fold, name):
        days = self.test_cv(fold)
        pick = [days[0], days[-1]]
        GD.roles_guard(pick, 'predict_cv', self.roles(fold))
        ent = self.by_name.get(name)
        if ent is None:
            raise Refused('S3_CHANNELS.json has no channel %s' % name)
        arr, ix = (ent['array'], int(ent['index']))
        got = []
        for y, i, _d in pick:
            ya = self.year_arrays(int(y))
            a = ya.f16 if arr == 'inputs_f16' else ya.f32
            got.append(np.asarray(a[int(i), ix]).astype(np.float32))
        if not np.array_equal(got[0], got[1]):
            raise Refused('%s differs between two days: not static' % name)
        return got[0]

    def static(self, fold, name):
        return self.static_grid(fold, name).ravel()[self.cells]

    def city_blocks_flat(self):
        cc = self.city_cells()
        if cc is None:
            return None
        from baselines import sealed as BS
        h, w = self.grid
        bl = BS.city_blocks(self.grid, self.mask, cc)
        return {c: [int(r) * w + int(k) for r, k in rc] for c, rc in bl.items()}

def gate_pd17(emul, pl, sources):
    p = os.path.join(emul, *pl['cpu_config'].split('/'), PD17_NAME)
    if not os.path.isfile(p):
        raise PD17Refused("the study design's PASS record %s is absent: no sealed result is opened before it" % PD17_NAME)
    rec = read_json(p)
    if rec.get('verdict') != 'PASS':
        raise PD17Refused("the study design's record does not read PASS: no sealed result is opened")
    if rec.get('S3_ARRAYS_sha256') != sources.sha256['S3_ARRAYS.json']:
        raise PD17Refused("the study design's record was made on another S3_ARRAYS.json: no sealed result is opened")
    return {'file': PD17_NAME, 'sha256': sha256_file(p), 'verdict': 'PASS'}

def read_opening(path):
    if path is None:
        return None
    d = read_json(path)
    if d.get('schema') != OPENING_SCHEMA or d.get('tier') != 'A' or (not re.match('^R-\\d+$', str(d.get('decided_by', '')))):
        raise LockboxRefused('the opening record is not {schema %s, tier A, decided_by R-<n>}' % OPENING_SCHEMA)
    return {'file': os.path.basename(path), 'sha256': sha256_file(path), 'decided_by': d['decided_by'], 'tier': 'A'}

def need_opening(opening):
    if not opening or not re.match('^R-\\d+$', str(opening.get('decided_by', ''))):
        raise LockboxRefused('the lockbox is read only under an opening record (its one opening): none given; nothing of the lockbox was read')

def claim_opening(eval_dir, opening, run_id):
    need_opening(opening)
    p = os.path.join(eval_dir, OPENED_MARKER)
    if os.path.isfile(p):
        d = read_json(p)
        if d.get('run_id') != run_id:
            raise LockboxRefused('the lockbox of the benchmark was opened before (%s): one opening only' % OPENED_MARKER)
        return d
    now = datetime.datetime.now(datetime.timezone.utc)
    d = {'schema': 'tier-a-lockbox-opened-1', 'run_id': run_id, 'decided_by': opening['decided_by'], 'opening_sha256': opening['sha256'], 'made_utc': now.strftime('%Y-%m-%d %H:%M:%S')}
    os.makedirs(eval_dir, exist_ok=True)
    _write_text(p, json.dumps(d, indent=1, sort_keys=True) + '\n')
    return d

def _verify_row(folder, row, what):
    name = str(row.get('path', '')).replace('\\', '/').rsplit('/', 1)[-1]
    p = os.path.join(folder, name)
    if not name or not os.path.isfile(p):
        raise Refused('%s: %s absent from its folder' % (what, name))
    if os.stat(p).st_size != int(row.get('bytes', -1)):
        raise Refused('%s: %s bytes differ from its DONE row' % (what, name))
    if sha256_file(p) != row.get('sha256'):
        raise Refused('%s: %s sha256 differs from its DONE row' % (what, name))
    return p

def _rows_by_name(done, what):
    files = done.get('files')
    if not isinstance(files, dict):
        raise Refused('%s: DONE file lists no files' % what)
    return files

def _order(days_info, want_dates, what):
    got = [str(d) for d in days_info]
    if len(set(got)) != len(got) or sorted(got) != sorted((str(d) for d in want_dates)):
        raise Refused("%s: its days are not the fold's %d days of the role" % (what, len(want_dates)))
    pos = {d: k for k, d in enumerate(got)}
    return np.array([pos[str(d)] for d in want_dates], dtype=np.int64)

def _load_pred(p, row, n_days, n_cells, what):
    a = np.load(p, mmap_mode='r')
    if a.dtype != np.float32 or tuple(a.shape) != (n_days, n_cells):
        raise Refused('%s: shape %s %s, expected (%d, %d) float32' % (what, list(a.shape), a.dtype, n_days, n_cells))
    if row.get('shape') is not None and list(row['shape']) != list(a.shape):
        raise Refused('%s: shape differs from its DONE row' % what)
    return a

def _check_info(info, sources, fold, what):
    if int(info.get('held_out_year', -1)) != int(fold):
        raise Refused('%s: held-out year %r, fold %d' % (what, info.get('held_out_year'), fold))
    if info.get('mask_sha256') != sources.mask_sha256:
        raise Refused('%s: mask sha256 differs' % what)
    if int(info.get('n_cells', -1)) != len(sources.cells):
        raise Refused('%s: %r cells, the mask %d' % (what, info.get('n_cells'), len(sources.cells)))

def read_network_cv(emul, pl, name, fold, sources):
    what = 'network %s' % name
    dfile = os.path.join(emul, *pl['ckpt'].split('/'), name, 'DONE.json')
    if not os.path.isfile(dfile):
        raise Refused('%s: no DONE.json (TRAIN/ckpt/%s): not read' % (what, name))
    done = read_json(dfile)
    if done.get('status') != 'done' or done.get('training') != name:
        raise Refused('%s: DONE.json is not done or names another training' % what)
    lab = done.get('labels') or {}
    if lab.get('fold') is not None and int(lab['fold']) != int(fold):
        raise Refused('%s: DONE.json labels fold %r, asked %d' % (what, lab.get('fold'), fold))
    files = _rows_by_name(done, what)
    if 'pred_masked.npy' not in files:
        raise Refused('%s: DONE.json lists no pred_masked.npy' % what)
    pdir = os.path.join(emul, *pl['pred_sealed'].split('/'), name)
    row = files['pred_masked.npy']
    p = _verify_row(pdir, row, what)
    info, ib, ish = read_json_sha(os.path.join(pdir, 'pred_info.json'))
    irow = (info.get('files') or {}).get('pred_masked.npy') or {}
    if irow.get('sha256') != row.get('sha256'):
        raise Refused("%s: pred_info.json's file row differs from DONE.json's" % what)
    _check_info(info, sources, fold, what)
    lb = info.get('lockbox') or {}
    if lb.get('held_out_is') != 'TEST_CV' or lb.get('sha256') != sources.sha256['LOCKBOX_DAYS.json']:
        raise Refused('%s: pred_info.json does not hold TEST_CV days of this lockbox' % what)
    days = sources.test_cv(fold)
    GD.roles_guard(days, 'predict_cv', sources.roles(fold))
    order = _order(info.get('days', []), [d for _y, _i, d in days], what)
    a = _load_pred(p, row, len(days), len(sources.cells), what)
    e = np.asarray(a, dtype=np.float64)[order]
    del a
    if not np.all(np.isfinite(e)):
        raise Refused('%s: non-finite predictions' % what)
    res = done.get('result') or {}
    keep = {k: res[k] for k in RESULT_KEYS if k in res}
    curve, log_rec = read_train_log(emul, pl, done, name)
    return (e, {'kind': 'network', 'name': name, 'fold': int(fold), 'file': 'pred_masked.npy', 'bytes': int(row['bytes']), 'sha256': row['sha256'], 'n_days': len(days), 'clipped_masked_total': info.get('clipped_masked_total'), 'result': keep, 'labels': lab, 'pred_info': {'file': 'pred_info.json', 'bytes': ib, 'sha256': ish, 'done_sha256': None}, 'inference': inference_record(info), 'train_log': log_rec, 'd11_curve': curve})

def inference_record(info):
    tot = info.get('seconds')
    ds = {str(k): (v or {}).get('seconds') for k, v in (info.get('day_sets') or {}).items()}

    def num(x):
        return isinstance(x, (int, float)) and (not isinstance(x, bool))
    cv = None
    if num(tot) and all((num(x) for x in ds.values())):
        cv = round(float(tot) - sum((float(x) for x in ds.values())), 1)
    return {'seconds_total': tot, 'seconds_day_sets': ds, 'seconds_test_cv': cv, 'infer_batch': info.get('infer_batch'), 'windows_per_day': info.get('windows_per_day'), 'n_days': info.get('n_days')}

def read_train_log(emul, pl, done, training):
    tl = done.get('train_log')
    if not tl:
        return (None, {'status': 'NOT READ', 'reason': 'DONE.json names no train_log'})
    name = str(tl).replace('\\', '/').rsplit('/', 1)[-1]
    places = []
    for k in TRAIN_LOG_PLACES:
        v = pl.get(k)
        if v and v not in places:
            places.append(v)
    rec = {'file': name, 'place': None, 'places': list(places), 'done_sha256': None, 'note': 'DONE.json names the log by path only (no sha256)'}
    if name != '%s.json' % training:
        rec.update(status='NOT READ', reason="the log's name is not the training's")
        return (None, rec)
    found = [q for q in places if os.path.isfile(os.path.join(emul, *q.split('/'), name))]
    if not found:
        rec.update(status='NOT READ', reason='absent from its places (%s)' % ', '.join(places))
        return (None, rec)
    if len(found) > 1:
        rec.update(status='NOT READ', places_found=found, reason='found in more than one place (%s): not read' % ', '.join(found))
        return (None, rec)
    rec['place'] = found[0]
    p = os.path.join(emul, *found[0].split('/'), name)
    try:
        d, nb, sh = read_json_sha(p)
    except Refused:
        rec.update(status='NOT READ', reason='not ASCII JSON')
        return (None, rec)
    rec.update(bytes=nb, sha256=sh)
    ev = d.get('evaluations') if isinstance(d, dict) else None
    if not isinstance(ev, list) or not all((isinstance(x, dict) and 'step' in x for x in ev)):
        rec.update(status='NOT READ', reason="no evaluations rows of the shared package's loop.py form")
        return (None, rec)
    rows = [{'step': x.get('step'), 'train_stat': x.get('train_stat'), 'val_loss': x.get('val_loss'), 'best_step': x.get('best_step')} for x in ev]
    rec.update(status='READ', n_evaluations=len(rows))
    return (rows, rec)

def _baseline_dir(emul, pl, key, name, fold):
    return os.path.join(emul, *pl[key].split('/'), name, 'fold%d' % int(fold))

def read_baseline_cv(emul, pl, name, fold, sources):
    what = 'baseline %s fold %d' % (name, fold)
    bdir = _baseline_dir(emul, pl, 'baselines_sealed', name, fold)
    dfile = os.path.join(bdir, 'DONE.json')
    if not os.path.isfile(dfile):
        raise Refused('%s: no DONE.json: not read' % what)
    done = read_json(dfile)
    if done.get('status') != 'done' or done.get('baseline') != name or int(done.get('fold', -1)) != int(fold):
        raise Refused('%s: DONE.json is not done or names another' % what)
    files = _rows_by_name(done, what)
    for k in ('pred_masked.npy', 'pred_info.json'):
        if k not in files:
            raise Refused('%s: DONE.json lists no %s' % (what, k))
    _verify_row(bdir, files['pred_info.json'], what)
    p = _verify_row(bdir, files['pred_masked.npy'], what)
    info = read_json(os.path.join(bdir, 'pred_info.json'))
    _check_info(info, sources, fold, what)
    days = sources.test_cv(fold)
    GD.roles_guard(days, 'predict_cv', sources.roles(fold))
    order = _order(info.get('days', []), [d for _y, _i, d in days], what)
    a = _load_pred(p, files['pred_masked.npy'], len(days), len(sources.cells), what)
    e = np.asarray(a, dtype=np.float64)[order]
    del a
    if not np.all(np.isfinite(e)):
        raise Refused('%s: non-finite predictions' % what)
    return (e, {'kind': 'baseline', 'name': name, 'fold': int(fold), 'file': 'pred_masked.npy', 'bytes': int(files['pred_masked.npy']['bytes']), 'sha256': files['pred_masked.npy']['sha256'], 'n_days': len(days), 'clipped_masked_total': info.get('clipped_at_0'), 'gbt_fit': done.get('gbt_fit')})

def read_lockbox(emul, pl, kind, name, fold, sources, opening):
    need_opening(opening)
    what = 'lockbox %s %s fold %d' % (kind, name, fold)
    if kind == 'baseline':
        ldir = os.path.join(emul, *pl['lockbox_sealed'].split('/'), 'baselines', name, 'fold%d' % int(fold))
    else:
        ldir = os.path.join(emul, *pl['lockbox_sealed'].split('/'), name)
    dfile = os.path.join(ldir, 'LOCKBOX_DONE.json')
    if not os.path.isfile(dfile):
        raise Refused('%s: no LOCKBOX_DONE.json: not read' % what)
    done = read_json(dfile)
    if done.get('status') != 'done' or int(done.get('fold', -1)) != int(fold) or done.get('baseline', done.get('training', name)) != name:
        raise Refused('%s: LOCKBOX_DONE.json is not done or names another' % what)
    files = _rows_by_name(done, what)
    for k in ('pred_lockbox.npy', 'lockbox_info.json'):
        if k not in files:
            raise Refused('%s: LOCKBOX_DONE.json lists no %s' % (what, k))
    _verify_row(ldir, files['lockbox_info.json'], what)
    p = _verify_row(ldir, files['pred_lockbox.npy'], what)
    info = read_json(os.path.join(ldir, 'lockbox_info.json'))
    _check_info(info, sources, fold, what)
    if (info.get('lockbox_file') or {}).get('sha256') != sources.sha256['LOCKBOX_DAYS.json']:
        raise Refused('%s: made with another LOCKBOX_DAYS.json' % what)
    days = sources.lockbox_days(fold, opening)
    GD.roles_guard(days, 'predict_lockbox', sources.roles(fold))
    want = [d for _y, _i, d in days]
    if info.get('dates_sha256') != dates_sha256(want):
        raise Refused("%s: dates sha256 differs from the roles' lockbox days" % what)
    order = _order(info.get('days', []), want, what)
    a = _load_pred(p, files['pred_lockbox.npy'], len(days), len(sources.cells), what)
    e = np.asarray(a, dtype=np.float64)[order]
    del a
    if not np.all(np.isfinite(e)):
        raise Refused('%s: non-finite predictions' % what)
    return (e, {'kind': kind, 'name': name, 'fold': int(fold), 'part': 'LOCKBOX', 'file': 'pred_lockbox.npy', 'bytes': int(files['pred_lockbox.npy']['bytes']), 'sha256': files['pred_lockbox.npy']['sha256'], 'n_days': len(days), 'gbt_fit': done.get('gbt_fit')})

def read_val_curve(emul, pl, fold):
    what = 'GBT-ES fold %d curve' % fold
    bdir = _baseline_dir(emul, pl, 'baselines_sealed', 'GBT-ES', fold)
    dfile = os.path.join(bdir, 'DONE.json')
    if not os.path.isfile(dfile):
        raise Refused('%s: no DONE.json: not read' % what)
    done = read_json(dfile)
    if done.get('status') != 'done' or done.get('baseline') != 'GBT-ES':
        raise Refused("%s: DONE.json is not GBT-ES's" % what)
    files = _rows_by_name(done, what)
    if 'val_curve.json' not in files:
        raise Refused('%s: DONE.json lists no val_curve.json' % what)
    p = _verify_row(bdir, files['val_curve.json'], what)
    return (read_json(p), {'file': 'val_curve.json', 'sha256': files['val_curve.json']['sha256']})
MODELS_SCHEMA = 'pe-models-1'
KINDS = ('network', 'baseline')
EARLY_P_NAMES = {(2015, 0): 'fold2015_baseline', (2016, 0): 'fold2016_baseline', (2017, 0): 'fold2017_baseline', (2018, 0): 'fold2018_baseline', (2019, 0): 'fold2019_baseline', (2016, 1): 'fold2016_baseline_seed1000', (2016, 2): 'fold2016_baseline_seed2000', (2019, 1): 'fold2019_baseline_seed1000'}

def read_models(path):
    d = read_json(path)
    if d.get('schema') != MODELS_SCHEMA or not isinstance(d.get('rows'), list):
        raise Refused('the models table is not {schema %s, rows}' % MODELS_SCHEMA)
    seen = set()
    rows = []
    for r in d['rows']:
        if r.get('kind') not in KINDS or not r.get('model') or (not r.get('name')) or (int(r.get('fold', 0)) not in FOLDS):
            raise Refused('the models table: a row lacks model, kind, name or a fold year')
        seed = r.get('seed')
        key = (r['model'], int(r['fold']), None if seed is None else int(seed))
        if key in seen:
            raise Refused('the models table lists %s fold %d seed %s twice' % key)
        seen.add(key)
        rows.append({'model': r['model'], 'kind': r['kind'], 'name': r['name'], 'fold': int(r['fold']), 'seed': key[2]})
    return (rows, sha256_file(path))

def _write_text(path, text):
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)
    os.replace(tmp, path)

def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    return o

def _cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(v)
    return str(v)

class Sealed:

    def __init__(self, folder):
        if os.path.exists(folder):
            raise Refused('the eval folder exists: a run writes a new one')
        os.makedirs(folder)
        self.folder = folder
        self.rows = []

    def _row(self, name):
        p = os.path.join(self.folder, name)
        r = {'file': name, 'bytes': os.stat(p).st_size, 'sha256': sha256_file(p)}
        self.rows.append(r)
        say('SEALED %s sha256 %s' % (name, r['sha256']))
        return r

    def _new(self, name):
        if '/' in name or '\\' in name or '..' in name:
            raise ValueError('a sealed name is a plain file name')
        p = os.path.join(self.folder, name)
        if os.path.exists(p):
            raise Refused('sealed file %s exists' % name)
        return p

    def json(self, name, obj):
        p = self._new(name)
        _write_text(p, json.dumps(plain(obj), indent=1, sort_keys=True, allow_nan=False) + '\n')
        return self._row(name)

    def csv(self, name, rows, fields):
        p = self._new(name)
        tmp = p + '.partial'
        with open(tmp, 'w', encoding='ascii', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(fields)
            for r in rows:
                w.writerow([_cell(plain(r.get(k))) for k in fields])
        os.replace(tmp, p)
        return self._row(name)

    def npy(self, name, arr):
        p = self._new(name)
        tmp = p + '.partial'
        with open(tmp, 'wb') as f:
            np.save(f, np.ascontiguousarray(arr))
        os.replace(tmp, p)
        return self._row(name)

    def done(self, record):
        p = self._new(DONE_NAME)
        d = dict(record)
        d['files'] = {r['file']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in self.rows}
        _write_text(p, json.dumps(plain(d), indent=1, sort_keys=True, allow_nan=False) + '\n')
        r = {'file': DONE_NAME, 'bytes': os.stat(p).st_size, 'sha256': sha256_file(p)}
        say('SEALED %s sha256 %s (%d files listed)' % (DONE_NAME, r['sha256'], len(self.rows)))
        return r
