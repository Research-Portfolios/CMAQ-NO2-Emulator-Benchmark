"""Shared parts of the refitted baselines: a fold that guards every read by the day's role, the cross-fitted weekday climatology read from its tables, and the writer of the held-back-day predictions with their done markers."""
import hashlib
import json
import os
import numpy as np
from baselines import common
from train import cfg
from train import guard as GD
MODES = {'train': 'fit', 'validate': 'validate', 'predict': 'predict_cv', 'predict_lockbox': 'predict_lockbox'}
LB_SUB = ('lockbox', 'baselines')

def iso(d):
    s = str(d)
    return '%s-%s-%s' % (s[:4], s[4:6], s[6:8]) if len(s) == 8 else s

def dates_sha256(dates):
    b = ''.join((x + '\n' for x in sorted((iso(d) for d in dates))))
    return hashlib.sha256(b.encode('ascii')).hexdigest()

def lockbox_doc(configs):
    from train import data
    if data.LOCKBOX_FILE != 'LOCKBOX_DAYS.json':
        raise ValueError('train.data.LOCKBOX_FILE is %r: run through code\\train_shim.py with the decided lockbox' % (data.LOCKBOX_FILE,))
    return cfg.read_json(configs.paths.config('LOCKBOX_DAYS.json'))

class FoldP7(common.Fold):

    def __init__(self, configs, paths, year):
        lb = lockbox_doc(configs)
        common.Fold.__init__(self, configs, paths, year)
        self.roles = GD.roles_of(configs.splits['folds'][str(year)], lb)
        self.val_days = list(self.split.val)
        dates = configs.arrays['years'][str(year)]['dates']
        self.lockbox_days = [(y, i, dates[i]) for y, i in GD.days_with(self.roles, 'LOCKBOX', year)]
        if set(((y, i) for y, i, _d in self.lockbox_days)) != set((k for k in self.split.lockbox_only if k[0] == int(year))):
            raise ValueError("fold %d: the LOCKBOX days differ from Split's lockbox set" % int(year))
        GD.roles_guard(self.train_days, 'fit', self.roles)
        GD.roles_guard(self.val_days, 'validate', self.roles)
        GD.roles_guard(self.held_days, 'predict_cv', self.roles)
        GD.roles_guard(self.lockbox_days, 'predict_lockbox', self.roles)

    def _ya(self, y, i, mode):
        if mode not in MODES:
            raise ValueError('mode %r' % (mode,))
        GD.roles_guard([(y, i)], MODES[mode], self.roles)
        if mode == 'predict_lockbox':
            if int(y) != self.year or (int(y), int(i)) not in self.split.lockbox_only:
                raise GD.GuardError('predict_lockbox: day %d/%d is not a lockbox day of %d' % (y, i, self.year))
            return self.held
        if mode == 'validate':
            self.split.guard(y, i, 'val')
            return self.years[y]
        return common.Fold._ya(self, y, i, mode)

    def target_cells(self, y, i, purpose='fit'):
        if purpose not in ('fit', 'validate'):
            raise GD.GuardError('the target is read for fit or validate only, not %r' % (purpose,))
        GD.roles_guard([(y, i)], purpose, self.roles)
        return common.Fold.target_cells(self, y, i)

class Table:

    def __init__(self, doy, r):
        self.doy, self.r = (doy, r)

class CrossC:

    def __init__(self, fold, index_path, arrays_dir):
        idx = cfg.read_json(index_path)
        self.index_path = index_path
        self.index_sha256 = cfg.sha256_file(index_path)
        y = fold.year
        cf = fold.configs
        lsha = cfg.sha256_file(cf.paths.config('LOCKBOX_DAYS.json'))
        bad = []
        if idx.get('schema') != 'ctables-1' or str(idx.get('fold')) != str(y):
            bad.append('schema or fold')
        if idx.get('mask_sha256') != fold.mask_sha256:
            bad.append('mask_sha256')
        if idx.get('splits_sha256') != cf.sha256.get('SPLITS.json'):
            bad.append('splits_sha256')
        if (idx.get('lockbox') or {}).get('sha256') != lsha:
            bad.append('lockbox sha256')
        want = ['all'] + ['wo%d' % t for t in fold.split.train_years]
        if sorted(idx.get('fits', {})) != sorted(want):
            bad.append('fits %s' % sorted(idx.get('fits', {})))
        if bad:
            raise ValueError('CTABLES_%d.json does not fit this run: %s' % (y, ', '.join(bad)))
        n = len(fold.cells)
        self.fits = {}
        for name in want:
            row = idx['fits'][name]
            got = []
            for k, shape in (('doy', (366, n)), ('r', (7, n))):
                p = os.path.join(arrays_dir, row[k]['file'])
                if not os.path.isfile(p) or cfg.sha256_file(p) != row[k]['sha256']:
                    raise ValueError('%s differs from CTABLES_%d.json' % (row[k]['file'], y))
                a = np.load(p)
                if a.shape != shape or a.dtype != np.float64:
                    raise ValueError('%s: shape %s %s' % (row[k]['file'], a.shape, a.dtype))
                got.append(a)
            if name != 'all' and int(name[2:]) in row['years']:
                raise ValueError('%s holds its own year' % name)
            self.fits[name] = Table(*got)
        self.train_years = set((int(t) for t in fold.split.train_years))
        self.year = y

    def fit_of(self, year):
        return 'wo%d' % int(year) if int(year) in self.train_years else 'all'

    def clim_wd(self, d):
        f = self.fits[self.fit_of(int(str(d)[:4]))]
        p = common.calendar_position(d)
        return np.maximum(f.doy[p - 1] * f.r[common.weekday(d)], 0.0)

    def describe(self):
        return {'index': os.path.basename(self.index_path), 'index_sha256': self.index_sha256, 'rule': 'C cross-fitted: wo<yr> on the training and validation days of yr, all on held-out days (TEST_CV and LOCKBOX); clipped at 0'}

def same_as_all(clim_fit, cross):
    a = cross.fits['all']
    return bool(np.array_equal(clim_fit.doy, a.doy, equal_nan=True) and np.array_equal(clim_fit.r, a.r))

def lb_dir(work_dir, name, year):
    return os.path.join(work_dir, *LB_SUB + (name, 'fold%d' % int(year)))

def _row(path, shape=None):
    r = {'path': cfg.rel(path), 'bytes': os.stat(path).st_size, 'sha256': common.sha256_file(path)}
    if shape is not None:
        r['shape'] = list(shape)
        r['dtype'] = 'float32'
    return r

class LockboxWriter:

    def __init__(self, work_dir, name, fold):
        if not fold.lockbox_days:
            raise ValueError('fold %d has no lockbox day' % fold.year)
        self.name, self.fold = (name, fold)
        self.dir = lb_dir(work_dir, name, fold.year)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, 'pred_lockbox.npy')
        self.tmp = self.path + '.partial'
        self.shape = (len(fold.lockbox_days), len(fold.cells))
        self.a = np.lib.format.open_memmap(self.tmp, mode='w+', dtype=np.float32, shape=self.shape)
        self.clipped = 0
        self.nonfinite = 0
        self.k = 0

    def write(self, k, row):
        row = np.asarray(row, dtype=np.float64)
        self.nonfinite += int((~np.isfinite(row)).sum())
        neg = row < 0
        self.clipped += int(neg.sum())
        self.a[k] = np.where(neg, 0.0, row).astype(np.float32)
        self.k += 1

    def close(self, extra):
        if self.k != self.shape[0]:
            raise RuntimeError('%s fold %d: %d of %d lockbox days written' % (self.name, self.fold.year, self.k, self.shape[0]))
        if self.nonfinite:
            raise RuntimeError('%s fold %d: %d non-finite lockbox predictions' % (self.name, self.fold.year, self.nonfinite))
        self.a.flush()
        del self.a
        os.replace(self.tmp, self.path)
        days = [d for _y, _i, d in self.fold.lockbox_days]
        info = {'schema': 'tier-a-lockbox-pred-1', 'baseline': self.name, 'fold': self.fold.year, 'held_out_year': self.fold.year, 'made': common.now(), 'units': 'ppb', 'days': days, 'dates_sha256': dates_sha256(days), 'n_days': self.shape[0], 'n_cells': self.shape[1], 'cell_order': "row-major flat indices of the mask (numpy.flatnonzero), as the U-Nets'", 'mask_sha256': self.fold.mask_sha256, 'clipped_at_0': self.clipped, 'training_days': len(self.fold.train_days), 'training_years': self.fold.split.train_years, 'lockbox_file': {'file': 'LOCKBOX_DAYS.json', 'sha256': self.fold.split.lockbox_sha256, 'key': 'lockbox'}, 'sealed': 'opened once by the evaluation only', 'files': {'pred_lockbox.npy': _row(self.path, self.shape)}}
        info.update(extra or {})
        common._write_json(os.path.join(self.dir, 'lockbox_info.json'), info)
        return info

def lockbox_write_done(work_dir, name, year, info, extra=None):
    d = {'schema': 'tier-a-lockbox-done-1', 'baseline': name, 'fold': int(year), 'status': 'done', 'ended': common.now(), 'files': dict(info['files'])}
    p = os.path.join(lb_dir(work_dir, name, year), 'lockbox_info.json')
    d['files']['lockbox_info.json'] = _row(p)
    d.update(extra or {})
    common._write_json(os.path.join(lb_dir(work_dir, name, year), 'LOCKBOX_DONE.json'), d)
    return d

def lockbox_done_ok(work_dir, name, year):
    p = os.path.join(lb_dir(work_dir, name, year), 'LOCKBOX_DONE.json')
    if not os.path.isfile(p):
        return False
    try:
        d = cfg.read_json(p)
    except cfg.ConfigError:
        return False
    for r in d.get('files', {}).values():
        fp = os.path.join(cfg.ROOT, r['path']) if not os.path.isabs(r['path']) else r['path']
        if not os.path.isfile(fp) or os.stat(fp).st_size != r['bytes'] or common.sha256_file(fp) != r['sha256']:
            return False
    return True

def write_json(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    common._write_json(path, d)
    return _row(path)

def plain(o):
    return json.loads(json.dumps(o, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
