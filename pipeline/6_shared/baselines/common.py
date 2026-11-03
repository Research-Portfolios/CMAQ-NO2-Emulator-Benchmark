"""Common parts of the baselines: a fold's training and held-out days and read-only arrays, the land cells, calendar position, weekday and season, the prediction writer with its done marker, and a free-memory floor of 1.5 GB."""
import datetime
import hashlib
import json
import os
import numpy as np
from train import cfg, data
NAMES = ('CLIM-C', 'CLIM-WD', 'MET-REG', 'GBT', 'GBT-RIDGE')
FLOOR_GB = 1.5

class MemoryFloor(Exception):
    pass

def memory_check(need_gb, what):
    import psutil
    free = psutil.virtual_memory().available / 1000000000.0
    if free - float(need_gb) < FLOOR_GB:
        raise MemoryFloor('%s needs about %.2f GB; %.2f GB free; the floor is %.1f GB' % (what, need_gb, free, FLOOR_GB))
    return free

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def sha256_file(path):
    return cfg.sha256_file(path)

def _date(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))

def calendar_position(s):
    d = _date(s)
    return datetime.date(2016, d.month, d.day).timetuple().tm_yday

def weekday(s):
    return _date(s).weekday()

def day_of_year(s):
    return _date(s).timetuple().tm_yday

def season(s):
    m = _date(s).month
    if m in (12, 1, 2):
        return 'winter'
    if m in (3, 4, 5):
        return 'spring'
    if m in (6, 7, 8):
        return 'summer'
    return 'fall'

def fold_training(year):
    return {'name': 'baselines_fold%d' % year, 'kind': 'fold', 'split': ('folds', str(year)), 'held_out_year': int(year), 'norm': 'fold%d' % year, 'net': 'baseline', 'seed': 0, 'arm': None}

class Fold:

    def __init__(self, configs, paths, year):
        self.configs = configs
        self.paths = paths
        self.year = int(year)
        self.split = data.Split(configs, fold_training(year))
        self.train_days = list(self.split.train)
        self.held_days = list(self.split.held)
        self.years = {y: data.YearArrays(configs, paths, y) for y in self.split.train_years}
        self.held = data.YearArrays(configs, paths, self.year, with_target=False)
        self.mask = data.load_mask(configs, paths)
        self.cells = data.mask_cells(self.mask)
        self.grid = self.held.grid
        self.mask_sha256 = configs.channels['mask']['sha256']
        by = {c['name']: c for c in configs.channels['channels']}
        self.chan = {n: (by[n]['array'], int(by[n]['index'])) for n in by}
        self.unfilled = {y: configs.unfilled(y) for y in cfg.FOLD_YEARS}

    def check_built(self, names):
        for y in self.split.train_years + [self.year]:
            bad = sorted(set(names) & self.unfilled[y])
            if bad:
                raise ValueError("fold %d: year %d's channels %s are not built yet (the fills stage)" % (self.year, y, bad))

    def target_cells(self, y, i):
        self.split.guard(y, i, 'train')
        return np.asarray(self.years[y].tgt[i], dtype=np.float64).ravel()[self.cells]

    def _ya(self, y, i, mode):
        self.split.guard(y, i, mode)
        return self.years[y] if mode == 'train' else self.held

    def channels_cells(self, y, i, names, mode):
        if set(names) & self.unfilled[int(y)]:
            self.check_built(names)
        ya = self._ya(y, i, mode)
        out = np.empty((len(names), len(self.cells)), dtype=np.float64)
        for k, n in enumerate(names):
            arr, ix = self.chan[n]
            a = ya.f16 if arr == 'inputs_f16' else ya.f32
            out[k] = np.asarray(a[i, ix], dtype=np.float64).ravel()[self.cells]
        return out

    def channels_grid(self, y, i, names, mode):
        if set(names) & self.unfilled[int(y)]:
            self.check_built(names)
        ya = self._ya(y, i, mode)
        h, w = self.grid
        out = np.empty((len(names), h, w), dtype=np.float64)
        for k, n in enumerate(names):
            arr, ix = self.chan[n]
            a = ya.f16 if arr == 'inputs_f16' else ya.f32
            out[k] = np.asarray(a[i, ix], dtype=np.float64)
        return out

    def held_out_target(self, paths):
        ya = data.YearArrays(self.configs, paths, self.year, with_target=True)
        out = np.empty((len(self.held_days), len(self.cells)), dtype=np.float64)
        for k, (y, i, _) in enumerate(self.held_days):
            out[k] = np.asarray(ya.tgt[i], dtype=np.float64).ravel()[self.cells]
        return out

def out_dir(work_dir, name, year):
    return os.path.join(work_dir, 'pred', 'baselines', name, 'fold%d' % int(year))

def _row(path, shape=None):
    r = {'path': cfg.rel(path), 'bytes': os.stat(path).st_size, 'sha256': sha256_file(path)}
    if shape is not None:
        r['shape'] = list(shape)
        r['dtype'] = 'float32'
    return r

def _write_json(path, d):
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    os.replace(tmp, path)

class PredWriter:

    def __init__(self, work_dir, name, fold):
        self.name, self.fold = (name, fold)
        self.dir = out_dir(work_dir, name, fold.year)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, 'pred_masked.npy')
        self.tmp = self.path + '.partial'
        self.shape = (len(fold.held_days), len(fold.cells))
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
            raise RuntimeError('%s fold %d: %d of %d days written' % (self.name, self.fold.year, self.k, self.shape[0]))
        if self.nonfinite:
            raise RuntimeError('%s fold %d: %d non-finite predictions' % (self.name, self.fold.year, self.nonfinite))
        self.a.flush()
        del self.a
        os.replace(self.tmp, self.path)
        info = {'schema': 'choi-baseline-pred-1', 'baseline': self.name, 'fold': self.fold.year, 'held_out_year': self.fold.year, 'made': now(), 'units': 'ppb', 'days': [d for _, _, d in self.fold.held_days], 'n_days': self.shape[0], 'n_cells': self.shape[1], 'cell_order': "row-major flat indices of the mask (numpy.flatnonzero), as the U-Nets'", 'mask_sha256': self.fold.mask_sha256, 'clipped_at_0': self.clipped, 'training_days': len(self.fold.train_days), 'training_years': self.fold.split.train_years, 'files': {'pred_masked.npy': _row(self.path, self.shape)}}
        info.update(extra or {})
        _write_json(os.path.join(self.dir, 'pred_info.json'), info)
        return info

def write_done(work_dir, name, year, info, extra=None):
    d = {'schema': 'choi-baseline-done-1', 'baseline': name, 'fold': int(year), 'status': 'done', 'ended': now(), 'files': dict(info['files'])}
    pi = os.path.join(out_dir(work_dir, name, year), 'pred_info.json')
    d['files']['pred_info.json'] = _row(pi)
    d.update(extra or {})
    _write_json(os.path.join(out_dir(work_dir, name, year), 'DONE.json'), d)
    return d

def done_ok(work_dir, name, year):
    p = os.path.join(out_dir(work_dir, name, year), 'DONE.json')
    if not os.path.isfile(p):
        return False
    try:
        d = cfg.read_json(p)
    except cfg.ConfigError:
        return False
    for r in d.get('files', {}).values():
        fp = os.path.join(cfg.ROOT, r['path']) if not os.path.isabs(r['path']) else r['path']
        if not os.path.isfile(fp) or os.stat(fp).st_size != r['bytes'] or sha256_file(fp) != r['sha256']:
            return False
    return True

def load_pred(work_dir, name, year):
    d = out_dir(work_dir, name, year)
    info = cfg.read_json(os.path.join(d, 'pred_info.json'))
    a = np.load(os.path.join(d, 'pred_masked.npy'), mmap_mode='r')
    return (a, info)

def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
