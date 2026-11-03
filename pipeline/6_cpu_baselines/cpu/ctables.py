"""Builds the climatology tables of each fold: the weekday climatology fitted on all training years and on each set of three, so that each training year gets values fitted without it (cross-fitted), and the tables of the two-year emission tests."""
import hashlib
import json
import math
import os
import sys
import time
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np
from train import cfg, data
from train import guard as GD
ORDER = ('2016', '2015', '2017', '2018', '2019', 'f6')
F6_KEYS = (('train2017_test2019', 2017, 2019), ('train2019_test2017', 2019, 2017))
SCHEMA = 'ctables-1'
FORMULA = 'C(i, d) = doy[p(d) - 1, i] x r[w(d), i], clipped at 0; p, w as baselines.common calendar_position, weekday'
RULED_BY = 'R-01'

def say(msg):
    print('%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)

def sha256_file(p):
    return cfg.sha256_file(p)

def iso(d):
    s = str(d)
    return '%s-%s-%s' % (s[:4], s[4:6], s[6:8]) if len(s) == 8 else s

def days_sha256(days):
    b = ''.join((x + '\n' for x in sorted((iso(d) for _y, _i, d in days))))
    return hashlib.sha256(b.encode('ascii')).hexdigest()

class Days:

    def __init__(self, cells, days, year, read, roles):
        self.cells = cells
        self.train_days = list(days)
        self.year = year
        self._read = read
        self._roles = roles

    def target_cells(self, y, i):
        GD.roles_guard([(y, i)], 'fit', self._roles)
        return self._read(y, i)

class Source:

    def __init__(self, configs, paths, fold):
        from baselines import common
        self.fold_name = fold
        lb = lockbox_doc(configs)
        if fold == 'f6':
            raise ValueError('use f6_sources()')
        y = int(fold)
        self.fold = common.Fold(configs, paths, y)
        self.year = y
        self.train_days = list(self.fold.train_days)
        self.train_years = list(self.fold.split.train_years)
        self.cells = self.fold.cells
        self.mask_sha256 = self.fold.mask_sha256
        self.roles = GD.roles_of(configs.splits['folds'][str(y)], lb)
        self.read = self.fold.target_cells

def f6_sources(configs, paths):
    lb = lockbox_doc(configs)
    mask = data.load_mask(configs, paths)
    cells = data.mask_cells(mask)
    out = {}
    for key, ytr, yte in F6_KEYS:
        t = {'name': 'ctables_f6_%s' % key, 'kind': 'f6', 'split': ('f6', key), 'held_out_year': yte, 'train_year': ytr, 'arm': 'TRUE'}
        sp = data.Split(configs, t)
        ya = data.YearArrays(configs, paths, ytr)
        roles = GD.roles_of(configs.splits['f6'][key], lb)

        def read(y, i, sp=sp, ya=ya):
            sp.guard(y, i, 'train')
            return np.asarray(ya.tgt[i], dtype=np.float64).ravel()[cells]
        out['f6_train%d' % ytr] = (list(sp.train), cells, read, roles, ytr)
    return (out, configs.channels['mask']['sha256'])

def lockbox_doc(configs):
    if data.LOCKBOX_FILE != 'LOCKBOX_DAYS.json':
        raise ValueError('train.data.LOCKBOX_FILE is %r: run through code\\train_shim.py with the decided lockbox' % (data.LOCKBOX_FILE,))
    return cfg.read_json(configs.paths.config('LOCKBOX_DAYS.json'))

def c_of(fit, d):
    from baselines import common
    p = common.calendar_position(d)
    return np.maximum(fit.doy[p - 1] * fit.r[common.weekday(d)], 0.0)

class Moments:

    def __init__(self):
        self.n, self.mean, self.m2 = (0, 0.0, 0.0)
        self.cmin, self.cmax = (math.inf, -math.inf)
        self.bad = 0

    def add(self, yv, c):
        x = yv - c
        fin = np.isfinite(x) & np.isfinite(c)
        self.bad += int((~fin).sum())
        if self.bad:
            return
        nb = x.size
        mb = float(x.mean())
        m2b = float(((x - mb) ** 2).sum())
        n = self.n + nb
        delta = mb - self.mean
        self.mean += delta * nb / n
        self.m2 += m2b + delta * delta * self.n * nb / n
        self.n = n
        self.cmin = min(self.cmin, float(c.min()))
        self.cmax = max(self.cmax, float(c.max()))

    def result(self, what):
        if self.bad or self.n == 0:
            raise ValueError('%s: %d non-finite y - C or C values (or no cell-day); nothing substituted' % (what, self.bad))
        return (math.sqrt(self.m2 / self.n), self.cmin, self.cmax)

def fit_days(src_days, name, train_years):
    if name == 'all':
        return list(src_days)
    yr = int(name[2:])
    if yr not in train_years:
        raise ValueError('%s: %d is not a training year' % (name, yr))
    out = [d for d in src_days if int(d[0]) != yr]
    if any((int(d[0]) == yr for d in out)):
        raise ValueError('%s holds a day of %d' % (name, yr))
    return out

def save_table(path, a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    with open(path + '.partial', 'wb') as f:
        np.save(f, a)
    os.replace(path + '.partial', path)
    return {'file': os.path.basename(path), 'sha256': sha256_file(path), 'shape': list(a.shape), 'dtype': 'float64'}

def write_json(path, d):
    with open(path + '.partial', 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(path + '.partial', path)
    return sha256_file(path)

def want_of(config_dir):
    return {'code_sha256': sha256_file(os.path.abspath(__file__)), 'splits_sha256': sha256_file(os.path.join(config_dir, 'SPLITS.json')), 'lockbox_sha256': sha256_file(os.path.join(config_dir, 'LOCKBOX_DAYS.json'))}

def index_ok(index_path, arrays_out, want=None):
    if not os.path.isfile(index_path):
        return False
    try:
        d = cfg.read_json(index_path)
    except cfg.ConfigError:
        return False
    if d.get('schema') != SCHEMA or not d.get('fits'):
        return False
    if want is not None and (d.get('code_sha256') != want['code_sha256'] or d.get('splits_sha256') != want['splits_sha256'] or (d.get('lockbox') or {}).get('sha256') != want['lockbox_sha256']):
        return False
    for f in d['fits'].values():
        for k in ('doy', 'r'):
            p = os.path.join(arrays_out, f[k]['file'])
            if not os.path.isfile(p) or sha256_file(p) != f[k]['sha256']:
                return False
    return True

def fit_row(fit, days, years, fold, name, arrays_out):
    base = os.path.join(arrays_out, 'CT_%s_%s' % (fold, name))
    return {'years': sorted((int(y) for y in years)), 'n_days': len(days), 'days_sha256': days_sha256(days), 'doy': save_table(base + '_doy.npy', fit.doy), 'r': save_table(base + '_r.npy', fit.r), 'calendar_positions_without_training_day': fit.n_pos_empty, 'weekday_factor_set_to_1': fit.n_r_one}

def head(configs, fold, mask_sha):
    lp = configs.paths.config('LOCKBOX_DAYS.json')
    return {'schema': SCHEMA, 'fold': fold, 'fits': {}, 'mask_sha256': mask_sha, 'splits_sha256': configs.sha256['SPLITS.json'], 'lockbox': {'file': 'LOCKBOX_DAYS.json', 'sha256': sha256_file(lp), 'decided_by': RULED_BY}, 'formula': FORMULA, 'code_sha256': sha256_file(os.path.abspath(__file__))}

def one_fold(configs, paths, fold, arrays_out, index_out):
    from baselines import clim
    t0 = time.time()
    src = Source(configs, paths, fold)
    names = ['all'] + ['wo%d' % yr for yr in src.train_years]
    fits, rows = ({}, {})
    doc = head(configs, fold, src.mask_sha256)
    for name in names:
        days = fit_days(src.train_days, name, src.train_years)
        view = Days(src.cells, days, src.year, src.read, src.roles)
        fits[name] = clim.ClimFit(view)
        years = sorted(set((int(d[0]) for d in days)))
        rows[name] = fit_row(fits[name], days, years, fold, name, arrays_out)
        say('CT fold %s %s: %d days of %s; doy %s and r %s written' % (fold, name, len(days), years, rows[name]['doy']['shape'], rows[name]['r']['shape']))
    mo = Moments()
    for y, i, d in src.train_days:
        GD.roles_guard([(y, i)], 'stat', src.roles)
        mo.add(src.read(y, i), c_of(fits['wo%d' % int(y)], d))
    s_f, cmin, cmax = mo.result('fold %s' % fold)
    doc['fits'] = rows
    doc.update({'s_f': s_f, 'c_min': cmin, 'c_max': cmax, 's_f_rule': "SD (ddof 0, float64) of y - C over the masked cells of the fold's TRAIN days, C cross-fitted (wo<yr> on the days of yr); c_min, c_max over the same cell-days", 'cell_days': mo.n, 'train_days': len(src.train_days), 'train_years': src.train_years, 'made_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    ip = os.path.join(index_out, 'CTABLES_%s.json' % fold)
    sha = write_json(ip, doc)
    say('CT fold %s: index %s sha16 %s; %d fits, %d cell-days, %.0f s' % (fold, os.path.basename(ip), sha[:16], len(rows), mo.n, time.time() - t0))
    return doc

def f6_fold(configs, paths, arrays_out, index_out):
    from baselines import clim
    t0 = time.time()
    srcs, mask_sha = f6_sources(configs, paths)
    doc = head(configs, 'f6', mask_sha)
    doc.update({'s_f': {}, 'c_min': {}, 'c_max': {}, 'cell_days': {}})
    for name, (days, cells, read, roles, ytr) in sorted(srcs.items()):
        view = Days(cells, days, ytr, read, roles)
        fit = clim.ClimFit(view)
        doc['fits'][name] = fit_row(fit, days, [ytr], 'f6', name, arrays_out)
        mo = Moments()
        for y, i, d in days:
            GD.roles_guard([(y, i)], 'stat', roles)
            mo.add(read(y, i), c_of(fit, d))
        s_f, cmin, cmax = mo.result('f6 %s' % name)
        doc['s_f'][name] = s_f
        doc['c_min'][name] = cmin
        doc['c_max'][name] = cmax
        doc['cell_days'][name] = mo.n
        say('CT fold f6 %s: %d days; doy and r written' % (name, len(days)))
    doc.update({'s_f_rule': "per fit, in-sample: SD (ddof 0, float64) of y - C over the masked cells of that fit's TRAIN days with C = that fit (T4, PROP); c_min, c_max over the same cell-days", 'made_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    ip = os.path.join(index_out, 'CTABLES_f6.json')
    sha = write_json(ip, doc)
    say('CT fold f6: index %s sha16 %s; %.0f s' % (os.path.basename(ip), sha[:16], time.time() - t0))
    return doc

def main(argv):
    o = {'--folds': ','.join(ORDER)}
    for k in range(0, len(argv) - 1, 2):
        o[argv[k]] = argv[k + 1]
    need = ('--config-dir', '--arrays-dir', '--arrays-out', '--index-out')
    if any((k not in o for k in need)):
        print('usage: ctables.py %s [--folds %s]' % (' '.join(('%s D' % k for k in need)), ','.join(ORDER)))
        return 1
    folds = [s for s in o['--folds'].split(',') if s]
    if any((f not in ORDER for f in folds)):
        print('unknown fold in %s' % folds)
        return 1
    for d in (o['--arrays-out'], o['--index-out']):
        os.makedirs(d, exist_ok=True)
    w = os.path.join(o['--arrays-out'], '_work')
    paths = cfg.Paths(o['--config-dir'], w, w, os.path.join(w, 'unused'), o['--arrays-dir'])
    configs = cfg.Configs(paths)
    done = 0
    want = want_of(o['--config-dir'])
    for f in folds:
        ip = os.path.join(o['--index-out'], 'CTABLES_%s.json' % f)
        if index_ok(ip, o['--arrays-out'], want):
            say('CT fold %s: done before (its index and tables verify by sha256, made by this code from these SPLITS.json and LOCKBOX_DAYS.json); skipped' % f)
            done += 1
            continue
        if f == 'f6':
            f6_fold(configs, paths, o['--arrays-out'], o['--index-out'])
        else:
            one_fold(configs, paths, f, o['--arrays-out'], o['--index-out'])
        done += 1
    say('CT end: %d of %d folds done' % (done, len(folds)))
    return 0 if done == len(folds) else 3
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
