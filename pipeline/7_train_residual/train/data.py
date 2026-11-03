"""The data reader of the residual trainings: the shared reader with the held-back days removed from every split, the cross-fitted weekday climatology read from its tables, and the residual target (the target minus the climatology, scaled) for the residual networks."""
import datetime
import math
import os
import numpy as np
from train import cfg
from train import guard as GD
LOCKBOX_FILE = None

class HeldOutError(AssertionError):
    pass

def date_month_weekday(s):
    d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return (d.month, d.weekday())

def iso_week(s):
    d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return d.isocalendar()[1]

class ChannelPlan:

    def __init__(self, configs, channel_set, log_scale, norm, emis_order=None, c_range=None):
        chans = configs.channels
        by_name = {c['name']: c for c in chans['channels']}
        self.channel_set = channel_set
        self.names = configs.channel_set(channel_set)
        if cfg.CLIMWD in self.names:
            raise cfg.ConfigError("S3_CHANNELS.json's %s set already names %s" % (channel_set, cfg.CLIMWD))
        self.log_scale = bool(log_scale)
        self.src = [(by_name[n]['array'], int(by_name[n]['index'])) for n in self.names]
        self.pos16 = [i for i, (a, _) in enumerate(self.src) if a == 'inputs_f16']
        self.idx16 = [self.src[i][1] for i in self.pos16]
        self.pos32 = [i for i, (a, _) in enumerate(self.src) if a == 'inputs_f32']
        self.idx32 = [self.src[i][1] for i in self.pos32]
        self.posl4 = [i for i, (a, _) in enumerate(self.src) if a == cfg.L4_ARRAY]
        self.idxl4 = [self.src[i][1] for i in self.posl4]
        self.is_log = [bool(by_name[n].get('log_scale_in_advanced')) and self.log_scale for n in self.names]
        if emis_order is None:
            order = chans.get('f6_emission_arrays', {}).get('channel_order', [])
            emis_order = [str(s).split()[0] for s in order]
        self.emis_names = list(emis_order)
        self.emis_pos = [self.names.index(n) for n in self.emis_names if n in self.names]
        self.emis_k = [k for k, n in enumerate(self.emis_names) if n in self.names]
        nch = norm['channels']
        self.lo, self.den, self.c = ([], [], [])
        self.zero_range = []
        for n, lg in zip(self.names, self.is_log):
            if n not in nch:
                raise cfg.ConfigError('NORM %s lacks channel %s' % (norm.get('name'), n))
            st = nch[n]
            if lg:
                for k in ('c', 'log10_min', 'log10_max'):
                    if k not in st:
                        raise cfg.ConfigError('NORM %s: %s lacks %s' % (norm.get('name'), n, k))
                lo, hi = (float(st['log10_min']), float(st['log10_max']))
                self.c.append(float(st['c']))
            else:
                lo, hi = (float(st['min']), float(st['max']))
                self.c.append(0.0)
            d = hi - lo
            if not d > 0:
                self.zero_range.append(n)
                d = 1.0
            self.lo.append(lo)
            self.den.append(d)
        self.c_pos = None
        if c_range is not None:
            lo, hi = (float(c_range[0]), float(c_range[1]))
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise cfg.ConfigError('the C range is not finite')
            self.c_pos = len(self.names)
            self.names = self.names + [cfg.CLIMWD]
            self.src.append(('climwd', 0))
            self.is_log.append(False)
            self.c.append(0.0)
            d = hi - lo
            if not d > 0:
                self.zero_range.append(cfg.CLIMWD)
                d = 1.0
            self.lo.append(lo)
            self.den.append(d)
        t = norm['target']
        self.t_lo = float(t['min'])
        self.t_den = float(t['max']) - float(t['min'])
        if not self.t_den > 0:
            raise cfg.ConfigError('NORM %s: target range %r' % (norm.get('name'), t))
        self.norm_name = norm.get('name')

    @property
    def n(self):
        return len(self.names)

    def describe(self):
        d = {'channel_set': self.channel_set, 'channels': self.names, 'n_channels': self.n, 'log_scale': self.log_scale, 'log_channels': [n for n, g in zip(self.names, self.is_log) if g], 'f16_channels': [self.names[i] for i in self.pos16], 'f32_channels': [self.names[i] for i in self.pos32], 'emission_order': self.emis_names, 'zero_range': self.zero_range, 'norm': self.norm_name, 'c_channel': self.c_pos is not None}
        if self.posl4:
            d['l4_channels'] = [self.names[i] for i in self.posl4]
        return d

    def to_ppb(self, field):
        return np.asarray(field, dtype=np.float64) * self.t_den + self.t_lo

class Scaler:

    def __init__(self, plan, device, s_f=None):
        import torch
        self.plan = plan
        self.device = device
        self.s_f = float(s_f) if s_f is not None else None
        f = torch.float32
        self.lo = torch.tensor(plan.lo, dtype=f, device=device).view(1, -1, 1, 1)
        self.den = torch.tensor(plan.den, dtype=f, device=device).view(1, -1, 1, 1)
        self.log_idx = [i for i, g in enumerate(plan.is_log) if g]
        self.c = torch.tensor([plan.c[i] for i in self.log_idx], dtype=f, device=device).view(1, -1, 1, 1)
        self.log_t = torch.tensor(self.log_idx, dtype=torch.long, device=device)
        self.t_lo = float(plan.t_lo)
        self.t_den = float(plan.t_den)

    def inputs(self, x):
        import torch
        if self.log_idx:
            x = x.clone()
            sub = torch.index_select(x, 1, self.log_t)
            x[:, self.log_idx] = torch.log10(torch.clamp(sub, min=0.0) + self.c)
        return (x - self.lo) / self.den

    def target(self, y, c=None):
        if self.s_f is not None:
            if c is None:
                raise ValueError('a residual target needs its C tile')
            return (y - c) / self.s_f
        return (y - self.t_lo) / self.t_den

def _open_npy(path, shape, dtype, nbytes=None):
    if not os.path.isfile(path):
        raise cfg.ConfigError('missing array: %s' % cfg.rel(path))
    if nbytes is not None and os.stat(path).st_size != int(nbytes):
        raise cfg.ConfigError('size of %s is %d, S3_ARRAYS.json %d' % (cfg.rel(path), os.stat(path).st_size, int(nbytes)))
    a = np.load(path, mmap_mode='r')
    if list(a.shape) != list(shape) or str(a.dtype) != str(dtype):
        raise cfg.ConfigError('%s: shape %s %s, expected %s %s' % (cfg.rel(path), list(a.shape), a.dtype, list(shape), dtype))
    return a

class YearArrays:

    def __init__(self, configs, paths, year, with_target=True, l4=False):
        ent = configs.arrays['years'][str(year)]
        self.year = int(year)
        self.dates = list(ent['dates'])
        self.index = {d: i for i, d in enumerate(self.dates)}
        fl = ent['files']
        self.f16 = _open_npy(paths.array(fl['inputs_f16']['path']), fl['inputs_f16']['shape'], fl['inputs_f16']['dtype'], fl['inputs_f16'].get('bytes'))
        self.f32 = _open_npy(paths.array(fl['inputs_f32']['path']), fl['inputs_f32']['shape'], fl['inputs_f32']['dtype'], fl['inputs_f32'].get('bytes'))
        self.tgt = None
        if with_target:
            self.tgt = _open_npy(paths.array(fl['target']['path']), fl['target']['shape'], fl['target']['dtype'], fl['target'].get('bytes'))
        self.fl4 = None
        if l4:
            r = configs.l4_file(year)
            self.fl4 = _open_npy(paths.array(r['path']), r['shape'], r['dtype'], r.get('bytes'))
        n = len(self.dates)
        for a in (self.f16, self.f32, self.tgt, self.fl4):
            if a is not None and a.shape[0] != n:
                raise cfg.ConfigError('year %d: %d dates, array %s' % (year, n, list(a.shape)))
        self.grid = tuple(self.f16.shape[2:])
        if tuple(self.f32.shape[2:]) != self.grid or (self.tgt is not None and tuple(self.tgt.shape[1:]) != self.grid) or (self.fl4 is not None and tuple(self.fl4.shape[2:]) != self.grid):
            raise cfg.ConfigError('year %d: grids differ' % year)

def load_mask(configs, paths):
    m = configs.channels['mask']
    p = paths.array(m['path'])
    if not os.path.isfile(p):
        raise cfg.ConfigError('missing mask: %s' % cfg.rel(p))
    if cfg.sha256_file(p) != m['sha256']:
        raise cfg.ConfigError('mask sha256 differs from S3_CHANNELS.json')
    a = np.load(p)
    if a.dtype != np.bool_ or int(a.sum()) != int(m['cells']):
        raise cfg.ConfigError('mask: dtype %s, %d cells, expected %s' % (a.dtype, int(a.sum()), m['cells']))
    return a

def mask_cells(mask):
    return np.flatnonzero(np.asarray(mask, dtype=bool).ravel())

def calendar_position(s):
    d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return datetime.date(2016, d.month, d.day).timetuple().tm_yday

def weekday(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).weekday()

def c_fit_of(training, split, year):
    if training.get('c_fit'):
        return str(training['c_fit'])
    if int(year) in split.train_years:
        return 'wo%d' % int(year)
    return 'all'

def needs_c(training):
    return training.get('target') == 'residual' or bool(training.get('c_channel'))

class ClimC:

    def __init__(self, configs, paths, training, split, mask):
        self.fold = str(training.get('ctables_fold') or training['held_out_year'])
        idx = cfg.ctables_index(paths, self.fold)
        self.index_sha256 = cfg.sha256_file(paths.config('CTABLES_%s.json' % self.fold))
        if idx.get('schema') != cfg.CTABLES_SCHEMA or str(idx.get('fold')) != self.fold:
            raise cfg.ConfigError('CTABLES_%s.json: schema or fold' % self.fold)
        want = {'mask_sha256': configs.channels['mask']['sha256'], 'splits_sha256': configs.sha256.get('SPLITS.json')}
        for k, v in want.items():
            if idx.get(k) != v:
                raise cfg.ConfigError("CTABLES_%s.json: %s differs from this run's" % (self.fold, k))
        lb = idx.get('lockbox') or {}
        if LOCKBOX_FILE is not None:
            lbx = lockbox_maps(configs)
            if lb.get('sha256') != lbx[2]:
                raise cfg.ConfigError("CTABLES_%s.json: the lockbox file's sha256 differs from this run's" % self.fold)
        self.mask = np.asarray(mask, dtype=bool)
        self.cells = mask_cells(self.mask)
        self.split = split
        self.training = training
        fits = sorted(set((c_fit_of(training, split, y) for y in list(split.train_years) + [split.held_year])))
        self.tables = {}
        self.files = {}
        for f in fits:
            if f not in idx.get('fits', {}):
                raise cfg.ConfigError('CTABLES_%s.json has no fit %s' % (self.fold, f))
            pair = []
            for part, rows in (('doy', 366), ('r', 7)):
                ent = idx['fits'][f][part]
                p = cfg.ctable_path(paths, ent['file'])
                if not os.path.isfile(p):
                    raise cfg.ConfigError('C table %s absent' % ent['file'])
                sha = cfg.sha256_file(p)
                if sha != ent['sha256']:
                    raise cfg.ConfigError('C table %s: sha256 differs from its index' % ent['file'])
                a = np.load(p, allow_pickle=False)
                if a.dtype != np.float64 or a.shape != (rows, len(self.cells)):
                    raise cfg.ConfigError('C table %s: %s %s, expected float64 %s' % (ent['file'], a.dtype, a.shape, (rows, len(self.cells))))
                pair.append(a)
                self.files[ent['file']] = sha
            self.tables[f] = tuple(pair)
        own = training.get('c_fit')
        sf, lo, hi = (idx.get('s_f'), idx.get('c_min'), idx.get('c_max'))
        if isinstance(sf, dict):
            if not own:
                raise cfg.ConfigError('CTABLES_%s.json gives s_f per fit and the training names none' % self.fold)
            sf, lo, hi = (sf[own], lo[own], hi[own])
        self.s_f, self.c_min, self.c_max = (float(sf), float(lo), float(hi))
        if not (self.s_f > 0 and math.isfinite(self.s_f)):
            raise cfg.ConfigError('CTABLES_%s.json: s_f %r' % (self.fold, sf))
        self._grids = {}

    def masked(self, year, date):
        doy, r = self.tables[c_fit_of(self.training, self.split, year)]
        return np.maximum(doy[calendar_position(date) - 1] * r[weekday(date)], 0.0)

    def day_grid(self, year, idx, date):
        key = (int(year), int(idx))
        g = self._grids.get(key)
        if g is None:
            g = np.zeros(self.mask.shape, dtype=np.float32)
            g.ravel()[self.cells] = self.masked(year, date).astype(np.float32)
            self._grids[key] = g
        return g

    def describe(self):
        return {'fold': self.fold, 'index_sha256': self.index_sha256, 'fits': sorted(self.tables), 'tables_sha256': dict(sorted(self.files.items())), 'formula': 'C(i, d) = doy[p(d) - 1, i] x r[w(d), i], clipped at 0', 's_f_c_range': 'from the index (values not printed)'}

def coverage_record(mask, rules=None, tile=cfg.TILE):
    rules = rules or {'P_A1_0.25': cfg.MIN_MASKED_FRACTION, 'E2_PIXR_1_of_4096': cfg.MIN_MASKED_FRACTION_ONE}
    m = np.asarray(mask, dtype=bool)
    out, maps = ({}, {})
    for key, frac in sorted(rules.items()):
        rr, cc, need, counts = valid_positions(m, tile, frac)
        cov = np.zeros(m.shape, dtype=np.int64)
        for r0, c0 in zip(rr.tolist(), cc.tolist()):
            cov[r0:r0 + tile, c0:c0 + tile] += 1
        vals = cov[m]
        pct = np.percentile(vals, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist() if vals.size else []
        share = float(np.mean(counts[rr, cc] / float(tile * tile))) if len(rr) else 0.0
        out[key] = {'min_masked_fraction': frac, 'min_masked_cells': int(need), 'valid_positions': int(len(rr)), 'mean_masked_share_of_a_valid_tile': share, 'cover_percentiles_0_1_5_25_50_75_95_99_100': pct, 'masked_cells_never_covered': int((vals == 0).sum()), 'masked_cells': int(m.sum())}
        maps[key] = cov.astype(np.int32)
    return (out, maps)

def blend_coverage(mask, floor=None):
    from choi import infer
    floor = infer.FLOOR if floor is None else float(floor)
    m = np.asarray(mask, dtype=bool)
    w = infer.blend_weights(m.shape)[:m.shape[0], :m.shape[1]]
    vals = w[m]
    return (int((vals <= floor).sum()), float(vals.min()) if vals.size else None)

class EmisArrays:

    def __init__(self, configs, paths, years):
        oth = configs.arrays['other']
        self.true, self.clim = ({}, {})
        for y in years:
            k = 'emis_true_%d.npy' % y
            if k not in oth:
                raise cfg.ConfigError('S3_ARRAYS.json lacks %s' % k)
            e = oth[k]
            self.true[y] = _open_npy(paths.array(e['path']), e['shape'], e['dtype'], e.get('bytes'))
            dk = 'emis_true_%d_days.json' % y
            dd = cfg.read_json(paths.array(oth[dk]['path']))
            days = dd if isinstance(dd, list) else dd.get('dates') or dd.get('days')
            ydates = configs.arrays['years'][str(y)]['dates']
            if list(days) != list(ydates):
                raise cfg.ConfigError("%s: its days are not the year's dates in order" % dk)
            k = 'emis_clim_%d.npy' % y
            e = oth[k]
            self.clim[y] = _open_npy(paths.array(e['path']), e['shape'], e['dtype'], e.get('bytes'))

class F6AArrays:

    def __init__(self, configs, paths, years):
        self.order, key = configs.f6a()
        self.key = key
        self.arr = {}
        for y in years:
            e = configs.arrays['years'][str(y)]['files'][key]
            a = _open_npy(paths.array(e['path']), e['shape'], e['dtype'], e.get('bytes'))
            nd = len(configs.arrays['years'][str(y)]['dates'])
            if a.shape[0] != nd or a.shape[1] != len(self.order) or str(a.dtype) != 'float32':
                raise cfg.ConfigError('climatological-emission %s %d: shape %s %s' % (key, y, list(a.shape), a.dtype))
            self.arr[y] = a

def arm_source(arm, day_year, train_year, test_year):
    if arm == 'TRUE':
        return ('true', day_year)
    if arm == 'OWN':
        return ('clim', day_year)
    if arm == 'OTHER':
        other = test_year if day_year == train_year else train_year
        return ('clim', other)
    if arm == 'F6A':
        return ('f6a', day_year)
    raise cfg.ConfigError('unknown arm %r' % (arm,))

def check_arm_text(configs, key, arm, train_year, test_year, norm):
    text = configs.splits['f6'][key]['arms'][arm]

    def fname(src):
        kind, y = src
        return 'emis_%s_%d.npy' % (kind, y)
    f_tr = fname(arm_source(arm, train_year, train_year, test_year))
    f_te = fname(arm_source(arm, test_year, train_year, test_year))
    i_tr = text.find(f_tr)
    i_te = text.find(f_te, i_tr + len(f_tr)) if i_tr >= 0 else -1
    i_trd = text.find('training days')
    i_ted = text.find('test days')
    if not (i_tr >= 0 and i_te > i_tr and (0 <= i_trd < i_ted)):
        raise cfg.ConfigError('f6 %s %s: arm text does not name %s for the training days and %s for the test days' % (key, arm, f_tr, f_te))
    es = str(norm.get('emission_source', ''))
    if f_tr not in es:
        raise cfg.ConfigError('NORM %s: emission_source does not name %s' % (norm.get('name'), f_tr))
    return {'train_days': f_tr, 'test_days': f_te}

def check_arm_channels(plan):
    held = [n for n in plan.names if n in cfg.EMISSION_CHANNELS]
    miss = [n for n in held if n not in plan.emis_names]
    if not plan.emis_names or miss or (not held):
        raise cfg.ConfigError("arm: the emission order %s lacks %s of the model's emission channels %s" % (plan.emis_names, miss, held))

def check_norm(norm, training, split):
    n, t, s = (norm, training, split)
    if t['kind'] == 'fold':
        want = 'f6a' if t.get('arm') == 'F6A' else 'fold'
        ok = n.get('kind') == want and int(n.get('held_out_year')) == s.held_year and (sorted((int(y) for y in n.get('training_years', []))) == s.train_years)
    else:
        ok = n.get('kind') == 'f6' and n.get('arm') == t['arm'] and (int(n.get('train_year')) == int(t['train_year'])) and (int(n.get('test_year')) == s.held_year)
    if not ok or int(n.get('training_days', -1)) != len(s.train):
        raise cfg.ConfigError('NORM %s does not match the training %s' % (n.get('name'), t['name']))

def check_built(configs, plan, years, arm):
    replaced = set(plan.emis_names) if arm else set()
    for y in years:
        bad = sorted((set(plan.names) & configs.unfilled(y)) - replaced)
        if bad:
            raise cfg.ConfigError('year %d: channels %s are not built yet (a partial year; the fills stage)' % (int(y), bad))
S18_L4 = "the arm of SALMAN's 18 needs L-4 ALBEDO, built before its training"

def s18_l4_ok(configs, departures=None):
    if 'L-4' in configs.channel_set('arm18'):
        return True
    return any(('L-4' in str(d) for d in departures or []))

def check_s18_l4(configs, training, departures=None):
    if training.get('net') == 's18' and (not s18_l4_ok(configs, departures)):
        raise cfg.ConfigError(S18_L4)

def restrict_years(split, years):
    if years is None:
        return list(split.train_years)
    ys = sorted(set((int(y) for y in years)))
    bad = [y for y in ys if y not in split.train_years]
    if not ys or bad:
        raise cfg.ConfigError('years %s: not a subset of the training years %s of %s' % (ys, split.train_years, split.key))
    return ys

def emission_order_for(configs, training):
    if training.get('arm') == 'F6A':
        return configs.f6a()[0]
    order = configs.channels.get('f6_emission_arrays', {}).get('channel_order', [])
    return [str(s).split()[0] for s in order]

def _days_of(part, years_arrays_dates):
    out = []
    for ys, idxs in sorted(part['index_by_year'].items()):
        y = int(ys)
        dates = years_arrays_dates[y]
        for i in idxs:
            out.append((y, int(i), dates[int(i)]))
    want = sorted(part['dates'])
    got = sorted((d for _, _, d in out))
    if got != want or len(out) != int(part['count']):
        raise cfg.ConfigError('SPLITS.json: index_by_year and dates differ')
    return out

def lockbox_maps(configs):
    if LOCKBOX_FILE is None:
        return None
    got = getattr(configs, '_lockbox', None)
    if got is not None:
        return got
    import tier_a_rules as R
    p = configs.paths.config(LOCKBOX_FILE)
    if not os.path.isfile(p):
        raise cfg.ConfigError('the lockbox is decided but %s is absent from the config folder' % LOCKBOX_FILE)
    doc = cfg.read_json(p)
    try:
        days, lock = (R.lockbox_days(doc), R.lockbox_only(doc))
    except ValueError as e:
        raise cfg.ConfigError('%s: %s' % (LOCKBOX_FILE, e))
    sha = cfg.sha256_file(p)
    configs.sha256[LOCKBOX_FILE] = sha
    got = (set(((y, i) for y, s in days.items() for i in s)), set(((y, i) for y, s in lock.items() for i in s)), sha)
    configs._lockbox = got
    return got
GUARD_PURPOSE = {'train': 'fit', 'stat': 'stat', 'val': 'validate', 'predict': 'predict_cv', 'predict_trains': 'stat', 'predict_val': 'validate'}

class Split:

    def __init__(self, configs, training):
        sec, key = training['split']
        if sec == 'f6a' and 'f6a' not in configs.splits:
            sec = 'folds'
        if sec == 'f6a':
            same = configs.splits['f6a'][key].get('same_days_as')
            if same and same != 'folds.%s' % key:
                raise cfg.ConfigError('SPLITS.json f6a.%s: same_days_as %r' % (key, same))
        s = configs.splits[sec][key]
        all_dates = {int(y): configs.arrays['years'][y]['dates'] for y in configs.arrays['years']}
        self.key = '%s.%s' % (sec, key)
        self.train = _days_of(s['train'], all_dates)
        self.val = _days_of(s['validation'], all_dates)
        self.held = _days_of(s['held_out'], all_dates)
        lbx = lockbox_maps(configs)
        self.lockbox_set = lbx[0] if lbx else set()
        self.lockbox_only = lbx[1] if lbx else set()
        self.lockbox_sha256 = lbx[2] if lbx else None
        self.removed = {}
        if lbx:
            for part in ('train', 'val', 'held'):
                was = getattr(self, part)
                keep = [x for x in was if (x[0], x[1]) not in self.lockbox_set]
                self.removed[part] = len(was) - len(keep)
                setattr(self, part, keep)
        self.held_year = int(training['held_out_year'])
        self.held_set = set(((y, i) for y, i, _ in self.held))
        self.held_dates = set((d for _, _, d in self.held))
        if any((y != self.held_year for y, _, _ in self.held)):
            raise HeldOutError('%s: held-out days outside %d' % (self.key, self.held_year))
        if sec in ('folds', 'f6a'):
            ty = set((int(y) for y in s['training_years']))
            if self.held_year in ty or int(s['held_out_year']) != self.held_year:
                raise HeldOutError('%s: the held-out year among the training years' % self.key)
            exp = set(cfg.FOLD_YEARS) - {self.held_year}
            if ty != exp:
                raise cfg.ConfigError('%s: training years %s' % (self.key, sorted(ty)))
        else:
            if int(s['test_year']) != self.held_year or int(s['train_year']) != int(training['train_year']):
                raise cfg.ConfigError('%s: years differ from the training' % self.key)
            ty = {int(s['train_year'])}
        self.train_years = sorted(ty)
        for name, part in (('train', self.train), ('validation', self.val)):
            for y, i, d in part:
                if (y, i) in self.held_set or d in self.held_dates or y == self.held_year:
                    raise HeldOutError('%s: held-out day %s in the %s list' % (self.key, d, name))
                if y not in ty:
                    raise cfg.ConfigError('%s: day %s outside the training years' % (self.key, d))
        tr = set(((y, i) for y, i, _ in self.train))
        va = set(((y, i) for y, i, _ in self.val))
        if tr & va:
            raise cfg.ConfigError('%s: training and validation overlap' % self.key)
        self.trains = [(y, i, d) for y, i, d in self.train if iso_week(d) % 5 == cfg.TRAINS_ISO_REMAINDER]
        self.trains_set = set(((y, i) for y, i, _ in self.trains))
        self.val_set = va
        lb_doc = None
        if lbx:
            lb_doc = cfg.read_json(configs.paths.config(LOCKBOX_FILE))
        self.roles = GD.roles_of(s, lb_doc)
        dates = all_dates[self.held_year]
        self.lockbox = sorted(((y, i, dates[i]) for y, i in self.lockbox_only if y == self.held_year and self.roles.get((y, i)) == 'LOCKBOX'))
        self.lockbox_days_set = set(((y, i) for y, i, _ in self.lockbox))
        for part, purpose in ((self.train, 'fit'), (self.val, 'validate'), (self.held, 'predict_cv'), (self.lockbox, 'predict_lockbox')):
            GD.roles_guard(part, purpose, self.roles)

    def guard(self, year, idx, mode):
        key = (int(year), int(idx))
        if mode == 'predict_lockbox':
            if key not in self.lockbox_days_set:
                raise HeldOutError('%s: day %d/%d is not a lockbox day of %d' % (self.key, year, idx, self.held_year))
            GD.roles_guard([key], 'predict_lockbox', self.roles)
            return
        if key in self.lockbox_set:
            raise HeldOutError('%s: lockbox or buffer day %d/%d asked for %s' % (self.key, year, idx, mode))
        if mode in GUARD_PURPOSE:
            GD.roles_guard([key], GUARD_PURPOSE[mode], self.roles)
        held = key in self.held_set or int(year) == self.held_year
        if mode in ('train', 'val', 'stat'):
            if held:
                raise HeldOutError('%s: held-out day %d/%d asked for %s' % (self.key, year, idx, mode))
        elif mode == 'predict':
            if key not in self.held_set:
                raise HeldOutError('%s: day %d/%d is not held out' % (self.key, year, idx))
        elif mode == 'predict_trains':
            if held or key not in self.trains_set:
                raise HeldOutError('%s: day %d/%d is not a TRAIN-S day' % (self.key, year, idx))
        elif mode == 'predict_val':
            if held or key not in self.val_set:
                raise HeldOutError('%s: day %d/%d is not a VAL day' % (self.key, year, idx))
        else:
            raise ValueError('mode %r' % (mode,))

    def describe(self):
        return {'split': self.key, 'train_days': len(self.train), 'validation_days': len(self.val), 'held_out_days': len(self.held), 'trains_days': len(self.trains), 'held_out_year': self.held_year, 'train_years': self.train_years, 'lockbox': {'file': LOCKBOX_FILE, 'sha256': self.lockbox_sha256, 'left_train': self.removed.get('train', 0), 'left_validation': self.removed.get('val', 0), 'left_held_out': self.removed.get('held', 0), 'held_out_is': 'TEST_CV'} if self.lockbox_sha256 else None}

def valid_positions(loss_mask, tile=cfg.TILE, min_frac=cfg.MIN_MASKED_FRACTION):
    m = np.asarray(loss_mask, dtype=np.int64)
    h, w = m.shape
    if h < tile or w < tile:
        raise ValueError('grid %s smaller than the tile' % (m.shape,))
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    counts = ii[tile:, tile:] - ii[:-tile, tile:] - ii[tile:, :-tile] + ii[:-tile, :-tile]
    need = int(math.ceil(min_frac * tile * tile))
    rr, cc = np.nonzero(counts >= need)
    return (rr.astype(np.int64), cc.astype(np.int64), need, counts)

def seed_sequences(seed, n=2):
    ss = np.random.SeedSequence(int(seed))
    return tuple(ss.spawn(n))

class TileSampler:

    def __init__(self, days, rows, cols, seed_seq):
        self.days = list(days)
        self.rows = rows
        self.cols = cols
        self.rng = np.random.Generator(np.random.PCG64(seed_seq))
        if not self.days or len(rows) == 0:
            raise ValueError('no days or no valid positions')

    def draw(self, n):
        di = self.rng.integers(0, len(self.days), size=n)
        pi = self.rng.integers(0, len(self.rows), size=n)
        return [(self.days[a][0], self.days[a][1], int(self.rows[b]), int(self.cols[b])) for a, b in zip(di, pi)]

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, st):
        self.rng.bit_generator.state = st

class PeakList:

    def __init__(self, years, train_days, loss_mask, counts, need, grid, tile=cfg.TILE, max_off=cfg.E6_MAX_OFFSET, pct=cfg.E6_PERCENTILE):
        cells = mask_cells(loss_mask)
        h, w = grid
        vals = np.empty((len(train_days), len(cells)), dtype=np.float32)
        for k, (y, i, _) in enumerate(train_days):
            vals[k] = np.asarray(years[y].tgt[i]).ravel()[cells]
        self.threshold = float(np.percentile(vals, pct))
        kk, cc = ([], [])
        for a in range(0, vals.shape[0], 64):
            k_, c_ = np.nonzero(vals[a:a + 64].astype(np.float64) >= self.threshold)
            kk.append(k_ + a)
            cc.append(c_)
        kk = np.concatenate(kk) if kk else np.zeros(0, dtype=np.int64)
        cc = np.concatenate(cc) if cc else np.zeros(0, dtype=np.int64)
        self.n_cell_days = int(vals.size)
        del vals
        rows, cols = (cells[cc] // w, cells[cc] % w)
        offs = np.arange(-max_off, max_off + 1)
        dr, dc = np.meshgrid(offs, offs, indexing='ij')
        dr, dc = (dr.ravel(), dc.ravel())
        self.max_off = int(max_off)
        self.tile = int(tile)
        self.grid = (int(h), int(w))
        entries, n_out = ([], 0)
        for k, r, c in zip(kk, rows, cols):
            r0 = np.clip(r + dr - tile // 2, 0, h - tile)
            c0 = np.clip(c + dc - tile // 2, 0, w - tile)
            ok = counts[r0, c0] >= need
            if not ok.any():
                n_out += 1
                continue
            entries.append((int(k), int(r), int(c)))
        self.days = [(train_days[k][0], train_days[k][1]) for k, _, _ in entries]
        self.cells = [(r, c) for _, r, c in entries]
        self.n_left_out = int(n_out)
        self.counts = counts
        self.need = int(need)
        if not self.cells:
            raise ValueError('the peak list is empty')

    def summary(self):
        import hashlib
        h = hashlib.sha256()
        for (y, i), (r, c) in zip(self.days, self.cells):
            h.update(('%d %d %d %d\n' % (y, i, r, c)).encode('ascii'))
        return {'percentile': cfg.E6_PERCENTILE, 'threshold_target_stored_units': self.threshold, 'cell_days_considered': self.n_cell_days, 'entries': len(self.cells), 'left_out_no_valid_offset': self.n_left_out, 'max_offset': self.max_off, 'entries_sha256': h.hexdigest()}

    def tile_for(self, rng, k):
        (y, i), (r, c) = (self.days[k], self.cells[k])
        h, w = self.grid
        t = self.tile
        while True:
            dr, dc = rng.integers(-self.max_off, self.max_off + 1, size=2)
            r0 = int(min(max(r + dr - t // 2, 0), h - t))
            c0 = int(min(max(c + dc - t // 2, 0), w - t))
            if self.counts[r0, c0] >= self.need:
                return (y, i, r0, c0)

class E6Sampler:

    def __init__(self, base, peaks, seed_seq):
        self.base = base
        self.peaks = peaks
        self.rng = np.random.Generator(np.random.PCG64(seed_seq))

    def draw(self, n):
        n0 = n - int(round(n * cfg.E6_PEAK_SHARE))
        out = self.base.draw(n0)
        for _ in range(n - n0):
            k = int(self.rng.integers(0, len(self.peaks.cells)))
            out.append(self.peaks.tile_for(self.rng, k))
        return out

    def state(self):
        return {'e0': self.base.state(), 'peak': self.rng.bit_generator.state}

    def set_state(self, st):
        self.base.set_state(st['e0'])
        self.rng.bit_generator.state = st['peak']

def block_mask(grid, row, col, size=cfg.BLOCK_SIZE):
    h, w = grid
    b = np.zeros((h, w), dtype=bool)
    r0, c0 = (row - size // 2, col - size // 2)
    b[max(0, r0):max(0, min(h, r0 + size)), max(0, c0):max(0, min(w, c0 + size))] = True
    return b

class TrainingData:

    def __init__(self, configs, paths, training, device='cpu', reader='memmap', val_tiles=cfg.VAL_TILES, block_cell=None, train_stat_tiles=cfg.TRAIN_STAT_TILES, stage=None, norm_check=True, years=None, departures=None):
        from train import plan as plan_mod
        self.training = training
        self.device = device
        self.split = Split(configs, training)
        net, chset, logs = plan_mod.network_spec(training['net'])
        check_s18_l4(configs, training, departures)
        self.norm = configs.norm(training['norm'])
        if norm_check:
            self._check_norm()
        self.read_years = restrict_years(self.split, years)
        self.years_restriction = list(self.read_years) if years is not None else None
        self.train_days = [x for x in self.split.train if x[0] in self.read_years]
        self.val_days = [x for x in self.split.val if x[0] in self.read_years]
        self.plan = ChannelPlan(configs, chset, logs, self.norm, emission_order_for(configs, training))
        check_built(configs, self.plan, self.read_years, training.get('arm'))
        self.years = {y: YearArrays(configs, paths, y, l4=bool(self.plan.posl4)) for y in self.read_years}
        grids = set((ya.grid for ya in self.years.values()))
        if len(grids) != 1:
            raise cfg.ConfigError('grids differ between years')
        self.grid = grids.pop()
        self.mask = load_mask(configs, paths)
        if tuple(self.mask.shape) != self.grid:
            raise cfg.ConfigError('mask grid %s, arrays %s' % (self.mask.shape, self.grid))
        self.climc = None
        self.residual = training.get('target') == 'residual'
        if needs_c(training):
            self.climc = ClimC(configs, paths, training, self.split, self.mask)
            if training.get('c_channel'):
                self.plan = ChannelPlan(configs, chset, logs, self.norm, emission_order_for(configs, training), c_range=(self.climc.c_min, self.climc.c_max))
        self.loss_mask = self.mask.copy()
        self.block = None
        if training.get('block'):
            if block_cell is None:
                raise cfg.ConfigError('block training without its cell')
            bm = block_mask(self.grid, int(block_cell[0]), int(block_cell[1]), training['block']['size'])
            self.block = {'row': int(block_cell[0]), 'col': int(block_cell[1]), 'cells_left_out': int((bm & self.mask).sum())}
            self.loss_mask = self.mask & ~bm
        self.rows, self.cols, self.need, self.counts = valid_positions(self.loss_mask, min_frac=float(training.get('min_masked_fraction', cfg.MIN_MASKED_FRACTION)))
        self.arm = training.get('arm')
        self.emis = None
        self.f6a = None
        self.arm_files = None
        if self.arm in cfg.F6_ARMS:
            key = training['split'][1]
            ty, te = (int(training['train_year']), self.split.held_year)
            self.arm_files = check_arm_text(configs, key, self.arm, ty, te, self.norm)
            check_arm_channels(self.plan)
            self.emis = EmisArrays(configs, paths, sorted({ty, te}))
            self.train_year, self.test_year = (ty, te)
        elif self.arm == 'F6A':
            check_arm_channels(self.plan)
            self.f6a = F6AArrays(configs, paths, self.read_years)
            self.arm_files = {'f6a_file_key': self.f6a.key, 'years': list(self.read_years)}
            self.train_year = self.test_year = None
        elif self.arm is not None:
            raise cfg.ConfigError('unknown arm %r' % (self.arm,))
        seqs = seed_sequences(training['seed'], 3)
        tr_seq, va_seq = (seqs[0], seqs[1])
        self.sampler = TileSampler([(y, i) for y, i, _ in self.train_days], self.rows, self.cols, tr_seq)
        self.peaks = None
        self.draw_sampler = self.sampler
        if training['net'] == 'e6':
            self.peaks = PeakList(self.years, self.train_days, self.loss_mask, self.counts, self.need, self.grid)
            self.draw_sampler = E6Sampler(self.sampler, self.peaks, seqs[2])
        vs = TileSampler([(y, i) for y, i, _ in self.val_days], self.rows, self.cols, va_seq)
        self.val_draws = vs.draw(int(val_tiles))
        for y, i, _, _ in self.val_draws:
            self.split.guard(y, i, 'val')
        ts = TileSampler([(y, i) for y, i, _ in self.train_days], self.rows, self.cols, np.random.SeedSequence(int(training['seed']) + cfg.STAT_SEED_OFFSET))
        self.stat_draws = ts.draw(int(train_stat_tiles))
        for y, i, _, _ in self.stat_draws:
            self.split.guard(y, i, 'stat')
        self.scaler = Scaler(self.plan, device, s_f=self.climc.s_f if self.residual else None)
        self.ppb_scale = self.climc.s_f if self.residual else self.plan.t_den
        if reader == 'device':
            self.reader = DeviceReader(self)
        elif reader == 'memmap':
            self.reader = MemmapReader(self, stage=stage)
        else:
            raise ValueError('reader %r' % (reader,))
        self.reader_kind = reader

    def _check_norm(self):
        check_norm(self.norm, self.training, self.split)

    def draw_batch(self, n):
        return self.draw_sampler.draw(n)

    def sampler_state(self):
        return self.draw_sampler.state()

    def set_sampler_state(self, st):
        self.draw_sampler.set_state(st)

    def emis_tile(self, year, idx, r0, c0, t):
        if self.arm == 'F6A':
            k = self.plan.emis_k
            return self.f6a.arr[year][idx, k, r0:r0 + t, c0:c0 + t]
        kind, y = arm_source(self.arm, year, self.train_year, self.test_year)
        k = self.plan.emis_k
        if kind == 'true':
            return self.emis.true[y][idx, k, r0:r0 + t, c0:c0 + t]
        mo, wd = date_month_weekday(self.years_dates(year)[idx])
        return self.emis.clim[y][mo - 1, wd, k, r0:r0 + t, c0:c0 + t]

    def years_dates(self, year):
        if year in self.years:
            return self.years[year].dates
        raise KeyError(year)

    def scale_raw(self, raw):
        c = raw[3] if len(raw) > 3 else None
        return (self.scaler.inputs(raw[0]), self.scaler.target(raw[1], c), raw[2])

    def c_tile(self, year, idx, r0, c0, t):
        g = self.climc.day_grid(year, idx, self.years_dates(year)[idx])
        return g[r0:r0 + t, c0:c0 + t]

    def batch(self, draws, mode):
        return self.scale_raw(self.reader.raw(draws, mode))

    def describe(self):
        d = {'split': self.split.describe(), 'channels': self.plan.describe(), 'grid': list(self.grid), 'mask_cells': int(self.mask.sum()), 'loss_mask_cells': int(self.loss_mask.sum()), 'valid_positions': int(len(self.rows)), 'min_masked_cells_per_tile': int(self.need), 'reader': self.reader_kind, 'reader_staging': getattr(self.reader, 'staging', None), 'val_tiles': len(self.val_draws), 'train_stat_tiles': len(self.stat_draws), 'train_stat_seed': int(self.training['seed']) + cfg.STAT_SEED_OFFSET, 'arm': self.arm, 'arm_files': self.arm_files, 'block': self.block, 'peak_list': self.peaks.summary() if self.peaks else None, 'years_read': list(self.read_years), 'years_restriction': self.years_restriction, 'augmentation': 'none', 'target': 'residual' if self.residual else 'direct', 'climc': self.climc.describe() if self.climc else None, 'min_masked_fraction': float(self.training.get('min_masked_fraction', cfg.MIN_MASKED_FRACTION)), 'roles': GD.counts(self.split.roles)}
        return d

class MemmapReader:

    def __init__(self, td, stage=None):
        import torch
        self.td = td
        dev = str(td.device)
        self.staging = bool(stage) if stage is not None else dev.startswith('cuda')
        self.pinned = self.staging and dev.startswith('cuda') and torch.cuda.is_available()
        self._bufs = {}
        if self.staging and td.arm and any((pos in td.plan.pos16 for pos in td.plan.emis_pos)):
            raise cfg.ConfigError("staging: an arm's emission channel is stored as float16 (the storage rule gives float32)")

    def _buf(self, key, shape, dtype):
        import torch
        b = self._bufs.get(key)
        if b is None or tuple(b.shape) != tuple(shape):
            b = torch.empty(shape, dtype=dtype, pin_memory=self.pinned)
            self._bufs[key] = b
        return b

    def raw(self, draws, mode):
        if self.staging:
            return self._raw_staged(draws, mode)
        import torch
        td, p, t = (self.td, self.td.plan, cfg.TILE)
        b = len(draws)
        x = np.empty((b, p.n, t, t), dtype=np.float32)
        y = np.empty((b, 1, t, t), dtype=np.float32)
        m = np.empty((b, 1, t, t), dtype=bool)
        cc = np.empty((b, 1, t, t), dtype=np.float32) if td.climc is not None else None
        for k, (yr, i, r0, c0) in enumerate(draws):
            td.split.guard(yr, i, mode)
            ya = td.years[yr]
            if p.pos16:
                x[k, p.pos16] = ya.f16[i, p.idx16, r0:r0 + t, c0:c0 + t]
            if p.pos32:
                x[k, p.pos32] = ya.f32[i, p.idx32, r0:r0 + t, c0:c0 + t]
            if p.posl4:
                x[k, p.posl4] = ya.fl4[i, p.idxl4, r0:r0 + t, c0:c0 + t]
            if td.arm:
                x[k, p.emis_pos] = td.emis_tile(yr, i, r0, c0, t)
            if cc is not None:
                cc[k, 0] = td.c_tile(yr, i, r0, c0, t)
                if p.c_pos is not None:
                    x[k, p.c_pos] = cc[k, 0]
            y[k, 0] = ya.tgt[i, r0:r0 + t, c0:c0 + t]
            m[k, 0] = td.loss_mask[r0:r0 + t, c0:c0 + t]
        dev = td.device
        out = (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(m).to(dev))
        if cc is not None and td.residual:
            out = out + (torch.from_numpy(cc).to(dev),)
        return out

    def _raw_staged(self, draws, mode):
        import torch
        td, p, t = (self.td, self.td.plan, cfg.TILE)
        b = len(draws)
        dev = td.device
        h16 = self._buf('f16', (b, len(p.pos16), t, t), torch.float16)
        h32 = self._buf('f32', (b, len(p.pos32), t, t), torch.float32)
        hy = self._buf('y', (b, 1, t, t), torch.float32)
        hm = self._buf('m', (b, 1, t, t), torch.bool)
        n16, n32 = (h16.numpy(), h32.numpy())
        ny, nm = (hy.numpy(), hm.numpy())
        hc = self._buf('c', (b, 1, t, t), torch.float32) if td.climc is not None else None
        nc = hc.numpy() if hc is not None else None
        hl4 = nl4 = None
        if p.posl4:
            dt = next(iter(td.years.values())).fl4.dtype
            hl4 = self._buf('l4', (b, len(p.posl4), t, t), torch.float16 if dt == np.float16 else torch.float32)
            nl4 = hl4.numpy()
        pos32 = {pos: j for j, pos in enumerate(p.pos32)}
        pos16 = {pos: j for j, pos in enumerate(p.pos16)}
        for k, (yr, i, r0, c0) in enumerate(draws):
            td.split.guard(yr, i, mode)
            ya = td.years[yr]
            if p.pos16:
                n16[k] = ya.f16[i, p.idx16, r0:r0 + t, c0:c0 + t]
            if p.pos32:
                n32[k] = ya.f32[i, p.idx32, r0:r0 + t, c0:c0 + t]
            if nl4 is not None:
                nl4[k] = ya.fl4[i, p.idxl4, r0:r0 + t, c0:c0 + t]
            if td.arm:
                e = td.emis_tile(yr, i, r0, c0, t)
                for j, pos in enumerate(p.emis_pos):
                    if pos in pos32:
                        n32[k, pos32[pos]] = e[j]
                    else:
                        n16[k, pos16[pos]] = e[j]
            ny[k, 0] = ya.tgt[i, r0:r0 + t, c0:c0 + t]
            nm[k, 0] = td.loss_mask[r0:r0 + t, c0:c0 + t]
            if nc is not None:
                nc[k, 0] = td.c_tile(yr, i, r0, c0, t)
        x = torch.empty((b, p.n, t, t), dtype=torch.float32, device=dev)
        if p.pos16:
            x[:, p.pos16] = h16.to(dev).to(torch.float32)
        if p.pos32:
            x[:, p.pos32] = h32.to(dev)
        if hl4 is not None:
            x[:, p.posl4] = hl4.to(dev).to(torch.float32)
        if nc is None:
            return (x, hy.to(dev).clone(), hm.to(dev).clone())
        c = hc.to(dev).clone()
        if p.c_pos is not None:
            x[:, p.c_pos] = c[:, 0]
        out = (x, hy.to(dev).clone(), hm.to(dev).clone())
        return out + ((c,) if td.residual else ())

class DeviceReader:

    def __init__(self, td, chunk=16):
        import torch
        self.td = td
        self.staging = None
        p, dev = (td.plan, td.device)
        need = {}
        for part, mode in ((td.train_days, 'train'), (td.val_days, 'val')):
            for yr, i, _ in part:
                td.split.guard(yr, i, mode)
                need.setdefault(yr, set()).add(i)
        self.local = {}
        self.x16, self.x32, self.xe, self.tg = ({}, {}, {}, {})
        self.cg = {}
        self.xl4 = {}
        for yr, s in sorted(need.items()):
            idx = sorted(s)
            self.local[yr] = {d: k for k, d in enumerate(idx)}
            ya = td.years[yr]
            parts16, parts32, partse, partst = ([], [], [], [])
            partsl4 = []
            for a in range(0, len(idx), chunk):
                sel = idx[a:a + chunk]
                if p.pos16:
                    parts16.append(torch.from_numpy(np.ascontiguousarray(ya.f16[sel][:, p.idx16])).to(dev))
                if p.pos32:
                    parts32.append(torch.from_numpy(np.ascontiguousarray(ya.f32[sel][:, p.idx32])).to(dev))
                if p.posl4:
                    partsl4.append(torch.from_numpy(np.ascontiguousarray(ya.fl4[sel][:, p.idxl4])).to(dev))
                if td.arm:
                    partse.append(torch.from_numpy(np.stack([np.ascontiguousarray(self._emis_day(yr, d)) for d in sel])).to(dev))
                partst.append(torch.from_numpy(np.ascontiguousarray(ya.tgt[sel])).to(dev))
            self.x16[yr] = torch.cat(parts16) if parts16 else None
            self.x32[yr] = torch.cat(parts32) if parts32 else None
            self.xl4[yr] = torch.cat(partsl4) if partsl4 else None
            self.xe[yr] = torch.cat(partse) if partse else None
            self.tg[yr] = torch.cat(partst)
            if td.climc is not None:
                dates = td.years[yr].dates
                self.cg[yr] = torch.cat([torch.from_numpy(np.stack([td.climc.day_grid(yr, d, dates[d]) for d in idx[a:a + chunk]])).to(dev) for a in range(0, len(idx), chunk)])
                td.climc._grids.clear()
        self.mask = torch.from_numpy(td.loss_mask).to(dev)

    def _emis_day(self, yr, i):
        td = self.td
        k = td.plan.emis_k
        if td.arm == 'F6A':
            return td.f6a.arr[yr][i, k]
        kind, y = arm_source(td.arm, yr, td.train_year, td.test_year)
        if kind == 'true':
            return td.emis.true[y][i, k]
        mo, wd = date_month_weekday(td.years[yr].dates[i])
        return td.emis.clim[y][mo - 1, wd, k]

    def raw(self, draws, mode):
        import torch
        td, p, t = (self.td, self.td.plan, cfg.TILE)
        b = len(draws)
        dev = td.device
        x = torch.empty((b, p.n, t, t), dtype=torch.float32, device=dev)
        y = torch.empty((b, 1, t, t), dtype=torch.float32, device=dev)
        m = torch.empty((b, 1, t, t), dtype=torch.bool, device=dev)
        c = torch.empty((b, 1, t, t), dtype=torch.float32, device=dev) if td.climc is not None else None
        for k, (yr, i, r0, c0) in enumerate(draws):
            td.split.guard(yr, i, mode)
            j = self.local[yr][i]
            if p.pos16:
                x[k, p.pos16] = self.x16[yr][j, :, r0:r0 + t, c0:c0 + t].to(torch.float32)
            if p.pos32:
                x[k, p.pos32] = self.x32[yr][j, :, r0:r0 + t, c0:c0 + t]
            if p.posl4:
                x[k, p.posl4] = self.xl4[yr][j, :, r0:r0 + t, c0:c0 + t].to(torch.float32)
            if td.arm:
                x[k, p.emis_pos] = self.xe[yr][j, :, r0:r0 + t, c0:c0 + t]
            if c is not None:
                c[k, 0] = self.cg[yr][j, r0:r0 + t, c0:c0 + t]
                if p.c_pos is not None:
                    x[k, p.c_pos] = c[k, 0]
            y[k, 0] = self.tg[yr][j, r0:r0 + t, c0:c0 + t]
            m[k, 0] = self.mask[r0:r0 + t, c0:c0 + t]
        if c is not None and td.residual:
            return (x, y, m, c)
        return (x, y, m)
DAY_SET_MODES = {'held_out': 'predict', 'TRAIN-S': 'predict_trains', 'VAL': 'predict_val', 'LOCKBOX': 'predict_lockbox', 'TILE_CHECK': 'predict_trains'}

class PredictData:

    def __init__(self, configs, paths, training, device='cpu', day_set='held_out'):
        from train import plan as plan_mod
        if day_set not in DAY_SET_MODES:
            raise ValueError('day set %r' % (day_set,))
        self.training = training
        self.device = device
        self.day_set = day_set
        self.mode = DAY_SET_MODES[day_set]
        self.split = Split(configs, training)
        net, chset, logs = plan_mod.network_spec(training['net'])
        self.norm = configs.norm(training['norm'])
        check_norm(self.norm, training, self.split)
        self.plan = ChannelPlan(configs, chset, logs, self.norm, emission_order_for(configs, training))
        if day_set == 'held_out':
            yrs = [self.split.held_year]
            self._days = list(self.split.held)
        elif day_set == 'LOCKBOX':
            yrs = [self.split.held_year]
            self._days = list(self.split.lockbox)
        elif day_set == 'TILE_CHECK':
            yrs = self.split.train_years
            self._days = list(self.split.trains[:1])
        else:
            yrs = self.split.train_years
            self._days = list(self.split.trains if day_set == 'TRAIN-S' else self.split.val)
        check_built(configs, self.plan, yrs, training.get('arm'))
        self.yas = {y: YearArrays(configs, paths, y, with_target=False, l4=bool(self.plan.posl4)) for y in yrs}
        self.ya = self.yas[yrs[0]]
        self.mask = load_mask(configs, paths)
        if tuple(self.mask.shape) != self.ya.grid:
            raise cfg.ConfigError('mask grid %s, arrays %s' % (self.mask.shape, self.ya.grid))
        self.cells = mask_cells(self.mask)
        self.mask_sha256 = configs.channels['mask']['sha256']
        self.arm = training.get('arm')
        self.train_year = self.test_year = None
        if self.arm in cfg.F6_ARMS:
            self.train_year = int(training['train_year'])
            self.test_year = self.split.held_year
            check_arm_text(configs, training['split'][1], self.arm, self.train_year, self.test_year, self.norm)
            check_arm_channels(self.plan)
            self.emis = EmisArrays(configs, paths, sorted({self.train_year, self.test_year}))
        elif self.arm == 'F6A':
            check_arm_channels(self.plan)
            self.f6a = F6AArrays(configs, paths, yrs)
        self.climc = None
        self.residual = training.get('target') == 'residual'
        if needs_c(training):
            self.climc = ClimC(configs, paths, training, self.split, self.mask)
            if training.get('c_channel'):
                self.plan = ChannelPlan(configs, chset, logs, self.norm, emission_order_for(configs, training), c_range=(self.climc.c_min, self.climc.c_max))
        self.scaler = Scaler(self.plan, device)

    def to_ppb_of(self, year, idx):
        if not self.residual:
            return self.plan.to_ppb
        date = self.yas[int(year)].dates[int(idx)]
        c64 = np.zeros(self.mask.shape, dtype=np.float64)
        c64.ravel()[self.cells] = self.climc.masked(year, date)
        s_f = float(self.climc.s_f)

        def f(field):
            return c64 + s_f * np.asarray(field, dtype=np.float64)
        return f

    def days(self):
        return list(self._days)

    def raw_day(self, year, idx, scale=None):
        import torch
        self.split.guard(year, idx, self.mode)
        p, ya = (self.plan, self.yas[int(year)])
        h, w = ya.grid
        x = np.empty((p.n, h, w), dtype=np.float32)
        if p.pos16:
            x[p.pos16] = ya.f16[idx, p.idx16]
        if p.pos32:
            x[p.pos32] = ya.f32[idx, p.idx32]
        if p.posl4:
            x[p.posl4] = ya.fl4[idx, p.idxl4]
        if self.arm == 'F6A':
            x[p.emis_pos] = self.f6a.arr[int(year)][idx, p.emis_k]
        elif self.arm:
            kind, y = arm_source(self.arm, year, self.train_year, self.test_year)
            k = p.emis_k
            if kind == 'true':
                x[p.emis_pos] = self.emis.true[y][idx, k]
            else:
                mo, wd = date_month_weekday(ya.dates[idx])
                x[p.emis_pos] = self.emis.clim[y][mo - 1, wd, k]
        if p.c_pos is not None:
            x[p.c_pos] = self.climc.day_grid(year, idx, ya.dates[idx])
        if scale is not None:
            names, factor = scale
            for n in names:
                if n in p.names:
                    j = p.names.index(n)
                    x[j] = x[j] * np.float32(factor)
        return torch.from_numpy(x).to(self.device)

    def scaled_day(self, year, idx, scale=None):
        return self.scaler.inputs(self.raw_day(year, idx, scale)[None])[0]
