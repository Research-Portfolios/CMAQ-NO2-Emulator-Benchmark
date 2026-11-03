"""Paths, fixed values and configuration files of the trainer, extended with the learning-rate period and patience of each training, the climatology tables and the albedo channel of the 18-input network."""
import hashlib
import json
import os
ROOT = '/content/emul'
RUN = os.path.join(ROOT, 'tmp', 'run')
W = os.path.join(ROOT, 'tmp', 'runner', 'work_build', 'work')
LIVE_CONFIG = os.path.join(RUN, 'config', 'p2')
ITEM_B_CONFIG = os.path.join(RUN, 'config')
LIVE_LOGS = os.path.join(RUN, 'logs')
LIVE_MARKER = os.path.join(ROOT, 'tmp', 'account_runs', 'train_launcher.log')
LABEL = 'train'
TILE = 64
MIN_MASKED_FRACTION = 0.25
BATCH = 64
STEPS_MAIN = 6000
STEPS_F6 = 3000
STEPS_MIN_DEPARTURE = 3000
LR = 0.001
LR_GAMMA = 0.93
LR_EVERY = 120
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-08
WEIGHT_DECAY = 0.0
EVAL_EVERY = 250
VAL_TILES = 1024
PATIENCE = 8
CKPT_EVERY = 250
MEM_EVERY = 250
MEM_PAUSE_GB = 1.5
MEM_PAUSE_MAX_S = 3600
MEM_POLL_S = 30
SEED_BASE = 20261016
FOLD_YEARS = (2015, 2016, 2017, 2018, 2019)
WIDTHS = (1.0, 0.5, 0.25)
DEVICES = ('cpu', 'xpu', 'cuda')
F6_PAIRS = ((2017, 2019), (2019, 2017))
F6_ARMS = ('TRUE', 'OTHER', 'OWN')
INFER_BATCH = 32
DISK_FLOOR_GB = 12.0
BLOCK_SIZE = 64
BLOCK_CITY = 'Dallas'
CITY_CELLS = 'CITY_CELLS.json'
TRAIN_STAT_TILES = 1024
STAT_SEED_OFFSET = 500000
E6_PEAK_SHARE = 0.5
E6_MAX_OFFSET = 16
E6_PERCENTILE = 99.9
TRAINS_ISO_REMAINDER = 2
F19_SCALE = 0.9
F19_CELLS = 1000
F19_SEED = 20261016
F19_DAYS_OF_MONTH = (1, 15)
F19_SCALED = ('E-1', 'E-2', 'E-3', 'P-1')
EMISSION_CHANNELS = ('E-1', 'E-2', 'E-3', 'E-4', 'P-1')
LOG_CHANNELS = ('M-9',) + EMISSION_CHANNELS
TEST_VALUE_KEYS = ('batch', 'eval_every', 'ckpt_every', 'mem_every', 'val_tiles', 'patience', 'train_stat_tiles')
L4_FILE = 'S3_L4_ARM18.json'
L4_ARRAY = 'inputs_l4'
L4_SCHEMA = 's3-l4-arm18-1'

class ConfigError(Exception):
    pass

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def rel(path, root=ROOT):
    p = os.path.abspath(path)
    r = os.path.abspath(root)
    if p.lower().startswith(r.lower() + os.sep):
        return p[len(r) + 1:]
    return p

class Paths:

    def __init__(self, config_dir, work_dir, logs_dir, marker, arrays_dir=None, root=ROOT):
        self.root = root
        self.config_dir = os.path.abspath(config_dir)
        self.work_dir = os.path.abspath(work_dir)
        self.logs_dir = os.path.abspath(logs_dir)
        self.marker = os.path.abspath(marker)
        self.arrays_dir = os.path.abspath(arrays_dir) if arrays_dir else None
        self.ckpt_dir = os.path.join(self.work_dir, 'ckpt')
        self.pred_dir = os.path.join(self.work_dir, 'pred')
        self.tmp_dir = os.path.join(self.work_dir, 'tmp')
        self.progress = os.path.join(self.logs_dir, 'TRAIN_PROGRESS_ASCII.txt')
        self.status = os.path.join(self.logs_dir, 'TRAIN_STATUS.json')
        self.train_logs = os.path.join(self.logs_dir, 'train')
        self.lock = os.path.join(self.tmp_dir, 'work.lock')

    def config(self, name):
        return os.path.join(self.config_dir, name)

    def array(self, path_from_root):
        if self.arrays_dir:
            name = str(path_from_root).replace('\\', '/').rsplit('/', 1)[-1]
            return os.path.join(self.arrays_dir, name)
        return os.path.join(self.root, path_from_root)

    def ckpt(self, training):
        return os.path.join(self.ckpt_dir, training)

    def pred(self, training):
        return os.path.join(self.pred_dir, training)

    def is_live(self):
        return os.path.normcase(self.config_dir) == os.path.normcase(os.path.abspath(LIVE_CONFIG)) and os.path.normcase(self.work_dir) == os.path.normcase(os.path.abspath(W)) and (os.path.normcase(self.logs_dir) == os.path.normcase(os.path.abspath(LIVE_LOGS))) and (os.path.normcase(self.marker) == os.path.normcase(os.path.abspath(LIVE_MARKER)))

    def as_dict(self):
        return {'config_dir': rel(self.config_dir, self.root), 'work_dir': rel(self.work_dir, self.root), 'logs_dir': rel(self.logs_dir, self.root), 'marker': rel(self.marker, self.root), 'arrays_dir': rel(self.arrays_dir, self.root) if self.arrays_dir else None}

def live_paths():
    return Paths(LIVE_CONFIG, W, LIVE_LOGS, LIVE_MARKER)

def read_json(path):
    if not os.path.isfile(path):
        raise ConfigError('missing: %s' % rel(path))
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        return json.loads(raw.decode('ascii'))
    except (UnicodeDecodeError, ValueError) as e:
        raise ConfigError('not ASCII JSON: %s (%s)' % (rel(path), type(e).__name__))

class Configs:

    def __init__(self, paths):
        self.paths = paths
        self.sha256 = {}
        self.arrays = self._load('S3_ARRAYS.json')
        self.channels = self._load('S3_CHANNELS.json')
        self.splits = self._load('SPLITS.json')
        self._norms = {}
        self.l4 = self._extend_l4()
        self._check()

    def _load(self, name):
        p = self.paths.config(name)
        d = read_json(p)
        self.sha256[name] = sha256_file(p)
        return d

    def norm(self, name):
        if name not in self._norms:
            fn = 'NORM_%s.json' % name
            if name.startswith('f6a_fold') and (not os.path.isfile(self.paths.config(fn))):
                alt = 'NORM_f6a_%s.json' % name[len('f6a_fold'):]
                if os.path.isfile(self.paths.config(alt)):
                    fn = alt
            self._norms[name] = self._l4_norm(name, fn, self._load(fn))
        return self._norms[name]

    def _extend_l4(self):
        p = self.paths.config(L4_FILE)
        if not os.path.isfile(p):
            return None
        x = self._load(L4_FILE)
        if x.get('schema') != L4_SCHEMA:
            raise ConfigError('%s: schema %r, expected %r' % (L4_FILE, x.get('schema'), L4_SCHEMA))
        ext = x.get('extends') or {}
        for name in ('S3_CHANNELS.json', 'S3_ARRAYS.json'):
            if ext.get(name) != self.sha256[name]:
                raise ConfigError('%s extends another %s (its sha256 differs from the file read)' % (L4_FILE, name))
        ch, after = (dict(x['channel']), x['after'])
        n = ch.get('name')
        c = self.channels
        if ch.get('array') != L4_ARRAY or n in c['order']:
            raise ConfigError('%s: channel %r (array %r) is not a new channel of array %s' % (L4_FILE, n, ch.get('array'), L4_ARRAY))
        for key in ('order', 'arm18_channels'):
            lst = list(c.get(key) or [])
            if after not in lst:
                raise ConfigError('%s: %s lacks %r' % (L4_FILE, key, after))
            lst.insert(lst.index(after) + 1, n)
            c[key] = lst
        chans = list(c['channels'])
        k = [i for i, e in enumerate(chans) if e['name'] == after]
        if len(k) != 1:
            raise ConfigError('%s: channels lack %r' % (L4_FILE, after))
        chans.insert(k[0] + 1, ch)
        c['channels'] = chans
        for y in FOLD_YEARS:
            if str(y) not in (x.get('files') or {}):
                raise ConfigError('%s names no array of %d' % (L4_FILE, y))
        return x

    def _l4_norm(self, name, fn, d):
        x = self.l4
        if not x or name not in (x.get('norm') or {}):
            return d
        e = x['norm'][name]
        if e.get('file') != fn or e.get('sha256') != self.sha256[fn]:
            raise ConfigError('%s: %s is not the NORM file its statistics were made for (name or sha256)' % (L4_FILE, fn))
        if int(d.get('training_days', -1)) != int(e['training_days']):
            raise ConfigError('%s: training_days of %s differ' % (L4_FILE, fn))
        n = x['channel']['name']
        if n in d['channels']:
            raise ConfigError('%s already holds %s' % (fn, n))
        d = dict(d)
        d['channels'] = dict(d['channels'])
        d['channels'][n] = dict(e['stats'])
        return d

    def l4_file(self, year):
        if not self.l4:
            raise ConfigError('the config folder holds no %s' % L4_FILE)
        return self.l4['files'][str(int(year))]

    def city_cells(self):
        p = self.paths.config(CITY_CELLS)
        if not os.path.isfile(p):
            return None
        d = self._load(CITY_CELLS)
        if isinstance(d.get('cells'), dict):
            return d
        if isinstance(d.get('cities'), list):
            cells, blocks = ({}, {})
            for c in d['cities']:
                cells[str(c['city'])] = {'row': int(c['row']), 'col': int(c['col'])}
                if isinstance(c.get('block_masked'), list):
                    blocks[str(c['city'])] = [[int(a), int(b)] for a, b in c['block_masked']]
            return {'cells': cells, 'block_masked': blocks, 'form': "stage (1)'s cities list"}
        raise ConfigError('CITY_CELLS.json: neither cells nor cities')

    def _check(self):
        a, c, s = (self.arrays, self.channels, self.splits)
        for k in ('years', 'other'):
            if k not in a:
                raise ConfigError('S3_ARRAYS.json lacks %r' % k)
        for k in ('order', 'baseline_channels', 'advanced_channels', 'channels', 'target', 'mask'):
            if k not in c:
                raise ConfigError('S3_CHANNELS.json lacks %r' % k)
        if not s.get('all_tests_pass', False):
            raise ConfigError('SPLITS.json: all_tests_pass is not true')
        folds = sorted((int(y) for y in s.get('folds', {})))
        if tuple(folds) != FOLD_YEARS:
            raise ConfigError("SPLITS.json folds %s, the design's %s" % (folds, FOLD_YEARS))
        for tr, te in F6_PAIRS:
            if 'train%d_test%d' % (tr, te) not in s.get('f6', {}):
                raise ConfigError('SPLITS.json lacks f6 train%d_test%d' % (tr, te))
        names = [ch['name'] for ch in c['channels']]
        if names != list(c['order']):
            raise ConfigError('S3_CHANNELS.json: channels and order differ')
        base = list(c['baseline_channels'])
        adv = list(c['advanced_channels'])
        if [n for n in adv if n in base] != base:
            raise ConfigError("baseline channels are not the advanced channels' first part in the table's order")
        for key in ('baseline_channels', 'advanced_channels', 'arm18_channels'):
            lst = list(c.get(key) or [])
            if any((n not in names for n in lst)):
                raise ConfigError('S3_CHANNELS.json %s names a channel not in order' % key)
            if [n for n in names if n in lst] != lst:
                raise ConfigError("S3_CHANNELS.json %s is not in the table's order" % key)
        l4n = self.l4['channel']['name'] if self.l4 else None
        for ch in c['channels']:
            if ch['array'] not in ('inputs_f16', 'inputs_f32') and (not (ch['name'] == l4n and ch['array'] == L4_ARRAY)):
                raise ConfigError('channel %s: array %r' % (ch['name'], ch['array']))
            want = ch['name'] in LOG_CHANNELS
            if bool(ch.get('log_scale_in_advanced')) != want:
                raise ConfigError('channel %s: log_scale_in_advanced %r, the study design gives %r' % (ch['name'], ch.get('log_scale_in_advanced'), want))
        for y in FOLD_YEARS:
            if str(y) not in a['years']:
                raise ConfigError('S3_ARRAYS.json lacks year %d' % y)

    def unfilled(self, year):
        f = self.arrays['years'][str(year)]['files'].get('inputs_f32', {})
        if not f.get('partial'):
            return set()
        return set((str(n) for n in f.get('unfilled_channels', [])))

    def channel_set(self, name):
        key = name + '_channels'
        if key not in self.channels:
            raise ConfigError('S3_CHANNELS.json lacks %r' % key)
        return list(self.channels[key])

    def f6a(self):
        d = self.channels.get('f6a_emission_arrays')
        if isinstance(d, dict) and 'channel_order' in d and ('file_key' in d or 'array' in d):
            order = [str(s).split()[0] for s in d['channel_order']]
            key = str(d.get('file_key') or d['array'])
        elif all(('inputs_c7' in self.arrays['years'][str(y)]['files'] for y in FOLD_YEARS)):
            order = [str(s).split()[0] for s in self.channels.get('f6_emission_arrays', {}).get('channel_order', [])]
            key = 'inputs_c7'
        else:
            raise ConfigError('S3_CHANNELS.json lacks f6a_emission_arrays {channel_order, file_key}, and S3_ARRAYS.json no inputs_c7')
        for y in FOLD_YEARS:
            if key not in self.arrays['years'][str(y)]['files']:
                raise ConfigError('S3_ARRAYS.json years.%d.files lacks %r' % (y, key))
        return (order, key)

def fold_index(year):
    return FOLD_YEARS.index(int(year))
MIN_MASKED_FRACTION_ONE = 1.0 / 4096.0
STEPS_SM_CHOICES = (16000, 12000)
STEPS_DESIGN = 6000
CTABLES_DIR = 'ctables'
CTABLES_SCHEMA = 'ctables-1'
CLIMWD = 'CLIMWD'

def lr_every_for(steps):
    return max(1, int(steps) // 50)

def patience_for(steps, base=PATIENCE):
    return max(1, int(base) * int(steps) // STEPS_DESIGN)

def ctables_index(paths, fold):
    return read_json(paths.config('CTABLES_%s.json' % fold))

def ctable_path(paths, name):
    base = paths.arrays_dir or paths.config_dir
    return os.path.join(base, CTABLES_DIR, str(name).replace('\\', '/').rsplit('/', 1)[-1])
