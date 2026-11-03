"""One of two independently written versions of a screen of the yearly NO2 target arrays for isolated high cell values (above 30 ppb) that stand out from their 5 x 5 neighborhood and from the same cell on nearby days; it checks its inputs against their hashes and writes the candidates and counts."""
import datetime
import hashlib
import os
import re
import shutil
import sys
import time
import traceback
import numpy as np
import numpy.lib.format as npy_format
import psutil
TEXT_SHA256 = '64e92465d289744cff04c8dfaf9ba54b71042650593e7abd96ac6fbf8418746f'
MASK_NAME = 'mask_conus_land.npy'
SUMS_NAME = 'SHA256SUMS_P2.txt'
SA22_SHA256 = {'target_2015.npy': '4d71de966523b306cdd792f9946b7bc0a256a9b040de31aaef9848e7e289fb0d', 'target_2016.npy': 'd2223d1c3761800b22e57bd241c3d1fb89bd2d66df15b1ebd8ac32894fa36d80', 'target_2017.npy': 'd4475144abb3d8d032e9a9b8a6e32a1662b5db20ff0cd2d50caa69edd8e1a120', 'target_2018.npy': 'dee9bdf5cb6b42f23232a4b79ecd1b042c223867b4483104fddd9dbc163301ff', 'target_2019.npy': '96da32426641d23de293df48ec9c359ed6081ad306e4f741473eb4973ad35ab6', MASK_NAME: 'ff3d553eaee09ac62fcddf52ab592055617f6dcb1a292f8fc2f5b23df5e466a1'}
GRID_ROWS = 299
GRID_COLS = 459
NPY_VERSION = (1, 0)
NPY_DATA_OFFSET = 128
TARGET_DESCR = '<f4'
MASK_DESCR = '|b1'
FIXTURE_MIN_SIDE = 5
B300_LEVEL = 300.0
L100_LEVEL = 100.0
L30_LEVEL = 30.0
FACTOR = 5.0
MARGIN = 20.0
MISSING_BOUND = -1e+30
NB_MIN = 3
OWN_HALF = 15
OWN_MIN = 15
MEM_GATE_GB = 5.6
MEM_STEP_S = 60
MEM_WAIT_MAX_S = 540
DISK_ROOT = '/content'
DISK_MIN_GB = 10.0
PEAK_BOUND_GB = 0.6
GIB = 1024 ** 3
COUNTS_NAME = 'COUNTS.csv'
FAIL_NAME = 'FAIL_LIST.csv'
WATCH_NAME = 'WATCH_LIST.csv'
UNTESTED_NAME = 'UNTESTED_LIST.csv'
LOG_NAME = 'RUN_LOG_ASCII.txt'
MANIFEST_NAME = 'MANIFEST_SHA256.txt'
OUT_NAMES = (COUNTS_NAME, FAIL_NAME, WATCH_NAME, UNTESTED_NAME, LOG_NAME, MANIFEST_NAME)
LIST_NAMES = (FAIL_NAME, WATCH_NAME, UNTESTED_NAME)
MANIFEST_ORDER = (COUNTS_NAME, FAIL_NAME, WATCH_NAME, UNTESTED_NAME, LOG_NAME)
LIST_COLS = ('year', 'day_index', 'date', 'row', 'col', 'y_ppb', 'nb_window', 'nb_count', 'nb_median_ppb', 'nb_max_ppb', 'own_count', 'own_median_ppb', 'nbr_made', 'own_made', 'tests_met', 'class')
COUNT_KEYS = ('n_present', 'n_nan', 'n_neginf', 'n_fill', 'n_neg', 'n_l30', 'n_l100', 'n_l100_nbr_made', 'n_l100_nbr_met', 'n_l100_own_made', 'n_l100_own_met', 'n_l30_nbr_made', 'n_l30_nbr_met', 'n_l30_own_made', 'n_l30_own_met', 'n_fail_b', 'n_fail_s', 'n_fail_bs', 'n_fail', 'n_untested', 'n_untested_no_nbr', 'n_untested_no_own', 'n_watch', 'n_watch_not_fail')
COUNT_COLS = ('year', 'month', 'n_days', 'n_mask_celldays') + COUNT_KEYS
POST_YEARS = (2015, 2017)
OTHER_YEARS = (2016, 2018, 2019)
RING1 = tuple(((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)))
RING2 = tuple(((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if max(abs(dr), abs(dc)) == 2))
OFFSETS = RING1 + RING2
PAD = 2
CHUNK = 4096
HASH_BLOCK = 1 << 22

class Stop(Exception):

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason
_USER_RE = re.compile('([\\\\/][Uu][Ss][Ee][Rr][Ss][\\\\/])[^\\\\/\\r\\n]+')
_HEX64 = re.compile('[0-9a-f]{64}')
_SUMS_LINE = re.compile('(\\S+)[ \\t]+\\*?(.*)$')

def _clean(text):
    text = _USER_RE.sub('\\1<user>', str(text))
    return text.encode('ascii', 'backslashreplace').decode('ascii')

def _now():
    t = time.localtime()
    off = t.tm_gmtoff or 0
    sign = '-' if off < 0 else '+'
    off = abs(off)
    return '%s UTC%s%02d:%02d (machine clock)' % (time.strftime('%Y-%m-%d %H:%M:%S', t), sign, off // 3600, off % 3600 // 60)

class RunLog:

    def __init__(self, path):
        self.path = path
        self._fh = open(path, 'xb')

    def line(self, text=''):
        for part in _clean(text).replace('\r', '').split('\n'):
            self._fh.write(part.rstrip(' ').encode('ascii') + b'\n')
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()

class ListWriters:

    def __init__(self, out):
        self.paths = {}
        self._fh = {}
        header = (','.join(LIST_COLS) + '\n').encode('ascii')
        for name in LIST_NAMES:
            path = os.path.join(out, name)
            fh = open(path, 'xb')
            fh.write(header)
            self.paths[name] = path
            self._fh[name] = fh

    def write(self, name, rows):
        if rows:
            self._fh[name].write(('\n'.join(rows) + '\n').encode('ascii'))

    def flush(self):
        for fh in self._fh.values():
            fh.flush()

    def close(self):
        for fh in self._fh.values():
            if not fh.closed:
                fh.close()

def _parse_args(argv):
    opts = {'inputs': None, 'out': None, 'years': None, 'fixture': False}
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--fixture':
            if opts['fixture']:
                raise Stop('--fixture is named twice')
            opts['fixture'] = True
            i += 1
            continue
        key = None
        val = None
        for k in ('inputs', 'out', 'years'):
            flag = '--' + k
            if a == flag:
                if i + 1 >= len(args):
                    raise Stop(flag + ' has no value')
                key, val = (k, args[i + 1])
                i += 2
                break
            if a.startswith(flag + '='):
                key, val = (k, a[len(flag) + 1:])
                i += 1
                break
        if key is None:
            raise Stop('unknown argument ' + repr(a))
        if opts[key] is not None:
            raise Stop('--' + key + ' is named twice')
        opts[key] = val
    for k in ('inputs', 'out', 'years'):
        if not opts[k]:
            raise Stop('--' + k + ' is required')
    return opts

def _out_folder_test(out):
    if os.path.exists(out) and (not os.path.isdir(out)):
        raise Stop('the output path exists and is not a folder')
    held = [n for n in OUT_NAMES if os.path.lexists(os.path.join(out, n))]
    if held:
        raise Stop('the output folder already holds ' + ', '.join(held) + '')

def _parse_years(text):
    years = []
    for tok in text.split(','):
        if not re.fullmatch('[0-9]{4}', tok):
            raise Stop('--years: ' + repr(tok) + ' is not a four-digit year')
        years.append(int(tok))
    for a, b in zip(years, years[1:]):
        if b == a:
            raise Stop('--years names a year twice')
        if b < a:
            raise Stop('--years is not strictly ascending')
    return years

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(HASH_BLOCK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def _read_sums(path):
    with open(path, 'rb') as f:
        data = f.read()
    text = data.decode('utf-8', errors='replace')
    return [ln.rstrip('\r') for ln in text.split('\n')]

def _sums_hits(lines, name):
    hits = []
    for line in lines:
        m = _SUMS_LINE.match(line)
        if m and m.group(2) == name:
            hits.append((m.group(1), line))
    return hits

def _read_header(path):
    with open(path, 'rb') as f:
        version = npy_format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = npy_format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = npy_format.read_array_header_2_0(f)
        else:
            return (version, None, None, None, None)
        offset = f.tell()
    return (version, shape, fortran, dtype, offset)

def _header_text(hdr):
    version, shape, fortran, dtype, offset = hdr
    return 'version %s, descr %s, fortran_order %s, shape %s, offset %s' % ('.'.join((str(v) for v in version)), dtype.str if dtype is not None else 'NA', fortran, tuple(shape) if shape is not None else 'NA', offset)

def _header_problems(hdr, size, want_descr, want_shape):
    version, shape, fortran, dtype, offset = hdr
    probs = []
    if version != NPY_VERSION:
        probs.append('version is not 1.0')
    if dtype is None or dtype.str != want_descr:
        probs.append('dtype is not ' + want_descr)
    if fortran is not False:
        probs.append('fortran_order is not False')
    if shape is None or tuple(shape) != tuple(want_shape):
        probs.append('shape is not %s' % (tuple(want_shape),))
    if offset != NPY_DATA_OFFSET:
        probs.append('data offset is not 128')
    if not probs:
        want_size = NPY_DATA_OFFSET + int(np.prod(shape)) * dtype.itemsize
        if size != want_size:
            probs.append('bytes are not 128 plus cells times item size')
    return probs

def _n_days_of(year):
    return (datetime.date(year + 1, 1, 1) - datetime.date(year, 1, 1)).days

def _check_inputs(inputs, years, fixture, log):
    files = [(MASK_NAME, None)] + [('target_%d.npy' % y, y) for y in years]
    paths = {}
    for name, _ in files:
        path = os.path.abspath(os.path.join(inputs, name))
        paths[name] = path
        if not os.path.isfile(path):
            log.line('input %s: MISSING at %s' % (name, path))
            raise Stop('missing input file ' + name)
    sums_path = os.path.abspath(os.path.join(inputs, SUMS_NAME))
    if not os.path.isfile(sums_path):
        log.line('sums file %s: MISSING at %s' % (SUMS_NAME, sums_path))
        raise Stop('missing input file ' + SUMS_NAME)
    log.line('sums file: %s, bytes %d, sha256 %s' % (sums_path, os.path.getsize(sums_path), _sha256_file(sums_path)))
    sums_lines = _read_sums(sums_path)
    for name, _ in files:
        path = paths[name]
        size = os.path.getsize(path)
        got = _sha256_file(path)
        log.line('input %s: path %s' % (name, path))
        log.line('  bytes %d' % size)
        log.line('  sha256 computed      %s' % got)
        if fixture:
            log.line("sha256 of the screen's list not required (--fixture)")
        else:
            want = SA22_SHA256.get(name)
            log.line("sha256 of the screen's list %s" % (want or 'NONE'))
            if want is None:
                raise Stop("no sha256 in the screen's list for " + name)
            if got != want:
                raise Stop("the sha256 of %s differs from the screen's list" % name)
        hits = _sums_hits(sums_lines, name)
        if not hits:
            log.line('  %s line: NONE' % SUMS_NAME)
            raise Stop('no line for %s in %s' % (name, SUMS_NAME))
        if len(hits) > 1:
            for _, line in hits:
                log.line('  %s line: %s' % (SUMS_NAME, line))
            raise Stop('repeated line for %s in %s' % (name, SUMS_NAME))
        line_hash, line = hits[0]
        log.line('  %s line: %s' % (SUMS_NAME, line))
        line_hash = line_hash.lower()
        if not _HEX64.fullmatch(line_hash) or line_hash != got:
            raise Stop('the sha256 of %s differs from its %s line' % (name, SUMS_NAME))
    mask_path = paths[MASK_NAME]
    hdr = _read_header(mask_path)
    log.line('header %s: %s' % (MASK_NAME, _header_text(hdr)))
    if fixture:
        shape = hdr[1]
        if shape is None or len(shape) != 2 or shape[0] < FIXTURE_MIN_SIDE or (shape[1] < FIXTURE_MIN_SIDE):
            raise Stop("the header of %s differs from the screen's rule (--fixture: a 2-D grid of at least 5 x 5)" % MASK_NAME)
        n_rows, n_cols = (int(shape[0]), int(shape[1]))
    else:
        n_rows, n_cols = (GRID_ROWS, GRID_COLS)
    probs = _header_problems(hdr, os.path.getsize(mask_path), MASK_DESCR, (n_rows, n_cols))
    if probs:
        raise Stop("the header of %s differs from the screen's rule: %s" % (MASK_NAME, '; '.join(probs)))
    year_paths = {}
    for name, year in files[1:]:
        path = paths[name]
        hdr = _read_header(path)
        log.line('header %s: %s' % (name, _header_text(hdr)))
        probs = _header_problems(hdr, os.path.getsize(path), TARGET_DESCR, (_n_days_of(year), n_rows, n_cols))
        if probs:
            raise Stop("the header of %s differs from the screen's rule: %s" % (name, '; '.join(probs)))
        year_paths[year] = path
    log.line('check before use: PASS for %d files' % len(files))
    return (mask_path, year_paths, n_rows, n_cols)

def _load_mask(mask_path, n_rows, n_cols, fixture, log):
    mm = np.load(mask_path, mmap_mode='r', allow_pickle=False)
    raw = np.array(mm.view(np.uint8))
    del mm
    only01 = bool((raw <= 1).all())
    mask = raw != 0
    del raw
    n_true = int(np.count_nonzero(mask))
    log.line('mask True count: %d (of %d cells)' % (n_true, n_rows * n_cols))
    log.line('mask bytes 0 or 1 only: %s' % ('YES' if only01 else 'NO'))
    if not 0 < n_true < n_rows * n_cols:
        log.line('mask count test: FAIL')
        raise Stop('mask test failed: the True count is not strictly between 0 and the grid cells')
    log.line('mask count test: PASS')
    if fixture:
        log.line('corner test: not made (--fixture)')
    else:
        corners = ((0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1))
        bad = [rc for rc in corners if mask[rc]]
        if bad:
            log.line('corner test: FAIL at %s' % (bad,))
            raise Stop('mask test failed: a corner cell is True')
        log.line('corner test: PASS (the four corners are False)')
    return (mask, n_true)

def _free_memory_gb():
    return psutil.virtual_memory().available / GIB

def _free_disk_gb():
    return shutil.disk_usage(DISK_ROOT).free / GIB

def memory_gate(year, fixture, log, read_free=_free_memory_gb, sleep=time.sleep):
    free = read_free()
    if fixture:
        log.line('year %d: free memory before it %.3f GB (--fixture: logged; no wait, no stop)' % (year, free))
        return free
    waited = 0
    while free < MEM_GATE_GB:
        if waited >= MEM_WAIT_MAX_S:
            log.line('year %d: free memory %.3f GB, under 5.6 GB after 9 minutes' % (year, free))
            raise Stop("free memory under 5.6 GB after the gate's 9 minutes")
        log.line('year %d: free memory %.3f GB, under 5.6 GB; waiting 60 s' % (year, free))
        sleep(MEM_STEP_S)
        waited += MEM_STEP_S
        free = read_free()
    log.line('year %d: free memory before it %.3f GB (gate PASS)' % (year, free))
    return free

def disk_gate(year, log, read_disk=_free_disk_gb):
    free = read_disk()
    if free < DISK_MIN_GB:
        log.line('year %d: free space on the work disk %.3f GB, under 10 GB' % (year, free))
        raise Stop('under 10 GB free on the work disk')
    log.line('year %d: free space on the work disk %.3f GB (gate PASS)' % (year, free))
    return free

def _missing_kinds(v):
    isnan = np.isnan(v)
    isneginf = np.isneginf(v)
    with np.errstate(invalid='ignore'):
        isfill = np.isfinite(v) & (v <= MISSING_BOUND)
    return (isnan, isneginf, isfill)

def _median_sorted(srt, k):
    k = np.asarray(k)
    lo = np.maximum((k - 1) // 2, 0)
    hi = np.minimum(np.maximum(k // 2, 0), srt.shape[0] - 1)
    a = np.take_along_axis(srt, lo[np.newaxis, :], axis=0)[0]
    b = np.take_along_axis(srt, hi[np.newaxis, :], axis=0)[0]
    with np.errstate(invalid='ignore', over='ignore'):
        even = (a + b) / 2.0
    return np.where(k % 2 == 1, a, even)

def evaluate_candidates(y, vpad, block, self_row, r, c):
    k = r.size
    yv = y[r, c]
    vals = np.empty((len(OFFSETS), k))
    for j, (dr, dc) in enumerate(OFFSETS):
        vals[j] = vpad[r + PAD + dr, c + PAD + dc]
    ok = ~np.isnan(vals)
    q3 = np.count_nonzero(ok[:8], axis=0)
    q5 = np.count_nonzero(ok, axis=0)
    nb_window = np.where(q3 >= NB_MIN, 3, np.where(q5 >= NB_MIN, 5, 0))
    nb_count = np.where(nb_window == 3, q3, q5)
    med3 = _median_sorted(np.sort(vals[:8], axis=0), q3)
    med5 = _median_sorted(np.sort(vals, axis=0), q5)
    max3 = np.max(np.where(ok[:8], vals[:8], -np.inf), axis=0)
    max5 = np.max(np.where(ok, vals, -np.inf), axis=0)
    nb_med = np.where(nb_window == 3, med3, np.where(nb_window == 5, med5, np.nan))
    nb_max = np.where(nb_window == 3, max3, np.where(nb_window == 5, max5, np.nan))
    nbr_made = nb_window != 0
    with np.errstate(invalid='ignore', over='ignore'):
        nbr_met = nbr_made & (yv > FACTOR * nb_med) & (yv > nb_max + MARGIN)
    own = np.array(block[:, r, c], dtype=np.float64)
    o_nan, o_neginf, o_fill = _missing_kinds(own)
    own[o_nan | o_neginf | o_fill] = np.nan
    own[self_row, :] = np.nan
    own_count = np.count_nonzero(~np.isnan(own), axis=0)
    own_made = own_count >= OWN_MIN
    own_med = np.where(own_made, _median_sorted(np.sort(own, axis=0), own_count), np.nan)
    with np.errstate(invalid='ignore', over='ignore'):
        own_met = own_made & (yv > FACTOR * own_med)
    b300 = yv > B300_LEVEL
    l100 = yv > L100_LEVEL
    fail_b = b300
    fail_s = l100 & nbr_met & own_met
    fail = fail_b | fail_s
    untested = ~fail_b & l100 & (~nbr_made | nbr_met) & (~own_made | own_met) & (~nbr_made | ~own_made)
    watch = nbr_met & own_met
    return {'y': yv, 'nb_window': nb_window, 'nb_count': nb_count, 'nb_med': nb_med, 'nb_max': nb_max, 'own_count': own_count, 'own_med': own_med, 'nbr_made': nbr_made, 'nbr_met': nbr_met, 'own_made': own_made, 'own_met': own_met, 'b300': b300, 'l100': l100, 'fail_b': fail_b, 'fail_s': fail_s, 'fail': fail, 'untested': untested, 'watch': watch}

def _fmt(x):
    return format(float(x), '.6f')

def _tests_met(res, j):
    parts = []
    if res['b300'][j]:
        parts.append('B300')
    if res['l100'][j]:
        parts.append('L100')
    if res['nbr_met'][j]:
        parts.append('NBR')
    if res['own_met'][j]:
        parts.append('OWN')
    parts.append('L30')
    return '+'.join(parts)

def _list_rows(year, d, date_iso, r, c, res):
    fail_rows, untested_rows, watch_rows = ([], [], [])
    listed = np.nonzero(res['fail'] | res['untested'] | res['watch'])[0]
    for j in listed:
        nbw = int(res['nb_window'][j])
        made_own = bool(res['own_made'][j])
        head = ','.join((str(year), str(d), date_iso, str(int(r[j])), str(int(c[j])), _fmt(res['y'][j]), str(nbw), str(int(res['nb_count'][j])), _fmt(res['nb_med'][j]) if nbw != 0 else 'NA', _fmt(res['nb_max'][j]) if nbw != 0 else 'NA', str(int(res['own_count'][j])), _fmt(res['own_med'][j]) if made_own else 'NA', '1' if nbw != 0 else '0', '1' if made_own else '0', _tests_met(res, j)))
        if res['fail'][j]:
            if res['fail_b'][j] and res['fail_s'][j]:
                cls = 'FAIL_BS'
            elif res['fail_b'][j]:
                cls = 'FAIL_B'
            else:
                cls = 'FAIL_S'
            fail_rows.append(head + ',' + cls)
        if res['untested'][j]:
            untested_rows.append(head + ',UNTESTED')
        if res['watch'][j]:
            watch_rows.append(head + ',WATCH')
    return (fail_rows, untested_rows, watch_rows)

def _new_tally():
    tally = {'n_days': 0}
    for key in COUNT_KEYS:
        tally[key] = 0
    return tally

def _add(tally, key, arr):
    tally[key] += int(np.count_nonzero(arr))

def _tally_candidates(tally, res):
    l100 = res['l100']
    _add(tally, 'n_l30', np.ones_like(l100))
    _add(tally, 'n_l100', l100)
    _add(tally, 'n_l100_nbr_made', l100 & res['nbr_made'])
    _add(tally, 'n_l100_nbr_met', l100 & res['nbr_met'])
    _add(tally, 'n_l100_own_made', l100 & res['own_made'])
    _add(tally, 'n_l100_own_met', l100 & res['own_met'])
    _add(tally, 'n_l30_nbr_made', res['nbr_made'])
    _add(tally, 'n_l30_nbr_met', res['nbr_met'])
    _add(tally, 'n_l30_own_made', res['own_made'])
    _add(tally, 'n_l30_own_met', res['own_met'])
    _add(tally, 'n_fail_b', res['fail_b'])
    _add(tally, 'n_fail_s', res['fail_s'])
    _add(tally, 'n_fail_bs', res['fail_b'] & res['fail_s'])
    _add(tally, 'n_fail', res['fail'])
    _add(tally, 'n_untested', res['untested'])
    _add(tally, 'n_untested_no_nbr', res['untested'] & ~res['nbr_made'])
    _add(tally, 'n_untested_no_own', res['untested'] & ~res['own_made'])
    _add(tally, 'n_watch', res['watch'])
    _add(tally, 'n_watch_not_fail', res['watch'] & ~res['fail'])

def evaluate_year(year, path, mask, writers):
    months = {m: _new_tally() for m in range(1, 13)}
    mm = np.load(path, mmap_mode='r', allow_pickle=False)
    if not isinstance(mm, np.memmap) or mm.mode != 'r':
        raise Stop('the year array did not open as a read-only memmap')
    n_days, n_rows, n_cols = mm.shape
    vpad = np.full((n_rows + 2 * PAD, n_cols + 2 * PAD), np.nan)
    inner = vpad[PAD:PAD + n_rows, PAD:PAD + n_cols]
    jan1 = datetime.date(year, 1, 1)
    for d in range(n_days):
        date = jan1 + datetime.timedelta(days=d)
        tally = months[date.month]
        tally['n_days'] += 1
        y = np.array(mm[d], dtype=np.float64)
        isnan, isneginf, isfill = _missing_kinds(y)
        cand = mask & ~(isnan | isneginf | isfill)
        _add(tally, 'n_present', cand)
        _add(tally, 'n_nan', mask & isnan)
        _add(tally, 'n_neginf', mask & isneginf)
        _add(tally, 'n_fill', mask & isfill)
        with np.errstate(invalid='ignore'):
            _add(tally, 'n_neg', cand & (y < 0.0))
            l30 = cand & (y > L30_LEVEL)
        del isnan, isneginf, isfill
        rr, cc = np.nonzero(l30)
        del l30
        if rr.size == 0:
            continue
        inner[...] = np.where(cand, y, np.nan)
        lo = max(0, d - OWN_HALF)
        hi = min(n_days - 1, d + OWN_HALF)
        block = mm[lo:hi + 1]
        date_iso = date.isoformat()
        for s in range(0, rr.size, CHUNK):
            r = rr[s:s + CHUNK]
            c = cc[s:s + CHUNK]
            res = evaluate_candidates(y, vpad, block, d - lo, r, c)
            _tally_candidates(tally, res)
            fail_rows, untested_rows, watch_rows = _list_rows(year, d, date_iso, r, c, res)
            writers.write(FAIL_NAME, fail_rows)
            writers.write(UNTESTED_NAME, untested_rows)
            writers.write(WATCH_NAME, watch_rows)
        writers.flush()
        del block
    del inner, vpad
    del mm
    return months

def _sum_tallies(tallies):
    total = _new_tally()
    for t in tallies:
        for key, val in t.items():
            total[key] += val
    return total

def count_rows(years, by_year, n_true):

    def row(label, month, t):
        return [label, month, t['n_days'], n_true * t['n_days']] + [t[k] for k in COUNT_KEYS]
    rows = []
    for year in years:
        for m in range(1, 13):
            rows.append(row(str(year), '%02d' % m, by_year[year][m]))
    year_tot = {}
    for year in years:
        year_tot[year] = _sum_tallies((by_year[year][m] for m in range(1, 13)))
        rows.append(row(str(year), 'ALL', year_tot[year]))
    for label, members in (('POST', POST_YEARS), ('OTHER', OTHER_YEARS), ('ALL', tuple(years))):
        held = [year_tot[y] for y in years if y in members]
        rows.append(row(label, 'ALL', _sum_tallies(held)))
    return rows

def _count_list_rows(path):
    n = 0
    with open(path, 'rb') as f:
        while True:
            block = f.read(HASH_BLOCK)
            if not block:
                break
            n += block.count(b'\n')
    return n - 1

def check_counts(rows, list_paths, log):
    ix = {name: i for i, name in enumerate(COUNT_COLS)}

    def g(r, name):
        return r[ix[name]]
    results = []
    results.append(('n_present + n_nan + n_neginf + n_fill = n_mask_celldays, every row', all((g(r, 'n_present') + g(r, 'n_nan') + g(r, 'n_neginf') + g(r, 'n_fill') == g(r, 'n_mask_celldays') for r in rows))))
    results.append(('n_fail = n_fail_b + n_fail_s - n_fail_bs, every row', all((g(r, 'n_fail') == g(r, 'n_fail_b') + g(r, 'n_fail_s') - g(r, 'n_fail_bs') for r in rows))))
    results.append(('n_l100_nbr_met <= n_l100_nbr_made <= n_l100 <= n_l30 <= n_present, every row', all((g(r, 'n_l100_nbr_met') <= g(r, 'n_l100_nbr_made') <= g(r, 'n_l100') <= g(r, 'n_l30') <= g(r, 'n_present') for r in rows))))
    all_rows = [r for r in rows if r[0] == 'ALL']
    all_row = all_rows[0] if len(all_rows) == 1 else None
    for name, key in ((FAIL_NAME, 'n_fail'), (UNTESTED_NAME, 'n_untested'), (WATCH_NAME, 'n_watch')):
        ok = all_row is not None and _count_list_rows(list_paths[name]) == g(all_row, key)
        results.append(('the rows of %s number %s of the row year ALL' % (name, key), ok))
    for text, ok in results:
        log.line('check %s: %s' % (text, 'PASS' if ok else 'FAIL'))
    failed = [text for text, ok in results if not ok]
    if failed:
        raise Stop('a check of the screen failed: ' + '; '.join(failed))

def write_counts(path, rows):
    lines = [','.join(COUNT_COLS)]
    for r in rows:
        lines.append(','.join((str(v) for v in r)))
    with open(path, 'xb') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii'))

def write_manifest(out):
    lines = []
    for name in MANIFEST_ORDER:
        lines.append('%s  %s' % (_sha256_file(os.path.join(out, name)), name))
    with open(os.path.join(out, MANIFEST_NAME), 'xb') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii'))

def _peak_wset():
    info = psutil.Process().memory_info()
    return getattr(info, 'peak_wset', None)

def _run(opts, years, out, log):
    fixture = opts['fixture']
    inputs = os.path.abspath(opts['inputs'])
    mask_path, year_paths, n_rows, n_cols = _check_inputs(inputs, years, fixture, log)
    mask, n_true = _load_mask(mask_path, n_rows, n_cols, fixture, log)
    writers = ListWriters(out)
    by_year = {}
    try:
        for year in years:
            memory_gate(year, fixture, log)
            disk_gate(year, log)
            t0 = time.time()
            by_year[year] = evaluate_year(year, year_paths[year], mask, writers)
            peak = _peak_wset()
            log.line('year %d: time taken %.1f s' % (year, time.time() - t0))
            if peak is None:
                log.line('year %d: peak working set after it: NA (no peak_wset on this system)' % year)
            else:
                log.line('year %d: peak working set after it: %d bytes (%.3f GB)' % (year, peak, peak / GIB))
                if peak > PEAK_BOUND_GB * GIB:
                    log.line('The peak working set is above 0.6 GB; not a stop')
            print('no2_screen_a: year %d done' % year, flush=True)
    finally:
        writers.close()
    rows = count_rows(years, by_year, n_true)
    check_counts(rows, writers.paths, log)
    write_counts(os.path.join(out, COUNTS_NAME), rows)
    log.line('COUNTS.csv written: %d rows' % len(rows))

def main(argv):
    try:
        opts = _parse_args(argv)
        out = os.path.abspath(opts['out'])
        _out_folder_test(out)
        years = _parse_years(opts['years'])
    except Stop as e:
        print('STOPPED: ' + _clean(e.reason) + '; nothing written', flush=True)
        return 1
    try:
        os.makedirs(out, exist_ok=True)
        log = RunLog(os.path.join(out, LOG_NAME))
    except OSError as e:
        print('STOPPED: the run folder or its log could not be made (%s)' % type(e).__name__, flush=True)
        return 1
    try:
        log.line('NO2 SCREEN, CODE A: RUN LOG')
        code_path = os.path.abspath(__file__)
        log.line('code: %s' % code_path)
        log.line('code sha256: %s' % _sha256_file(code_path))
        log.line('text sha256 written from: %s' % TEXT_SHA256)
        prog = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ''
        if prog and os.path.isfile(prog):
            log.line('program run (sys.argv[0], the runner): %s, sha256 %s' % (prog, _sha256_file(prog)))
        log.line('command line: %s' % ' '.join(getattr(sys, 'orig_argv', sys.argv)))
        log.line('main argv: %s' % ' '.join(argv))
        log.line('mode: %s' % ('--fixture' if opts['fixture'] else 'real'))
        log.line('years: %s' % ','.join((str(y) for y in years)))
        log.line('start: %s' % _now())
        log.line('python %s; numpy %s; psutil %s' % (sys.version.split()[0], np.__version__, psutil.__version__))
        _run(opts, years, out, log)
    except Stop as e:
        log.line('end: %s' % _now())
        log.line('STATUS: STOPPED: %s' % e.reason)
        log.close()
        print('STOPPED: ' + _clean(e.reason), flush=True)
        return 1
    except (Exception, KeyboardInterrupt) as e:
        log.line('traceback:')
        log.line(traceback.format_exc())
        log.line('end: %s' % _now())
        log.line('STATUS: STOPPED: exception %s' % type(e).__name__)
        log.close()
        print('STOPPED: exception %s (traceback in the run log)' % type(e).__name__, flush=True)
        return 1
    log.line('end: %s' % _now())
    log.line('STATUS: PASS')
    log.close()
    try:
        write_manifest(out)
    except Exception as e:
        print('STOPPED: MANIFEST_SHA256.txt could not be written (%s)' % type(e).__name__, flush=True)
        return 1
    print('no2_screen_a: STATUS: PASS', flush=True)
    return 0
if __name__ == '__main__':
    print('STOPPED: this code runs only through its runner code_a\\wait_screen_a.py')
    sys.exit(2)
