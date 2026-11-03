"""The second, independently written version of the same screen of the NO2 target, which classifies the cells one at a time by explicit loops so that its results can be compared with the first version's."""
import calendar
import datetime
import hashlib
import os
import re
import shutil
import sys
import time
import traceback
import numpy as np
import numpy.lib.format as npformat
import psutil
TEXT_SHA256 = '64e92465d289744cff04c8dfaf9ba54b71042650593e7abd96ac6fbf8418746f'
EXPECTED_SHA256 = {'target_2015.npy': '4d71de966523b306cdd792f9946b7bc0a256a9b040de31aaef9848e7e289fb0d', 'target_2016.npy': 'd2223d1c3761800b22e57bd241c3d1fb89bd2d66df15b1ebd8ac32894fa36d80', 'target_2017.npy': 'd4475144abb3d8d032e9a9b8a6e32a1662b5db20ff0cd2d50caa69edd8e1a120', 'target_2018.npy': 'dee9bdf5cb6b42f23232a4b79ecd1b042c223867b4483104fddd9dbc163301ff', 'target_2019.npy': '96da32426641d23de293df48ec9c359ed6081ad306e4f741473eb4973ad35ab6', 'mask_conus_land.npy': 'ff3d553eaee09ac62fcddf52ab592055617f6dcb1a292f8fc2f5b23df5e466a1'}
MASK_NAME = 'mask_conus_land.npy'
SUMS_NAME = 'SHA256SUMS_P2.txt'
GRID_ROWS = 299
GRID_COLS = 459
NPY_VERSION = (1, 0)
NPY_OFFSET = 128
TARGET_DTYPE = '<f4'
MASK_DTYPE = '|b1'
FIXTURE_MIN_SIDE = 5
B300_LEVEL = 300.0
L100_LEVEL = 100.0
L30_LEVEL = 30.0
FACTOR = 5.0
MARGIN = 20.0
FILL_BOUND = -1e+30
NEG_INF = float('-inf')
NB_MIN = 3
OWN_HALF = 15
OWN_MIN = 15
MEM_GATE_GB = 5.6
MEM_STEP_S = 60
MEM_MAX_WAIT_S = 9 * 60
DISK_ROOT = '/content'
DISK_MIN_GB = 10.0
PEAK_BOUND_GB = 0.6
POST_YEARS = (2015, 2017)
OTHER_YEARS = (2016, 2018, 2019)
OUTPUT_FILES = ('COUNTS.csv', 'FAIL_LIST.csv', 'WATCH_LIST.csv', 'UNTESTED_LIST.csv', 'RUN_LOG_ASCII.txt', 'MANIFEST_SHA256.txt')
MANIFEST_ORDER = ('COUNTS.csv', 'FAIL_LIST.csv', 'WATCH_LIST.csv', 'UNTESTED_LIST.csv', 'RUN_LOG_ASCII.txt')
LIST_HEADER = ('year', 'day_index', 'date', 'row', 'col', 'y_ppb', 'nb_window', 'nb_count', 'nb_median_ppb', 'nb_max_ppb', 'own_count', 'own_median_ppb', 'nbr_made', 'own_made', 'tests_met', 'class')
COUNT_FIELDS = ('n_days', 'n_mask_celldays', 'n_present', 'n_nan', 'n_neginf', 'n_fill', 'n_neg', 'n_l30', 'n_l100', 'n_l100_nbr_made', 'n_l100_nbr_met', 'n_l100_own_made', 'n_l100_own_met', 'n_l30_nbr_made', 'n_l30_nbr_met', 'n_l30_own_made', 'n_l30_own_met', 'n_fail_b', 'n_fail_s', 'n_fail_bs', 'n_fail', 'n_untested', 'n_untested_no_nbr', 'n_untested_no_own', 'n_watch', 'n_watch_not_fail')
COUNTS_HEADER = ('year', 'month') + COUNT_FIELDS

class EarlyStop(Exception):
    pass

class Stop(Exception):
    pass

def _free_mem_gb():
    return psutil.virtual_memory().available / 1024 ** 3

def _free_disk_gb():
    return shutil.disk_usage(DISK_ROOT).free / 1024 ** 3

def _peak_wset_bytes():
    info = psutil.Process().memory_info()
    return getattr(info, 'peak_wset', None)
_sleep = time.sleep
_USERS_RE = re.compile('([\\\\/]+[Uu][Ss][Ee][Rr][Ss][\\\\/]+)[^\\\\/\\r\\n\'\\"]+')

def _clean(text):
    text = _USERS_RE.sub('\\1<user>', str(text))
    return text.encode('ascii', 'backslashreplace').decode('ascii')

def _say(text):
    sys.stdout.write(_clean(text) + '\n')
    sys.stdout.flush()

def _now_text():
    now = datetime.datetime.now().astimezone()
    off = now.utcoffset()
    total = int(off.total_seconds()) if off is not None else 0
    sign = '+' if total >= 0 else '-'
    total = abs(total)
    offset = 'UTC%s%02d:%02d' % (sign, total // 3600, total % 3600 // 60)
    zone = time.localtime().tm_zone or ''
    if ' ' in zone:
        zone = ''.join((word[0] for word in zone.split() if word))
    return '%s %s (%s, machine clock)' % (now.strftime('%Y-%m-%d %H:%M:%S'), zone, offset)

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def _fmt(x):
    return format(x, '.6f')

def _is_present(v):
    if v != v:
        return False
    if v == NEG_INF:
        return False
    if v <= FILL_BOUND:
        return False
    return True

def _median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    a = s[n // 2 - 1]
    b = s[n // 2]
    return (a + b) / 2.0

def _maximum(values):
    best = values[0]
    for v in values[1:]:
        if v > best:
            best = v
    return best

class _RunLog:

    def __init__(self, path):
        self._path = path
        self._fh = open(path, 'x', encoding='ascii', newline='\n')

    def put(self, text=''):
        if self._fh.closed:
            self._fh = open(self._path, 'a', encoding='ascii', newline='\n')
        for line in str(text).split('\n'):
            self._fh.write(_clean(line.rstrip('\r').rstrip()) + '\n')
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()

def _parse_argv(argv):
    opts = {'inputs': None, 'out': None, 'years': None, 'fixture': False}
    seen = set()
    k = 0
    argv = list(argv)
    while k < len(argv):
        tok = argv[k]
        if tok in ('--inputs', '--out', '--years'):
            if tok in seen:
                raise EarlyStop('argument named twice: ' + tok)
            if k + 1 >= len(argv):
                raise EarlyStop('argument without its value: ' + tok)
            seen.add(tok)
            opts[tok[2:]] = argv[k + 1]
            k += 2
        elif tok == '--fixture':
            if tok in seen:
                raise EarlyStop('argument named twice: --fixture')
            seen.add(tok)
            opts['fixture'] = True
            k += 1
        else:
            raise EarlyStop('unknown argument: ' + tok)
    for name in ('inputs', 'out', 'years'):
        if opts[name] is None:
            raise EarlyStop('missing argument: --' + name)
    return opts

def _parse_years(text):
    tokens = text.split(',')
    years = []
    for tok in tokens:
        if re.fullmatch('[0-9]{4}', tok) is None:
            raise EarlyStop('--years is not a comma-separated list of four-digit years')
        years.append(int(tok))
    if len(set(years)) != len(years):
        raise EarlyStop('--years names a year twice')
    for k in range(1, len(years)):
        if not years[k] > years[k - 1]:
            raise EarlyStop('--years is not strictly ascending')
    return years

def _test_out_folder(out_dir):
    if os.path.exists(out_dir) and (not os.path.isdir(out_dir)):
        raise EarlyStop('--out names a path that is not a folder')
    if os.path.isdir(out_dir):
        for name in OUTPUT_FILES:
            if os.path.exists(os.path.join(out_dir, name)):
                raise EarlyStop('an output file is already in the run folder: ' + name)

def _n_days(year):
    return 366 if calendar.isleap(year) else 365

def _read_sums(folder):
    path = os.path.join(folder, SUMS_NAME)
    if not os.path.isfile(path):
        raise Stop('missing input file: ' + SUMS_NAME)
    with open(path, 'rb') as fh:
        raw = fh.read()
    text = raw.decode('ascii', 'replace')
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    entries = []
    for line in lines:
        if line.endswith('\r'):
            line = line[:-1]
        if len(line) > 66 and line[64:66] == '  ':
            entries.append((line[66:], line[:64], line))
        else:
            entries.append((None, None, line))
    return (path, entries)

def _sums_line_for(entries, name):
    hits = [e for e in entries if e[0] == name]
    if not hits:
        raise Stop('missing line in ' + SUMS_NAME + ' for ' + name)
    if len(hits) > 1:
        raise Stop('repeated line in ' + SUMS_NAME + ' for ' + name)
    return hits[0]

def _check_sha(name, computed, sums_entry, fixture):
    if re.fullmatch('[0-9a-f]{64}', computed) is None:
        raise Stop('sha256 of ' + name + ' not 64 lowercase hex characters')
    if not fixture:
        want = EXPECTED_SHA256.get(name)
        if want is None or computed != want:
            raise Stop('sha256 of ' + name + "differs from the screen's list")
    field = sums_entry[1].lower()
    if re.fullmatch('[0-9a-f]{64}', field) is None or computed != field:
        raise Stop('sha256 of ' + name + ' differs from ' + SUMS_NAME)

def _read_header(path, name):
    try:
        with open(path, 'rb') as fh:
            version = tuple(npformat.read_magic(fh))
            if version == (1, 0):
                shape, fortran, dtype = npformat.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran, dtype = npformat.read_array_header_2_0(fh)
            else:
                return {'version': version, 'shape': None, 'fortran': None, 'dtype': None, 'itemsize': None, 'offset': None}
            offset = fh.tell()
    except ValueError:
        raise Stop('the header of ' + name + ' could not be read as .npy')
    return {'version': version, 'shape': tuple((int(x) for x in shape)), 'fortran': bool(fortran), 'dtype': dtype.str, 'itemsize': int(dtype.itemsize), 'offset': int(offset)}

def _check_header(name, hdr, want_shape, want_dtype, size):
    bad = []
    if hdr['version'] != NPY_VERSION:
        bad.append('version')
    if hdr['dtype'] != want_dtype:
        bad.append('dtype')
    if hdr['fortran'] is not False:
        bad.append('order')
    if hdr['shape'] != want_shape:
        bad.append('shape')
    if hdr['offset'] != NPY_OFFSET:
        bad.append('data offset')
    if not bad:
        cells = 1
        for n in hdr['shape']:
            cells *= n
        if size != NPY_OFFSET + cells * hdr['itemsize']:
            bad.append('file bytes')
    if bad:
        raise Stop('the header of ' + name + "differs from the screen's rule: " + ', '.join(bad))

def _header_text(hdr):
    return 'version %s, dtype %s, fortran_order %s, shape %s, data offset %s' % (hdr['version'], hdr['dtype'], hdr['fortran'], hdr['shape'], hdr['offset'])

def _open_memmap(path, name):
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    if not isinstance(arr, np.memmap) or arr.mode != 'r':
        raise Stop(name + ' was not opened as a read-only memory map')
    return arr

def _mask_flags(arr):
    return (np.asarray(arr).view(np.uint8) != 0).tolist()

def _mask_test(flags, rows, cols, fixture):
    true_count = 0
    for r in range(rows):
        row = flags[r]
        for c in range(cols):
            if row[c]:
                true_count += 1
    count_ok = 0 < true_count < rows * cols
    if fixture:
        corner_ok = None
    else:
        corner_ok = not flags[0][0] and (not flags[0][cols - 1]) and (not flags[rows - 1][0]) and (not flags[rows - 1][cols - 1])
    return (true_count, count_ok, corner_ok)

def _new_counts():
    return dict(((f, 0) for f in COUNT_FIELDS))

def _evaluate(arr, grid, flags, rows, cols, n_days, d, r, c, y):
    q3 = []
    q5 = []
    for dr in (-2, -1, 0, 1, 2):
        r2 = r + dr
        if r2 < 0 or r2 > rows - 1:
            continue
        flag_row = flags[r2]
        grid_row = grid[r2]
        for dc in (-2, -1, 0, 1, 2):
            c2 = c + dc
            if c2 < 0 or c2 > cols - 1:
                continue
            if dr == 0 and dc == 0:
                continue
            if not flag_row[c2]:
                continue
            v = grid_row[c2]
            if not _is_present(v):
                continue
            q5.append(v)
            if -1 <= dr <= 1 and -1 <= dc <= 1:
                q3.append(v)
    if len(q3) >= NB_MIN:
        nb_values, nb_window, nb_count = (q3, 3, len(q3))
    elif len(q5) >= NB_MIN:
        nb_values, nb_window, nb_count = (q5, 5, len(q5))
    else:
        nb_values, nb_window, nb_count = ([], 0, len(q5))
    nbr_made = nb_window != 0
    if nbr_made:
        nb_med = _median(nb_values)
        nb_max = _maximum(nb_values)
        nbr_met = y > FACTOR * nb_med and y > nb_max + MARGIN
    else:
        nb_med = None
        nb_max = None
        nbr_met = False
    lo = max(0, d - OWN_HALF)
    hi = min(n_days - 1, d + OWN_HALF)
    series = arr[lo:hi + 1, r, c].tolist()
    own_values = []
    for k in range(lo, hi + 1):
        if k == d:
            continue
        v = series[k - lo]
        if _is_present(v):
            own_values.append(v)
    own_count = len(own_values)
    own_made = own_count >= OWN_MIN
    if own_made:
        own_med = _median(own_values)
        own_met = y > FACTOR * own_med
    else:
        own_med = None
        own_met = False
    b300 = y > B300_LEVEL
    l100 = y > L100_LEVEL
    l30 = y > L30_LEVEL
    return {'nb_window': nb_window, 'nb_count': nb_count, 'nb_med': nb_med, 'nb_max': nb_max, 'own_count': own_count, 'own_med': own_med, 'nbr_made': nbr_made, 'nbr_met': nbr_met, 'own_made': own_made, 'own_met': own_met, 'b300': b300, 'l100': l100, 'l30': l30}

def _classify(ev):
    fail_b = ev['b300']
    fail_s = ev['l100'] and ev['nbr_made'] and ev['nbr_met'] and ev['own_made'] and ev['own_met']
    if fail_b and fail_s:
        fail_class = 'FAIL_BS'
    elif fail_b:
        fail_class = 'FAIL_B'
    elif fail_s:
        fail_class = 'FAIL_S'
    else:
        fail_class = None
    nbr_ok = ev['nbr_met'] if ev['nbr_made'] else True
    own_ok = ev['own_met'] if ev['own_made'] else True
    untested = not fail_b and ev['l100'] and nbr_ok and own_ok and (not ev['nbr_made'] or not ev['own_made'])
    watch = ev['l30'] and ev['nbr_made'] and ev['nbr_met'] and ev['own_made'] and ev['own_met']
    names = []
    if ev['b300']:
        names.append('B300')
    if ev['l100']:
        names.append('L100')
    if ev['nbr_made'] and ev['nbr_met']:
        names.append('NBR')
    if ev['own_made'] and ev['own_met']:
        names.append('OWN')
    if ev['l30']:
        names.append('L30')
    return {'fail_b': fail_b, 'fail_s': fail_s, 'fail_class': fail_class, 'untested': untested, 'watch': watch, 'tests_met': '+'.join(names)}

def _row_text(year, d, date_text, r, c, y, ev, tests_met, cls):
    nb_on = ev['nb_window'] != 0
    own_on = ev['own_count'] >= OWN_MIN
    fields = [str(year), str(d), date_text, str(r), str(c), _fmt(y), str(ev['nb_window']), str(ev['nb_count']), _fmt(ev['nb_med']) if nb_on else 'NA', _fmt(ev['nb_max']) if nb_on else 'NA', str(ev['own_count']), _fmt(ev['own_med']) if own_on else 'NA', '1' if ev['nbr_made'] else '0', '1' if ev['own_made'] else '0', tests_met, cls]
    return ','.join(fields)

def screen_year(arr, year, rows, cols, flags, cells_by_row, emit):
    n_days = _n_days(year)
    jan1 = datetime.date(year, 1, 1)
    months = dict(((m, _new_counts()) for m in range(1, 13)))
    neg_inf = NEG_INF
    fill_bound = FILL_BOUND
    l30_level = L30_LEVEL
    for d in range(n_days):
        day = jan1 + datetime.timedelta(days=d)
        date_text = day.isoformat()
        cnt = months[day.month]
        cnt['n_days'] += 1
        grid = arr[d].tolist()
        n_present = n_nan = n_neginf = n_fill = n_neg = 0
        candidates = []
        for r in range(rows):
            grid_row = grid[r]
            for c in cells_by_row[r]:
                v = grid_row[c]
                if v != v:
                    n_nan += 1
                elif v == neg_inf:
                    n_neginf += 1
                elif v <= fill_bound:
                    n_fill += 1
                else:
                    n_present += 1
                    if v < 0.0:
                        n_neg += 1
                    if v > l30_level:
                        candidates.append((r, c, v))
        cnt['n_present'] += n_present
        cnt['n_nan'] += n_nan
        cnt['n_neginf'] += n_neginf
        cnt['n_fill'] += n_fill
        cnt['n_neg'] += n_neg
        fail_rows = []
        watch_rows = []
        untested_rows = []
        for r, c, y in candidates:
            ev = _evaluate(arr, grid, flags, rows, cols, n_days, d, r, c, y)
            cl = _classify(ev)
            cnt['n_l30'] += 1
            if ev['nbr_made']:
                cnt['n_l30_nbr_made'] += 1
                if ev['nbr_met']:
                    cnt['n_l30_nbr_met'] += 1
            if ev['own_made']:
                cnt['n_l30_own_made'] += 1
                if ev['own_met']:
                    cnt['n_l30_own_met'] += 1
            if ev['l100']:
                cnt['n_l100'] += 1
                if ev['nbr_made']:
                    cnt['n_l100_nbr_made'] += 1
                    if ev['nbr_met']:
                        cnt['n_l100_nbr_met'] += 1
                if ev['own_made']:
                    cnt['n_l100_own_made'] += 1
                    if ev['own_met']:
                        cnt['n_l100_own_met'] += 1
            if cl['fail_b']:
                cnt['n_fail_b'] += 1
            if cl['fail_s']:
                cnt['n_fail_s'] += 1
            if cl['fail_b'] and cl['fail_s']:
                cnt['n_fail_bs'] += 1
            if cl['fail_class'] is not None:
                cnt['n_fail'] += 1
                fail_rows.append((r, c, _row_text(year, d, date_text, r, c, y, ev, cl['tests_met'], cl['fail_class'])))
            if cl['untested']:
                cnt['n_untested'] += 1
                if not ev['nbr_made']:
                    cnt['n_untested_no_nbr'] += 1
                if not ev['own_made']:
                    cnt['n_untested_no_own'] += 1
                untested_rows.append((r, c, _row_text(year, d, date_text, r, c, y, ev, cl['tests_met'], 'UNTESTED')))
            if cl['watch']:
                cnt['n_watch'] += 1
                if cl['fail_class'] is None:
                    cnt['n_watch_not_fail'] += 1
                watch_rows.append((r, c, _row_text(year, d, date_text, r, c, y, ev, cl['tests_met'], 'WATCH')))
        fail_rows.sort(key=lambda t: (t[0], t[1]))
        watch_rows.sort(key=lambda t: (t[0], t[1]))
        untested_rows.sort(key=lambda t: (t[0], t[1]))
        emit([t[2] for t in fail_rows], [t[2] for t in watch_rows], [t[2] for t in untested_rows])
        del grid
    return months

def _sum_counts(parts):
    total = _new_counts()
    for part in parts:
        for f in COUNT_FIELDS:
            total[f] += part[f]
    return total

def _count_rows(years, by_year, true_count):
    out = []
    year_all = {}
    for year in years:
        for m in range(1, 13):
            cnt = by_year[year][m]
            cnt['n_mask_celldays'] = true_count * cnt['n_days']
            out.append((str(year), '%02d' % m, cnt))
    for year in years:
        year_all[year] = _sum_counts([by_year[year][m] for m in range(1, 13)])
        out.append((str(year), 'ALL', year_all[year]))
    out.append(('POST', 'ALL', _sum_counts([year_all[y] for y in years if y in POST_YEARS])))
    out.append(('OTHER', 'ALL', _sum_counts([year_all[y] for y in years if y in OTHER_YEARS])))
    out.append(('ALL', 'ALL', _sum_counts([year_all[y] for y in years])))
    return out

def _list_rows_in(path):
    n = 0
    with open(path, 'rb') as fh:
        for _line in fh:
            n += 1
    return n - 1

def _identity_checks(rows_out, out_dir):
    ok1 = ok2 = ok3 = True
    for _y, _m, k in rows_out:
        if k['n_present'] + k['n_nan'] + k['n_neginf'] + k['n_fill'] != k['n_mask_celldays']:
            ok1 = False
        if k['n_fail'] != k['n_fail_b'] + k['n_fail_s'] - k['n_fail_bs']:
            ok2 = False
        if not k['n_l100_nbr_met'] <= k['n_l100_nbr_made'] <= k['n_l100'] <= k['n_l30'] <= k['n_present']:
            ok3 = False
    all_row = rows_out[-1][2]
    ok4 = _list_rows_in(os.path.join(out_dir, 'FAIL_LIST.csv')) == all_row['n_fail']
    ok5 = _list_rows_in(os.path.join(out_dir, 'UNTESTED_LIST.csv')) == all_row['n_untested']
    ok6 = _list_rows_in(os.path.join(out_dir, 'WATCH_LIST.csv')) == all_row['n_watch']
    return [('n_present + n_nan + n_neginf + n_fill = n_mask_celldays, every row', ok1), ('n_fail = n_fail_b + n_fail_s - n_fail_bs, every row', ok2), ('n_l100_nbr_met <= n_l100_nbr_made <= n_l100 <= n_l30 <= n_present, every row', ok3), ('rows of FAIL_LIST.csv = n_fail of row year ALL', ok4), ('rows of UNTESTED_LIST.csv = n_untested of row year ALL', ok5), ('rows of WATCH_LIST.csv = n_watch of row year ALL', ok6)]

def _gate(log, fixture):
    free = _free_mem_gb()
    log.put('  free memory before the year: %.2f GB' % free)
    if not fixture:
        waited = 0
        while free < MEM_GATE_GB:
            if waited >= MEM_MAX_WAIT_S:
                raise Stop("free memory under 5.6 GB after the gate's 9 minutes")
            _sleep(MEM_STEP_S)
            waited += MEM_STEP_S
            free = _free_mem_gb()
            log.put('  free memory after %d s of waiting: %.2f GB' % (waited, free))
    else:
        log.put('  (--fixture: the memory gate logs only)')
    disk = _free_disk_gb()
    log.put('free space on the work disk before the year: %.2f GB' % disk)
    if disk < DISK_MIN_GB:
        raise Stop('under 10 GB free on the work disk')

def _run(opts, years, out_dir, log):
    fixture = opts['fixture']
    inputs = opts['inputs']
    code_path = os.path.abspath(__file__)
    log.put('RUN LOG OF THE NO2 SCREEN, CODE B')
    log.put('code: ' + code_path)
    log.put('code sha256: ' + _sha256_file(code_path))
    log.put('text written from: sha256 ' + TEXT_SHA256)
    log.put('command line: ' + ' '.join([sys.executable] + (['-P'] if getattr(sys.flags, 'safe_path', False) else []) + (['-B'] if sys.flags.dont_write_bytecode else []) + list(sys.argv[:1]) + list(opts['argv'])))
    log.put('start: ' + _now_text())
    log.put('python %s, numpy %s, psutil %s' % (sys.version.split()[0], np.__version__, psutil.__version__))
    log.put('mode: ' + ('--fixture' if fixture else 'real'))
    names = [MASK_NAME] + ['target_%d.npy' % y for y in years]
    paths = dict(((n, os.path.join(inputs, n)) for n in names))
    for n in names:
        if not os.path.isfile(paths[n]):
            raise Stop('missing input file: ' + n)
    sums_path, entries = _read_sums(inputs)
    log.put('sums file: ' + sums_path)
    log.put('INPUT FILES')
    for n in names:
        size = os.path.getsize(paths[n])
        log.put('file: ' + paths[n])
        log.put('  bytes: %d' % size)
        computed = _sha256_file(paths[n])
        log.put('  sha256 computed: ' + computed)
        if fixture:
            log.put("sha256 of the screen's list: not required (--fixture)")
        else:
            log.put("sha256 of the screen's list: " + EXPECTED_SHA256.get(n, '(none named)'))
        named = [e for e in entries if e[0] == n]
        if not named:
            log.put('  sums line read: (none)')
        for e in named:
            log.put('  sums line read: ' + e[2])
        hit = _sums_line_for(entries, n)
        _check_sha(n, computed, hit, fixture)
        log.put('  sha256 check: PASS')
    mask_hdr = _read_header(paths[MASK_NAME], MASK_NAME)
    log.put('header of ' + MASK_NAME + ': ' + _header_text(mask_hdr))
    if fixture:
        shp = mask_hdr['shape']
        if shp is None or len(shp) != 2 or shp[0] < FIXTURE_MIN_SIDE or (shp[1] < FIXTURE_MIN_SIDE):
            raise Stop('the header of ' + MASK_NAME + "differs from the screen's rule: shape (--fixture: two axes, each at least 5)")
        rows, cols = shp
    else:
        rows, cols = (GRID_ROWS, GRID_COLS)
    _check_header(MASK_NAME, mask_hdr, (rows, cols), MASK_DTYPE, os.path.getsize(paths[MASK_NAME]))
    log.put('  header check: PASS')
    for y in years:
        n = 'target_%d.npy' % y
        hdr = _read_header(paths[n], n)
        log.put('header of ' + n + ': ' + _header_text(hdr))
        _check_header(n, hdr, (_n_days(y), rows, cols), TARGET_DTYPE, os.path.getsize(paths[n]))
        log.put('  header check: PASS')
    log.put('grid: %d rows x %d columns' % (rows, cols))
    mask_arr = _open_memmap(paths[MASK_NAME], MASK_NAME)
    flags = _mask_flags(mask_arr)
    del mask_arr
    true_count, count_ok, corner_ok = _mask_test(flags, rows, cols, fixture)
    log.put('MASK')
    log.put('  True count: %d' % true_count)
    log.put('  count test (0 < count < %d): %s' % (rows * cols, 'PASS' if count_ok else 'FAIL'))
    if corner_ok is None:
        log.put('  corner test: not made (--fixture)')
    else:
        log.put('  corner test: ' + ('PASS' if corner_ok else 'FAIL'))
    if not count_ok or corner_ok is False:
        raise Stop('the mask test failed')
    cells_by_row = []
    for r in range(rows):
        cells_by_row.append([c for c in range(cols) if flags[r][c]])
    handles = {}
    for n in ('FAIL_LIST.csv', 'WATCH_LIST.csv', 'UNTESTED_LIST.csv'):
        handles[n] = open(os.path.join(out_dir, n), 'x', encoding='ascii', newline='\n')
        handles[n].write(','.join(LIST_HEADER) + '\n')
        handles[n].flush()

    def emit(fail_rows, watch_rows, untested_rows):
        for n, rows_ in (('FAIL_LIST.csv', fail_rows), ('WATCH_LIST.csv', watch_rows), ('UNTESTED_LIST.csv', untested_rows)):
            if rows_:
                handles[n].write(''.join((t + '\n' for t in rows_)))
                handles[n].flush()
    by_year = {}
    try:
        for y in years:
            log.put('YEAR %d' % y)
            _gate(log, fixture)
            t0 = time.time()
            arr = _open_memmap(paths['target_%d.npy' % y], 'target_%d.npy' % y)
            if tuple(arr.shape) != (_n_days(y), rows, cols) or arr.dtype.str != TARGET_DTYPE:
                raise Stop('target_%d.npy opened with another shape or dtype' % y)
            by_year[y] = screen_year(arr, y, rows, cols, flags, cells_by_row, emit)
            del arr
            log.put('  time taken: %.1f s' % (time.time() - t0))
            peak = _peak_wset_bytes()
            if peak is None:
                log.put('  peak working set after the year: not available')
            else:
                log.put('  peak working set after the year: %d bytes (%.3f GB)' % (peak, peak / 1024 ** 3))
                if peak / 1024 ** 3 > PEAK_BOUND_GB:
                    log.put('  The peak working set is above the bound of 0.6 GB; not a stop')
    finally:
        for fh in handles.values():
            fh.close()
    rows_out = _count_rows(years, by_year, true_count)
    checks = _identity_checks(rows_out, out_dir)
    log.put('CHECKS OF THE SCREEN')
    for text, ok in checks:
        log.put('  ' + ('PASS' if ok else 'FAIL') + ': ' + text)
    if not all((ok for _t, ok in checks)):
        raise Stop('a check of the screen failed')
    with open(os.path.join(out_dir, 'COUNTS.csv'), 'x', encoding='ascii', newline='\n') as fh:
        fh.write(','.join(COUNTS_HEADER) + '\n')
        for yt, mt, k in rows_out:
            fh.write(','.join([yt, mt] + [str(k[f]) for f in COUNT_FIELDS]) + '\n')
    log.put('COUNTS.csv written: %d rows' % len(rows_out))

def _write_manifest(out_dir):
    lines = []
    for n in MANIFEST_ORDER:
        lines.append(_sha256_file(os.path.join(out_dir, n)) + '  ' + n)
    with open(os.path.join(out_dir, 'MANIFEST_SHA256.txt'), 'x', encoding='ascii', newline='\n') as fh:
        fh.write(''.join((line + '\n' for line in lines)))

def main(argv):
    argv = list(argv)
    try:
        opts = _parse_argv(argv)
        years = _parse_years(opts['years'])
        out_dir = os.path.abspath(opts['out'])
        _test_out_folder(out_dir)
    except EarlyStop as exc:
        _say('STATUS: STOPPED: %s (nothing written)' % exc)
        return 1
    opts['argv'] = argv
    try:
        os.makedirs(out_dir, exist_ok=True)
        log = _RunLog(os.path.join(out_dir, 'RUN_LOG_ASCII.txt'))
    except Exception as exc:
        _say('STATUS: STOPPED: the run folder or its run log could not be opened (%s)' % type(exc).__name__)
        return 1
    try:
        _run(opts, years, out_dir, log)
        log.put('end: ' + _now_text())
        log.put('STATUS: PASS')
        log.close()
        _write_manifest(out_dir)
    except Stop as exc:
        try:
            log.put('end: ' + _now_text())
            log.put('STATUS: STOPPED: %s' % exc)
            log.close()
        except Exception:
            pass
        _say('STATUS: STOPPED: %s' % exc)
        return 1
    except Exception as exc:
        try:
            log.put('TRACEBACK')
            log.put(traceback.format_exc())
            log.put('end: ' + _now_text())
            log.put('STATUS: STOPPED: an exception (%s); its traceback is above' % type(exc).__name__)
            log.close()
        except Exception:
            pass
        _say('STATUS: STOPPED: an exception (%s); see the run log' % type(exc).__name__)
        return 1
    _say('STATUS: PASS')
    return 0
