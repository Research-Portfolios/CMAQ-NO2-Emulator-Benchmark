"""Fixed rules, places and helpers of the Drive extraction: archive kinds and families, the days of each year, the value sample, time stamps, hashing, text masking and wrapping, memory reads, and JSON and log writing."""
import datetime
import hashlib
import json
import os
import re
import sys
ROOT = '/content/c7a'
CODE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(CODE)
VENDOR = os.path.join(CODE, 'vendor')
RUN_DIR = os.path.join(PKG, 'run')
STOP_FILE = os.path.join(PKG, 'STOP')
H_INPUT = ROOT + '/INPUT'
OUT_ROOT = ROOT + '/extracts_c7a'
XFER_OUT = ROOT + '/xfer_out/extracts_c7a'
XFER_IN_CROSS = ROOT + '/xfer_in/cross_route'
CROSS_DIR = ROOT + '/cross_route_c7a'
RCRUN = ROOT + '/k4'
K4_TABLES = (RCRUN + '/K4_%s.csv', PKG + '/K4_%s_AUTHORS.csv')
PINS = PKG + '/pins/pins_%s_idfree.csv'
PINS_SHA16 = {'2015': 'f2d3b06fda610202', '2016': '114d6983c1c300c1', '2018': 'e06798265e852512'}
T2_SCRIPT = PKG + '/ref/wait_t2.py'
T2_SHA16 = '147657ed76268292'
SLOT_SCRIPT = PKG + '/ref/wait_slot.py'
SLOT_SHA16 = 'e1b4b55719884bec'
VENV_PY = ROOT + '/venv/bin/python'
S2A_PLAN = PKG + '/config/S2A_PLAN_S3.json'
S2A_PLAN_SHA16 = None
YEARS = ('2015', '2016', '2018')
GRID = (299, 459)
GRID_ATTRS = {'NCOLS': 459, 'NROWS': 299, 'GDTYP': 2, 'P_ALP': 33.0, 'P_BET': 45.0, 'P_GAM': -97.0, 'XCENT': -97.0, 'YCENT': 40.0, 'XORIG': -2556000.0, 'YORIG': -1728000.0, 'XCELL': 12000.0, 'YCELL': 12000.0}
R_EARTH = 6370000.0
MERGED_SPECIES = ('NO', 'NO2', 'HONO', 'VOC_INV')
PT_SPECIES = ('NO', 'NO2', 'HONO')
DAILY_SECTORS = ('ptegu', 'cmv_c1c2_12', 'cmv_c3_12')
REP_SECTORS = ('ptnonipm', 'pt_oilgas', 'othpt')
SECTORS = DAILY_SECTORS + REP_SECTORS
N_HOURS = 24
LAYER = 0
BASE_EXPECTED = 1
UNITS_OK = {'NO': ('moles/s', 'mol/s'), 'NO2': ('moles/s', 'mol/s'), 'HONO': ('moles/s', 'mol/s'), 'VOC_INV': ('g/s',)}
REP_MIN, REP_MAX = (4, 7)
REP_K6_1_N = 8
REP_K6_1_FIRST = 4
REP_K6_1_WINDOW = (('1220', '1231'), ('0101', '0102'))
K2_LOW, K2_HIGH = (0.5, 2.0)
MEM_FLOOR_GB = 4.6
PEAK_GB = 1.0
PEAK_STACKS = 2685954
MEM_WAIT_S = 300
T2_EVERY_S = 900
T2_FAILS_STOP = 3
POLL_S = 300
T3_DEADLINE = '2026-10-23 00:00:00 -0500'
CF_DEADLINE = '2026-10-24 12:00:00 -0500'
C7A_DEADLINE = '2026-10-26 00:00:00 -0500'
VC_MERGED_MMDD = ('0115', '0415', '0701', '1015')
VC_DAILY_MMDD = '0701'
VC_REP_MONTH = '07'
K5_MMDD = '0701'
RE_MERGED = re.compile('(?:^|/)emis_mole_all_(\\d{8})_12US1_nobeis_norwc_[^/]*\\.(nc4|ncf)$')
RE_PT = re.compile('(?:^|/)inln_mole_(%s)_(\\d{8})_12US1_[^/]*\\.(nc4|ncf)$' % '|'.join(SECTORS))
RE_GROUPS = re.compile('(?:^|/)stack_groups_(%s)_12US1_[^/]*\\.(nc4|ncf)$' % '|'.join(SECTORS))
_HOME = os.path.expanduser('~')
_NAME = os.path.basename(_HOME.rstrip('\\/'))

def kind_of_tar(name):
    if name.endswith('.contents.txt'):
        return 'contents'
    if 'stackgroups' in name:
        return 'stackgroups'
    if 'merged_nobeis_norwc' in name:
        return 'merged'
    if 'ptsectors_plus_rwc' in name:
        return 'ptsectors'
    return 'other'

def family_month(name):
    k = kind_of_tar(name)
    m = re.search('_(\\d{2})_EQUATES_v1\\.0\\.tar$', name)
    return (k, m.group(1) if m and k != 'stackgroups' else '')

def classify(name, year):
    m = RE_MERGED.search(name)
    if m:
        return ('merged', None, m.group(1), m.group(2))
    m = RE_PT.search(name)
    if m:
        return ('pt', m.group(1), m.group(2), m.group(3))
    m = RE_GROUPS.search(name)
    if m:
        return ('groups', m.group(1), None, m.group(2))
    return None

def days_of_year(y):
    d = datetime.date(int(y), 1, 1)
    out = []
    while d.year == int(y):
        out.append(d.strftime('%Y%m%d'))
        d += datetime.timedelta(days=1)
    return out

def yyyyddd(d):
    x = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    return x.year * 1000 + x.timetuple().tm_yday

def value_sample(year, rep_dates):
    out = [('merged', None, year + md) for md in VC_MERGED_MMDD]
    out += [('pt', s, year + VC_DAILY_MMDD) for s in DAILY_SECTORS]
    for s in REP_SECTORS:
        jul = sorted((d for d in rep_dates.get(s, []) if d[:6] == year + VC_REP_MONTH))
        out.append(('pt', s, jul[0] if jul else None))
    return out

def rep_month_ok(dates):
    dates = list(dates)
    distinct = sorted(set(dates))
    repeated = [d for d in distinct if dates.count(d) > 1]
    beyond = distinct[REP_K6_1_FIRST:]
    n = len(dates)
    if repeated:
        return (False, '%d members, %s repeated' % (n, ' '.join(repeated)), repeated, beyond)
    if REP_MIN <= n <= REP_MAX:
        return (True, '%d members, %d to %d' % (n, REP_MIN, REP_MAX), repeated, beyond)
    if n == REP_K6_1_N:
        out = [d for d in beyond if not any((a <= d[4:8] <= b for a, b in REP_K6_1_WINDOW))]
        if not out:
            return (True, '%d members, the %d beyond the first %d from 20 Dec to 2 Jan' % (n, len(beyond), REP_K6_1_FIRST), repeated, beyond)
        return (False, '%d members, %s beyond the first %d outside 20 Dec to 2 Jan' % (n, ' '.join(out), REP_K6_1_FIRST), repeated, beyond)
    return (False, '%d members, not %d to %d or %d' % (n, REP_MIN, REP_MAX, REP_K6_1_N), repeated, beyond)

def now():
    return datetime.datetime.now().astimezone()

def stamp(t=None):
    return (t or now()).strftime('%Y-%m-%d %H:%M:%S %z')

def parse_stamp(s):
    return datetime.datetime.strptime(s.strip(), '%Y-%m-%d %H:%M:%S %z')

def mask(s):
    s = str(s)
    if _HOME and len(_HOME) > 3:
        s = re.sub(re.escape(_HOME), '<home>', s, flags=re.I)
        s = re.sub(re.escape(_HOME.replace('\\', '/')), '<home>', s, flags=re.I)
    if _NAME and len(_NAME) >= 3:
        s = re.sub(re.escape(_NAME), '<name>', s, flags=re.I)
    return s.replace(chr(64), '<at>')

def sha256_file(path, piece=8 << 20):
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(piece)
            if not b:
                break
            h.update(b)
            n += len(b)
    return (h.hexdigest(), n)

def sha16(path):
    return sha256_file(path)[0][:16]

def read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()

def mem_free_gb():
    import psutil
    return psutil.virtual_memory().available / float(1 << 30)

def wrap(text, width=78, indent='  '):
    words = str(text).split(' ')
    lines, cur = ([], '')
    for w in words:
        while len(w) > width - len(indent):
            if cur:
                lines.append(cur)
                cur = ''
            lines.append((indent if lines else '') + w[:width - len(indent)])
            w = w[width - len(indent):]
        trial = cur + ' ' + w if cur else (indent if lines else '') + w
        if len(trial) > width:
            lines.append(cur)
            cur = indent + w
        else:
            cur = trial
    if cur or not lines:
        lines.append(cur)
    return lines

def check_lines(lines):
    for ln in lines:
        if len(ln) > 78 or any((ord(c) > 126 or ord(c) < 32 for c in ln)):
            raise ValueError('record line not 7-bit or over 78: %r' % ln)
        if chr(64) in ln:
            raise ValueError('an at sign in a record line')

def append_record(path, text):
    lines = []
    for para in mask(text).split('\n'):
        lines.extend(wrap(para))
    check_lines(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return lines

def write_json(path, obj):
    s = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True)
    if chr(64) in s:
        raise ValueError('an at sign in %s' % path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + '.partial'
    with open(part, 'w', encoding='ascii', newline='\n') as f:
        f.write(s + '\n')
    os.replace(part, path)

def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='ascii') as f:
        return json.load(f)

def use_vendor():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import choi
    from choi import tario, stacks
    from choi_dl import common, s3io
    for mod in (choi, tario, stacks, common, s3io):
        f = os.path.abspath(mod.__file__)
        if not f.lower().startswith(VENDOR.lower() + os.sep):
            raise RuntimeError('reader %s not from the vendor folder' % mod.__name__)
    return (tario, stacks, s3io)
