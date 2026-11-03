"""Common parts of the preprocessing: places, the channel table, the inventory, the reader of a raw RSIG answer (read whole, size and sha256 checked, decompressed and opened in memory), the mean of a cell's four corners for dot-grid fields, dates, weekday classes, clock and memory reads, and log helpers."""
import datetime
import gzip
import hashlib
import json
import os
import sys
CODE = '/content/emul/run/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np
from choi_dl import common
ROOT = common.ROOT
W = common.W
RUN = common.RUN
CFG = os.path.join(RUN, 'config')
LOGS = os.path.join(RUN, 'logs')
ARRAYS = os.path.join(W, 'arrays')
WORK_S3 = os.path.join(W, 'work', 's3')
TMP_S3 = os.path.join(W, 'tmp', 's3')
INVENTORY = os.path.join(CFG, 'S2C_INVENTORY.json')
REPORT = os.path.join(LOGS, 'S3_PREPROCESS_ASCII.txt')
NROWS, NCOLS = (299, 459)
NULL_LIMIT = -9e+36
YEARS = (2015, 2016, 2017, 2018, 2019)
EMIS_YEARS = (2017, 2019)
SERIES = {'T1': ('NO2', 'ppbV', False), 'M1': ('PRES', 'Pa', False), 'M2': ('SFC_TMP', 'C', False), 'M3': ('WSPD10', 'm s-1', False), 'M4': ('UWIND', 'm s-1', True), 'M5': ('VWIND', 'm s-1', True), 'M6': ('SOL_RAD', 'W m-2', False), 'M7': ('QV', 'kg kg-1', False), 'M8': ('PBLH', 'm', False), 'M9': ('precip', 'cm', False), 'X6': ('PBLH', 'm', False), 'T2': ('DAILY_NO2', 'ppbV', False)}
C2_UNITS = {'T1': ('ppbV = ppb', ('ppbV',)), 'M1': ('Pa', ('Pa',)), 'M2': ('C', ('C',)), 'M3': ('m/s', ('m s-1',)), 'M4': ('m/s', ('m s-1',)), 'M5': ('m/s', ('m s-1',)), 'M6': ('W/m2', ('W m-2',)), 'M7': ('kg/kg', ('kg kg-1',)), 'M8': ('m', ('m',)), 'M9': ('cm', ('cm',)), 'X6': ('m', ('m',)), 'T2': ('ppbV', ('ppbV',))}
CHANNELS = [('M-1', 'PRES layer 1', 'RSIG M1 metcro3d.PRES L1 DAILY_MEAN', 'Pa', 'float16_affine'), ('M-2', 'SFC_TMP', 'RSIG M2 aconc.SFC_TMP DAILY_MEAN', 'C', 'float16'), ('M-3', 'WSPD10', 'RSIG M3 aconc.WSPD10 DAILY_MEAN', 'm/s', 'float16'), ('M-4', 'UWIND layer 1, dot to cell', 'RSIG M4 metdot3d.UWIND L1 DAILY_MEAN, mean of the 4 corners', 'm/s', 'float16'), ('M-5', 'VWIND layer 1, dot to cell', 'RSIG M5 metdot3d.VWIND L1 DAILY_MEAN, mean of the 4 corners', 'm/s', 'float16'), ('M-6', 'SOL_RAD', 'RSIG M6 aconc.SOL_RAD DAILY_MEAN', 'W/m2', 'float16'), ('M-7', 'QV layer 1', 'RSIG M7 metcro3d.QV L1 DAILY_MEAN', 'kg/kg', 'float16'), ('M-8', 'PBLH', 'RSIG M8 aconc.PBLH DAILY_MEAN', 'm', 'float16'), ('M-9', 'precip', 'RSIG M9 aconc.precip DAILY_MEAN', 'cm', 'float32'), ('L-1', 'HT', 'MCIP GRIDCRO2D HT (static)', 'm', 'float16'), ('L-2', 'LWMASK', 'MCIP GRIDCRO2D LWMASK (static)', '1', 'float16'), ('L-3', 'urban fraction', 'MCIP GRIDCRO2D PURB / 100 (static)', 'fraction', 'float16'), ('L-5', 'VEG', 'MCIP METCRO2D VEG, mean of records 00-23 UTC of the 15th of each month of 2016 (monthly)', 'fraction', 'float16'), ('L-6', 'DLUSE', 'MCIP GRIDCRO2D DLUSE (static), the dominant land-use category code read as a number', '1', 'float16'), ('E-1', 'NO', 'climatology of merged + point and marine NO', 'mol/s', 'float32'), ('E-2', 'NO2', 'climatology of merged + point and marine NO2', 'mol/s', 'float32'), ('E-3', 'HONO', 'climatology of merged + point and marine HONO', 'mol/s', 'float32'), ('E-4', 'VOC_INV', 'climatology of merged VOC_INV', 'g/s', 'float32'), ('X-1', 'PBLH, previous day', 'M-8 of the day before', 'm', 'float16'), ('X-2', 'WSPD10, previous day', 'M-3 of the day before', 'm/s', 'float16'), ('X-3', 'UWIND, previous day', 'M-4 of the day before', 'm/s', 'float16'), ('X-4', 'VWIND, previous day', 'M-5 of the day before', 'm/s', 'float16'), ('X-5', 'SFC_TMP, previous day', 'M-2 of the day before', 'C', 'float16'), ('X-6', 'PBLH daily maximum', 'RSIG X6 aconc.PBLH DAILY_MAX', 'm', 'float16')]
CH_NAMES = [c[0] for c in CHANNELS]
CH_INDEX = {c[0]: i for i, c in enumerate(CHANNELS)}
MET_SERIES = {'M-1': 'M1', 'M-2': 'M2', 'M-3': 'M3', 'M-4': 'M4', 'M-5': 'M5', 'M-6': 'M6', 'M-7': 'M7', 'M-8': 'M8', 'M-9': 'M9', 'X-6': 'X6'}
LAG_OF = {'X-1': 'M-8', 'X-2': 'M-3', 'X-3': 'M-4', 'X-4': 'M-5', 'X-5': 'M-2'}
EMIS = ('E-1', 'E-2', 'E-3', 'E-4')
EMIS_SPECIES = {'E-1': 'NO', 'E-2': 'NO2', 'E-3': 'HONO', 'E-4': 'VOC_INV'}
LOG_CHANNELS = ('M-9', 'E-1', 'E-2', 'E-3', 'E-4')
PRES_OFFSET = 50000.0
PRES_SCALE = 60000.0
CLASS_OF_WEEKDAY = {0: 'Mon', 1: 'TueFri', 2: 'TueFri', 3: 'TueFri', 4: 'TueFri', 5: 'Sat', 6: 'Sun'}
DAILY_SECTORS = ('ptegu', 'cmv_c1c2_12', 'cmv_c3_12')
REP_SECTORS = ('ptnonipm', 'pt_oilgas', 'othpt')

def path_abs(rel_path):
    return os.path.join(ROOT, rel_path)

def now_local():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def mem_gb():
    import psutil
    return psutil.virtual_memory().available / 1000000000.0

def disk_free_gb():
    import shutil
    return shutil.disk_usage(W).free / 1000000000.0

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def file_info(path):
    st = os.stat(path)
    return {'path': common.rel(path), 'bytes': int(st.st_size), 'sha256': sha256_file(path)}

def days_of_year(y):
    d = datetime.date(y, 1, 1)
    out = []
    while d.year == y:
        out.append(d)
        d += datetime.timedelta(days=1)
    return out

def yyyyddd(d):
    return d.year * 1000 + d.timetuple().tm_yday

def from_yyyyddd(v):
    v = int(v)
    return datetime.date(v // 1000, 1, 1) + datetime.timedelta(days=v % 1000 - 1)

def ymd(d):
    return d.strftime('%Y%m%d')

def is_validation(d):
    return d.isocalendar()[1] % 5 == 0

def load_inventory():
    with open(INVENTORY, 'r', encoding='ascii') as f:
        return json.load(f)

def inventory_inputs(inv=None):
    inv = inv or load_inventory()
    out = {}
    for e in inv['files']:
        if e.get('s3_input') is True and e.get('verified') is True:
            out.setdefault(e['kind'], []).append(e)
    return out

def rsig_index(inputs):
    out = {}
    for e in inputs.get('rsig', []):
        key = (e['series'], e['period'])
        if key in out:
            raise ValueError('two answers for %s %s' % key)
        out[key] = e
    return out
GRID_ATTRS = ('NCOLS', 'NROWS', 'NLAYS', 'NVARS', 'TSTEP', 'SDATE', 'STIME', 'GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG', 'XCELL', 'YCELL', 'VGTYP', 'VGTOP', 'FTYPE')

def _attr(ds, name):
    try:
        v = getattr(ds, name)
    except AttributeError:
        return None
    if isinstance(v, bytes):
        return v.decode('ascii', 'replace')
    if hasattr(v, 'tolist'):
        v = v.tolist()
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    return v

def decompress_answer(entry, check_raw=True):
    p = path_abs(entry['path'])
    with open(p, 'rb') as f:
        raw = f.read()
    chk = {'raw_bytes_ok': len(raw) == int(entry['bytes'])}
    if check_raw:
        chk['raw_sha256_ok'] = sha256_bytes(raw) == entry['sha256']
    data = gzip.decompress(raw)
    del raw
    chk['dec_bytes_ok'] = len(data) == int(entry['decompressed_bytes'])
    chk['dec_sha256_ok'] = sha256_bytes(data) == entry['decompressed_sha256']
    bad = [k for k, v in chk.items() if not v]
    if bad:
        raise ValueError('%s: %s' % (entry['path'], ', '.join(bad)))
    return (data, chk)

def read_answer(entry, var, coords=False, check_raw=True):
    import netCDF4
    data, chk = decompress_answer(entry, check_raw=check_raw)
    ds = netCDF4.Dataset('inmemory.nc', mode='r', memory=data)
    try:
        out = {'format': ds.file_format, 'checks': chk, 'variables': [str(v) for v in ds.variables], 'dims': {str(k): len(d) for k, d in ds.dimensions.items()}, 'attrs': {}}
        for a in GRID_ATTRS:
            v = _attr(ds, a)
            if v is not None:
                out['attrs'][a] = v
        v = ds.variables[var]
        v.set_auto_maskandscale(False)
        out['units'] = str(getattr(v, 'units', '')).strip()
        out['var_dims'] = [str(d) for d in v.dimensions]
        out['values'] = np.array(v[:])
        t = ds.variables['TFLAG']
        t.set_auto_maskandscale(False)
        out['tflag'] = np.array(t[:])
        if coords:
            for c in ('LONGITUDE', 'LATITUDE'):
                if c in ds.variables:
                    cv = ds.variables[c]
                    cv.set_auto_maskandscale(False)
                    a = np.array(cv[:])
                    while a.ndim > 2:
                        a = a[0]
                    out[c] = a
    finally:
        ds.close()
        del data
    return out

def dot_to_cell(a):
    a = np.asarray(a, dtype=np.float64)
    return 0.25 * (a[..., :-1, :-1] + a[..., 1:, :-1] + a[..., :-1, 1:] + a[..., 1:, 1:])

def write_json(path, obj):
    s = json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=True)
    s = s.replace('\r\n', '\n')
    if '@' in s:
        raise ValueError('an at sign in %s' % path)
    part = path + '.partial'
    with open(part, 'w', encoding='ascii', newline='\n') as f:
        f.write(s + '\n')
    os.replace(part, path)
    return file_info(path)

def check_record_text(text):
    bad = []
    for i, line in enumerate(text.split('\n'), 1):
        if len(line) > 78:
            bad.append('line %d has %d characters' % (i, len(line)))
        if any((ord(c) > 126 or (ord(c) < 32 and c != '\n') for c in line)):
            bad.append('line %d has a non-ASCII or control character' % i)
        if '@' in line:
            bad.append('line %d has an at sign' % i)
    return bad

def append_record(path, lines):
    text = '\n'.join(lines) + '\n'
    bad = check_record_text(text.rstrip('\n'))
    if bad:
        raise ValueError('record lines: ' + '; '.join(bad[:5]))
    with open(path, 'a', encoding='ascii', newline='\n') as f:
        f.write(text)

def wrap(first, text, cont='  '):
    return common.wrap(first, text, 78, cont)

def run_main(fn):
    import traceback
    try:
        fn()
    except SystemExit:
        raise
    except BaseException:
        sys.stderr.write(common.mask(traceback.format_exc()))
        sys.stderr.flush()
        raise SystemExit(1)
