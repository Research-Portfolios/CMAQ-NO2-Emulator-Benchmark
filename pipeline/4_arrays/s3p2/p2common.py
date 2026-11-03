"""Common parts of the second pass: its folders, the channel table with the point and marine NOx channel after the emission channels, the default storage types, JSON and file helpers, dates, clock, memory and disk reads, and a log."""
import datetime
import json
import os
import sys
CODE = '/content/emul/run/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np
from choi_dl import common
from s3 import s3common as S
ROOT = common.ROOT
W = common.W
RUN = common.RUN
ARRAYS_P2 = os.path.join(W, 'arrays_p2')
WORK_P2 = os.path.join(ARRAYS_P2, 'work')
TMP_P2 = os.path.join(W, 'tmp', 's3p2')
CFG_P2 = os.path.join(RUN, 'config', 'p2')
CFG_B = S.CFG
ARRAYS_B = S.ARRAYS
LOGS = S.LOGS
REPORT = os.path.join(LOGS, 'S3P2_ASCII.txt')
D4_RECORD = os.path.join(LOGS, 'S3P2_D4_ASCII.txt')
MARKER = os.path.join(LOGS, 'S3P2_DONE_ASCII.txt')
NE_ZIPS = os.path.join(W, 'natural_earth', 'zips_p2')
NE_EXTRACTED = os.path.join(NE_ZIPS, 'extracted')
NE_SHAPES = os.path.join(W, 'natural_earth', 'shapefiles', 'natural_earth', 'cultural')
NROWS, NCOLS = (S.NROWS, S.NCOLS)
G = (NROWS, NCOLS)
YEARS = S.YEARS
EMIS_YEARS = S.EMIS_YEARS
C7A_YEARS = (2015, 2016, 2018)
P1_ENTRY = ('P-1', 'STACK NOx', 'the six point and marine sectors (ptegu, ptnonipm, pt_oilgas, othpt, cmv_c1c2_12, cmv_c3_12): per stack the daily mean 00-23 UTC of NO + NO2 + HONO summed into its cell', 'mol/s', 'float32')
SOURCE_P2 = {'E-1': "merged layer-1 NO only, each year's own day; the emission climatology in inputs_c7", 'E-2': "merged layer-1 NO2 only, each year's own day; the emission climatology in inputs_c7", 'E-3': "merged layer-1 HONO only, each year's own day; the emission climatology in inputs_c7", 'E-4': "merged layer-1 VOC_INV, each year's own day; the emission climatology in inputs_c7", 'P-1': "point and marine NO + NO2 + HONO, each year's own day, the representative-day sectors by the weekday classes; the emission climatology in inputs_c7"}

def _channels():
    out = []
    for c in S.CHANNELS:
        name, label, source, unit, st = c
        if name in SOURCE_P2:
            source = SOURCE_P2[name]
        out.append((name, label, source, unit, st))
        if name == 'E-4':
            out.append(P1_ENTRY[:2] + (SOURCE_P2['P-1'],) + P1_ENTRY[3:])
    return out
CHANNELS = _channels()
CH_NAMES = [c[0] for c in CHANNELS]
CH_INDEX = {c[0]: i for i, c in enumerate(CHANNELS)}
DEFAULT_STORAGE = {c[0]: c[4] for c in CHANNELS}
EMIS = ('E-1', 'E-2', 'E-3', 'E-4', 'P-1')
EMIS_SPECIES = {'E-1': 'NO', 'E-2': 'NO2', 'E-3': 'HONO', 'E-4': 'VOC_INV', 'P-1': 'NO+NO2+HONO'}
LOG_CHANNELS = ('M-9', 'E-1', 'E-2', 'E-3', 'E-4', 'P-1')
MET_SERIES = S.MET_SERIES
LAG_OF = S.LAG_OF
FLOAT16_SMALLEST_NORMAL = float(np.finfo(np.float16).smallest_normal)

def ensure_dirs():
    for d in (ARRAYS_P2, WORK_P2, TMP_P2, CFG_P2):
        os.makedirs(d, exist_ok=True)

def jload(p):
    with open(p, 'r', encoding='ascii') as f:
        return json.load(f)

def write_json(path, obj):
    return S.write_json(path, obj)

def file_info(path):
    return S.file_info(path)

def npy_info(path):
    a = np.load(path, mmap_mode='r')
    info = S.file_info(path)
    info.update({'shape': list(a.shape), 'dtype': str(a.dtype)})
    del a
    return info

def now_local():
    return S.now_local()

def mem_gb():
    return S.mem_gb()

def disk_free_gb():
    return S.disk_free_gb()

def days_of_year(y):
    return S.days_of_year(y)

def ymd(d):
    return S.ymd(d)

def from_ymd(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))

def run_main(fn):
    S.run_main(fn)

def log(msg):
    print('%s %s' % (now_local(), common.mask(msg)), flush=True)
