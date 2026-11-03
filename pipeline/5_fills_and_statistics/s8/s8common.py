"""Common parts of the fill: places, the year's extracts with their checks, holidays, the daily emission fields formed as for 2017 and 2019 (gridded NO, NO2, HONO and VOC; the point and marine NOx sum over six sectors), the channel indices, and memory and disk reads."""
import collections
import csv
import datetime
import hashlib
import io
import json
import os
import sys
CODE = '/content/emul/run/code'
for _p in (CODE, os.path.join(CODE, 's2a')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from s3 import s3common as S
from s3p2 import p2common as P
from s3p2 import p2_emis as PE
import s2a_s3_pt as S2PT
ROOT = S.common.ROOT
W = S.common.W
RUN = S.common.RUN
G = P.G
EMIS = P.EMIS
MERGED_SPECIES = ('NO', 'NO2', 'HONO', 'VOC_INV')
PT_SPECIES = ('NO', 'NO2', 'HONO')
MERGED_UNITS = ('moles/s', 'moles/s', 'moles/s', 'g/s')
MOL_UNITS = ('moles/s', 'mol/s')
DAILY = tuple(S.DAILY_SECTORS)
REP = tuple(S.REP_SECTORS)
CLASSES = ('Mon', 'TueFri', 'Sat', 'Sun')
HOURS = '00-23 UTC'
NULL_LIMIT = -9e+36
ARRAYS_P2 = P.ARRAYS_P2
WORK_P2 = P.WORK_P2
CFG_P2 = P.CFG_P2
LOGS = os.path.join(RUN, 'logs')

def paths(year):
    y = str(year)
    ext = os.path.join(W, 'extract_c7a', y)
    return {'ext': ext, 'copylog': os.path.join(LOGS, 'B2C_C7A_COPY_%s.json' % y), 'line': os.path.join(ext, 'C7A_4_LINE_%s_ASCII.txt' % y), 'manifest': os.path.join(ext, 'MANIFEST_%s.csv' % y), 'sums': os.path.join(ext, 'SHA256SUMS_%s.txt' % y), 'array': os.path.join(ARRAYS_P2, 'inputs_f32_%s.npy' % y), 'mask': os.path.join(ARRAYS_P2, 'mask_conus_land.npy'), 'scratch': os.path.join(W, 'tmp', 's8_%s' % y), 'report': os.path.join(LOGS, 'S8_%s_ASCII.txt' % y), 'd4record': os.path.join(LOGS, 'S8_%s_D4_ASCII.txt' % y), 'work': lambda name: os.path.join(WORK_P2, name % y)}

def now():
    return datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')

def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def jload(p):
    with open(p, 'r', encoding='ascii') as f:
        return json.load(f)

def write_json(path, obj):
    return S.write_json(path, obj)

def fwd(p):
    return p.replace('\\', '/')

class Inputs:

    def __init__(self, year):
        self.year = int(year)
        self.p = paths(year)
        self.log = jload(self.p['copylog'])
        self.rows = []
        with open(self.p['manifest'], 'r', encoding='ascii', newline='') as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        self.by_path = {}
        for r in self.rows:
            k = fwd(r['path'])
            if k in self.by_path:
                raise ValueError('manifest: a path twice: ' + k)
            self.by_path[k] = r
        self.sums = {}
        with open(self.p['sums'], 'r', encoding='ascii') as f:
            for line in f:
                line = line.rstrip('\r\n')
                if not line.strip():
                    continue
                parts = line.split('  ')
                if len(parts) != 3:
                    raise ValueError('sums line form')
                sha, nb, rel = parts
                if rel in self.sums:
                    raise ValueError('sums: a path twice: ' + rel)
                self.sums[rel] = (sha, int(nb))
        self.reads = collections.Counter()
        self.verified = set()

    def preconditions(self):
        out = {'copylog': S.common.rel(self.p['copylog']), 'copylog_sha256': sha256_file(self.p['copylog']), 'copylog_bytes': os.stat(self.p['copylog']).st_size, 'complete': self.log.get('complete') is True, 'completed_at': self.log.get('completed_at'), 'files': len(self.log['files']), 'status': dict(collections.Counter((str(v.get('status')) for v in self.log['files'].values()))), 'unmatched_listed': len(self.log.get('unmatched_listed', [])), 'missing_from_sums': len(self.log.get('missing_from_sums', []))}
        with open(self.p['line'], 'r', encoding='ascii') as f:
            first = f.readline().rstrip('\r\n')
        want = 'C-7a (4) %d: PASS' % self.year
        out['line_file'] = S.common.rel(self.p['line'])
        out['line_file_sha256'] = sha256_file(self.p['line'])
        out['line_begins_pass'] = first.startswith(want)
        out['met'] = bool(out['complete'] and out['line_begins_pass'] and (out['status'] == {'ok': out['files']}))
        return out

    def rel_merged(self, d):
        return 'emis_merged/emis_merged_%s.npz' % d

    def rel_pt(self, s, d):
        return 'pt_marine/%s/pt_%s_%s.npz' % (s, s, d)

    def read(self, rel):
        e = self.log['files'].get(rel)
        if e is None or e.get('status') != 'ok':
            raise ValueError('not ok in the copy log: ' + rel)
        r = self.by_path.get(rel)
        if r is None:
            raise ValueError('not in the manifest: ' + rel)
        sm = self.sums.get(rel)
        if sm is None:
            raise ValueError('not in the sums: ' + rel)
        full = os.path.join(self.p['ext'], *rel.split('/'))
        with open(full, 'rb') as f:
            b = f.read()
        h = hashlib.sha256(b).hexdigest()
        n = len(b)
        if not (h == e['sha256'] == r['sha256'] == sm[0] and n == int(e['bytes']) == int(r['bytes']) == sm[1]):
            raise ValueError('extract differs from its log, manifest or sums: ' + rel)
        self.reads[rel] += 1
        self.verified.add(rel)
        return (np.load(io.BytesIO(b), allow_pickle=False), r)

    def pt_dates(self, sector):
        out = sorted((r['date'] for r in self.rows if r['kind'] == 'pt' and r['sector'] == sector))
        return out

    def merged_dates(self):
        return sorted((r['date'] for r in self.rows if r['kind'] == 'merged'))

def holidays(year):
    out = collections.defaultdict(list)
    for h in S2PT.federal_holidays(int(year)):
        out[h['legal']].append(h['name'] + ' (legal date)')
        if h['observed'] != h['legal']:
            out[h['observed']].append(h['name'] + ' (observed)')
    return dict(out)

def plan_holidays_check():
    plan = jload(os.path.join(S.CFG, 'S2A_PLAN_S3.json'))
    res = {}
    for y, lst in plan['pt_marine']['federal_holidays'].items():
        mine = S2PT.federal_holidays(int(y))
        res[y] = sorted(((h['legal'], h['observed'], h['name']) for h in lst)) == sorted(((h['legal'], h['observed'], h['name']) for h in mine))
    return res

def as_str(z, k):
    return str(z[k].tolist())

def check_merged(z, r, d, year):
    bad = []
    sp = [str(x) for x in z['species'].tolist()]
    un = [str(x) for x in z['units'].tolist()]
    if tuple(sp) != MERGED_SPECIES:
        bad.append('species %s' % sp)
    if tuple(un) != MERGED_UNITS:
        if not (all((u in MOL_UNITS for u in un[:3])) and un[3] == 'g/s'):
            bad.append('units %s' % un)
    if as_str(z, 'hours') != HOURS:
        bad.append('hours')
    if as_str(z, 'date') != d:
        bad.append('date')
    dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    if int(z['tflag_date'].tolist()) != S.yyyyddd(dd):
        bad.append('tflag_date')
    if as_str(z, 'p1') != 'on':
        bad.append('p1')
    for k in MERGED_SPECIES:
        a = z[k]
        if a.shape != G or a.dtype != np.float32:
            bad.append('%s %s %s' % (k, a.shape, a.dtype))
    if r['kind'] != 'merged' or r['date'] != d or r['hours'] != HOURS or (r['units'].split() != un) or (r['species'].split() != sp):
        bad.append('manifest row')
    if int(d[:4]) != int(year):
        bad.append('year')
    return bad

def check_pt(z, r, s, d, year):
    bad = []
    sp = [str(x) for x in z['species'].tolist()]
    un = [str(x) for x in z['units'].tolist()]
    if tuple(sp) != PT_SPECIES:
        bad.append('species %s' % sp)
    if len(un) != 3 or not all((u in MOL_UNITS for u in un)):
        bad.append('units %s' % un)
    if as_str(z, 'hours') != HOURS:
        bad.append('hours')
    if as_str(z, 'date') != d:
        bad.append('date')
    dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    if int(z['tflag_date'].tolist()) != S.yyyyddd(dd):
        bad.append('tflag_date')
    if as_str(z, 'p1') != 'on':
        bad.append('p1')
    if as_str(z, 'sector') != s:
        bad.append('sector')
    if int(z['base'].tolist()) != 1:
        bad.append('base')
    for k in PT_SPECIES + ('P1_NOX',):
        a = z[k]
        if a.shape != G or a.dtype != np.float32:
            bad.append('%s %s %s' % (k, a.shape, a.dtype))
    if r['kind'] != 'pt' or r['sector'] != s or r['date'] != d or (r['hours'] != HOURS) or (r['p1'] != 'on') or (r['species'].split() != sp) or (r['units'].split() != un):
        bad.append('manifest row')
    if int(d[:4]) != int(year):
        bad.append('year')
    return bad

class Months:

    def __init__(self, inp, hol):
        self.inp = inp
        self.hol = hol
        self.year = inp.year
        self.members = {s: inp.pt_dates(s) for s in REP}
        self.left_out = []
        self.fallbacks = []
        self.classes = {}
        for s in REP:
            for m in range(1, 13):
                mem = [d for d in self.members[s] if int(d[4:6]) == m]
                by = collections.defaultdict(list)
                for d in mem:
                    if d in hol:
                        self.left_out.append('%s %s' % (s, d))
                        continue
                    dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
                    by[S.CLASS_OF_WEEKDAY[dd.weekday()]].append(d)
                key = '%s_%d_%02d' % (s, self.year, m)
                self.classes[key] = {c: list(by.get(c, [])) for c in CLASSES}
                for c in CLASSES:
                    if not by.get(c):
                        self.fallbacks.append('%s %d-%02d %s -> TueFri' % (s, self.year, m, c))
                        if not by.get('TueFri'):
                            raise ValueError('%s %d-%02d: no TueFri member' % (s, self.year, m))

    def use(self, s, m, c):
        cl = self.classes['%s_%d_%02d' % (s, self.year, m)]
        return cl[c] or cl['TueFri']

    def means(self, m):
        out = {}
        for s in REP:
            out[s] = {}
            for c in CLASSES:
                use = self.use(s, m, c)
                acc = np.zeros(G, np.float64)
                acc_sp = np.zeros(G, np.float64)
                for d in use:
                    z, r = self.inp.read(self.inp.rel_pt(s, d))
                    bad = check_pt(z, r, s, d, self.year)
                    if bad:
                        raise ValueError('%s %s: %s' % (s, d, bad))
                    acc += z['P1_NOX'].astype(np.float64)
                    acc_sp += PE.p1_of(z)
                out[s][c] = (acc / len(use), acc_sp / len(use))
        return out

def day_fields(inp, d, cmeans):
    y = inp.year
    z, r = inp.read(inp.rel_merged(d))
    bad = check_merged(z, r, d, y)
    if bad:
        raise ValueError('merged %s: %s' % (d, bad))
    f = np.zeros((len(EMIS),) + G, np.float64)
    for k, sp in enumerate(MERGED_SPECIES):
        f[k] = z[sp]
    parts = {'merged': f[0] + f[1] + f[2]}
    p1 = np.zeros(G, np.float64)
    sp1 = np.zeros(G, np.float64)
    for s in DAILY:
        zp, rp = inp.read(inp.rel_pt(s, d))
        bad = check_pt(zp, rp, s, d, y)
        if bad:
            raise ValueError('%s %s: %s' % (s, d, bad))
        add = zp['P1_NOX'].astype(np.float64)
        p1 += add
        parts[s] = add
        sp1 += PE.p1_of(zp)
    dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    c = S.CLASS_OF_WEEKDAY[dd.weekday()]
    for s in REP:
        add, add_sp = cmeans[s][c]
        p1 += add
        parts[s] = add
        sp1 += add_sp
    f[4] = p1
    return (f, sp1, parts)

def f32_index():
    ch = jload(os.path.join(CFG_P2, 'S3_CHANNELS.json'))
    idx = {}
    for e in ch['channels']:
        if e['array'] == 'inputs_f32':
            idx[e['name']] = int(e['index'])
    if sorted(idx.values()) != list(range(len(idx))):
        raise ValueError('inputs_f32 indices are not 0..n-1')
    others = sorted((c for c in idx if c not in EMIS))
    return (idx, others)

def steps32(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    big = np.maximum(np.abs(a), np.abs(b))
    sp = np.spacing(big).astype(np.float64)
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    out = np.zeros(a.shape, np.float64)
    ne = a != b
    out[ne] = diff[ne] / sp[ne]
    return out

def mem_gb():
    return S.mem_gb()

def disk_free_gb():
    import shutil
    return shutil.disk_usage(W).free / 1000000000.0
