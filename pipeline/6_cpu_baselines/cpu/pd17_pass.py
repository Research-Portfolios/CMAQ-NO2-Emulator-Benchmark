"""A read-only pass over every built input array before any fit: checks its sha256, counts non-finite, fill and negative values per channel (stopping on non-finite or fill values), and tests the four-corner mean of the dot-grid winds on a synthetic case."""
import csv
import datetime
import hashlib
import json
import os
import sys
import time
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np
NULL_LIMIT = -9e+36
LIMIT32 = np.float32(NULL_LIMIT)
EMIS = ('E-1', 'E-2', 'E-3', 'E-4', 'P-1')
C7A_YEARS = (2015, 2016, 2018)
SECTORS = ('ptegu', 'cmv_c1c2_12', 'cmv_c3_12', 'ptnonipm', 'pt_oilgas', 'othpt')
MERGED_SPECIES = 'NO NO2 HONO VOC_INV'
PT_SPECIES = 'NO NO2 HONO'
DOT_SHAPE = (300, 460)

def say(msg):
    print('%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)

def jload(p):
    with open(p, 'rb') as f:
        return json.loads(f.read().decode('ascii'))

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def base_name(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]

def order_names(doc, key):
    d = doc.get(key)
    if not isinstance(d, dict) or not isinstance(d.get('channel_order'), list):
        raise ValueError('S3_CHANNELS.json lacks %s.channel_order' % key)
    names = [str(s).split()[0] for s in d['channel_order']]
    if tuple(names) != EMIS:
        raise ValueError('S3_CHANNELS.json %s: channels %s' % (key, names))
    return names

def array_list(arrays_doc, channels_doc):
    by = {'inputs_f16': {}, 'inputs_f32': {}}
    for ch in channels_doc['channels']:
        by[ch['array']][int(ch['index'])] = ch['name']
    lay = {k: [v[i] for i in sorted(v)] for k, v in by.items()}
    for k, v in by.items():
        if sorted(v) != list(range(len(v))):
            raise ValueError('S3_CHANNELS.json %s indices are not 0 to n-1' % k)
    c7 = order_names(channels_doc, 'f6a_emission_arrays')
    f6 = order_names(channels_doc, 'f6_emission_arrays')
    out = []
    for y in sorted(arrays_doc['years']):
        fl = arrays_doc['years'][y]['files']
        for k in ('inputs_f16', 'inputs_f32', 'inputs_c7'):
            if k not in fl:
                raise ValueError('S3_ARRAYS.json years.%s lacks %s' % (y, k))
            out.append(('%s_%s' % (k, y), fl[k], c7 if k == 'inputs_c7' else lay[k]))
    for nm in sorted(arrays_doc.get('other', {})):
        if nm.startswith(('emis_true_', 'emis_clim_')) and nm.endswith('.npy'):
            out.append((nm[:-4], arrays_doc['other'][nm], f6))
    return out

def scan_array(path, names):
    a = np.load(path, mmap_mode='r')
    if a.ndim < 3 or a.shape[-3] != len(names):
        raise ValueError('%s: shape %s, %d channels named' % (base_name(path), list(a.shape), len(names)))
    lead = a.shape[:-3]
    nc = len(names)
    n = np.zeros(nc, np.int64)
    nf = np.zeros(nc, np.int64)
    fl = np.zeros(nc, np.int64)
    ng = np.zeros(nc, np.int64)
    for idx in np.ndindex(*lead):
        b = np.asarray(a[idx])
        fin = np.isfinite(b)
        n += b[0].size
        nf += np.count_nonzero(~fin, axis=(1, 2))
        with np.errstate(invalid='ignore'):
            fl += np.count_nonzero(fin & (b <= LIMIT32), axis=(1, 2))
            ng += np.count_nonzero(fin & (b < 0), axis=(1, 2))
    del a
    return {nm: {'n': int(n[k]), 'nonfinite': int(nf[k]), 'fill': int(fl[k]), 'negative': int(ng[k])} for k, nm in enumerate(names)}

def check_manifest(path, year):
    bad, cnt, seen = ([], {}, set())
    days = set()
    d = datetime.date(year, 1, 1)
    while d.year == year:
        days.add(d.strftime('%Y%m%d'))
        d += datetime.timedelta(days=1)
    with open(path, 'r', encoding='ascii', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        bad.append('no rows')
    need = ('path', 'kind', 'sector', 'date', 'species', 'p1')
    if rows and any((k not in rows[0] for k in need)):
        return (['the columns %s are not all there' % (need,)], cnt)
    for r in rows:
        k, s, dt = (r['kind'], r['sector'], r['date'])
        p = r['path'].replace('\\', '/')
        why = []
        if r['p1'] != 'on':
            why.append('p1 %r' % r['p1'])
        if dt not in days:
            why.append('date outside %d' % year)
        if k == 'merged':
            if s != '':
                why.append('a merged row with a sector')
            if p != 'emis_merged/emis_merged_%s.npz' % dt:
                why.append('merged path')
            if r['species'] != MERGED_SPECIES:
                why.append('merged species')
        elif k == 'pt':
            if s not in SECTORS:
                why.append('sector %r' % s)
            if p != 'pt_marine/%s/pt_%s_%s.npz' % (s, s, dt):
                why.append('pt path')
            if r['species'] != PT_SPECIES:
                why.append('pt species')
        else:
            why.append('kind %r' % k)
        key = (k, s, dt)
        if key in seen:
            why.append('(kind, sector, date) twice')
        seen.add(key)
        cnt['%s %s' % (k, s or '-')] = cnt.get('%s %s' % (k, s or '-'), 0) + 1
        if why:
            bad.append('%s: %s' % (p, '; '.join(why)))
    return (bad, cnt)

def corner_test():
    from s3 import s3common as S
    h, w = DOT_SHAPE
    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    lin = 3.0 + 2.0 * rr - 0.25 * cc
    got = S.dot_to_cell(lin)
    want = 3.0 + 2.0 * (np.arange(h - 1)[:, None] + 0.5) - 0.25 * (np.arange(w - 1)[None, :] + 0.5)
    ok_lin = got.shape == (h - 1, w - 1) and bool(np.allclose(got, want, rtol=0, atol=1e-12))
    rng = np.random.Generator(np.random.PCG64(20261025))
    f = rng.normal(size=(2, h, w)).astype(np.float32)
    got2 = S.dot_to_cell(f)
    picks = [(0, 0, 0), (1, h - 2, w - 2), (0, 137, 211), (1, 0, w - 2), (0, h - 2, 0)]
    ok_rand = got2.shape == (2, h - 1, w - 1) and got2.dtype == np.float64
    for k, i, j in picks:
        hand = (float(f[k, i, j]) + float(f[k, i + 1, j]) + float(f[k, i, j + 1]) + float(f[k, i + 1, j + 1])) / 4.0
        ok_rand = ok_rand and abs(float(got2[k, i, j]) - hand) <= 1e-12
    return {'linear_field_exact': ok_lin, 'random_field_five_cells': ok_rand, 'pass': bool(ok_lin and ok_rand), 'function': 's3\\s3common.py dot_to_cell'}

def run(config_dir, arrays_dir, manifest_dir, c7a_years=C7A_YEARS):
    t0 = time.time()
    ap = os.path.join(config_dir, 'S3_ARRAYS.json')
    cp = os.path.join(config_dir, 'S3_CHANNELS.json')
    arrays_doc, channels_doc = (jload(ap), jload(cp))
    rec = {'schema': 'pd17-pass-1', 'design': "the study design's read-only check of the inputs", 'made_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'S3_ARRAYS_sha256': sha256_file(ap), 'S3_CHANNELS_sha256': sha256_file(cp), 'rules': {'fill': 'a finite value at or below -9.0e36 (s3common NULL_LIMIT)', 'stop': 'any non-finite value or fill in an input array; any manifest row against (b); the four-corner test failing', 'record_only': 'negative values (count per channel)'}, 'arrays': {}, 'manifests': {}, 'reasons': []}
    for name, ent, names in array_list(arrays_doc, channels_doc):
        p = os.path.join(arrays_dir, base_name(ent['path']))
        row = {'file': base_name(ent['path']), 'bytes': None, 'sha256': None, 'sha256_as_S3_ARRAYS': False}
        rec['arrays'][name] = row
        if not os.path.isfile(p):
            rec['reasons'].append('%s absent' % row['file'])
            continue
        row['bytes'] = os.stat(p).st_size
        row['sha256'] = sha256_file(p)
        row['sha256_as_S3_ARRAYS'] = row['sha256'] == ent.get('sha256')
        if not row['sha256_as_S3_ARRAYS']:
            rec['reasons'].append('%s: sha256 differs from S3_ARRAYS.json' % row['file'])
            continue
        row['channels'] = scan_array(p, names)
        for c, v in row['channels'].items():
            if v['nonfinite'] or v['fill']:
                rec['reasons'].append('%s %s: %d non-finite, %d fills' % (row['file'], c, v['nonfinite'], v['fill']))
        say('scanned %s (%d bytes, sha16 %s): %d channels' % (row['file'], row['bytes'], row['sha256'][:16], len(names)))
    for y in c7a_years:
        p = os.path.join(manifest_dir, 'MANIFEST_%d.csv' % y)
        if not os.path.isfile(p):
            rec['manifests'][str(y)] = {'file': os.path.basename(p), 'present': False}
            rec['reasons'].append('MANIFEST_%d.csv absent' % y)
            continue
        bad, cnt = check_manifest(p, y)
        rec['manifests'][str(y)] = {'file': os.path.basename(p), 'present': True, 'bytes': os.stat(p).st_size, 'sha256': sha256_file(p), 'rows_by_kind_sector': cnt, 'problems': len(bad), 'first_problems': bad[:8]}
        for b in bad[:8]:
            rec['reasons'].append('MANIFEST_%d.csv %s' % (y, b))
        say('manifest %d: %d rows, %d problems' % (y, sum(cnt.values()), len(bad)))
    rec['four_corner_test'] = corner_test()
    if not rec['four_corner_test']['pass']:
        rec['reasons'].append("the four-corner mean's synthetic test fails")
    rec['verdict'] = 'FAIL' if rec['reasons'] else 'PASS'
    rec['seconds'] = round(time.time() - t0, 1)
    return rec

def write_record(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.partial', 'w', encoding='ascii', newline='\n') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(path + '.partial', path)
    return sha256_file(path)

def main(argv):
    o = {}
    for k in range(0, len(argv) - 1, 2):
        o[argv[k]] = argv[k + 1]
    need = ('--config-dir', '--arrays-dir', '--manifest-dir', '--out')
    if any((k not in o for k in need)) or len(argv) != 2 * len(need):
        print('usage: pd17_pass.py %s' % ' '.join(('%s D' % k for k in need)))
        return 1
    rec = run(o['--config-dir'], o['--arrays-dir'], o['--manifest-dir'])
    sha = write_record(o['--out'], rec)
    say('read-only check %s: %d arrays, %d manifests, %d reasons; record %s sha16 %s' % (rec['verdict'], len(rec['arrays']), len(rec['manifests']), len(rec['reasons']), os.path.basename(o['--out']), sha[:16]))
    for r in rec['reasons'][:12]:
        say('READ-ONLY CHECK REASON: ' + r)
    return 0 if rec['verdict'] == 'PASS' else 4
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
