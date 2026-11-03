"""Builds the expected set of downloaded files from the two download plans through the download code's own functions that list the items, then verifies every file on disk against its provenance row (sha256, size and checks) and writes the results as JSON, changing no data file."""
import collections
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
CODE = '/content/emul/run/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common
MIB = 1 << 20
OUT = os.path.join(common.W_TMP, 's2c')
LOG = os.path.join(OUT, 'verify_log.txt')
PLAN_R = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_RSIG.json')
PLAN_S = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_S3.json')
DL_STATUS = tuple((os.path.join(common.RUN_LOGS, n) for n in ('DL_STATUS_C5.json', 'DL_STATUS_RSIG.json')))
NE_DIR = os.path.join(common.NATURAL_EARTH, 'shapefiles', 'natural_earth')
NE_SETS = (('cultural', 'ne_10m_admin_0_boundary_lines_land'), ('cultural', 'ne_10m_admin_1_states_provinces'), ('cultural', 'ne_10m_admin_1_states_provinces_lakes'), ('cultural', 'ne_10m_populated_places'), ('cultural', 'ne_50m_admin_0_boundary_lines_land'), ('cultural', 'ne_50m_admin_0_countries'), ('physical', 'ne_10m_coastline'), ('physical', 'ne_10m_lakes'), ('physical', 'ne_50m_coastline'), ('physical', 'ne_50m_lakes'), ('physical', 'ne_50m_land'))
NE_EXTS = ('.cpg', '.dbf', '.prj', '.shp', '.shx')
REQ_RSIG = ('path', 'source', 'retrieved_utc', 'retrieved_local', 'http_status', 'bytes', 'sha256', 'version')
REQ_WORK = ('path', 'source', 'retrieved_utc', 'retrieved_local', 'bytes', 'sha256', 'version', 'member_sha256')
REQ_S3RAW = ('path', 'source', 'byte_range', 'retrieved_utc', 'retrieved_local', 'http_status', 'bytes', 'sha256', 'version', 'etag', 'last_modified')
REQ_EXTRACT = REQ_S3RAW + ('species', 'hours', 'member_sha256')
REQ_NE = ('path', 'source', 'retrieved_utc', 'retrieved_local', 'http_status', 'bytes', 'sha256', 'version', 'etag', 'last_modified')
HEX64 = re.compile('^[0-9a-f]{64}$')
GRID = (299, 459)

def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = '%s %s' % (common.now_pair()[1][:19], common.mask(msg))
    with open(LOG, 'ab') as f:
        f.write((line + '\n').encode('ascii', 'replace'))
    print(line, flush=True)

def absp(rel):
    return os.path.join(common.ROOT, rel)

def _init():
    common.set_temp()

def _npz_check(path, spec):
    import numpy as np
    out = {'problems': [], 'nonfinite': 0, 'negative': 0, 'zero_fields': 0}
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        need = list(spec['species']) + ['species', 'units', 'hours', 'date', 'member', 'tflag_date']
        if spec['kind'] == 'pt_marine':
            need += ['sector', 'n_stacks', 'n_summed', 'n_outside', 'base']
        miss = [k for k in need if k not in names]
        if miss:
            out['problems'].append('keys missing: %s' % ','.join(miss))
        for sp in spec['species']:
            if sp not in names:
                continue
            a = z[sp]
            if a.shape != GRID or a.dtype != np.float32:
                out['problems'].append('%s shape %s dtype %s' % (sp, a.shape, a.dtype))
                continue
            nf = int(np.count_nonzero(~np.isfinite(a)))
            ng = int(np.count_nonzero(a < 0))
            out['nonfinite'] += nf
            out['negative'] += ng
            if not np.any(a != 0):
                out['zero_fields'] += 1
        if 'date' in names and str(z['date']) != spec['date']:
            out['problems'].append('date %s, file name %s' % (str(z['date']), spec['date']))
        if 'tflag_date' in names:
            d = datetime.date(int(spec['date'][:4]), int(spec['date'][4:6]), int(spec['date'][6:]))
            want = d.year * 1000 + d.timetuple().tm_yday
            if int(z['tflag_date']) != want:
                out['problems'].append('tflag_date %s, expected %d' % (int(z['tflag_date']), want))
        if 'hours' in names and str(z['hours']) != '00-23 UTC':
            out['problems'].append('hours %s' % str(z['hours']))
        if 'species' in names and [str(x) for x in z['species']] != list(spec['species']):
            out['problems'].append('species %s' % [str(x) for x in z['species']])
        if spec['kind'] == 'pt_marine':
            if 'sector' in names and str(z['sector']) != spec['sector']:
                out['problems'].append('sector %s' % str(z['sector']))
            if 'n_stacks' in names and spec.get('n_stacks') is not None and (int(z['n_stacks']) != int(spec['n_stacks'])):
                out['problems'].append('n_stacks %d, plan %d' % (int(z['n_stacks']), int(spec['n_stacks'])))
            for k in ('n_stacks', 'n_summed', 'n_outside'):
                if k in names:
                    out[k] = int(z[k])
    return out

def work(task):
    path = task['abs']
    res = {'path': task['path'], 'exists': False}
    t0 = time.time()
    try:
        if not os.path.exists(path):
            return res
        res['exists'] = True
        res['size'] = os.stat(path).st_size
        h = hashlib.sha256()
        n = 0
        with open(path, 'rb') as f:
            while True:
                b = f.read(MIB)
                if not b:
                    break
                h.update(b)
                n += len(b)
        res['sha256'] = h.hexdigest()
        res['read_bytes'] = n
        if task.get('gz'):
            hm = hashlib.sha256()
            m = 0
            try:
                with gzip.open(path, 'rb') as g:
                    while True:
                        b = g.read(MIB)
                        if not b:
                            break
                        hm.update(b)
                        m += len(b)
                res['member_sha256'] = hm.hexdigest()
                res['member_bytes'] = m
            except Exception as e:
                res['gz_error'] = '%s: %s' % (type(e).__name__, common.mask(str(e))[:120])
        if task.get('npz'):
            try:
                res['npz'] = _npz_check(path, task['npz'])
            except Exception as e:
                res['npz'] = {'problems': ['open: %s: %s' % (type(e).__name__, common.mask(str(e))[:120])]}
    except Exception as e:
        res['error'] = '%s: %s' % (type(e).__name__, common.mask(str(e))[:160])
    res['seconds'] = round(time.time() - t0, 3)
    return res

def load_rows(folder):
    from choi_dl import prov
    return prov.load(folder)

def missing_fields(row, req):
    if row is None:
        return list(req)
    return [k for k in req if k not in row or str(row[k]).strip() == '']

def br_len(br):
    try:
        a, b = br.split('-')
        return int(b) - int(a) + 1
    except Exception:
        return None

def build_expected():
    from choi_dl import dl_rsig, dl_s3, rsig
    plan_r = json.load(open(PLAN_R, 'rb'))
    plan_s = json.load(open(PLAN_S, 'rb'))
    ent = []
    files = {}
    rows = {k: load_rows(v) for k, v in (('rsig', rsig.RAW_RSIG), ('work', rsig.WORK_RSIG), ('mcip', dl_s3.MCIP_DIR), ('s3meta', dl_s3.S3META), ('stackgroups', dl_s3.SG_DIR), ('emis_merged', dl_s3.EM_DIR), ('pt_marine', dl_s3.PT_DIR), ('ne', common.NATURAL_EARTH))}
    work_by_raw = {}
    for p, r in rows['work'].items():
        src = str(r.get('source', ''))
        m = re.match('^gzip decompression of (\\S+)', src)
        if m:
            work_by_raw[m.group(1)] = p

    def add_file(rel, kind, row, req, extra=None):
        t = {'path': rel, 'abs': absp(rel), 'kind': kind, 'row_missing': missing_fields(row, req), 'row_sha256': row.get('sha256') if row else None, 'row_bytes': int(row['bytes']) if row and str(row.get('bytes', '')).isdigit() else None, 'checks': []}
        if extra:
            t.update(extra)
        files[rel] = t
        return t
    items = dl_rsig.build(plan_r)
    held = dl_rsig.held_ids(plan_r, items)
    for it in items:
        fl = rsig.raw_files(it)
        url = rsig.url_for(it)
        e = {'kind': 'rsig', 'id': it['id'], 'series': it['key'], 'sid': it['sid'], 'steps': it['steps'], 'var': it['var'], 'files': [], 'held': it['id'] in held, 'folder': common.rel(rsig.RAW_RSIG)}
        for p in fl:
            rel = common.rel(p)
            r = rows['rsig'].get(rel)
            t = add_file(rel, 'rsig', r, REQ_RSIG, {'gz': p.endswith('.nc.gz'), 'item': it['id']})
            if r is not None:
                if r.get('source') != url:
                    t['checks'].append('source differs from url_for(item)')
                note = str(r.get('note', ''))
                t['note_ok'] = note.startswith('check ok')
                t['note'] = note[:120]
                if t['note_ok']:
                    mm = re.match('check ok: (\\S+) (.+) (\\d+)x(\\d+) (\\d+) steps', note)
                    if not mm:
                        t['checks'].append('note not parsed')
                    else:
                        if mm.group(1) != it['var']:
                            t['checks'].append('note var %s' % mm.group(1))
                        if int(mm.group(5)) != it['steps']:
                            t['checks'].append('note steps %s, expected %d' % (mm.group(5), it['steps']))
                if r.get('member_sha256'):
                    t['want_member'] = r['member_sha256']
                    t['want_member_from'] = 'row member_sha256'
                elif rel in work_by_raw:
                    wr = rows['work'][work_by_raw[rel]]
                    t['want_member'] = wr.get('sha256')
                    t['want_member_from'] = 'row of %s' % work_by_raw[rel]
                    t['want_member_bytes'] = wr.get('bytes')
                mn = re.search('in W\\\\tmp, (\\d+) bytes', str(r.get('member_note', '')))
                if mn:
                    t['want_member_bytes'] = mn.group(1)
            e['files'].append(rel)
        ent.append(e)
    test_ids = sorted((k for k in plan_r.get('request_seconds', {}) if k.startswith('TEST_')))
    for tid in test_ids:
        m = re.match('^(.*)_(\\d{8})(_try\\d+)?$', tid)
        folder = m.group(1) if m else tid
        rel = common.rel(os.path.join(rsig.RAW_RSIG, folder, tid + '.nc.gz'))
        r = rows['rsig'].get(rel)
        t = add_file(rel, 'rsig_test', r, REQ_RSIG, {'gz': True, 'item': tid})
        if r is not None:
            t['note_ok'] = str(r.get('note', '')).startswith('check ok')
            t['note'] = str(r.get('note', ''))[:120]
            if r.get('member_sha256'):
                t['want_member'] = r['member_sha256']
                t['want_member_from'] = 'row member_sha256'
            elif rel in work_by_raw:
                wr = rows['work'][work_by_raw[rel]]
                t['want_member'] = wr.get('sha256')
                t['want_member_from'] = 'row of %s' % work_by_raw[rel]
                t['want_member_bytes'] = wr.get('bytes')
        ent.append({'kind': 'rsig_test', 'id': tid, 'files': [rel], 'folder': common.rel(rsig.RAW_RSIG)})
    for p, r in rows['work'].items():
        t = add_file(p, 'rsig_work', r, REQ_WORK)
        src = str(r.get('source', ''))
        mm = re.match('^gzip decompression of (\\S+)', src)
        if not mm:
            t['checks'].append('source does not name its raw answer')
        else:
            rr = rows['rsig'].get(mm.group(1))
            if rr is None:
                t['checks'].append('its raw answer has no row')
            elif rr.get('sha256') != r.get('member_sha256'):
                t['checks'].append("member_sha256 is not its raw answer's sha256")
        ent.append({'kind': 'rsig_work', 'id': os.path.basename(p), 'files': [p], 'folder': common.rel(rsig.WORK_RSIG)})
    keys = plan_s['keys']
    s3items = dl_s3.build_items(plan_s)
    for part, iid, spec in s3items:
        path = dl_s3.item_path(part, spec)
        rel = common.rel(path)
        if part == 'mcip':
            r = rows['mcip'].get(rel)
            t = add_file(rel, 'mcip', r, REQ_S3RAW)
            key = spec['key']
            want_src = 's3://%s/%s (member %s)' % ('cmas-equates', key, spec['member'].lstrip('./'))
        elif part == 'stackgroups':
            r = rows['stackgroups'].get(rel)
            t = add_file(rel, 'stackgroups', r, REQ_S3RAW)
            key = spec['key']
            want_src = 's3://cmas-equates/%s' % key
            if r is not None and str(r.get('bytes')) != str(spec['size']):
                t['checks'].append('bytes %s, plan size %s' % (r.get('bytes'), spec['size']))
        else:
            r = rows[part].get(rel)
            sp = plan_s['emis_merged']['species'] if part == 'emis_merged' else plan_s['pt_marine']['species']
            npz = {'kind': part, 'species': sp, 'date': spec['date']}
            if part == 'pt_marine':
                npz['sector'] = spec['sector']
                npz['n_stacks'] = plan_s['pt_marine']['stack_counts'].get('%s_%s' % (spec['sector'], spec['year']))
            t = add_file(rel, part, r, REQ_EXTRACT, {'npz': npz})
            key = spec['key']
            want_src = 's3://cmas-equates/%s (member %s)' % (key, spec['name'])
            if r is not None:
                if r.get('species') != ' '.join(sp):
                    t['checks'].append('species %s' % r.get('species'))
                if r.get('hours') != '00-23 UTC':
                    t['checks'].append('hours %s' % r.get('hours'))
                if not HEX64.match(str(r.get('member_sha256', ''))):
                    t['checks'].append('member_sha256 not a sha256')
                bl = br_len(str(r.get('byte_range', '')))
                size = spec.get('size')
                if size is not None and bl != int(size):
                    t['checks'].append('byte range length %s, plan size %s' % (bl, size))
        if r is not None:
            if r.get('source') != want_src:
                t['checks'].append('source differs from the plan')
            pk = keys.get(key)
            if pk is None:
                t['checks'].append('the plan has no HEAD of the key')
            elif str(r.get('etag')) != str(pk['etag']):
                t['checks'].append("etag differs from the plan's HEAD")
        e = {'kind': part, 'id': iid, 'files': [rel], 'folder': common.rel(dl_s3.FOLDER[part])}
        if part in ('emis_merged', 'pt_marine'):
            e['date'] = spec['date']
            e['year'] = spec['date'][:4]
        if part == 'pt_marine':
            e['sector'] = spec['sector']
            e['member_size'] = spec.get('size')
        ent.append(e)
    for key in sorted(keys):
        if not key.endswith('.contents.txt'):
            continue
        path = os.path.join(dl_s3.S3META, key.split('/')[-1])
        rel = common.rel(path)
        r = rows['s3meta'].get(rel)
        t = add_file(rel, 's3meta', r, REQ_S3RAW)
        if r is not None:
            if r.get('source') != 's3://cmas-equates/%s' % key:
                t['checks'].append('source differs from the plan')
            if str(r.get('etag')) != str(keys[key]['etag']):
                t['checks'].append("etag differs from the plan's HEAD")
            if str(r.get('bytes')) != str(keys[key]['size']):
                t['checks'].append('bytes %s, plan size %s' % (r.get('bytes'), keys[key]['size']))
        ent.append({'kind': 's3meta', 'id': key.split('/')[-1], 'files': [rel], 'folder': common.rel(dl_s3.S3META)})
    for cat, name in NE_SETS:
        for ext in NE_EXTS:
            rel = common.rel(os.path.join(NE_DIR, cat, name + ext))
            r = rows['ne'].get(rel)
            t = add_file(rel, 'ne', r, REQ_NE)
            if r is not None and (not str(r.get('source', '')).startswith('https://naturalearth.s3.amazonaws.com/')):
                t['checks'].append('source not the Natural Earth bucket')
            ent.append({'kind': 'ne', 'id': name + ext, 'files': [rel], 'folder': common.rel(common.NATURAL_EARTH)})
    orphans = {'rows': [], 'files': [], 'backups': []}
    for k, rr in rows.items():
        for p in rr:
            if p not in files:
                orphans['rows'].append(p)
    for top in (common.RAW, common.EXTRACT, os.path.join(common.WORK, 'rsig'), common.NATURAL_EARTH):
        for dp, dn, fn in os.walk(top):
            for f in fn:
                if f in ('PROVENANCE.jsonl', 'PROVENANCE_ASCII.txt'):
                    continue
                rel = common.rel(os.path.join(dp, f))
                if '_Backup_PriorTo' in f:
                    orphans['backups'].append(rel)
                elif rel not in files:
                    orphans['files'].append(rel)
    return (plan_r, plan_s, ent, files, orphans, held)

def main(argv):
    common.set_temp()
    os.makedirs(OUT, exist_ok=True)
    workers = 4
    if '--workers' in argv:
        workers = int(argv[argv.index('--workers') + 1])
    only = None
    if '--only' in argv:
        with open(argv[argv.index('--only') + 1], 'rb') as f:
            only = set((x.strip() for x in f.read().decode('ascii').split('\n') if x.strip()))
    tag = ''
    if '--tag' in argv:
        tag = '_' + re.sub('[^A-Za-z0-9]', '', argv[argv.index('--tag') + 1])
    t0 = time.time()
    log('S2C VERIFY start; workers %d; only %s; tag %r' % (workers, len(only) if only else 'all', tag))
    plan_r, plan_s, ent, files, orphans, held = build_expected()
    by_kind = collections.Counter((e['kind'] for e in ent))
    log('EXPECTED entries: %s; files %d; held rsig %d; orphan rows %d, orphan files %d' % (', '.join(('%s %d' % kv for kv in sorted(by_kind.items()))), len(files), len(held), len(orphans['rows']), len(orphans['files'])))
    with open(os.path.join(OUT, 'expected%s.json' % tag), 'wb') as f:
        f.write(json.dumps({'entries': ent, 'orphans': orphans, 'held': held}, sort_keys=True, indent=0).encode('ascii'))
    for dl_status in DL_STATUS:
        try:
            with open(dl_status, 'rb') as f:
                st = json.loads(f.read().decode('ascii', 'replace'))
            bad = {}
            for q, v in st.get('queues', {}).items():
                if v.get('failed') or v.get('refused'):
                    bad[q] = {'failed': v.get('failed'), 'refused': v.get('refused')}
            log('%s failed or refused: %s' % (os.path.basename(dl_status), json.dumps(bad, sort_keys=True)))
        except Exception as e:
            log('%s not read: %s' % (os.path.basename(dl_status), type(e).__name__))
    tasks = [t for t in files.values() if only is None or t['path'] in only]
    tasks.sort(key=lambda t: -(t.get('row_bytes') or 0))
    results = {}
    done = 0
    nbytes = 0
    with mp.get_context('spawn').Pool(workers, initializer=_init) as pool:
        for res in pool.imap_unordered(work, tasks, chunksize=1):
            results[res['path']] = res
            done += 1
            nbytes += res.get('size', 0) or 0
            if done % 250 == 0 or done == len(tasks):
                log('hashed %d of %d files, %.2f GB, %.0f s' % (done, len(tasks), nbytes / 1000000000.0, time.time() - t0))
    for rel, t in files.items():
        if rel not in results:
            continue
        r = results[rel]
        prob = list(t['checks'])
        if not r.get('exists'):
            prob.append('file missing')
        else:
            if r.get('error'):
                prob.append('read error: %s' % r['error'])
            if t['row_missing']:
                prob.append('row incomplete: %s' % ','.join(t['row_missing']))
            if t['row_bytes'] is not None and r.get('size') != t['row_bytes']:
                prob.append('size %s, row %s' % (r.get('size'), t['row_bytes']))
            if t['row_sha256'] and r.get('sha256') != t['row_sha256']:
                prob.append('sha256 differs from the row')
            if t.get('gz'):
                if r.get('gz_error'):
                    prob.append('gzip: %s' % r['gz_error'])
                elif t.get('want_member'):
                    if r.get('member_sha256') != t['want_member']:
                        prob.append('decompressed sha256 differs (%s)' % t.get('want_member_from'))
                    wb = t.get('want_member_bytes')
                    if wb is not None and str(r.get('member_bytes')) != str(wb):
                        prob.append('decompressed bytes %s, row %s' % (r.get('member_bytes'), wb))
            if r.get('npz'):
                prob.extend(('npz: ' + x for x in r['npz'].get('problems', [])))
        t['problems'] = prob
        t['verified'] = not prob
        t['sha256'] = r.get('sha256')
        t['size'] = r.get('size')
        t['member_sha256_now'] = r.get('member_sha256')
        t['member_bytes_now'] = r.get('member_bytes')
        t['npz'] = r.get('npz')
        t['seconds'] = r.get('seconds')
    for e in ent:
        if e['kind'] == 'rsig':
            fs = [files[p] for p in e['files']]
            use = None
            for t in fs:
                if t.get('note_ok') and t.get('verified'):
                    use = t['path']
                    break
            e['use'] = use
            e['present'] = bool(fs)
            e['verified'] = use is not None
            e['tries'] = len(fs)
            e['problems'] = [] if use else ['no raw answer'] if not fs else ['no try with a check-ok row that re-verifies']
            for t in fs:
                if t['path'] != use and t.get('problems'):
                    e.setdefault('other_tries', []).append({'path': t['path'], 'problems': t['problems'], 'note': t.get('note')})
        else:
            t = files[e['files'][0]]
            e['present'] = bool(results.get(t['path'], {}).get('exists'))
            e['verified'] = bool(t.get('verified'))
            e['problems'] = t.get('problems', ['not hashed'])
    out = {'made': common.now_pair(), 'workers': workers, 'seconds': round(time.time() - t0, 1), 'entries': ent, 'files': files, 'orphans': orphans, 'held': held}
    with open(os.path.join(OUT, 'verify_results%s.json' % tag), 'wb') as f:
        f.write(json.dumps(out, sort_keys=True, indent=0).encode('ascii'))
    summ = collections.OrderedDict()
    for e in ent:
        s = summ.setdefault(e['kind'], {'expected': 0, 'present': 0, 'verified': 0})
        s['expected'] += 1
        s['present'] += int(bool(e['present']))
        s['verified'] += int(bool(e['verified']))
    for k, s in summ.items():
        log('SUMMARY %s: expected %d, present %d, verified %d' % (k, s['expected'], s['present'], s['verified']))
    nf = collections.Counter((t['kind'] for t in files.values()))
    nv = collections.Counter((t['kind'] for t in files.values() if t.get('verified')))
    for k in sorted(nf):
        log('FILES %s: %d, verified %d' % (k, nf[k], nv[k]))
    pc = collections.Counter()
    for t in files.values():
        for x in t.get('problems', []):
            pc[t['kind'], ' '.join(x.split()[:4])] += 1
    for (k, x), n in sorted(pc.items()):
        log('PROBLEM %s: %r x%d' % (k, x, n))
    shown = 0
    for e in ent:
        if not e['verified'] or e.get('other_tries'):
            shown += 1
            if shown <= 60:
                log('ENTRY %s %s: %s; other tries %s' % (e['kind'], e['id'], '; '.join(e['problems']) or 'verified', json.dumps(e.get('other_tries', []))[:300]))
    log('ENTRIES not verified or with other tries: %d' % shown)
    ex = collections.Counter()
    for t in files.values():
        if t.get('npz'):
            ex['nonfinite'] += t['npz'].get('nonfinite', 0) or 0
            ex['negative'] += t['npz'].get('negative', 0) or 0
            ex['zero_fields'] += t['npz'].get('zero_fields', 0) or 0
            ex['npz_files'] += 1
    log('NPZ: %s' % json.dumps(dict(ex), sort_keys=True))
    log('ORPHANS rows %d: %s' % (len(orphans['rows']), '; '.join(orphans['rows'][:20])))
    log('ORPHANS files %d: %s' % (len(orphans['files']), '; '.join(orphans['files'][:20])))
    log('BACKUPS beside data files %d: %s' % (len(orphans['backups']), '; '.join(orphans['backups'][:20])))
    log('S2C VERIFY end; %.0f s' % (time.time() - t0))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
