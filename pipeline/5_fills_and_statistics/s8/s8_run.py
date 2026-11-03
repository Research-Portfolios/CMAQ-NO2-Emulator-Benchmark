"""Fills the emission channels of one year of 2015, 2016 or 2018 in its float32 input array: checks the disk, the extracts and the array before writing, fills each day whole from the extracts, and verifies the result, each step writing its JSON result."""
import collections
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
sys.path.insert(0, '/content/emul/run/code')
sys.path.insert(0, '/content/emul/run/code/s8')
import numpy as np
import s8common as C
from s3 import s3_build as B
from s3 import s3common as S
from s3p2 import p2_roundtrip as RT
GB = 1000000000.0
STOP_FRACTION = 0.001

def log(msg):
    print('%s %s' % (C.now(), S.common.mask(msg)), flush=True)

def wjson(name, y, obj):
    p = C.paths(y)['work'](name)
    info = C.write_json(p, obj)
    log('written %s (%d bytes, sha16 %s)' % (info['path'], info['bytes'], info['sha256'][:16]))
    return info

def year_days(y):
    return [S.ymd(d) for d in S.days_of_year(int(y))]

def add_report(y, lines):
    S.append_record(C.paths(y)['report'], lines)

def cmd_disk(y):
    n = len(year_days(y))
    ne = len(C.EMIS)
    cfg = {k: os.stat(os.path.join(C.CFG_P2, k)).st_size for k in ('S3_ARRAYS.json', 'S3_CHANNELS.json')}
    need = collections.OrderedDict()
    need['inputs_f32 completed in place'] = 0
    need['smallk_s8 (5 x K float32, K 2000000)'] = ne * B.K_SMALL * 4 + 8192
    need['daystats_s8 (2 x n x 5 float64 and keys)'] = 2 * n * ne * 8 + n + n * 32 + ne * 12 + 8192
    need['json work files (10 x 2 MB, an upper bound)'] = 10 * 2000000
    need['configs: backups, new versions, the list'] = 2 * (cfg['S3_ARRAYS.json'] + cfg['S3_CHANNELS.json']) + 65536 + 8192
    need['records (report, input-check record, marker; 2 MB)'] = 2000000
    need['scratch logs (20 MB, an upper bound)'] = 20000000
    total = int(sum(need.values()))
    free = shutil.disk_usage(C.W).free
    asked = total + 30 * GB
    met = free >= asked
    out = {'made': C.now(), 'year': int(y), 'rule': "at least 30 GB more than the stage's remaining need; else stop", 'gb': 'decimal (1e9 bytes)', 'need_bytes': need, 'need_total_bytes': total, 'need_gb': total / GB, 'free_bytes': int(free), 'free_gb': free / GB, 'asked_gb': asked / GB, 'verdict': 'MET' if met else 'NOT MET'}
    wjson('disk_s8_%s.json', y, out)
    log('disk: need %.3f GB, asked %.3f GB, work disk free %.3f GB: %s' % (total / GB, asked / GB, free / GB, out['verdict']))
    return 0 if met else 2

def arr_counts(a, mask):
    a = np.asarray(a)
    fin = np.isfinite(a)
    fill = fin & (a <= C.NULL_LIMIT)
    g = a[fin & ~fill]
    return {'nonfinite': int((~fin).sum()), 'nonfinite_mask': int((~fin & mask).sum()), 'fill': int(fill.sum()), 'fill_mask': int((fill & mask).sum()), 'negative': int((g < 0).sum()), 'negative_mask': int(((a < 0) & fin & ~fill & mask).sum()), 'zero': int((a == 0).sum()), 'all_zero': bool((a == 0).all()), 'min': float(g.min()) if g.size else None, 'max': float(g.max()) if g.size else None}

def k6_month_ok(dates):
    if len(set(dates)) != len(dates):
        return False
    n = len(dates)
    if 4 <= n <= 7:
        return True
    if n == 8:
        rest = sorted(dates)[4:]
        return all((d[4:6] == '12' and int(d[6:]) >= 20 or (d[4:6] == '01' and int(d[6:]) <= 2) for d in rest))
    return False

def cmd_d4(y):
    t0 = time.time()
    inp = C.Inputs(y)
    pre = inp.preconditions()
    log('preconditions: %s' % ('MET' if pre['met'] else 'NOT MET'))
    if not pre['met']:
        wjson('d4_s8_%s.json', y, {'made': C.now(), 'preconditions': pre, 'verdict': 'STOP'})
        return 2
    p = C.paths(y)
    mask = np.load(p['mask'])
    arrays = C.jload(os.path.join(C.CFG_P2, 'S3_ARRAYS.json'))
    mask_sha = C.sha256_file(p['mask'])
    mask_ok = mask_sha == arrays['other']['mask_conus_land.npy']['sha256']
    days = year_days(y)
    hol = C.holidays(y)
    res = {'made': C.now(), 'year': int(y), 'preconditions': pre, 'mask': {'path': S.common.rel(p['mask']), 'sha256': mask_sha, 'equal_to_S3_ARRAYS': mask_ok, 'cells': int(mask.sum())}, 'holidays': hol, 'plan_holidays_equal_s2a': C.plan_holidays_check(), 'manifest': {'path': S.common.rel(p['manifest']), 'sha256': C.sha256_file(p['manifest']), 'rows': len(inp.rows)}, 'sums': {'path': S.common.rel(p['sums']), 'sha256': C.sha256_file(p['sums']), 'lines': len(inp.sums)}, 'groups': {}, 'problems': []}
    groups = collections.OrderedDict()
    groups['merged'] = [r for r in inp.rows if r['kind'] == 'merged']
    for s in C.DAILY + C.REP:
        groups[s] = [r for r in inp.rows if r['kind'] == 'pt' and r['sector'] == s]
    other = [r for r in inp.rows if r['kind'] not in ('merged', 'pt') or (r['kind'] == 'pt' and r['sector'] not in C.DAILY + C.REP)]
    res['rows_outside_the_six_sectors_and_merged'] = len(other)
    if other:
        res['problems'].append('manifest rows outside merged and the six sectors: %d' % len(other))
    for g, rows in groups.items():
        dates = [r['date'] for r in rows]
        cnt = collections.Counter(dates)
        rec = {'files': len(rows), 'dates': len(cnt), 'duplicates': sorted((d for d, k in cnt.items() if k > 1)), 'not_this_year': sorted((d for d in cnt if d[:4] != str(y))), 'problems': [], 'species': {}, 'units': {}, 'hours': {}, 'keys': {}, 'stacks': {}, 'summed': {}, 'outside_total': 0, 'outside_files': 0, 'counts': {}, 'masked_missing': {}}
        if g == 'merged' or g in C.DAILY:
            miss = [d for d in days if d not in cnt]
            rec['missing_days'] = miss
            if miss:
                res['problems'].append('%s: %d days missing' % (g, len(miss)))
        else:
            mm = {}
            for m in range(1, 13):
                ds = sorted((d for d in dates if int(d[4:6]) == m))
                mm['%02d' % m] = {'n': len(ds), 'dates': ds, 'k6_ok': k6_month_ok(ds)}
                if not k6_month_ok(ds):
                    res['problems'].append('%s %d-%02d: %d members' % (g, int(y), m, len(ds)))
            rec['members_a_month'] = mm
            rec['members_on_holidays'] = sorted((d for d in dates if d in hol))
        if rec['duplicates'] or rec['not_this_year']:
            res['problems'].append('%s: duplicates or other years' % g)
        names = C.MERGED_SPECIES if g == 'merged' else C.PT_SPECIES + ('P1_NOX',)
        tot = {k: collections.Counter() for k in names}
        allzero = collections.Counter()
        for r in sorted(rows, key=lambda x: x['date']):
            d = r['date']
            rel = inp.rel_merged(d) if g == 'merged' else inp.rel_pt(g, d)
            if C.fwd(r['path']) != rel:
                rec['problems'].append('path form %s' % r['path'])
                continue
            z, _r = inp.read(rel)
            bad = C.check_merged(z, r, d, y) if g == 'merged' else C.check_pt(z, r, g, d, y)
            if bad:
                rec['problems'].append('%s: %s' % (d, bad))
            ks = ' '.join(sorted(z.files))
            rec['keys'][ks] = rec['keys'].get(ks, 0) + 1
            sp = ' '.join((str(x) for x in z['species'].tolist()))
            un = ' '.join((str(x) for x in z['units'].tolist()))
            rec['species'][sp] = rec['species'].get(sp, 0) + 1
            rec['units'][un] = rec['units'].get(un, 0) + 1
            hr = C.as_str(z, 'hours')
            rec['hours'][hr] = rec['hours'].get(hr, 0) + 1
            if g != 'merged':
                for k, col in (('n_stacks', 'stacks'), ('n_summed', 'summed')):
                    v = str(int(z[k].tolist()))
                    rec[col][v] = rec[col].get(v, 0) + 1
                    if str(r[k]) != v:
                        rec['problems'].append('%s: %s differs from the manifest' % (d, k))
                no = int(z['n_outside'].tolist())
                rec['outside_total'] += no
                rec['outside_files'] += int(no > 0)
                if str(r['n_outside']) != str(no):
                    rec['problems'].append('%s: n_outside differs' % d)
            for k in names:
                c = arr_counts(z[k], mask)
                for kk in ('nonfinite', 'nonfinite_mask', 'fill', 'fill_mask', 'negative', 'negative_mask', 'zero'):
                    tot[k][kk] += c[kk]
                if c['all_zero']:
                    allzero[k] += 1
                for kk in ('min', 'max'):
                    v = c[kk]
                    if v is None:
                        continue
                    cur = tot[k].get(kk + '_v')
                    if cur is None or (v < cur if kk == 'min' else v > cur):
                        tot[k][kk + '_v'] = v
        for k in names:
            rec['counts'][k] = dict(tot[k])
            rec['counts'][k]['all_zero_files'] = int(allzero[k])
            miss = tot[k]['nonfinite_mask'] + tot[k]['fill_mask']
            rec['masked_missing'][k] = int(miss)
            if tot[k]['nonfinite'] or tot[k]['fill']:
                res['problems'].append('%s %s: non-finite %d, fill %d' % (g, k, tot[k]['nonfinite'], tot[k]['fill']))
        if rec['problems']:
            res['problems'].append('%s: %d file problems' % (g, len(rec['problems'])))
        res['groups'][g] = rec
        log('d4 %s: %d files, %d dates, problems %d' % (g, rec['files'], rec['dates'], len(rec['problems'])))
    try:
        mo = C.Months(inp, hol)
        res['classes'] = mo.classes
        res['holiday_members_left_out'] = mo.left_out
        res['fallbacks'] = mo.fallbacks
    except ValueError as e:
        res['problems'].append('weekday classes: %s' % e)
    res['reads'] = int(sum(inp.reads.values()))
    res['files_verified'] = len(inp.verified)
    if not mask_ok:
        res['problems'].append("the mask differs from S3_ARRAYS.json's")
    res['verdict'] = 'PASS' if not res['problems'] else 'STOP'
    res['seconds'] = round(time.time() - t0, 1)
    res['memory_gb_end'] = round(C.mem_gb(), 3)
    wjson('d4_s8_%s.json', y, res)
    log('d4: %d files verified, %d reads; problems %d: %s' % (res['files_verified'], res['reads'], len(res['problems']), res['verdict']))
    return 0 if res['verdict'] == 'PASS' else 2

def partial_shas(y):
    name = 'inputs_f32_%s.npy' % y
    lst = C.jload(os.path.join(C.CFG_P2, 'S3C_LIST.json'))
    row = [e for e in lst['files'] if e['path'].replace('\\', '/').split('/')[-1] == name]
    arrays = C.jload(os.path.join(C.CFG_P2, 'S3_ARRAYS.json'))
    f32 = arrays['years'][str(y)]['files']['inputs_f32']
    lp = os.path.join(C.LOGS, 'B2_S3C_LOG.json')
    absent = not os.path.isfile(lp)
    if absent:
        log('pre: %s absent (no copy step on this machine): the partial array is tested against S3C_LIST.json and S3_ARRAYS.json only' % S.common.rel(lp))
    slog = {'sent': []} if absent else C.jload(lp)
    sent = [e for e in slog['sent'] if e['name'] == 'inputs_f32_%s_PARTIAL.npy' % y]
    return {'S3C_LIST.json': row[0]['sha256'] if len(row) == 1 else None, 'S3C_LIST_partial': bool(row and row[0].get('partial')), 'S3_ARRAYS.json': f32.get('sha256'), 'S3_ARRAYS_partial': f32.get('partial'), 'B2_S3C_LOG.json': sent[0].get('sha256') if len(sent) == 1 else None, 'B2_S3C_LOG_absent': absent, 'f32': f32}

def recorded_states(y):
    p = C.paths(y)['report']
    out = []
    if not os.path.isfile(p):
        return out
    with open(p, 'r', encoding='ascii') as f:
        lines = f.read().split('\n')
    for i, line in enumerate(lines):
        if 'ARRAY STATE' in line and i + 1 < len(lines):
            v = lines[i + 1].strip()
            if len(v) == 64 and all((c in '0123456789abcdef' for c in v)):
                out.append(v)
    return out

def state_of(y, sha):
    ps = partial_shas(y)
    want = ps['S3C_LIST.json']
    third = want if ps['B2_S3C_LOG_absent'] else ps['B2_S3C_LOG.json']
    if want and sha == want == ps['S3_ARRAYS.json'] == third:
        return ('FRESH', ps)
    if sha in recorded_states(y):
        return ('RESUME', ps)
    return ('STOP', ps)

def other_hashes(arr, oi):
    out = []
    for i in range(arr.shape[0]):
        b = np.ascontiguousarray(arr[i, oi]).tobytes()
        out.append(hashlib.sha256(b).hexdigest())
    return out

def header_hash(path, offset):
    with open(path, 'rb') as f:
        b = f.read(offset)
    return (hashlib.sha256(b).hexdigest(), len(b))

def cmd_pre(y):
    t0 = time.time()
    p = C.paths(y)
    path = p['array']
    st = os.stat(path)
    th = time.time()
    sha = C.sha256_file(path)
    hash_s = round(time.time() - th, 1)
    state, ps = state_of(y, sha)
    f32 = ps['f32']
    idx, others = C.f32_index()
    ei = [idx[c] for c in C.EMIS]
    oi = [idx[c] for c in others]
    arr = np.load(path, mmap_mode='r')
    shape_ok = list(arr.shape) == f32['shape'] and str(arr.dtype) == f32['dtype'] and (arr.shape[0] == len(year_days(y))) and (arr.shape[1] == len(idx)) and (tuple(arr.shape[2:]) == C.G)
    unf_ok = f32.get('unfilled_channels') == list(C.EMIS) and f32.get('unfilled_indices') == ei
    hh, hn = header_hash(path, arr.offset)
    nan_emis = []
    nan_other = []
    for i in range(arr.shape[0]):
        nan_emis.append([int(np.isnan(arr[i, j]).sum()) for j in ei])
        nan_other.append([int(np.isnan(arr[i, j]).sum()) for j in oi])
    full = int(np.prod(C.G))
    all_nan_days = sum((1 for v in nan_emis if all((x == full for x in v))))
    no_nan_days = sum((1 for v in nan_emis if all((x == 0 for x in v))))
    oh = other_hashes(arr, oi)
    del arr
    res = {'made': C.now(), 'year': int(y), 'array': S.common.rel(path), 'bytes': int(st.st_size), 'sha256': sha, 'hash_seconds': hash_s, 'state': state, 'partial': {k: v for k, v in ps.items() if k != 'f32'}, 'recorded_states_in_report': recorded_states(y), 'shape': list(f32['shape']), 'dtype': f32['dtype'], 'shape_ok': shape_ok, 'unfilled_entry_ok': unf_ok, 'index': idx, 'emis_indices': ei, 'other_channels': others, 'other_indices': oi, 'header_bytes': hn, 'header_sha256': hh, 'nan_emis_per_day': nan_emis, 'nan_other_per_day': nan_other, 'days_all_nan': all_nan_days, 'days_no_nan': no_nan_days, 'other_hashes_per_day': oh, 'seconds': round(time.time() - t0, 1)}
    stop = state == 'STOP' or not shape_ok or (not unf_ok) or any((x for v in nan_other for x in v))
    if state == 'FRESH' and all_nan_days != len(nan_emis):
        stop = True
    res['verdict'] = 'STOP' if stop else state
    wjson('pre_s8_%s.json', y, res)
    log('pre: sha256 %s (%.1f s), state %s; shape ok %s; days all NaN %d, no NaN %d; other NaN %d; verdict %s' % (sha[:16], hash_s, state, shape_ok, all_nan_days, no_nan_days, sum((x for v in nan_other for x in v)), res['verdict']))
    return 2 if stop else 0

def cmd_fill(y):
    t0 = time.time()
    p = C.paths(y)
    path = p['array']
    d4 = C.jload(p['work']('d4_s8_%s.json'))
    pre = C.jload(p['work']('pre_s8_%s.json'))
    if d4.get('verdict') != 'PASS' or pre.get('verdict') not in ('FRESH', 'RESUME'):
        log('fill: STOP (d4 %s, pre %s)' % (d4.get('verdict'), pre.get('verdict')))
        return 2
    sha0 = C.sha256_file(path)
    state, _ps = state_of(y, sha0)
    log("fill: the array's sha256 before the first write %s: %s" % (sha0[:16], state))
    if state == 'STOP':
        return 2
    idx, others = C.f32_index()
    ei = [idx[c] for c in C.EMIS]
    inp = C.Inputs(y)
    hol = C.holidays(y)
    mo = C.Months(inp, hol)
    days = year_days(y)
    prog = {'made': C.now(), 'year': int(y), 'start_sha256': sha0, 'start_state': state, 'months': {}, 'written': 0, 'done_earlier_equal': 0, 'partial_days_rewritten': [], 'stop': None}
    arr = np.load(path, mmap_mode='r+')
    try:
        for m in range(1, 13):
            mem = B.wait_memory()
            cm = mo.means(m)
            nw = ns = 0
            for i, d in enumerate(days):
                if int(d[4:6]) != m:
                    continue
                f, _sp1, _parts = C.day_fields(inp, d, cm)
                v = f.astype(np.float32)
                if not np.isfinite(v).all():
                    prog['stop'] = '%s: a non-finite value formed' % d
                    raise RuntimeError(prog['stop'])
                cur = np.stack([np.asarray(arr[i, j]) for j in ei])
                isn = np.isnan(cur)
                if not isn.any():
                    if np.array_equal(cur.view(np.uint32), v.view(np.uint32)):
                        ns += 1
                        continue
                    prog['stop'] = '%s: a done day differs' % d
                    raise RuntimeError(prog['stop'])
                if not isn.all():
                    prog['partial_days_rewritten'].append(d)
                for k, j in enumerate(ei):
                    arr[i, j] = v[k]
                nw += 1
            arr.flush()
            th = time.time()
            sha = C.sha256_file(path)
            prog['months']['%02d' % m] = {'written': nw, 'done_earlier_equal': ns, 'memory_gb_start': round(mem, 3), 'memory_gb_end': round(C.mem_gb(), 3), 'sha256_after_flush': sha, 'hash_seconds': round(time.time() - th, 1), 'at': C.now()}
            prog['written'] += nw
            prog['done_earlier_equal'] += ns
            add_report(y, ['  ARRAY STATE after %s-%02d (fill, flushed, %s):' % (y, m, C.now()[11:19]), '    ' + sha])
            wjson('fill_s8_%s.json', y, prog)
            log('fill %s-%02d: written %d, done earlier %d; memory %.2f GB; sha256 %s' % (y, m, nw, ns, C.mem_gb(), sha[:16]))
    except RuntimeError as e:
        arr.flush()
        del arr
        prog['seconds'] = round(time.time() - t0, 1)
        wjson('fill_s8_%s.json', y, prog)
        log('fill: STOP: %s' % e)
        return 2
    del arr
    prog['end'] = C.now()
    prog['seconds'] = round(time.time() - t0, 1)
    prog['reads'] = int(sum(inp.reads.values()))
    prog['holiday_members_left_out'] = mo.left_out
    prog['fallbacks'] = mo.fallbacks
    wjson('fill_s8_%s.json', y, prog)
    log('fill done: written %d, done earlier %d, partial rewritten %d; %.0f s' % (prog['written'], prog['done_earlier_equal'], len(prog['partial_days_rewritten']), prog['seconds']))
    return 0

def cmd_after(y):
    t0 = time.time()
    p = C.paths(y)
    path = p['array']
    pre = C.jload(p['work']('pre_s8_%s.json'))
    idx, others = C.f32_index()
    ei = [idx[c] for c in C.EMIS]
    oi = [idx[c] for c in others]
    st = os.stat(path)
    arr = np.load(path, mmap_mode='r')
    hh, hn = header_hash(path, arr.offset)
    oh = other_hashes(arr, oi)
    nan_left = []
    for i in range(arr.shape[0]):
        nan_left.append([int(np.isnan(arr[i, j]).sum()) for j in ei])
    del arr
    sha = C.sha256_file(path)
    same = [a == b for a, b in zip(oh, pre['other_hashes_per_day'])]
    res = {'made': C.now(), 'year': int(y), 'array': S.common.rel(path), 'bytes': int(st.st_size), 'bytes_equal_before': int(st.st_size) == pre['bytes'], 'sha256': sha, 'sha256_before': pre['sha256'], 'header_sha256': hh, 'header_equal_before': hh == pre['header_sha256'] and hn == pre['header_bytes'], 'other_hashes_per_day': oh, 'other_days_equal': int(sum(same)), 'other_days_differ': [i for i, s in enumerate(same) if not s], 'nan_left_per_day': nan_left, 'nan_left_total': int(sum((sum(v) for v in nan_left))), 'seconds': round(time.time() - t0, 1)}
    ok = res['bytes_equal_before'] and res['header_equal_before'] and (res['other_days_equal'] == len(oh) == len(pre['other_hashes_per_day'])) and (res['nan_left_total'] == 0)
    res['verdict'] = 'PASS' if ok else 'FAIL'
    wjson('after_s8_%s.json', y, res)
    log('after: sha256 %s, bytes %d; other indices equal on %d of %d days; header equal %s; NaN left %d: %s' % (sha[:16], st.st_size, res['other_days_equal'], len(oh), res['header_equal_before'], res['nan_left_total'], res['verdict']))
    return 0 if ok else 2

def savez_partial(path, **kw):
    part = path + '.partial'
    with open(part, 'wb') as f:
        np.savez(f, **kw)
    os.replace(part, path)
    return S.file_info(path)

def cmd_verify(y):
    t0 = time.time()
    start = C.now()
    p = C.paths(y)
    path = p['array']
    idx, _others = C.f32_index()
    ei = [idx[c] for c in C.EMIS]
    inp = C.Inputs(y)
    hol = C.holidays(y)
    mo = C.Months(inp, hol)
    days = year_days(y)
    n = len(days)
    mask = np.load(p['mask'])
    mcells = int(mask.sum())
    arr = np.load(path, mmap_mode='r')
    rt = {c: RT.ChanYear('float32') for c in C.EMIS}
    r16 = {c: RT.ChanYear('float16') for c in C.EMIS}
    small = {c: B.SmallK(B.K_SMALL) for c in C.EMIS}
    daymin = np.full((n, len(C.EMIS)), np.nan)
    daymax = np.full((n, len(C.EMIS)), np.nan)
    isval = np.array([S.is_validation(d) for d in S.days_of_year(int(y))])
    cnt = {c: collections.Counter() for c in C.EMIS}
    ext = {c: {'min': None, 'max': None} for c in C.EMIS}
    parts_names = ['merged'] + list(C.DAILY) + list(C.REP)
    nox = {q: {'mask': 0.0, 'grid': 0.0} for q in parts_names + ['total']}
    sp_days = []
    sp_tot = collections.Counter()
    sp_max = 0.0
    mism = []
    mem_min = C.mem_gb()
    for pas in ('a', 'b'):
        for m in range(1, 13):
            mem_min = min(mem_min, B.wait_memory())
            cm = mo.means(m)
            for i, d in enumerate(days):
                if int(d[4:6]) != m:
                    continue
                f, sp1, parts = C.day_fields(inp, d, cm)
                if pas == 'b':
                    for k, c in enumerate(C.EMIS):
                        rt[c].pass_b(f[k])
                        r16[c].pass_b(f[k])
                    continue
                v = f.astype(np.float32)
                stored = np.stack([np.asarray(arr[i, j]) for j in ei])
                if not np.array_equal(stored.view(np.uint32), v.view(np.uint32)):
                    mism.append(d)
                for k, c in enumerate(C.EMIS):
                    a = f[k]
                    fin = np.isfinite(a)
                    fill = fin & (a <= C.NULL_LIMIT)
                    cnt[c]['n'] += a.size
                    cnt[c]['n_mask'] += mcells
                    cnt[c]['nonfinite'] += int((~fin).sum())
                    cnt[c]['nonfinite_mask'] += int((~fin & mask).sum())
                    cnt[c]['fill'] += int(fill.sum())
                    cnt[c]['fill_mask'] += int((fill & mask).sum())
                    cnt[c]['negative'] += int((a < 0).sum())
                    cnt[c]['negative_mask'] += int(((a < 0) & mask).sum())
                    cnt[c]['zero'] += int((a == 0).sum())
                    s32 = stored[k]
                    cnt[c]['stored_nonfinite'] += int((~np.isfinite(s32)).sum())
                    cnt[c]['stored_negative'] += int((s32 < 0).sum())
                    lo, hi = (float(s32.min()), float(s32.max()))
                    daymin[i, k] = lo
                    daymax[i, k] = hi
                    if ext[c]['min'] is None or lo < ext[c]['min']:
                        ext[c]['min'] = lo
                    if ext[c]['max'] is None or hi > ext[c]['max']:
                        ext[c]['max'] = hi
                    if not isval[i]:
                        small[c].add(s32)
                    rt[c].pass_a(a)
                    r16[c].pass_a(a)
                parts['total'] = parts['merged'] + f[4]
                for q, a in parts.items():
                    nox[q]['mask'] += float(a[mask].sum())
                    nox[q]['grid'] += float(a.sum())
                s32 = sp1.astype(np.float32)
                stp = C.steps32(s32, v[4])
                nd = int((s32 != v[4]).sum())
                mx = float(stp.max())
                sp_days.append({'date': d, 'differ': nd, 'max_steps': mx})
                sp_tot['cells'] += int(s32.size)
                sp_tot['differ'] += nd
                sp_tot['equal'] += int(s32.size) - nd
                sp_tot['one_step'] += int(((stp > 0) & (stp <= 1)).sum())
                sp_tot['two_steps'] += int(((stp > 1) & (stp <= 2)).sum())
                sp_tot['more'] += int((stp > 2).sum())
                sp_max = max(sp_max, mx)
            log('verify pass %s %s-%02d done; memory %.2f GB' % (pas, y, m, C.mem_gb()))
        if pas == 'a':
            for c in C.EMIS:
                rt[c].end_a()
                r16[c].end_a()
    del arr
    files = {}
    roundtrip = {}
    for c in C.EMIS:
        r = rt[c].result()
        r['float16_plain_cast_for_the_record'] = r16[c].result()
        roundtrip[c] = r
    rt_doc = {'made': C.now(), 'year': int(y), 'rule': 'the storage rule: every stored day of the year, the five channels that stage (8) fills; float32 by that rule, so no result moves a channel', 'threshold': RT.THRESH, 'k': RT.K, 'channels': roundtrip, 'source': 'the float64 fields before the cast (s8common.day_fields), the plain float16 cast for the record'}
    files['roundtrip'] = wjson('roundtrip_s8_%s.json', y, rt_doc)
    yd = {'extracted_days': n, 'nox_mean_mask': nox['total']['mask'] / n, 'nox_mean_grid': nox['total']['grid'] / n, 'parts': {}}
    for q in parts_names:
        yd['parts'][q] = {'nox_mean_mask': nox[q]['mask'] / n, 'nox_mean_grid': nox[q]['grid'] / n, 'share_mask': nox[q]['mask'] / nox['total']['mask'], 'share_grid': nox[q]['grid'] / nox['total']['grid']}
    pm = [q for q in parts_names if q != 'merged']
    yd['point_and_marine_share_mask'] = sum((yd['parts'][q]['share_mask'] for q in pm))
    yd['point_and_marine_share_grid'] = sum((yd['parts'][q]['share_grid'] for q in pm))
    d5 = {'run': 'emul', 'stage': 'the fills stage, the emission fields of %s' % y, 'made': C.now(), 'design': "the study design's emission fields (E-1 + E-2 + E-3 + P-1); P-1 by the ONE DEFINITION (the extracts' P1_NOX); stage (1)'s p2_emis.py form", 'units': "mol/s (NO + NO2 + HONO), summed over cells; the mean over the year's extracted days", 'mask': S.file_info(p['mask']), 'mask_cells': mcells, 'grid_cells': int(mask.size), 'sectors': parts_names[1:], 'years': {str(y): yd}}
    files['d5'] = wjson('d5_s8_%s.json', y, d5)
    spd = {'made': C.now(), 'year': int(y), 'what': "for the record only: the species-grid P-1 (the six sectors' (NO + NO2) + HONO of the float32 species grids in float64, the same class means, cast once to float32) against the stored P-1 (the extracts' P1_NOX); d in float32 steps of the larger value's spacing", 'totals': dict(sp_tot), 'max_steps': sp_max, 'days': sp_days, 'days_with_a_difference': sum((1 for x in sp_days if x['differ']))}
    files['p1species'] = wjson('p1species_s8_%s.json', y, spd)
    ds_path = p['work']('daystats_s8_%s.npz')
    files['daystats'] = savez_partial(ds_path, daymin=daymin, daymax=daymax, isval=isval, dates=np.array(days), channels=np.array(list(C.EMIS)))
    sk, npos = ({}, {})
    for c, s in small.items():
        vv, npc = s.result()
        sk[c.replace('-', '')] = vv
        npos[c] = npc
    sk_path = p['work']('smallk_s8_%s.npz')
    files['smallk'] = savez_partial(sk_path, **sk)
    log('written %s and %s' % (files['daystats']['path'], files['smallk']['path']))
    stop = []
    for c in C.EMIS:
        miss = cnt[c]['nonfinite_mask'] + cnt[c]['fill_mask']
        frac = miss / float(cnt[c]['n_mask'])
        cnt[c]['masked_missing'] = miss
        cnt[c]['masked_missing_fraction'] = frac
        if frac > STOP_FRACTION:
            stop.append('%s: %.6f of masked cell-days missing' % (c, frac))
        if cnt[c]['nonfinite'] or cnt[c]['stored_nonfinite']:
            stop.append('%s: non-finite values' % c)
    if mism:
        stop.append('stored days differing from their fields: %d' % len(mism))
    build = {'year': int(y), 'start': start, 'end': C.now(), 'seconds': round(time.time() - t0, 1), 'days': n, 'channels': list(C.EMIS), 'indices': ei, 'source': "W\\extract_c7a\\%s\\ (each day's own merged extract for E-1 to E-4; P-1 the six sectors' P1_NOX by the ONE DEFINITION and the weekday classes)" % y, 'file': S.file_info(path), 'validation_days': int(isval.sum()), 'smallk_npos_nonvalidation': npos, 'k_small': B.K_SMALL, 'daystats': files['daystats'], 'smallk': files['smallk'], 'memory_gb_min': round(mem_min, 3), 'memory_gb_end': round(C.mem_gb(), 3), 'disk_free_gb_end': round(C.disk_free_gb(), 3)}
    files['build'] = wjson('build_s8_%s.json', y, build)
    ver = {'made': C.now(), 'year': int(y), 'start': start, 'stored_days_equal_to_fields': n - len(mism), 'stored_days_differing': mism, 'd4_counts': {c: dict(cnt[c]) for c in C.EMIS}, 'extremes': ext, 'p1_species_totals': dict(sp_tot), 'p1_species_max_steps': sp_max, 'roundtrip_verdicts': {c: roundtrip[c]['verdict'] for c in C.EMIS}, 'holiday_members_left_out': mo.left_out, 'fallbacks': mo.fallbacks, 'reads': int(sum(inp.reads.values())), 'files': files, 'stop': stop, 'verdict': 'PASS' if not stop else 'STOP', 'memory_gb_min': round(mem_min, 3), 'seconds': round(time.time() - t0, 1)}
    wjson('verify_s8_%s.json', y, ver)
    log('verify: stored days equal %d of %d; P-1 species-grid differ %d cells, max %.3f steps; stop %s: %s' % (n - len(mism), n, sp_tot['differ'], sp_max, stop, ver['verdict']))
    return 0 if not stop else 2

def main():
    a = sys.argv[1:]
    if len(a) != 2 or a[0] not in ('disk', 'd4', 'pre', 'fill', 'after', 'verify'):
        sys.exit('usage: disk|d4|pre|fill|after|verify YEAR')
    S.common.set_temp()
    os.makedirs(C.paths(a[1])['scratch'], exist_ok=True)
    fn = {'disk': cmd_disk, 'd4': cmd_d4, 'pre': cmd_pre, 'fill': cmd_fill, 'after': cmd_after, 'verify': cmd_verify}[a[0]]
    log('start %s %s; memory %.3f GB; work disk free %.3f GB' % (a[0], a[1], C.mem_gb(), C.disk_free_gb()))
    rc = fn(a[1])
    log('end %s %s: exit %d; memory %.3f GB' % (a[0], a[1], rc, C.mem_gb()))
    sys.exit(rc)
if __name__ == '__main__':
    S.run_main(main)
