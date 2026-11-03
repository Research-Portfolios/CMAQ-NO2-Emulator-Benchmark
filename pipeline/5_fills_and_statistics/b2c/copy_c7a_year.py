"""Copies one year's Drive extracts into the working folder file by file once the year's check has passed, testing each file's size and sha256 against the year's sums and manifest, copying a mismatch once more, and stopping when the free disk is below the need plus 30 GB."""
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
ROOT = '/content/emul'
W = os.path.join(ROOT, 'tmp', 'runner', 'work_build', 'work')
SRC_ROOT = '/content/drive/MyDrive/PROJECT_DATA/EMUL/EXTRACTS'
PACE = None
LOGDIR = os.path.join(ROOT, 'tmp', 'run', 'logs')
CHUNK = 1 << 22
MARGIN_GB = 30.0
PUBLISHED = 'published'

def now():
    return datetime.datetime.now()

def stamp():
    return now().strftime('%Y-%m-%d %H:%M:%S')

def pace_open():
    if PACE is None:
        return (True, 'no pace on Colab: OPEN')
    try:
        with open(PACE, 'r', encoding='ascii', errors='replace') as f:
            first = f.readline().strip()
    except OSError:
        return (False, 'no pace file: CLOSED')
    m = re.fullmatch('OPEN UNTIL (\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2})', first)
    if not m:
        return (False, 'pace reads %r: CLOSED' % first[:40])
    until = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M')
    if until <= now():
        return (False, 'window ended %s: CLOSED' % m.group(1))
    return (True, 'OPEN UNTIL %s' % m.group(1))

def norm(p):
    p = p.strip().replace('\\', '/')
    while p.startswith('./'):
        p = p[2:]
    return p

def match_key(listed, rels):
    n = norm(listed)
    if n in rels:
        return n
    tails = [r for r in rels if n.endswith('/' + r) or r.endswith('/' + n)]
    return tails[0] if len(tails) == 1 else None

def source_of(src, rel, year):
    if rel == 'STATE_%s.json' % year:
        pub = os.path.join(src, PUBLISHED, rel)
        if os.path.isfile(pub):
            return pub
    return os.path.join(src, rel)

def sha_local(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(CHUNK), b''):
            h.update(c)
    return h.hexdigest()

def copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = hashlib.sha256()
    n = 0
    part = dst + '.partial'
    with open(src, 'rb') as fi, open(part, 'wb') as fo:
        for c in iter(lambda: fi.read(CHUNK), b''):
            fo.write(c)
            h.update(c)
            n += len(c)
        fo.flush()
        os.fsync(fo.fileno())
    return (part, n, h.hexdigest())

def save(logp, log):
    tmp = logp + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(log, f, indent=1)
    os.replace(tmp, logp)

def main():
    a = sys.argv[1:]
    if not a or not re.fullmatch('20\\d\\d', a[0]):
        sys.exit('usage: YEAR [--extra GB] [--disk-only]')
    year = a[0]
    extra = float(a[a.index('--extra') + 1]) if '--extra' in a else 0.0
    src = os.path.join(SRC_ROOT, year)
    dst = os.path.join(W, 'extract_c7a', year)
    line_p = os.path.join(src, 'C7A_4_LINE_%s_ASCII.txt' % year)
    if not os.path.isfile(line_p):
        print('%s: no line file: not copied' % year)
        return 2
    with open(line_p, 'r', encoding='ascii', errors='replace') as f:
        first = f.readline().strip()
    if not first.startswith('C-7a (4) %s: PASS' % year):
        print('%s: line reads %r: not copied' % (year, first[:40]))
        return 2
    sums_p = os.path.join(src, 'SHA256SUMS_%s.txt' % year)
    man_p = os.path.join(src, 'MANIFEST_%s.csv' % year)
    if not (os.path.isfile(sums_p) and os.path.isfile(man_p)):
        print('%s: SHA256SUMS or MANIFEST absent: not copied' % year)
        return 1
    rels = []
    for d, dn, fs in os.walk(src):
        if d == src:
            dn[:] = [x for x in dn if x != PUBLISHED]
        for n in fs:
            rels.append(norm(os.path.relpath(os.path.join(d, n), src)))
    rels.sort()
    sums = {}
    with open(sums_p, 'r', encoding='ascii', errors='replace') as f:
        for ln in f:
            parts = ln.strip().split(None, 2)
            if len(parts) != 3:
                continue
            k = match_key(parts[2], rels)
            sums[k if k else 'UNMATCHED:' + norm(parts[2])] = (parts[0].lower(), int(parts[1]))
    man = {}
    with open(man_p, 'r', encoding='ascii', errors='replace', newline='') as f:
        for row in csv.DictReader(f):
            k = match_key(row.get('path', ''), rels)
            try:
                b = int(row.get('bytes', '') or -1)
            except ValueError:
                b = -1
            man[k if k else 'UNMATCHED:' + norm(row.get('path', ''))] = ((row.get('sha256') or '').lower(), b)
    unmatched = sorted([k for k in list(sums) + list(man) if k.startswith('UNMATCHED:')])
    logp = os.path.join(LOGDIR, 'B2C_C7A_COPY_%s.json' % year)
    log = {'year': year, 'line_first_40': first[:40], 'files': {}, 'unmatched_listed': unmatched, 'complete': False}
    if os.path.isfile(logp):
        with open(logp, 'r', encoding='ascii') as f:
            log = json.load(f)
        log['unmatched_listed'] = unmatched
    done = {k for k, e in log['files'].items() if e.get('status') == 'ok' and os.path.isfile(os.path.join(dst, k)) and (os.stat(os.path.join(dst, k)).st_size == e['bytes'])}
    need = sum((b for k, (_, b) in sums.items() if not k.startswith('UNMATCHED:') and k not in done and (b > 0)))
    os.makedirs(W, exist_ok=True)
    free = shutil.disk_usage(W).free
    need_gb = need / 1000000000.0 + extra
    print('%s: %d files in the folder, %d in the sums, %d in the manifest, %d listed but unmatched' % (year, len(rels), len(sums), len(man), len(unmatched)))
    print('DISK: need %.3f GB (copies %.3f GB + extra %.3f GB), free %.3f GB, asked %.3f GB: %s' % (need_gb, need / 1000000000.0, extra, free / 1000000000.0, need_gb + MARGIN_GB, 'MET' if free / 1000000000.0 >= need_gb + MARGIN_GB else 'NOT MET'))
    log.setdefault('disk', []).append({'at': stamp(), 'need_gb': round(need_gb, 3), 'free_gb': round(free / 1000000000.0, 3), 'margin_gb': MARGIN_GB})
    if '--disk-only' in a:
        return 0
    if free / 1000000000.0 < need_gb + MARGIN_GB:
        save(logp, log)
        return 4
    order = [r for r in rels if r in sums] + [r for r in rels if r not in sums]
    meta = {'SHA256SUMS_%s.txt' % year, 'MANIFEST_%s.csv' % year, 'C7A_4_LINE_%s_ASCII.txt' % year}
    order = [r for r in order if r not in meta] + [r for r in order if r in meta]
    for rel in order:
        if rel in done:
            continue
        ok, why = pace_open()
        if not ok:
            print('PACE %s; stopped before %s' % (why, rel))
            save(logp, log)
            return 3
        s = source_of(src, rel, year)
        t = os.path.join(dst, rel)
        if os.path.exists(t):
            print('%s: present in W but not done in the log: not overwritten; recorded' % rel)
            log.setdefault('present_not_logged', []).append(rel)
            save(logp, log)
            return 1
        tries = 0
        while True:
            tries += 1
            t0 = stamp()
            part, n, got = copy_one(s, t)
            vs = 'not listed'
            vm = 'not listed'
            good = True
            if rel in sums:
                vs = 'EQUAL' if sums[rel] == (got, n) else 'DIFFERENT'
                good = good and vs == 'EQUAL'
            if rel in man:
                mh, mb = man[rel]
                vm = 'EQUAL' if mh == got and mb == n else 'DIFFERENT'
                good = good and vm == 'EQUAL'
            if rel in meta:
                good = True if vs != 'DIFFERENT' and vm != 'DIFFERENT' else False
            entry = {'bytes': n, 'sha256': got, 'start': t0, 'end': stamp(), 'sums': vs, 'manifest': vm, 'tries': tries}
            note = ''
            if s != os.path.join(src, rel):
                entry['source'] = PUBLISHED + '/' + rel
                note = ' (from %s/)' % PUBLISHED
            if good:
                os.replace(part, t)
                entry['status'] = 'ok'
                log['files'][rel] = entry
                save(logp, log)
                print('%s %d %s sums %s manifest %s%s' % (rel, n, got[:16], vs, vm, note))
                break
            entry['status'] = 'mismatch'
            log['files'][rel] = entry
            save(logp, log)
            print('%s %d %s sums %s manifest %s%s: MISMATCH (try %d)' % (rel, n, got[:16], vs, vm, note, tries))
            if tries >= 2:
                print('%s: the year stops (second mismatch; .partial kept)' % year)
                return 1
            os.replace(part, part + '.try1')
    missing = sorted((k for k in sums if not k.startswith('UNMATCHED:') and log['files'].get(k, {}).get('status') != 'ok'))
    log['complete'] = not missing and (not unmatched)
    log['missing_from_sums'] = missing
    log['completed_at'] = stamp()
    save(logp, log)
    print('%s: %s; %d files ok; missing %d; unmatched %d' % (year, 'WHOLE AND EQUAL' if log['complete'] else 'NOT WHOLE', sum((1 for e in log['files'].values() if e.get('status') == 'ok')), len(missing), len(unmatched)))
    return 0 if log['complete'] else 1
if __name__ == '__main__':
    sys.exit(main())
