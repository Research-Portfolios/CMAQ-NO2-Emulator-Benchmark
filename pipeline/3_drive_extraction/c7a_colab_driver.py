"""Driver of the Drive extraction stage on Colab: copies each pinned emission archive of 2015, 2016 and 2018 from a copy of the CMAS Google Drive folder to the runtime's disk, one at a time, tests its md5 and sha256 against the pins, runs the extraction on it, and copies each year's outputs to the Drive."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import queue
import shutil
import sys
import threading
import time
import zlib
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7a_common as C
import c7a_extract as E
DRIVE = '/content/drive/MyDrive'
COPYDIR_INPUT = DRIVE + '/EQUATES_v1.0_CMAQ_12US1_copy_rev/INPUT'
EMUL = DRIVE + '/PROJECT_DATA/EMUL'
EXTRACTS = EMUL + '/EXTRACTS'
RUN_DRIVE = EXTRACTS + '/_run'
KEPT = os.path.join(C.PKG, 'ref', 'MANIFEST_%s_KEPT.csv')
LOCAL_LOG = os.path.join(C.ROOT, 'driver', 'COLAB_DRIVER_LOG_ASCII.txt')
PINS_IDFREE_SHA256 = {'2015': 'e92eead1ff0eedc9a6bd20d75a4ffdf00d050e8d6deebfb126c100400d19ff1d', '2016': 'dc07a5ead15fbe9e9904f602c64d40a7451d5b274220fbf660448c9b37af2008', '2018': '9facfe1c49c93574439b1ab98abf80cb63e0b450bf9493b874c53d9d300bb3c3'}
K4_FIELDS = ['path', 'source_link', 'date_saved', 'bytes', 'sha256', 'md5', 'version', 'source', 'copy_id', 'pinned_bytes', 'pinned_sha256', 'pinned_md5', 'k4', 'hashed_at', 'rclone_secs']
FINISH_MERGED = ('01', '04', '07', '10')
FINISH_PT = ('07',)
GB = 1000000000.0
RESERVE_GB = 8.0
PIECE = 64 << 20
NO_T2_WHY = 'T-2 not read on the Colab VM: it watches the archive disk, which this route does not touch'

def NO_T2():
    return (0, NO_T2_WHY)

def log(text, year=None):
    line = '%s %s' % (C.stamp(), text)
    C.append_record(LOCAL_LOG, line)
    if year:
        C.append_record(os.path.join(C.OUT_ROOT, year, 'COLAB_LOG_%s_ASCII.txt' % year), line)
    print(C.mask(line)[:400], flush=True)

def under_emul(path):
    p = os.path.abspath(path)
    if not (p == EMUL or p.startswith(EMUL + '/')) or '..' in path:
        raise SystemExit(9)
    return p

def sha256_of(path):
    return C.sha256_file(path)[0]

def free_gb(path):
    return shutil.disk_usage(path).free / GB

def to_drive(src, dst):
    under_emul(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + '.partial'
    shutil.copyfile(src, part)
    os.replace(part, dst)
    n = os.path.getsize(dst)
    if n != os.path.getsize(src):
        raise IOError('Drive copy size differs: %s' % os.path.basename(dst))
    return n

def from_drive(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + '.partial'
    shutil.copyfile(src, part)
    os.replace(part, dst)

def mirror_year(y, done):
    ydir = os.path.join(C.OUT_ROOT, y)
    ddir = os.path.join(EXTRACTS, y)
    n = 0
    mp = os.path.join(ydir, 'MANIFEST_%s.csv' % y)
    if os.path.exists(mp):
        with open(mp, newline='', encoding='ascii') as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            rel = r['path'].replace('\\', '/')
            if rel in done:
                continue
            to_drive(os.path.join(ydir, rel), os.path.join(ddir, rel))
            done.add(rel)
            n += 1
    for name in sorted(os.listdir(ydir)):
        p = os.path.join(ydir, name)
        if os.path.isfile(p) and name != 'STATE_%s.json' % y and (not name.endswith('.partial')):
            to_drive(p, os.path.join(ddir, name))
            n += 1
    for d, dd in ((C.RUN_DIR, RUN_DRIVE), (os.path.dirname(LOCAL_LOG), RUN_DRIVE)):
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isfile(p) and (not name.endswith('.partial')) and (name != 'LOCK'):
                    to_drive(p, os.path.join(dd, name))
                    n += 1
    kp = os.path.join(C.RCRUN, 'K4_%s_DONE.csv' % y)
    if os.path.exists(kp):
        to_drive(kp, os.path.join(RUN_DRIVE, os.path.basename(kp)))
        n += 1
    sp = os.path.join(ydir, 'STATE_%s.json' % y)
    if os.path.exists(sp):
        to_drive(sp, os.path.join(ddir, 'STATE_%s.json' % y))
        n += 1
    return n

def write_rows(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.partial', 'w', newline='', encoding='ascii') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(path + '.partial', path)

def restore_year(y):
    ydir = os.path.join(C.OUT_ROOT, y)
    ddir = os.path.join(EXTRACTS, y)
    done = set()
    n = 0
    own = ('STATE_%s.json' % y, 'MANIFEST_%s.csv' % y, 'K4_%s_DONE.csv' % y)
    if os.path.isdir(RUN_DRIVE):
        for name in sorted(os.listdir(RUN_DRIVE)):
            p = os.path.join(RUN_DRIVE, name)
            if not os.path.isfile(p) or name.endswith('.partial'):
                continue
            if name == os.path.basename(LOCAL_LOG):
                dst = LOCAL_LOG
            elif name.startswith('K4_'):
                dst = os.path.join(C.RCRUN, name)
            else:
                dst = os.path.join(C.RUN_DIR, name)
            if name in own or not os.path.exists(dst):
                from_drive(p, dst)
    if not os.path.isdir(ddir):
        return (0, done)
    for name in ('MANIFEST_%s.csv' % y, 'STATE_%s.json' % y):
        p = os.path.join(ddir, name)
        if os.path.exists(p):
            from_drive(p, os.path.join(ydir, name))
            if name.startswith('STATE_'):
                from_drive(p, os.path.join(C.RUN_DIR, name))
    mp = os.path.join(ydir, 'MANIFEST_%s.csv' % y)
    rows, fields = ([], E.MANIFEST_FIELDS)
    if os.path.exists(mp):
        with open(mp, newline='', encoding='ascii') as fh:
            rd = csv.DictReader(fh)
            rows = list(rd)
            fields = rd.fieldnames or fields
    bad, bad_rows = (set(), [])
    for r in rows:
        rel = r['path'].replace('\\', '/')
        dst = os.path.join(ydir, rel)
        src = os.path.join(ddir, rel)
        if not (os.path.exists(dst) and sha256_of(dst) == r['sha256']):
            if os.path.exists(src):
                from_drive(src, dst)
            if not (os.path.exists(dst) and sha256_of(dst) == r['sha256']):
                bad.add(str(r.get('tar')))
                bad_rows.append(r)
                continue
            n += 1
        if os.path.exists(src) and os.path.getsize(src) == int(r.get('bytes') or -1):
            done.add(rel)
    if bad_rows:
        rows = [r for r in rows if r not in bad_rows]
        for r in bad_rows:
            rel = r['path'].replace('\\', '/')
            if os.path.exists(os.path.join(ydir, rel)):
                os.remove(os.path.join(ydir, rel))
        write_rows(mp, rows, fields)
        write_rows(os.path.join(C.RUN_DIR, 'MANIFEST_%s.csv' % y), rows, fields)
        log('RESTORE %s: %d manifest rows dropped, their extracts not on the Drive (tars %s)' % (y, len(bad_rows), ' '.join(sorted(bad))[:200]), y)
    for name in sorted(os.listdir(ddir)):
        p = os.path.join(ddir, name)
        if os.path.isfile(p) and (not name.endswith('.partial')) and (name not in ('MANIFEST_%s.csv' % y, 'STATE_%s.json' % y)):
            dst = os.path.join(ydir, name)
            if not os.path.exists(dst):
                from_drive(p, dst)
    sp = os.path.join(ydir, 'STATE_%s.json' % y)
    st = C.read_json(sp)
    if st is None:
        return (n, done)
    k4p = os.path.join(C.RCRUN, 'K4_%s_DONE.csv' % y)
    k4 = read_k4(k4p)
    k4names = set((os.path.basename(r['path']) for r in k4 if r.get('k4') == 'PASS'))
    reset = []
    for t, info in sorted(st.get('tars', {}).items()):
        if info.get('status') != 'done':
            continue
        ok = t in k4names and t not in bad
        if ok and C.kind_of_tar(t) != 'stackgroups':
            nrow = sum((1 for r in rows if r['tar'] == t))
            ok = nrow == int(info.get('members_done') or 0)
        if not ok:
            reset.append(t)
    if reset:
        if st.get('year_status'):
            log('RESTORE %s FAILED: the year is %s but tars %s lack their K4 row or extracts on the Drive; a manual decision is needed' % (y, str(st['year_status'])[:40], ' '.join(reset[:5])), y)
            raise SystemExit(8)
        k = 1
        while os.path.exists(sp[:-5] + '_Backup_PriorToRestoreReset%d.json' % k):
            k += 1
        shutil.copyfile(sp, sp[:-5] + '_Backup_PriorToRestoreReset%d.json' % k)
        for t in reset:
            del st['tars'][t]
            st['issues'] = [i for i in st.get('issues', []) if t not in i]
        gone = [r for r in rows if r['tar'] in reset]
        rows = [r for r in rows if r['tar'] not in reset]
        for r in gone:
            rel = r['path'].replace('\\', '/')
            done.discard(rel)
            if os.path.exists(os.path.join(ydir, rel)):
                os.remove(os.path.join(ydir, rel))
        write_rows(mp, rows, fields)
        write_rows(os.path.join(C.RUN_DIR, 'MANIFEST_%s.csv' % y), rows, fields)
        write_k4(k4p, [r for r in k4 if os.path.basename(r['path']) not in reset])
        C.write_json(sp, st)
        C.write_json(os.path.join(C.RUN_DIR, 'STATE_%s.json' % y), st)
        log('RESTORE %s: %d tars reset to run again (done in STATE but without their K4 row or all their extracts on the Drive): %s; %d manifest rows removed; STATE backup _Backup_PriorToRestoreReset%d' % (y, len(reset), ' '.join(reset), len(gone), k), y)
    return (n, done)

def drive_path(y, name):
    return os.path.join(COPYDIR_INPUT, y, 'emis', name)

def farm_path(y, name):
    return os.path.join(C.H_INPUT, y, 'emis', name)

def is_staged(y, name):
    p = farm_path(y, name)
    return os.path.isfile(p) and (not os.path.islink(p))

def link(y, name):
    p = farm_path(y, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.islink(p):
        return
    if os.path.exists(p):
        os.remove(p)
    os.symlink(drive_path(y, name), p)

def unstage(y, name):
    p = farm_path(y, name)
    if is_staged(y, name):
        os.remove(p)
    link(y, name)

def copy_hash(src, dst):
    for k in (1, 2, 3):
        try:
            return copy_hash_once(src, dst)
        except OSError as e:
            if os.path.exists(dst + '.partial'):
                os.remove(dst + '.partial')
            log('READ ERROR %s (try %d of 3): %s' % (os.path.basename(src), k, C.mask(str(e))[:160]))
            if k == 3:
                raise
            time.sleep(60)

def copy_hash_once(src, dst):
    t0 = time.time()
    md5, sha = (hashlib.md5(), hashlib.sha256())
    n = 0
    part = dst + '.partial'
    with open(src, 'rb') as fi, open(part, 'wb') as fo:
        while True:
            b = fi.read(PIECE)
            if not b:
                break
            md5.update(b)
            sha.update(b)
            fo.write(b)
            n += len(b)
    if os.path.islink(dst):
        os.unlink(dst)
    os.replace(part, dst)
    return (n, md5.hexdigest(), sha.hexdigest(), time.time() - t0)

def stage(y, name, pin):
    dst = farm_path(y, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    need = int(pin['bytes']) / GB + RESERVE_GB
    if is_staged(y, name):
        unstage(y, name)
    if free_gb('/content') < need:
        log('DISK %s %s: %.1f GB free on the VM, %.1f GB needed (the file and %.1f GB spare); stopped before the copy ' % (y, name, free_gb('/content'), need, RESERVE_GB), y)
        raise SystemExit(7)
    tries = []
    for attempt in (1, 2):
        n, md5, sha, secs = copy_hash(drive_path(y, name), dst)
        ok = n == int(pin['bytes']) and md5 == pin['md5'] and (sha == pin['sha256'])
        tries.append({'bytes': n, 'md5': md5, 'sha256': sha, 'seconds': round(secs, 1), 'equal_to_pin': ok})
        if ok:
            break
    if not ok:
        unstage(y, name)
    return (ok, {'tries': tries, 'bytes': n, 'md5': md5, 'sha256': sha, 'seconds': round(secs, 1), 'at': C.stamp()})

def k4_row(y, name, pin, info):
    ok = info['bytes'] == int(pin['bytes']) and info['md5'] == pin['md5'] and (info['sha256'] == pin['sha256'])
    return {'path': farm_path(y, name), 'source_link': 'COPYDIR INPUT/%s/emis/%s, read by path through the Ultra My Drive shortcut (no id)' % (y, name), 'date_saved': info['at'], 'bytes': str(info['bytes']), 'sha256': info['sha256'], 'md5': info['md5'], 'version': 'EQUATES v1.0', 'source': 'COPYDIR (a Drive copy of the CMAS original), staged on the Colab VM', 'copy_id': '-', 'pinned_bytes': str(pin['bytes']), 'pinned_sha256': pin['sha256'], 'pinned_md5': pin['md5'], 'k4': 'PASS' if ok else 'FAIL', 'hashed_at': info['at'], 'rclone_secs': '%.1f' % info['seconds']}

def write_k4(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.partial', 'w', newline='', encoding='ascii') as fh:
        w = csv.DictWriter(fh, fieldnames=K4_FIELDS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(path + '.partial', path)

def read_k4(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='ascii') as fh:
        return list(csv.DictReader(fh))

class Prefetch:

    def __init__(self, y, pins):
        self.y, self.pins = (y, pins)
        self.q = queue.Queue()
        self.name = None
        self.th = None

    def start(self, name):
        self.name = name

        def work():
            try:
                self.q.put(stage(self.y, name, self.pins[name]))
            except BaseException as e:
                self.q.put((False, {'error': "%s: %s (exit 7 is the VM's disk)" % (type(e).__name__, C.mask(str(e))[:200])}))
        self.th = threading.Thread(target=work, daemon=True)
        self.th.start()

    def take(self, name):
        if self.name != name or self.th is None:
            return None
        res = self.q.get()
        self.th.join()
        self.name, self.th = (None, None)
        return res
PC_ZLIB = '1.3.1'

def sha256_os0(path):
    b = bytearray(C.read_bytes(path))
    e = b.rfind(b'PK\x05\x06')
    if e < 0:
        return None
    n = int.from_bytes(b[e + 10:e + 12], 'little')
    off = int.from_bytes(b[e + 16:e + 20], 'little')
    for _ in range(n):
        if b[off:off + 4] != b'PK\x01\x02':
            return None
        b[off + 5] = 0
        off += 46 + sum((int.from_bytes(b[off + k:off + k + 2], 'little') for k in (28, 30, 32)))
    return hashlib.sha256(b).hexdigest()

def compare(y):
    ydir = os.path.join(C.OUT_ROOT, y)
    with open(os.path.join(ydir, 'MANIFEST_%s.csv' % y), newline='', encoding='ascii') as fh:
        new = {(r['kind'], r['sector'], r['date']): r for r in csv.DictReader(fh)}
    with open(KEPT % y, newline='', encoding='ascii') as fh:
        old = {(r['kind'], r['sector'], r['date']): r for r in csv.DictReader(fh)}
    fields = ('bytes', 'tar', 'tar_sha256', 'member', 'member_offset', 'member_size', 'member_sha256', 'species', 'units', 'hours', 'version', 'n_stacks', 'n_summed', 'n_outside', 'base', 'p1')
    out, counts, differ = ([], {}, [])
    for key in sorted(set(new) | set(old)):
        n, o = (new.get(key), old.get(key))
        s0 = ''
        if n and o:
            s0 = sha256_os0(os.path.join(ydir, *n['path'].replace('\\', '/').split('/'))) or ''
            v = 'MATCH' if n['sha256'] == o['sha256'] else 'MATCH (OS BYTE)' if s0 == o['sha256'] else 'DIFFER'
            eq = [f for f in fields if n.get(f) != o.get(f)]
            path_eq = n['path'].replace('\\', '/') == o['path'].replace('\\', '/')
        elif n:
            v, eq, path_eq = ('NO KEPT ROW', [], None)
        else:
            v, eq, path_eq = ('NOT REBUILT', [], None)
        counts[v] = counts.get(v, 0) + 1
        if v == 'DIFFER':
            differ.append('%s %s %s (fields differing: %s)' % (key[0], key[1] or '-', key[2], ' '.join(eq) or 'none'))
        out.append({'kind': key[0], 'sector': key[1], 'date': key[2], 'verdict': v, 'sha256_new': n['sha256'] if n else '', 'sha256_new_os0': s0, 'sha256_kept': o['sha256'] if o else '', 'fields_differing': ' '.join(eq), 'path_equal': '' if path_eq is None else str(path_eq)})
    cp = os.path.join(ydir, 'COMPARE_%s.csv' % y)
    with open(cp, 'w', newline='', encoding='ascii') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ['kind'], lineterminator='\n')
        w.writeheader()
        for r in out:
            w.writerow(r)
    kept_sha = sha256_of(KEPT % y)
    text = "COMPARE %s (each extract's sha256 against the kept manifest of the deleted local extracts, work/MANIFEST_%s.csv on the work disk, sha256 %s): %s. MATCH: byte-equal. MATCH (OS BYTE): equal once each zip central-directory entry's OS byte is set to 0 (Windows), the one byte CPython's zipfile writes by platform (0 on the PC, 3 on the VM), so every array and name is byte-equal. DIFFER: reported, not hidden; its fields differing name the manifest fields beside the file hash. zlib %s on the VM, %s on the PC." % (y, y, kept_sha[:16], ', '.join(('%s %d' % (k, counts[k]) for k in sorted(counts))), zlib.ZLIB_RUNTIME_VERSION, PC_ZLIB)
    lines = [text]
    if differ:
        lines.append('DIFFER (first 50 of %d): %s' % (len(differ), '; '.join(differ[:50])))
    p = os.path.join(ydir, 'COMPARE_%s_ASCII.txt' % y)
    if os.path.exists(p):
        os.remove(p)
    for ln in lines:
        C.append_record(p, ln)
    return (counts, differ)

def verify_drive(y):
    ddir = os.path.join(EXTRACTS, y)
    sp = os.path.join(ddir, 'SHA256SUMS_%s.txt' % y)
    ok, bad = (0, [])
    with open(sp, encoding='ascii') as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 3:
                continue
            sha, nb, rel = (p[0], int(p[1]), p[2])
            f = os.path.join(ddir, 'published', rel) if rel == 'STATE_%s.json' % y else os.path.join(ddir, rel)
            if not os.path.exists(f) or os.path.getsize(f) != nb or sha256_of(f) != sha:
                bad.append(rel)
            else:
                ok += 1
    return (ok, bad)

def tar_order(pins):
    tars = sorted((n for n in pins if n.endswith('.tar')))
    keep = keep_set(tars)
    sg = [t for t in tars if C.kind_of_tar(t) == 'stackgroups']
    rest = [t for t in tars if t not in keep]
    fin = [t for t in tars if t in keep and t not in sg]
    return (sg + rest + fin, keep)

def keep_set(tars):
    keep = set()
    for t in tars:
        k, mm = C.family_month(t)
        if k == 'stackgroups' or (k == 'merged' and mm in FINISH_MERGED) or (k == 'ptsectors' and mm in FINISH_PT):
            keep.add(t)
    return keep

def do_year(y, p1, on_fail, prefetch_mode):
    ddir = os.path.join(EXTRACTS, y)
    under_emul(ddir)
    st_drive = None
    if os.path.exists(os.path.join(ddir, 'STATE_%s.json' % y)):
        with open(os.path.join(ddir, 'STATE_%s.json' % y), encoding='ascii') as fh:
            st_drive = json.load(fh)
    vs = str((st_drive or {}).get('year_status') or '')
    if vs == 'PUBLISHED' and os.path.exists(os.path.join(ddir, 'COMPARE_%s_ASCII.txt' % y)):
        log('YEAR %s: PUBLISHED on the Drive with its COMPARE; skipped' % y)
        return 'PUBLISHED'
    if vs.startswith(('FAILED', 'HELD')):
        log('YEAR %s: %s on the Drive; a manual decision is needed; not run again here' % (y, vs[:60]))
        return 'STOP' if on_fail == 'stop' else 'NEXT'
    os.makedirs(os.path.join(C.OUT_ROOT, y), exist_ok=True)
    nr, done = restore_year(y)
    log('YEAR %s: begin; %d extracts restored from the Drive and checked by sha256; VM disk free %.1f GB' % (y, nr, free_gb('/content')), y)
    pins, _ = E.pins_of(y, C.PINS, check=False)
    names = sorted(pins)
    tars = [n for n in names if n.endswith('.tar')]
    if len(names) != 50 or len(tars) != 25:
        log('PIN FAIL %s: %d pinned files, %d tars, not 50 and 25' % (y, len(names), len(tars)), y)
        raise SystemExit(3)
    miss = [n for n in names if not os.path.exists(drive_path(y, n)) or os.path.getsize(drive_path(y, n)) != int(pins[n]['bytes'])]
    if miss:
        log('PIN FAIL %s: %d pinned files absent from COPYDIR or of another size there (%s)' % (y, len(miss), '; '.join(miss[:5])), y)
        raise SystemExit(3)
    log('COPYDIR %s: the 50 pinned files present by path, sizes equal to the pins (%d bytes in all)' % (y, sum((int(pins[n]['bytes']) for n in names))), y)
    for n in names:
        if not is_staged(y, n):
            link(y, n)
    k4_live = os.path.join(C.RCRUN, 'K4_%s.csv' % y)
    k4_done = os.path.join(C.RCRUN, 'K4_%s_DONE.csv' % y)
    done_rows = read_k4(k4_done)
    rows_c = []
    for n in names:
        if n.endswith('.tar'):
            continue
        ok, info = stage(y, n, pins[n])
        if not ok:
            log('PIN FAIL %s %s: the staged copy differs from its pin after two reads (%s)' % (y, n, json.dumps(info['tries'])), y)
            raise SystemExit(3)
        rows_c.append(k4_row(y, n, pins[n], info))
    log('CONTENTS %s: 25 contents files staged, each equal to its pin by bytes, md5 and sha256' % y, y)
    order, keep = tar_order(pins)
    for t in tars:
        if t not in keep and is_staged(y, t):
            unstage(y, t)
    emis = os.path.join(C.H_INPUT, y, 'emis')
    for name in os.listdir(emis):
        if name.endswith('.partial'):
            os.remove(os.path.join(emis, name))
    need_keep = sum((int(pins[t]['bytes']) for t in keep)) / GB
    big = max((int(pins[t]['bytes']) for t in tars if t not in keep)) / GB
    free = free_gb('/content')
    staged_keep = sum((int(pins[t]['bytes']) for t in keep if is_staged(y, t))) / GB
    if free + staged_keep < need_keep + RESERVE_GB:
        log('DISK %s: %.1f GB free on the VM (%.1f GB of kept tars already staged); the year needs %.1f GB (the %d tars K5 and the value check read, %.1f GB, and %.1f GB spare): choose a runtime with more disk (High-RAM); stopped' % (y, free, staged_keep, need_keep + RESERVE_GB, len(keep), need_keep, RESERVE_GB), y)
        raise SystemExit(7)
    pre_ok = prefetch_mode == 'on' or (prefetch_mode == 'auto' and free + staged_keep >= need_keep + big + RESERVE_GB)
    log('PLAN %s: %d tars in order (the stack groups tar first; the %d tars K5 and the value check read last, kept on the VM); prefetch %s; VM disk free %.1f GB' % (y, len(order), len(keep), 'on' if pre_ok else 'off', free), y)
    run = E.Run([y], p1, t2=NO_T2, check_pins=False)
    log('RUN %s: code %s' % (y, json.dumps(run.code_sha16, sort_keys=True)), y)
    fin_st = run.state(y)['tars']
    done_rows = [r for r in done_rows if fin_st.get(os.path.basename(r['path']), {}).get('status') in ('done', 'failed')]
    write_k4(k4_done, done_rows)
    write_k4(k4_live, done_rows + rows_c)
    pf = Prefetch(y, pins) if pre_ok else None
    todo = [t for t in order if run.state(y)['tars'].get(t, {}).get('status') not in ('done', 'failed')]
    failed = [t for t in order if run.state(y)['tars'].get(t, {}).get('status') == 'failed']
    if failed:
        log('K FAIL %s: tars already failed K2 or K3 (%s)' % (y, '; '.join(failed[:5])), y)
        return 'STOP' if on_fail == 'stop' else 'NEXT'

    def ensure_kept(only_sg):
        for k in order:
            if k in keep and (not is_staged(y, k)) and (not only_sg or C.kind_of_tar(k) == 'stackgroups') and (run.state(y)['tars'].get(k, {}).get('status') == 'done'):
                ok, info = stage(y, k, pins[k])
                log('RESTAGE %s %s (done, kept on the VM): %s, %.1f s' % (y, k, 'equal to its pin' if ok else 'DIFFERS AFTER TWO READS: PIN FAIL', info['seconds']), y)
                if not ok:
                    raise SystemExit(3)
    ensure_kept(True)
    for i, t in enumerate(todo):
        if t in keep and C.kind_of_tar(t) != 'stackgroups':
            ensure_kept(False)
        res = pf.take(t) if pf else None
        if res is not None and (not res[0]) and str(res[1].get('error', '')).startswith('SystemExit: 7'):
            log("READ-AHEAD %s %s refused for the VM's disk; staged now instead (REVERIFY)" % (y, t), y)
            res = None
        if res is None:
            res = stage(y, t, pins[t])
        ok, info = res
        if not ok:
            log('PIN FAIL (or read error) %s %s: %s; nothing of it is used' % (y, t, json.dumps(info.get('tries') or info)[:300]), y)
            unstage(y, t)
            mirror_year(y, done)
            raise SystemExit(3)
        row = k4_row(y, t, pins[t], info)
        write_k4(k4_live, done_rows + rows_c + [row])
        log('STAGED %s %s: %d bytes in %.1f s (%.1f MB/s), md5 and sha256 equal to the pin' % (y, t, info['bytes'], info['seconds'], info['bytes'] / 1000000.0 / max(info['seconds'], 0.1)), y)
        nxt = todo[i + 1] if i + 1 < len(todo) else None
        if pf and nxt and (not is_staged(y, nxt)):
            pf.start(nxt)
        r = run.run_once()
        run.write_status(r)
        st = run.state(y)
        tst = st['tars'].get(t, {}).get('status')
        log('TAR %s %s: Run pass %s, tar %s, %s members, %s s' % (y, t, r, tst, st['tars'].get(t, {}).get('members_done'), st['tars'].get(t, {}).get('seconds')), y)
        if tst in ('done', 'failed'):
            done_rows = [x for x in done_rows if x['path'] != row['path']]
            done_rows.append(row)
            write_k4(k4_done, done_rows)
        if t not in keep:
            unstage(y, t)
        nm = mirror_year(y, done)
        log('DRIVE %s: %d files copied to EXTRACTS/%s and EXTRACTS/_run' % (y, nm, y), y)
        if tst == 'failed' or r not in ('worked', 'idle', 'done'):
            if pf and pf.th is not None:
                pf.take(pf.name)
            log('STOP %s at %s: tar %s, Run pass %s (K2 or K3, or a gate); a manual decision is needed' % (y, t, tst, r), y)
            return 'STOP' if on_fail == 'stop' else 'NEXT'
        if tst != 'done':
            log('STOP %s at %s: the Run did not take the staged tar (pass %s); nothing more is run' % (y, t, r), y)
            return 'STOP'
    st = run.state(y)
    if not st.get('year_status'):
        ensure_kept(False)
        write_k4(k4_live, done_rows + rows_c)
        r = run.run_once()
        run.write_status(r)
        st = run.state(y)
        log('FINISH PASS %s: Run pass %s' % (y, r), y)
    vs = str(st.get('year_status') or '')
    log('YEAR %s: %s; line %s' % (y, vs or 'open', st.get('line4')), y)
    mirror_year(y, done)
    if vs != 'PUBLISHED':
        log('STOP %s: the year is %s; K1 to K6 or the value check did not pass; a manual decision is needed' % (y, vs or 'open'), y)
        return 'STOP' if on_fail == 'stop' else 'NEXT'
    sums = os.path.join(C.XFER_OUT, y, 'SHA256SUMS_%s.txt' % y)
    if not os.path.exists(sums):
        pub = run.publish(y)
        log('REPUBLISH %s on this VM: %s' % (y, json.dumps(pub)[:200]), y)
    to_drive(sums, os.path.join(ddir, 'SHA256SUMS_%s.txt' % y))
    to_drive(os.path.join(C.XFER_OUT, y, 'STATE_%s.json' % y), os.path.join(ddir, 'published', 'STATE_%s.json' % y))
    ok_n, bad = verify_drive(y)
    log('DRIVE CHECK %s: %d files of SHA256SUMS_%s.txt read back from EXTRACTS/%s equal by bytes and sha256; %d not (%s)' % (y, ok_n, y, y, len(bad), '; '.join(bad[:5])), y)
    counts, differ = compare(y)
    log('COMPARE %s: %s' % (y, ', '.join(('%s %d' % (k, counts[k]) for k in sorted(counts)))), y)
    mirror_year(y, done)
    for n in names:
        if is_staged(y, n):
            unstage(y, n)
    shutil.rmtree(os.path.join(C.XFER_OUT, y), ignore_errors=True)
    log('FREED %s: the staged tars removed from the VM; VM disk free %.1f GB' % (y, free_gb('/content')), y)
    mirror_year(y, done)
    if bad:
        return 'STOP'
    return 'PUBLISHED'

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='2018,2015,2016')
    ap.add_argument('--on-fail', choices=('stop', 'next'), default='stop')
    ap.add_argument('--prefetch', choices=('auto', 'on', 'off'), default='auto')
    a = ap.parse_args(argv)
    import fcntl
    os.makedirs(os.path.dirname(LOCAL_LOG), exist_ok=True)
    lk = open(os.path.join(os.path.dirname(LOCAL_LOG), 'LOCK'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ANOTHER DRIVER RUNS ON THIS VM: not started')
        return 10
    for p in (EXTRACTS, RUN_DRIVE):
        under_emul(p)
    if not os.path.isdir(COPYDIR_INPUT):
        print('NO COPYDIR: %s is not a folder (the shortcut in My Drive)' % COPYDIR_INPUT)
        return 9
    os.makedirs(RUN_DRIVE, exist_ok=True)
    old_log = os.path.join(RUN_DRIVE, os.path.basename(LOCAL_LOG))
    if os.path.exists(old_log) and (not os.path.exists(LOCAL_LOG)):
        from_drive(old_log, LOCAL_LOG)
    for y in a.years.split(','):
        raw = C.read_bytes(C.PINS % y)
        h = hashlib.sha256(raw).hexdigest()
        if h != PINS_IDFREE_SHA256[y]:
            print('PINS %s: sha256 %s, not the recorded %s' % (y, h[:16], PINS_IDFREE_SHA256[y][:16]))
            return 3
    go = E.go_line()
    if go is None:
        print("NO GO: GO_ASCII.txt lacks 'FREEZE: GO' or P-1")
        return 5
    sp = E.selftest_pins()
    vers = sp.get('versions', {})
    log('START driver: years %s, on-fail %s, prefetch %s; GO %s, P-1 %s; selftest %s %s; versions %s; zlib %s; platform %s; T-2: %s' % (a.years, a.on_fail, a.prefetch, go['freeze'], go['p1'], 'PASS' if sp['ok'] else 'FAIL', sp.get('bad'), json.dumps(vers, sort_keys=True), zlib.ZLIB_RUNTIME_VERSION, platform.platform(), NO_T2_WHY))
    if not sp['ok']:
        print('SELFTEST FAILED: %s' % sp['bad'])
        return 6
    for y in a.years.split(','):
        res = do_year(y, go['p1'] == 'on', a.on_fail, a.prefetch)
        log('YEAR %s RESULT: %s' % (y, res))
        if res == 'STOP':
            log('END: stopped at %s (see EXTRACTS/%s/COLAB_LOG_%s_ASCII.txt)' % (y, y, y))
            to_drive(LOCAL_LOG, os.path.join(RUN_DRIVE, os.path.basename(LOCAL_LOG)))
            return 4
    log('END: every year done (%s)' % a.years)
    to_drive(LOCAL_LOG, os.path.join(RUN_DRIVE, os.path.basename(LOCAL_LOG)))
    return 0
if __name__ == '__main__':
    sys.exit(main())
