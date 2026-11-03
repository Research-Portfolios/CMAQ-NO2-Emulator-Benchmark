"""Driver of the Amazon Web Services stage on Colab: runs the S3 download queue on its plan, which reads the public CMAS EQUATES bucket anonymously by ranged requests and writes the 2016 MCIP files, the daily emission extracts of 2017 and 2019 and the stack groups archives with their provenance rows, while restoring and mirroring its outputs on the Drive."""
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import zlib
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from choi_dl import common, dl_s3
from choi_dl.progress import Progress
DRIVE = '/content/drive/MyDrive'
EMUL = DRIVE + '/PROJECT_DATA/EMUL'
D_W = EMUL + '/W'
D_RUN = EMUL + '/RUN'
PLAN = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_S3.json')
REF = os.path.join(common.RUN, 'ref')
PAIRS = ((os.path.join(common.RAW, 'mcip'), D_W + '/raw/mcip', None), (os.path.join(common.RAW, 'emis'), D_W + '/raw/emis', None), (os.path.join(common.RAW, 's3meta'), D_W + '/raw/s3meta', None), (common.EXTRACT, D_W + '/extract', None), (common.WORK, D_W + '/work', lambda n: n == 'dl_refused_s3.txt'), (common.RUN_LOGS, D_RUN + '/logs', lambda n: n.startswith(('C5_', 'COMPARE_C5', 'DL_PROGRESS_C5', 'DL_STATUS_C5'))), (common.RUN_PROV, D_RUN + '/provenance', lambda n: n.startswith('s3_')))
PARTS = ('emis_merged', 'pt_marine', 'mcip', 'stackgroups', 's3meta')
LOG = os.path.join(common.RUN_LOGS, 'C5_COLAB_DRIVER_ASCII.txt')
EVERY_S = 120

def log(text):
    u, loc = common.now_pair()
    lines = common.wrap('%s ' % u, common.mask(text))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'ab') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii', 'replace'))
    print('\n'.join(lines), flush=True)

def under_emul(p):
    if not (p == EMUL or p.startswith(EMUL + '/')) or '..' in p:
        raise SystemExit(9)
    return p

def skip(test, path):
    name = os.path.basename(path)
    return name.endswith('.partial') or name.endswith('.lock') or (test is not None and (not test(name)))

def copy(src, dst, to_drive):
    if to_drive:
        under_emul(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)
SEEN = {}
MLOCK = threading.Lock()

def mirror():
    n = 0
    with MLOCK:
        for loc, dd, test in PAIRS:
            if not os.path.isdir(loc):
                continue
            for dp, dn, fn in os.walk(loc):
                for f in fn:
                    p = os.path.join(dp, f)
                    if skip(test, p):
                        continue
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    sig = (st.st_size, st.st_mtime_ns)
                    if SEEN.get(p) == sig:
                        continue
                    dst = os.path.join(dd, os.path.relpath(p, loc))
                    try:
                        copy(p, dst, True)
                    except OSError as e:
                        log('MIRROR ERROR %s: %s' % (f, str(e)[:120]))
                        continue
                    SEEN[p] = sig
                    n += 1
    return n

def clean_jsonl(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    lines = data.split(b'\n')
    keep = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            json.loads(ln.decode('ascii'))
            keep.append(ln)
        except (ValueError, UnicodeDecodeError):
            pass
    out = b'\n'.join(keep) + (b'\n' if keep else b'')
    if out != data:
        with open(path, 'wb') as fh:
            fh.write(out)
        return True
    return False

def restore():
    n, fixed = (0, 0)
    for loc, dd, test in PAIRS:
        if not os.path.isdir(dd):
            continue
        for dp, dn, fn in os.walk(dd):
            for f in fn:
                src = os.path.join(dp, f)
                if skip(test, src):
                    continue
                dst = os.path.join(loc, os.path.relpath(src, dd))
                if os.path.exists(dst):
                    continue
                copy(src, dst, False)
                n += 1
                if f.endswith('.jsonl') and clean_jsonl(dst):
                    fixed += 1
                st = os.stat(dst)
                SEEN[dst] = (st.st_size, st.st_mtime_ns)
    return (n, fixed)

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_os0(path):
    with open(path, 'rb') as fh:
        b = bytearray(fh.read())
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

def rows_of(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, 'rb') as fh:
        for ln in fh.read().decode('ascii', 'replace').split('\n'):
            if ln.strip():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                out[r.get('path', '').replace('\\', '/')] = r
    return out

def checks(plan):
    keys = plan.get('keys', {})
    folders = {'emis_merged': dl_s3.EM_DIR, 'pt_marine': dl_s3.PT_DIR, 'mcip': dl_s3.MCIP_DIR, 'stackgroups': dl_s3.SG_DIR, 's3meta': dl_s3.S3META}
    counts, out, etag_bad, md5 = ({}, [], [], {'equal': 0, 'differ': 0})
    for part in PARTS:
        new = rows_of(os.path.join(folders[part], 'PROVENANCE.jsonl'))
        old = rows_of(os.path.join(REF, 'KEPT_s3_%s_PROVENANCE.jsonl' % part))
        for path, r in new.items():
            src = str(r.get('source', ''))
            key = src.split('/', 3)[-1].split(' (member ')[0] if src.startswith('s3://') else ''
            if not key or keys.get(key, {}).get('etag') != r.get('etag'):
                etag_bad.append(path)
            et = str(r.get('etag') or '').strip('"')
            full = os.path.join(common.ROOT, *path.split('/'))
            if part in ('stackgroups', 's3meta') and et and ('-' not in et) and os.path.exists(full):
                md5['equal' if md5_of(full) == et else 'differ'] += 1
        for path in sorted(set(new) | set(old)):
            n, o = (new.get(path), old.get(path))
            s0 = ''
            if n and o:
                full = os.path.join(common.ROOT, *path.split('/'))
                if n.get('sha256') == o.get('sha256'):
                    v = 'MATCH'
                else:
                    s0 = sha256_os0(full) or '' if path.endswith('.npz') and os.path.exists(full) else ''
                    v = 'MATCH (OS BYTE)' if s0 and s0 == o.get('sha256') else 'DIFFER'
                mem = '' if 'member_sha256' not in o else str(n.get('member_sha256') == o.get('member_sha256'))
            elif n:
                v, mem = ('NO KEPT ROW', '')
            else:
                v, mem = ('NOT REBUILT', '')
            counts[part, v] = counts.get((part, v), 0) + 1
            out.append({'part': part, 'path': path, 'verdict': v, 'sha256_new': (n or {}).get('sha256', ''), 'sha256_new_os0': s0, 'sha256_kept': (o or {}).get('sha256', ''), 'member_sha256_equal': mem, 'etag_equal': '' if not (n and o) else str(n.get('etag') == o.get('etag'))})
    cp = os.path.join(common.RUN_LOGS, 'COMPARE_C5.csv')
    with open(cp + '.partial', 'w', newline='', encoding='ascii') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ['part'], lineterminator='\n')
        w.writeheader()
        for r in out:
            w.writerow(r)
    os.replace(cp + '.partial', cp)
    text = ["COMPARE C5 (each rebuilt row against the local machine's kept row, run\\provenance\\s3_<part>_PROVENANCE.jsonl on the work disk): " + '; '.join(('%s %s %d' % (p, v, c) for (p, v), c in sorted(counts.items()))) + ". ETags unequal to the plan: %d. Single-part objects by MD5 against the ETag: %d equal, %d differ. zlib %s on the VM, 1.3.1 on the PC (the local machine's not recorded)." % (len(etag_bad), md5['equal'], md5['differ'], zlib.ZLIB_RUNTIME_VERSION)]
    p = os.path.join(common.RUN_LOGS, 'COMPARE_C5_ASCII.txt')
    with open(p, 'w', encoding='ascii', newline='\n') as fh:
        for t in text:
            fh.write('\n'.join(common.wrap('', t)) + '\n')
    return (counts, etag_bad, md5)

def main():
    os.makedirs(common.RUN_LOGS, exist_ok=True)
    lk = open(os.path.join(common.RUN_LOGS, 'C5_DRIVER.lock'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ANOTHER DRIVER RUNS ON THIS VM: not started')
        return 10
    for _, dd, _t in PAIRS:
        under_emul(dd)
    common.set_temp()
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(143))
    n, fixed = restore()
    log('C5 START: %d files restored from EMUL/W and EMUL/RUN (%d provenance files had a cut last line, dropped); VM disk free %.1f GB; zlib %s' % (n, fixed, shutil.disk_usage('/content').free / 1000000000.0, zlib.ZLIB_RUNTIME_VERSION))
    plan = json.load(open(PLAN, 'rb'))
    stop = threading.Event()

    def loop():
        while not stop.wait(EVERY_S):
            mirror()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc = 1
    try:
        progress = Progress(progress_path=os.path.join(common.RUN_LOGS, 'DL_PROGRESS_C5_ASCII.txt'), status_path=os.path.join(common.RUN_LOGS, 'DL_STATUS_C5.json'))
        res = dl_s3.run(plan, progress)
        bad = len(res.get('failed', [])) + len(res.get('refused', []))
        log('C5 DL_S3 RESULT: done %s, skipped %s, failed %d, refused %d, held %d, bytes %s, stop %s' % (res.get('done'), res.get('skipped'), len(res.get('failed', [])), len(res.get('refused', [])), len(res.get('held', [])), res.get('bytes'), res.get('stop')))
        with open(os.path.join(common.RUN_LOGS, 'C5_RESULT.json'), 'w', encoding='ascii') as fh:
            json.dump(res, fh, indent=1, sort_keys=True, default=str)
        counts, etag_bad, md5 = checks(plan)
        log('C5 CHECKS: ETags unequal to the plan %d; single-part MD5 equal %d, differ %d; see RUN/logs/COMPARE_C5_ASCII.txt' % (len(etag_bad), md5['equal'], md5['differ']))
        rc = 0 if bad == 0 and (not etag_bad) and (md5['differ'] == 0) else 1
    finally:
        stop.set()
        th.join(timeout=5)
        log('C5 END: exit %d; mirror at the end' % rc)
        log('C5 MIRROR: %d files copied to EMUL/W and EMUL/RUN' % mirror())
    return rc
if __name__ == '__main__':
    sys.exit(main())
