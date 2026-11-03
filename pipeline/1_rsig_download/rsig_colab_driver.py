"""Driver of the RSIG download stage on Colab: runs the download queue on the RSIG plan (677 requests), restores earlier outputs from the Drive and mirrors new ones there every 120 s, and at the end compares each file's sha256 with the hashes kept from an earlier download of the same requests, writing the comparison as text and CSV."""
import csv
import fcntl
import json
import os
import shutil
import signal
import sys
import threading
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from choi_dl import common, dl_rsig
from choi_dl.progress import Progress
DRIVE = '/content/drive/MyDrive'
EMUL = DRIVE + '/PROJECT_DATA/EMUL'
RAW_RSIG = os.path.join(common.RAW, 'rsig')
PLAN = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_RSIG.json')
KEPT = os.path.join(common.RUN, 'ref', 'KEPT_rsig_HASHES.csv')
LOG = os.path.join(common.RUN_LOGS, 'RSIG_COLAB_DRIVER_ASCII.txt')
EVERY_S = 120
PAIRS = ((RAW_RSIG, EMUL + '/W/raw/rsig'), (os.path.join(common.WORK, 'dl_refused_rsig.txt'), EMUL + '/W/work/dl_refused_rsig.txt'))
RUN_FILES = ('DL_PROGRESS_RSIG_ASCII.txt', 'DL_STATUS_RSIG.json', 'RSIG_COLAB_DRIVER_ASCII.txt', 'COMPARE_RSIG_ASCII.txt', 'COMPARE_RSIG.csv', 'RSIG_VM_FREEZE_ASCII.txt')

def log(text):
    u, loc = common.now_pair()
    lines = common.wrap('%s ' % u, common.mask(text))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'ab') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii', 'replace'))
    print('\n'.join(lines), flush=True)

def under_emul(p):
    if not p.startswith(EMUL + '/') or '..' in p:
        raise SystemExit(9)
    return p

def copy(src, dst, to_drive):
    if to_drive:
        under_emul(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)

def pairs():
    out = list(PAIRS)
    names = list(RUN_FILES)
    if os.path.isdir(common.RUN_LOGS):
        names += [n for n in sorted(os.listdir(common.RUN_LOGS)) if n.startswith('RSIG_VM_FREEZE_')]
    for name in names:
        out.append((os.path.join(common.RUN_LOGS, name), EMUL + '/RUN/logs/' + name))
    for name in ('rsig_PROVENANCE.jsonl', 'rsig_PROVENANCE_ASCII.txt'):
        out.append((os.path.join(common.RUN_PROV, name), EMUL + '/RUN/provenance/' + name))
    return out
SEEN = {}
MLOCK = threading.Lock()

def files_of(loc, dd):
    if os.path.isdir(loc):
        for dp, dn, fn in os.walk(loc):
            for f in fn:
                p = os.path.join(dp, f)
                yield (p, os.path.join(dd, os.path.relpath(p, loc)))
    elif os.path.isfile(loc):
        yield (loc, dd)

def mirror():
    n = 0
    with MLOCK:
        for loc, dd in pairs():
            for p, dst in files_of(loc, dd):
                if p.endswith('.partial'):
                    continue
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                sig = (st.st_size, st.st_mtime_ns)
                if SEEN.get(p) == sig:
                    continue
                try:
                    copy(p, dst, True)
                except OSError as e:
                    log('MIRROR ERROR %s: %s' % (os.path.basename(p), str(e)[:120]))
                    continue
                SEEN[p] = sig
                n += 1
    return n

def clean_jsonl(path):
    data = open(path, 'rb').read()
    keep = []
    for ln in data.split(b'\n'):
        if ln.strip():
            try:
                json.loads(ln.decode('ascii'))
                keep.append(ln)
            except (ValueError, UnicodeDecodeError):
                pass
    out = b'\n'.join(keep) + (b'\n' if keep else b'')
    if out != data:
        open(path, 'wb').write(out)
        return True
    return False

def restore():
    n, fixed = (0, 0)
    for loc, dd in pairs():
        src_files = []
        if os.path.isdir(dd):
            for dp, dn, fn in os.walk(dd):
                for f in fn:
                    s = os.path.join(dp, f)
                    src_files.append((s, os.path.join(loc, os.path.relpath(s, dd))))
        elif os.path.isfile(dd):
            src_files.append((dd, loc))
        for s, dst in src_files:
            if s.endswith('.partial'):
                continue
            if os.path.exists(dst):
                continue
            copy(s, dst, False)
            n += 1
            if dst.endswith('.jsonl') and clean_jsonl(dst):
                fixed += 1
            st = os.stat(dst)
            SEEN[dst] = (st.st_size, st.st_mtime_ns)
    return (n, fixed)

def compare():
    new = {}
    pj = os.path.join(RAW_RSIG, 'PROVENANCE.jsonl')
    if os.path.exists(pj):
        for ln in open(pj, 'rb').read().decode('ascii', 'replace').split('\n'):
            if ln.strip():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                new[r.get('path', '').replace('\\', '/')] = r
    old = {}
    with open(KEPT, newline='', encoding='ascii') as fh:
        for r in csv.DictReader(fh):
            old[r['path'].replace('\\', '/')] = r
    counts, out = ({}, [])
    for path in sorted(set(new) | set(old)):
        n, o = (new.get(path), old.get(path))
        if n and o:
            if n.get('sha256') == o['sha256']:
                v = 'MATCH'
            elif o['member_sha256'] and n.get('member_sha256') == o['member_sha256']:
                v = 'MATCH (DECOMPRESSED)'
            else:
                v = 'DIFFER'
        elif n:
            v = 'NO KEPT ROW'
        else:
            v = 'NOT REBUILT'
        counts[v] = counts.get(v, 0) + 1
        out.append({'path': path, 'verdict': v, 'sha256_new': (n or {}).get('sha256', ''), 'sha256_kept': (o or {}).get('sha256', ''), 'member_sha256_new': (n or {}).get('member_sha256', ''), 'member_sha256_kept': (o or {}).get('member_sha256', '')})
    cp = os.path.join(common.RUN_LOGS, 'COMPARE_RSIG.csv')
    with open(cp, 'w', newline='', encoding='ascii') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ['path'], lineterminator='\n')
        w.writeheader()
        for r in out:
            w.writerow(r)
    text = "COMPARE RSIG (each answer's row against the local machine's kept hashes, run\\provenance\\rsig_PROVENANCE.jsonl on the work disk): " + ', '.join(('%s %d' % (k, counts[k]) for k in sorted(counts))) + '. MATCH (DECOMPRESSED): the gzip bytes differ, the netCDF bytes inside are equal. DIFFER: reported, not hidden.'
    with open(os.path.join(common.RUN_LOGS, 'COMPARE_RSIG_ASCII.txt'), 'w', encoding='ascii', newline='\n') as fh:
        fh.write('\n'.join(common.wrap('', text)) + '\n')
    return counts

def main():
    os.makedirs(common.RUN_LOGS, exist_ok=True)
    lk = open(os.path.join(common.RUN_LOGS, 'RSIG_DRIVER.lock'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ANOTHER DRIVER RUNS ON THIS VM: not started')
        return 10
    for _, dd in pairs():
        under_emul(dd)
    common.set_temp()
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(143))
    n, fixed = restore()
    log('RSIG START: %d files restored from the Drive (%d provenance files had a cut last line, dropped); VM disk free %.1f GB' % (n, fixed, shutil.disk_usage('/content').free / 1000000000.0))
    plan = json.load(open(PLAN, 'rb'))
    stop = threading.Event()

    def loop():
        while not stop.wait(EVERY_S):
            mirror()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc = 1
    try:
        progress = Progress(progress_path=os.path.join(common.RUN_LOGS, 'DL_PROGRESS_RSIG_ASCII.txt'), status_path=os.path.join(common.RUN_LOGS, 'DL_STATUS_RSIG.json'))
        res = dl_rsig.run(plan, progress)
        bad = len(res.get('failed', [])) + len(res.get('refused', []))
        log('RSIG RESULT: done %s, skipped %s, failed %d, refused %d, stop %s' % (res.get('done'), res.get('skipped'), len(res.get('failed', [])), len(res.get('refused', [])), res.get('stop')))
        counts = compare()
        log('RSIG COMPARE: %s' % ', '.join(('%s %d' % (k, counts[k]) for k in sorted(counts))))
        rc = 0 if bad == 0 else 1
    finally:
        stop.set()
        th.join(timeout=5)
        log('RSIG END: exit %d' % rc)
        log('RSIG MIRROR: %d files copied to the Drive' % mirror())
    return rc
if __name__ == '__main__':
    sys.exit(main())
