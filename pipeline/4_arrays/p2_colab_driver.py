"""Driver of the arrays stage on Colab: waits for the RSIG and Amazon Web Services stages to finish, restores their outputs from the Drive, runs the stage's 22 steps in order as separate processes (verification, inventory, Natural Earth, grid and mask, input checks, emissions, round trip, splits, arrays, index, statistics, checks and file list), stops at the first failure, and mirrors outputs to the Drive."""
import csv
import fcntl
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from choi_dl import common
DRIVE = '/content/drive/MyDrive'
EMUL = DRIVE + '/PROJECT_DATA/EMUL'
D_P2 = EMUL + '/W_P2'
D_RUN = EMUL + '/RUN'
W = common.W
RUN = common.RUN
LOGS = common.RUN_LOGS
CFG = common.RUN_CONFIG
STEPS_DIR = os.path.join(LOGS, 'P2_STEPS')
LOG = os.path.join(LOGS, 'P2_COLAB_DRIVER_ASCII.txt')
DONE = os.path.join(LOGS, 'P2_DONE_ASCII.txt')
REF = os.path.join(RUN, 'ref')
PLAN_S = os.path.join(CFG, 'S2A_PLAN_S3.json')
INV = os.path.join(CFG, 'S2C_INVENTORY.json')
WORK_P2 = os.path.join(W, 'arrays_p2', 'work')
S2C_TMP = os.path.join(W, 'tmp', 's2c')
EVERY_S = 120
WAIT_EVERY_S = 300
WAIT_MAX_S = 7200
NEED_AFTER_GB = 56.2
NE_GB = 0.2
SCRATCH_GB = 1.0
INPUT_DIRS = ('raw/rsig', 'raw/mcip', 'raw/emis/stackgroups', 'raw/s3meta', 'extract/emis_merged', 'extract/pt_marine')
INPUT_FILES = ('work/dl_refused_s3.txt', 'work/dl_refused_rsig.txt')
OTHER_LOGS = ('C5_COLAB_DRIVER_ASCII.txt', 'DL_PROGRESS_C5_ASCII.txt', 'DL_STATUS_C5.json', 'RSIG_COLAB_DRIVER_ASCII.txt', 'DL_PROGRESS_RSIG_ASCII.txt', 'DL_STATUS_RSIG.json')
NEED_LOGS = ('DL_PROGRESS_C5_ASCII.txt', 'DL_PROGRESS_RSIG_ASCII.txt')
OWN_LOG_PREFIX = ('P2_', 'COMPARE_P2', 'S3P2_', 'S2C_')
FIX = 'in the Drive web page, rename the W folder that holds raw/rsig to W_RSIG, then choose Run all again (the scan then sees both)'

def _own_log(rel):
    first = rel.split('/')[0]
    return first.startswith(OWN_LOG_PREFIX) and first not in OTHER_LOGS
PAIRS = ((os.path.join(W, 'arrays_p2'), D_P2 + '/arrays_p2', None), (os.path.join(W, 'natural_earth'), D_P2 + '/natural_earth', None), (S2C_TMP, D_P2 + '/work/s2c', None), (os.path.join(W, 'tmp', 's3p2'), D_P2 + '/work/s3p2', None), (os.path.join(W, 'raw', 's3meta'), D_P2 + '/raw/s3meta', None), (os.path.join(W, 'extract', 'pt_marine'), D_P2 + '/extract/pt_marine', None), (os.path.join(W, 'work', 's2a'), D_P2 + '/work/s2a', None), (os.path.join(CFG, 'p2'), D_RUN + '/config/p2', None), (CFG, D_RUN + '/config', lambda r: r == 'S2C_INVENTORY.json'), (LOGS, D_RUN + '/logs', _own_log), (common.RUN_PROV, D_RUN + '/provenance', lambda r: r.startswith('ne_')))
NOT_OUTPUTS = (LOG, DONE)

def log(text):
    u, loc = common.now_pair()
    lines = common.wrap('%s ' % u, common.mask(text))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'ab') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii', 'replace'))
    print('\n'.join(lines), flush=True)

def under(p):
    ok = (p.startswith(D_P2 + '/') or p.startswith(D_RUN + '/')) and '..' not in p
    if not ok:
        raise SystemExit(9)
    return p

def skip_name(name):
    return name.endswith('.partial') or name.endswith('.lock')

def walk(top, test=None):
    if not os.path.isdir(top):
        return
    for dp, dn, fn in os.walk(top):
        dn.sort()
        for f in sorted(fn):
            if skip_name(f):
                continue
            p = os.path.join(dp, f)
            r = os.path.relpath(p, top).replace(os.sep, '/')
            if test is None or test(r):
                yield (r, p)

def copy(src, dst, to_drive):
    if to_drive:
        under(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)
SEEN = {}
MLOCK = threading.Lock()
CHILD = [None]
LOCK_FD = [None]
FROM_W = set()

def stop_child():
    p = CHILD[0]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=60)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=60)
            except Exception:
                pass

def sig_of(p):
    st = os.stat(p)
    return (st.st_size, st.st_mtime_ns)

def mirror():
    n, err = (0, 0)
    with MLOCK:
        for loc, dd, test in PAIRS:
            for r, p in walk(loc, test):
                if loc == CFG and '/' in r:
                    continue
                try:
                    sig = sig_of(p)
                except OSError:
                    continue
                if SEEN.get(p) == sig:
                    continue
                try:
                    copy(p, dd + '/' + r, True)
                except OSError as e:
                    err += 1
                    log('MIRROR ERROR %s: %s' % (os.path.basename(p), str(e)[:120]))
                    continue
                SEEN[p] = sig
                n += 1
    return (n, err)

def snapshot():
    out = {}
    for loc, dd, test in PAIRS:
        for r, p in walk(loc, test):
            if loc == CFG and '/' in r or p in NOT_OUTPUTS or p.startswith(STEPS_DIR + os.sep):
                continue
            try:
                out[p] = sig_of(p)
            except OSError:
                pass
    return out

def clean_jsonl(path):
    with open(path, 'rb') as fh:
        data = fh.read()
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
        with open(path, 'wb') as fh:
            fh.write(out)
        return True
    return False

def drivers_done():
    why = []
    for name, tag in (('C5_COLAB_DRIVER_ASCII.txt', 'C5'), ('RSIG_COLAB_DRIVER_ASCII.txt', 'RSIG')):
        p = D_RUN + '/logs/' + name
        if not os.path.isfile(p):
            why.append('%s is not on the Drive' % name)
            continue
        with open(p, 'rb') as fh:
            lines = fh.read().decode('ascii', 'replace').split('\n')
        starts = [i for i, ln in enumerate(lines) if tag + ' START:' in ln]
        ends = [i for i, ln in enumerate(lines) if tag + ' END: exit ' in ln]
        last_start = starts[-1] if starts else -1
        if not ends or ends[-1] < last_start:
            why.append('%s: no END line after its last START (running)' % name)
        elif tag + ' END: exit 0' not in lines[ends[-1]]:
            why.append('%s: its last END line is not exit 0' % name)
    return why

def wait_for_drivers():
    t0 = time.time()
    while True:
        why = drivers_done()
        if not why:
            log('ARRAYS WAIT: both download drivers ended exit 0')
            return True
        if time.time() - t0 >= WAIT_MAX_S:
            log('ARRAYS WAIT ENDED after %.1f h: %s' % ((time.time() - t0) / 3600, '; '.join(why)))
            return False
        if any(('is not exit 0' in w for w in why)):
            log('ARRAYS WAIT ENDED: %s (a driver ended with an error)' % '; '.join(why))
            return False
        log('ARRAYS WAITS (polls every 5 min, up to 2 h): %s' % '; '.join(why))
        time.sleep(WAIT_EVERY_S)

def w_copies():
    out = []
    for n in sorted(os.listdir(EMUL)):
        if n == 'W_P2' or not os.path.isdir(os.path.join(EMUL, n)):
            continue
        if n == 'W' or n.startswith(('W ', 'W(', 'W_')):
            out.append(n)
    return out

def restore_own_logs():
    n = 0
    for r, src in walk(D_RUN + '/logs', _own_log):
        dst = os.path.join(LOGS, *r.split('/'))
        if not os.path.exists(dst):
            copy(src, dst, False)
            n += 1
            SEEN[dst] = sig_of(dst)
        elif os.path.getsize(dst) == os.path.getsize(src):
            SEEN[dst] = sig_of(dst)
    return n

def prime_seen(held):
    from_w = {os.path.join(W, 'raw', 's3meta'): 'raw/s3meta/', os.path.join(W, 'extract', 'pt_marine'): 'extract/pt_marine/'}
    for loc, dd, test in PAIRS:
        for r, p in walk(loc, test):
            if p in SEEN or (loc == CFG and '/' in r):
                continue
            size = os.path.getsize(p)
            dst = dd + '/' + r
            same = os.path.isfile(dst) and os.path.getsize(dst) == size
            if not same and loc in from_w:
                same = any((s == size for _c, s, _p in held.get(from_w[loc] + r, [])))
            if same:
                SEEN[p] = sig_of(p)

def restore_plan():
    jobs = {}
    keep = covered()

    def add(dst, src):
        if dst not in jobs and (not os.path.exists(dst)):
            jobs[dst] = (src, os.path.getsize(src))
    for loc, dd, test in PAIRS:
        if loc == LOGS:
            continue
        for r, src in walk(dd, test):
            if loc == CFG and '/' in r:
                continue
            dst = os.path.join(loc, *r.split('/'))
            if common.rel(dst) in keep:
                add(dst, src)
    copies = w_copies()
    held = {}
    for c in copies:
        for d in INPUT_DIRS + INPUT_FILES:
            base = os.path.join(EMUL, c, *d.split('/'))
            if os.path.isfile(base):
                held.setdefault(d, []).append((c, os.path.getsize(base), base))
            for r, src in walk(base):
                held.setdefault(d + '/' + r, []).append((c, os.path.getsize(src), src))
    refused = sorted((r for r, lst in held.items() if len(set((s for _c, s, _p in lst))) > 1))
    missing = [d for d in INPUT_DIRS if not any((r.startswith(d + '/') for r in held))]
    for r, lst in sorted(held.items()):
        dst = os.path.join(W, *r.split('/'))
        FROM_W.add(dst)
        add(dst, lst[0][2])
    for r, src in walk(D_RUN + '/provenance'):
        dst = os.path.join(common.RUN_PROV, r)
        if '/' not in r and (not r.startswith('ne_') or common.rel(dst) in keep):
            add(dst, src)
    for name in OTHER_LOGS:
        src = D_RUN + '/logs/' + name
        if os.path.isfile(src):
            jobs[os.path.join(LOGS, name)] = (src, os.path.getsize(src))
    missing += ['RUN/logs/' + n for n in NEED_LOGS if not os.path.isfile(D_RUN + '/logs/' + n)]
    return (jobs, copies, refused, missing, held)

def run_restore(jobs):
    items = sorted(jobs.items())
    done, nbytes, fixed = ([0], [0], [0])
    lk = threading.Lock()

    def one(it):
        dst, (src, size) = it
        copy(src, dst, False)
        f = dst.endswith('.jsonl') and clean_jsonl(dst)
        with lk:
            SEEN[dst] = sig_of(dst)
            done[0] += 1
            nbytes[0] += size
            fixed[0] += int(bool(f))
            if done[0] % 500 == 0:
                log('ARRAYS RESTORE: %d of %d files, %.2f GB' % (done[0], len(items), nbytes[0] / 1000000000.0))
    with ThreadPoolExecutor(4) as ex:
        for _ in ex.map(one, items):
            pass
    return (done[0], nbytes[0], fixed[0])

def step0():
    jobs, copies, refused, missing, held = restore_plan()
    log('ARRAYS RESTORE PLAN: W copies on the Drive %s; files to copy %d, %.2f GB' % (copies or 'none', len(jobs), sum((s for _p, s in jobs.values())) / 1000000000.0))
    if refused:
        log('ARRAYS RESTORE REFUSED: %d relative paths are held by two W copies with different sizes (first: %s); a manual decision is needed which copy is whole' % (len(refused), refused[0]))
        return 5
    if missing:
        log('ARRAYS RESTORE STOPPED: not in any W copy on the Drive: %s. The fix: %s.' % (', '.join(missing), FIX))
        return 5
    free = shutil.disk_usage('/content').free / 1000000000.0
    to_copy = sum((s for _p, s in jobs.values())) / 1000000000.0
    a2 = os.path.join(W, 'arrays_p2') + os.sep
    p2_have = sum((s for d, (_p, s) in jobs.items() if d.startswith(a2))) + sum((os.path.getsize(p) for _r, p in walk(a2.rstrip(os.sep))))
    need = to_copy + NE_GB + SCRATCH_GB + max(0.0, NEED_AFTER_GB - p2_have / 1000000000.0)
    try:
        import psutil
        vm = psutil.virtual_memory()
        mem = 'memory %.1f GB total, %.1f GB available; %d CPUs' % (vm.total / 1000000000.0, vm.available / 1000000000.0, os.cpu_count() or 0)
    except Exception:
        mem = 'memory not read; %d CPUs' % (os.cpu_count() or 0)
    log("ARRAYS DISK AND MEMORY: VM disk free %.1f GB; needed %.1f GB (inputs to copy %.2f GB + Natural Earth %.1f GB + scratch %.1f GB + stage (1)'s ask %.1f GB less its files already made %.2f GB); %s" % (free, need, to_copy, NE_GB, SCRATCH_GB, NEED_AFTER_GB, p2_have / 1000000000.0, mem))
    if free < need:
        log('ARRAYS DISK TOO SMALL: %.1f GB free, about %.1f GB needed: choose a runtime with a larger disk' % (free, need))
        return 7
    try:
        n, nb, fixed = run_restore(jobs)
    except OSError as e:
        log('ARRAYS RESTORE ERROR (a copy from the Drive failed): %s: %s; a second Run all copies only what is missing' % (type(e).__name__, str(e)[:160]))
        return 1
    prime_seen(held)
    log('ARRAYS RESTORE DONE: %d files, %.2f GB copied to the VM (%d provenance files had a cut last line, dropped); no hash pass (s2c_verify checks every input); VM disk free now %.1f GB' % (n, nb / 1000000000.0, fixed, shutil.disk_usage('/content').free / 1000000000.0))
    return 0

def jload(p):
    with open(p, 'rb') as fh:
        return json.loads(fh.read().decode('ascii'))

def marker_path(nn, name):
    return os.path.join(STEPS_DIR, '%s_%s_DONE.json' % (nn, name))

def _marker_ok(nn, name):
    p = marker_path(nn, name)
    try:
        return os.path.isfile(p) and jload(p).get('exit') == 0
    except ValueError:
        return False

def _out(o):
    return (o[0], o[1]) if isinstance(o, list) else (o, None)

def skip_done(nn, name):
    p = marker_path(nn, name)
    if not os.path.isfile(p):
        return False
    try:
        m = jload(p)
    except ValueError:
        return False
    if m.get('exit') != 0:
        return False
    for o in m.get('outputs', []):
        r, size = _out(o)
        q = os.path.join(common.ROOT, *r.split('/'))
        if not os.path.isfile(q) or (size is not None and os.path.getsize(q) != size):
            return False
    return True

def covered(skip=None):
    keep = set()
    if os.path.isdir(STEPS_DIR):
        for f in sorted(os.listdir(STEPS_DIR)):
            if not f.endswith('_DONE.json') or f == skip:
                continue
            try:
                m = jload(os.path.join(STEPS_DIR, f))
            except ValueError:
                continue
            if m.get('exit') == 0:
                keep.update((_out(o)[0] for o in m.get('outputs', [])))
    return keep

def write_marker(nn, name, rc, sec, outputs, note=''):
    os.makedirs(STEPS_DIR, exist_ok=True)
    m = {'step': nn, 'name': name, 'utc': common.now_pair()[0], 'exit': rc, 'seconds': round(sec, 1), 'outputs': sorted(outputs), 'note': note}
    p = marker_path(nn, name)
    with open(p + '.partial', 'w', encoding='ascii', newline='\n') as fh:
        fh.write(json.dumps(m, indent=1) + '\n')
    os.replace(p + '.partial', p)

def run_child(nn, name, argv):
    os.makedirs(STEPS_DIR, exist_ok=True)
    os.makedirs(common.W_TMP, exist_ok=True)
    env = dict(os.environ, TMP=common.W_TMP, TEMP=common.W_TMP, TMPDIR=common.W_TMP, CARTOPY_DATA_DIR=common.NATURAL_EARTH)
    cmd = [sys.executable, '-I', '-B'] + argv
    shown = ' '.join(['python -I -B'] + [common.rel(a) if os.path.isabs(a) else a for a in argv])
    out = os.path.join(STEPS_DIR, '%s_%s_stdout.txt' % (nn, name))
    lines = []
    with open(out, 'ab') as fo:
        fo.write(('--- %s %s\n' % (common.now_pair()[0], shown)).encode('ascii', 'replace'))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', env=env, cwd=common.W_TMP, start_new_session=True, pass_fds=(LOCK_FD[0],) if LOCK_FD[0] is not None else ())
        CHILD[0] = p
        try:
            for ln in p.stdout:
                t = common.mask(ln.rstrip('\n'))
                lines.append(t)
                print('  ' + t, flush=True)
                fo.write((t + '\n').encode('ascii', 'replace'))
            rc = p.wait()
        except BaseException:
            stop_child()
            raise
        finally:
            CHILD[0] = None
    return (rc, '\n'.join(lines))

def script(rel):
    return os.path.join(HERE, *rel.split('/'))

def check_inventory(_text):
    inv = jload(INV)
    st = inv['stop']
    bad = [] if st['verdict'] == 'PASS' else ['S2C_INVENTORY.json verdict %s: %s' % (st['verdict'], '; '.join(st['reasons'][:8]))]
    try:
        inventory_beside(inv)
    except Exception as e:
        log('ARRAYS INVENTORY RECORD not made: %s' % type(e).__name__)
    return bad

def inventory_beside(inv):
    old = jload(os.path.join(REF, 'KEPT_S2C_INVENTORY.json'))
    nk, ok_ = (inv['sectors']['kept'], old['sectors']['kept'])
    log('ARRAYS INVENTORY BESIDE THE KEPT ONE: sectors kept %s (local machine %s): %s' % (nk, ok_, 'EQUAL' if sorted(nk) == sorted(ok_) else 'DIFFER'))
    log('ARRAYS INVENTORY: sectors dropped %s (local machine %s)' % (sorted(inv['sectors']['dropped']), sorted(old['sectors']['dropped'])))
    for y in sorted(set(inv['day_sets']['emis_merged']) | set(old['day_sets']['emis_merged'])):
        a = inv['day_sets']['emis_merged'].get(y, {}).get('dates', [])
        b = old['day_sets']['emis_merged'].get(y, {}).get('dates', [])
        log('ARRAYS INVENTORY: emis_merged %s: %d days (local machine %d); %s' % (y, len(a), len(b), 'EQUAL' if a == b else 'DIFFER'))
    pn, po = (inv['day_sets']['pt_marine'], old['day_sets']['pt_marine'])
    for s in sorted(set(pn) | set(po)):
        for y in sorted(set(pn.get(s, {})) | set(po.get(s, {}))):
            a = pn.get(s, {}).get(y, {}).get('dates', [])
            b = po.get(s, {}).get(y, {}).get('dates', [])
            log('ARRAYS INVENTORY: pt_marine %s %s: %d days (local machine %d); %s' % (s, y, len(a), len(b), 'EQUAL' if a == b else 'DIFFER'))

    def inputs(i):
        c = {}
        for e in i['files']:
            if e.get('s3_input') is True and e.get('verified') is True:
                c[e['kind']] = c.get(e['kind'], 0) + 1
        return c
    log('ARRAYS INVENTORY: stage (1) inputs by kind %s (local machine %s)' % (json.dumps(inputs(inv), sort_keys=True), json.dumps(inputs(old), sort_keys=True)))

def check_d4(_text):
    s = jload(os.path.join(WORK_P2, 'd4_summary.json'))
    return [] if s.get('verdict') == 'PASS' else ['input-check verdict %s (over the limit %s; failing checks %s; grid differs %s)' % (s.get('verdict'), s.get('over_limit'), s.get('checks_failed'), s.get('grid_differ'))]

def check_selftest(text):
    return [] if 'SELFTEST PASS' in text else ['SELFTEST PASS not printed']

def check_splits(_text):
    s = jload(os.path.join(CFG, 'p2', 'SPLITS.json'))
    return [] if s.get('all_tests_pass') is True else ['SPLITS.json all_tests_pass is %s' % s.get('all_tests_pass')]

def check_checks(text):
    c = jload(os.path.join(WORK_P2, 'checks_p2.json'))
    bad = []
    if (c.get('second_reading') or {}).get('pass') is not True:
        bad.append("(1) the second reading's pass is not true")
    if 'all equal: True' not in text:
        bad.append("(2) the lags: 'all equal: True' not printed")
    for y, v in sorted((c.get('finite') or {}).items()):
        if v.get('nonfinite_extremes_built'):
            bad.append('(3) %s: %s non-finite day extremes' % (y, v['nonfinite_extremes_built']))
        if v.get('unfilled_extremes_all_nan') is False:
            bad.append('(3) %s: unfilled slots not all NaN' % y)
    new = c.get('new') or {}
    for y, v in sorted((new.get('L6') or {}).items()):
        if v.get('days_differing'):
            bad.append('(4) L-6 %s: %s days differ' % (y, v['days_differing']))
    for y, v in sorted((new.get('E') or {}).items()):
        if v.get('E_exact') != v.get('days') or v.get('P1_within_4_steps') != v.get('days'):
            bad.append('(4) E and P-1 %s: E exact %s, P-1 within 4 steps %s of %s days' % (y, v.get('E_exact'), v.get('P1_within_4_steps'), v.get('days')))
    for y, v in sorted((new.get('c7') or {}).items()):
        if v.get('equal_to_clim') != v.get('days'):
            bad.append('(4) emission climatology %s: %s of %s days equal' % (y, v.get('equal_to_clim'), v.get('days')))
    for y, v in sorted((new.get('unfilled') or {}).items()):
        if v.get('emission_slots_all_nan') not in (None, v.get('days')) or v.get('other_float32_channels_finite') != v.get('days'):
            bad.append('(4) unfilled %s: %s' % (y, json.dumps(v, sort_keys=True)[:120]))
    return bad

def step_refetch(nn, name):
    vr = jload(os.path.join(S2C_TMP, 'verify_results_post.json'))
    ids = sorted((e['id'] for e in vr['entries'] if e['kind'] == 'pt_marine' and (not e['verified'])))
    if not ids:
        for n in ('refetch_result.json', 'refetch_state.json'):
            p = os.path.join(S2C_TMP, n)
            if not os.path.exists(p):
                with open(p, 'w', encoding='ascii', newline='\n') as fh:
                    fh.write('{}\n')
        log('ARRAYS STEP %s %s: no point and marine item failed; {} written where absent' % (nn, name))
        return (0, '')
    log('ARRAYS STEP %s %s: %d point and marine items failed: refetched by s2c_refetch.py, then s2c_verify.py --tag post again' % (nn, name, len(ids)))
    rc, t1 = run_child(nn, name, [script('s2c/s2c_refetch.py')] + ids)
    if rc != 0:
        return (rc, t1)
    rc, t2 = run_child(nn, name, [script('s2c/s2c_verify.py'), '--tag', 'post', '--workers', str(WORKERS)])
    return (rc, t1 + '\n' + t2)
WORKERS = max(2, min(8, os.cpu_count() or 2))
STEPS = [('01', 'contents', [HERE + '/p2_colab_driver.py', '--step', 'contents'], None, False), ('02', 'natural_earth', [script('s2a/s2a_s3_ne.py')], None, False), ('03', 'rsig_headers', [HERE + '/p2_colab_driver.py', '--step', 'headers'], None, False), ('04', 's2c_verify', [script('s2c/s2c_verify.py'), '--tag', 'post', '--workers', str(WORKERS)], None, True), ('05', 's2c_refetch', step_refetch, None, True), ('06', 's2c_inventory', [script('s2c/s2c_inventory.py')], check_inventory, True), ('07', 'p2_disk_start', [script('s3p2/p2_disk.py'), 'start'], None, True), ('08', 'p2_ne_fetch', [script('s3p2/p2_ne_fetch.py')], None, True), ('09', 'p2_grid_mask', [script('s3p2/p2_grid_mask.py')], None, True), ('10', 'p2_d4', [script('s3p2/p2_d4.py'), '--workers', '3'], None, True), ('11', 'p2_d4_record', [script('s3p2/p2_d4_record.py')], check_d4, True), ('12', 'p2_emis', [script('s3p2/p2_emis.py')], None, True), ('13', 'p2_roundtrip_selftest', [script('s3p2/p2_roundtrip.py'), '--selftest'], check_selftest, True), ('14', 'p2_roundtrip', [script('s3p2/p2_roundtrip.py')], None, True), ('15', 'p2_splits', [script('s3p2/p2_splits.py')], check_splits, True), ('16', 'p2_build', [script('s3p2/p2_build.py')], None, True), ('17', 'p2_c7', [script('s3p2/p2_c7.py')], None, True), ('18', 'p2_index', [script('s3p2/p2_index.py')], None, True), ('19', 'p2_norm', [script('s3p2/p2_norm.py')], None, True), ('20', 'p2_checks', [script('s3p2/p2_checks.py')], check_checks, True), ('21', 'p2_s3c_list', [script('s3p2/p2_s3c_list.py')], None, True), ('22', 'compare_p2', [HERE + '/p2_colab_driver.py', '--step', 'compare'], None, False)]

def run_steps():
    passed = 0
    for nn, name, argv, test, gating in STEPS:
        if skip_done(nn, name):
            log('ARRAYS STEP %s %s: SKIPPED (its marker and files are present)' % (nn, name))
            passed += 1
            continue
        mp = marker_path(nn, name)
        if os.path.isfile(mp):
            try:
                old = jload(mp)
            except ValueError:
                old = {}
            own = (os.path.join(W, 'arrays_p2') + os.sep, os.path.join(CFG, 'p2') + os.sep)
            for o in old.get('outputs', []):
                q = os.path.join(common.ROOT, *_out(o)[0].split('/'))
                if q.startswith(own) and os.path.isfile(q):
                    os.remove(q)
            os.remove(mp)
            log("ARRAYS STEP %s %s: its marker's files were not all here: its stage (1) files removed; it runs again" % (nn, name))
        log('ARRAYS STEP %s %s: start; VM disk free %.1f GB' % (nn, name, shutil.disk_usage('/content').free / 1000000000.0))
        before = snapshot()
        t0 = time.time()
        if callable(argv):
            rc, text = argv(nn, name)
        else:
            rc, text = run_child(nn, name, argv)
        sec = time.time() - t0
        bad = []
        if rc == 0 and test is not None:
            try:
                bad = test(text)
            except Exception as e:
                bad = ['the result could not be read: %s' % type(e).__name__]
        after = snapshot()
        others = covered(skip=os.path.basename(mp))
        outs = [[common.rel(p), s[0]] for p, s in sorted(after.items()) if before.get(p) != s or (p not in FROM_W and common.rel(p) not in others)]
        n_m, e_m = mirror()
        if e_m:
            n_m, e_m = mirror()
        if rc == 0 and (not bad) and e_m:
            log('ARRAYS STEP %s %s: %d of its files did not reach the Drive: no marker; the driver stops (a second Run all mirrors again)' % (nn, name, e_m))
            return (1, passed, len(STEPS))
        if rc == 0 and (not bad):
            write_marker(nn, name, 0, sec, outs)
            log('ARRAYS STEP %s %s: DONE in %.0f s; %d files written' % (nn, name, sec, len(outs)))
            passed += 1
            continue
        why = 'exit %d' % rc if rc != 0 else 'FAIL: ' + '; '.join(bad)
        if not gating:
            log('ARRAYS STEP %s %s: NOT DONE (%s); recorded, not a stop (a record or a non-input step); a second Run all tries it again' % (nn, name, why))
            continue
        log('ARRAYS STEP %s %s FAILED (%s): the driver stops here; see RUN/logs/P2_STEPS/%s_%s_stdout.txt' % (nn, name, why, nn, name))
        return (1, passed, len(STEPS))
    return (0, passed, len(STEPS))

def sub_contents():
    from choi_dl import dl_s3
    from choi_dl.progress import Progress
    plan = jload(PLAN_S)
    prog = Progress(progress_path=os.path.join(LOGS, 'P2_CONTENTS_PROGRESS_ASCII.txt'), status_path=os.path.join(LOGS, 'P2_CONTENTS_STATUS.json'))
    try:
        ctx = dl_s3.Ctx(plan, prog, dl_s3.Rows())
    except Exception as e:
        print('CONTENTS NOT FETCHED: the client: %s' % type(e).__name__)
        return 3
    ok = 0
    for y in sorted(plan['stackgroups']):
        key = plan['stackgroups'][y]['key']
        name = key.split('/')[-1][:-4] + '.contents.txt'
        try:
            members = ctx.contents(key)
            p = os.path.join(dl_s3.S3META, name)
            print('CONTENTS %s: %d bytes, %d members listed' % (name, os.path.getsize(p), len(members)))
            ok += 1
        except Exception as e:
            print('CONTENTS NOT FETCHED %s: %s: %s' % (name, type(e).__name__, QUERY_RE.sub(lambda m: m.group(1) + '?<query>', common.mask(str(e)))[:160]))
    return 0 if ok == len(plan['stackgroups']) else 3
HDR_ATTRS = ('CDATE', 'CTIME', 'WDATE', 'WTIME', 'HISTORY', 'EXEC_ID')
PROFILE_RE = re.compile('(?i)(/(?:home|users)/|[a-z]:\\\\+users\\\\+)[^/\\\\\\s]+')
QUERY_RE = re.compile('(?i)\\b((?:https?|ftp)://[^\\s?#]*)[?#]\\S*')

def _clean(v):
    if isinstance(v, bytes):
        v = v.decode('ascii', 'replace')
    if hasattr(v, 'tolist'):
        v = v.tolist()
    s = ' '.join(str(v).split())
    s = QUERY_RE.sub(lambda m: m.group(1) + '?<query>', s)
    s = PROFILE_RE.sub(lambda m: m.group(1) + '<name>', common.mask(s))
    s = ''.join((c if 32 <= ord(c) < 127 else '?' for c in s))
    return s[:160]

def sub_headers():
    import netCDF4
    root = os.path.join(common.RAW, 'rsig')
    out = ['RSIG ANSWER HEADERS (a record; COMPARE_RULING_DRAFT 2.6): one rebuilt answer per series, its I/O API global attributes ' + ', '.join(HDR_ATTRS) + ' (free text masked, cut at 160).']
    for sid in sorted(os.listdir(root)):
        d = os.path.join(root, sid)
        if not os.path.isdir(d) or sid.startswith(('LAG_', 'TEST_')):
            continue
        fs = sorted((f for f in os.listdir(d) if f.endswith('.nc.gz') and '_try' not in f))
        if not fs:
            continue
        try:
            with open(os.path.join(d, fs[0]), 'rb') as fh:
                data = gzip.decompress(fh.read())
            ds = netCDF4.Dataset('inmemory.nc', mode='r', memory=data)
            try:
                have = set(ds.ncattrs())
                vals = ['%s=%s' % (a, _clean(ds.getncattr(a)) if a in have else '(absent)') for a in HDR_ATTRS]
            finally:
                ds.close()
            t = '%s: %s' % (fs[0], '; '.join(vals))
        except Exception as e:
            t = '%s: not read (%s)' % (fs[0], type(e).__name__)
        print(t, flush=True)
        out.append(t)
    p = os.path.join(LOGS, 'P2_RSIG_HEADERS_ASCII.txt')
    with open(p, 'w', encoding='ascii', newline='\n') as fh:
        for t in out:
            fh.write('\n'.join(common.wrap('', t)) + '\n')
    return 0
DROP_KEYS = ('made', 'S3_ARRAYS_sha256')

def _strip_made(o):
    if isinstance(o, dict):
        return {k: _strip_made(v) for k, v in o.items() if k not in DROP_KEYS}
    if isinstance(o, list):
        return [_strip_made(v) for v in o]
    return o

def _leaves(o, at=''):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _leaves(o[k], '%s/%s' % (at, k))
    elif isinstance(o, list):
        yield (at + '[]', ('list', len(o)))
        for i, v in enumerate(o):
            yield from _leaves(v, '%s[%d]' % (at, i))
    else:
        yield (at, o)

def _num(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool))

def content_compare(new_p, old_p):
    a = dict(_leaves(_strip_made(jload(new_p))))
    b = dict(_leaves(_strip_made(jload(old_p))))
    res = {'numbers': 0, 'numbers_differ': 0, 'max_abs': 0.0, 'max_at': '', 'lists_differ': 0, 'values_differ': 0, 'value_keys': [], 'keys_one_side': len(set(a) ^ set(b)), 'rows': []}
    for k in sorted(set(a) & set(b)):
        x, y = (a[k], b[k])
        if isinstance(x, tuple) or isinstance(y, tuple):
            if x != y:
                res['lists_differ'] += 1
        elif _num(x) and _num(y):
            res['numbers'] += 1
            if x != x and y != y:
                continue
            d = abs(float(x) - float(y))
            if d != 0 or x != y:
                res['numbers_differ'] += 1
                res['rows'].append((k, x, y, d))
                if d >= res['max_abs']:
                    res['max_abs'], res['max_at'] = (d, k)
        elif isinstance(x, str) and isinstance(y, str):
            if x.replace('\\', '/') != y.replace('\\', '/'):
                res['values_differ'] += 1
                if len(res['value_keys']) < 5:
                    res['value_keys'].append(k)
        elif x != y:
            res['values_differ'] += 1
            if len(res['value_keys']) < 5:
                res['value_keys'].append(k)
    res['equal'] = not (res['numbers_differ'] or res['lists_differ'] or res['values_differ'] or res['keys_one_side'])
    return res

def sub_compare():
    new = jload(os.path.join(CFG, 'p2', 'S3C_LIST.json'))
    old = jload(os.path.join(REF, 'KEPT_S3C_LIST.json'))

    def rows(lst):
        out = {}
        for part in ('files', 'configs_by_outbox'):
            for r in lst.get(part, []):
                n = r['path'].replace('\\', '/').split('/')[-1]
                out[n] = dict(r, part=part)
        return out
    nr, orow = (rows(new), rows(old))

    def order(n):
        r = nr.get(n) or orow.get(n)
        k = r.get('kind', '')
        return (0 if k == 'target' else 1 if k == 'mask' else 2, r.get('year', 0), n)
    names = sorted(set(nr) | set(orow), key=order)
    content = lambda n, r: r.get('kind') in ('norm', 'city_cells') or n == 'SPLITS.json'
    csv_rows, text, counts = ([], [], {})
    tm = {'MATCH': 0, 'DIFFER': 0}
    mask_v = 'NOT LISTED'
    for n in names:
        a, b = (nr.get(n), orow.get(n))
        r = a or b
        base = {'part': r['part'], 'name': n, 'kind': r.get('kind', 'config'), 'year': r.get('year', ''), 'sha256_new': (a or {}).get('sha256', ''), 'sha256_kept': (b or {}).get('sha256', ''), 'bytes_new': (a or {}).get('bytes', ''), 'bytes_kept': (b or {}).get('bytes', ''), 'key': '', 'value_new': '', 'value_kept': '', 'abs_diff': ''}
        if a and (not b):
            v = 'NO KEPT ROW'
        elif b and (not a):
            v = 'NOT REBUILT'
        elif a['sha256'] == b['sha256']:
            v = 'MATCH'
        elif content(n, r) and os.path.isfile(os.path.join(REF, 'KEPT_' + n)):
            c = content_compare(os.path.join(common.ROOT, *a['path'].replace('\\', '/').split('/')), os.path.join(REF, 'KEPT_' + n))
            v = 'MATCH (CONTENT)' if c['equal'] else 'DIFFER (CONTENT)'
            text.append('  %s: %s; numbers %d, differing %d, largest absolute difference %r at %s; lists differing %d; other values differing %d %s; keys on one side %d' % (n, v, c['numbers'], c['numbers_differ'], c['max_abs'], c['max_at'] or '-', c['lists_differ'], c['values_differ'], c['value_keys'], c['keys_one_side']))
            for k, x, y, d in c['rows']:
                csv_rows.append(dict(base, verdict='NUMBER DIFFERS', key=k, value_new=repr(x), value_kept=repr(y), abs_diff=repr(d)))
        else:
            v = 'DIFFER'
        if r.get('kind') == 'target' and v in tm:
            tm[v] += 1
        if r.get('kind') == 'mask':
            mask_v = v
        counts[v] = counts.get(v, 0) + 1
        csv_rows.insert(len([x for x in csv_rows if x['key'] == '']), dict(base, verdict=v))
    head = "COMPARE (a record, not a gate): config/p2/S3C_LIST.json of this VM against the local machine's kept run\\config\\p2\\S3C_LIST.json (ref/KEPT_S3C_LIST.json). THE NO2 SCREEN'S INPUTS: targets MATCH %d, DIFFER %d of 5; mask %s. All rows: %s." % (tm['MATCH'], tm['DIFFER'], mask_v, '; '.join(('%s %d' % kv for kv in sorted(counts.items()))))
    lines = [head, 'BY FILE (targets and mask first):']
    for x in csv_rows:
        if x['key'] == '':
            lines.append('  %s %s: %s' % (x['kind'], x['name'], x['verdict']))
    lines.append('CONTENT COMPARISONS (made and S3_ARRAYS_sha256 removed; S3_ARRAYS.json itself has its own row by sha256; \\ read as /):')
    lines += text or ['  none (no content row differed by sha256)']
    with open(os.path.join(LOGS, 'COMPARE_P2_ASCII.txt'), 'w', encoding='ascii', newline='\n') as fh:
        for t in lines:
            fh.write('\n'.join(common.wrap('', t)) + '\n')
    fields = ['part', 'name', 'kind', 'year', 'verdict', 'sha256_new', 'sha256_kept', 'bytes_new', 'bytes_kept', 'key', 'value_new', 'value_kept', 'abs_diff']
    with open(os.path.join(LOGS, 'COMPARE_P2.csv'), 'w', newline='', encoding='ascii') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for x in csv_rows:
            w.writerow(x)
    with open(os.path.join(LOGS, 'COMPARE_P2_SUMMARY.json'), 'w', encoding='ascii', newline='\n') as fh:
        fh.write(json.dumps({'targets_match': tm['MATCH'], 'targets_differ': tm['DIFFER'], 'mask': mask_v, 'rows': counts}, sort_keys=True) + '\n')
    print(head, flush=True)
    return 0
SUBS = {'contents': sub_contents, 'headers': sub_headers, 'compare': sub_compare}

def on_term(_s, _f):
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)
    stop_child()
    sys.exit(143)

def log_to_drive():
    try:
        copy(LOG, D_RUN + '/logs/' + os.path.basename(LOG), True)
    except OSError:
        pass

def main(argv):
    if '--step' in argv:
        common.set_temp()
        return SUBS[argv[argv.index('--step') + 1]]()
    if not os.path.isdir(EMUL):
        print('THE DRIVE IS NOT MOUNTED (PROJECT_DATA/EMUL not found under /content/drive/MyDrive): run cell 1 first')
        return 9
    os.makedirs(LOGS, exist_ok=True)
    lk = open(os.path.join(LOGS, 'P2_DRIVER.lock'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ANOTHER DRIVER RUNS ON THIS VM: not started')
        return 10
    LOCK_FD[0] = lk.fileno()
    common.set_temp()
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_term)
    n_own = restore_own_logs()
    log('ARRAYS START: the arrays stage (stage (1) with the inventory and Natural Earth); %d of its own log files restored; steps %d' % (n_own, len(STEPS)))
    if os.path.isfile(DONE) and all((_marker_ok(nn, name) for nn, name, _a, _t, g in STEPS if g)):
        with open(DONE, 'rb') as fh:
            line = fh.read().decode('ascii', 'replace').strip()
        log("ARRAYS ALREADY DONE (P2_DONE_ASCII.txt and every gating step's marker on the Drive): nothing to do; %s" % line)
        return 0
    if not wait_for_drivers():
        log('ARRAYS END: exit 6 (the download drivers did not both end exit 0; see the line above)')
        log_to_drive()
        return 6
    rc0 = step0()
    if rc0 != 0:
        log('ARRAYS END: exit %d (before the steps)' % rc0)
        log_to_drive()
        return rc0
    if any((not skip_done(nn, name) for nn, name, _a, _t, _g in STEPS)):
        for p in (DONE, under(D_RUN + '/logs/' + os.path.basename(DONE))):
            if os.path.isfile(p):
                os.remove(p)
                log('ARRAYS: the earlier %s removed (a step runs again)' % ('RUN/logs/P2_DONE_ASCII.txt' + (' on the Drive' if p.startswith(EMUL) else '')))
    stop = threading.Event()

    def loop():
        while not stop.wait(EVERY_S):
            mirror()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc, passed, total = (1, 0, len(STEPS))
    try:
        rc, passed, total = run_steps()
    finally:
        stop.set()
        th.join(timeout=5)
        if rc == 0:
            n1, e1 = mirror()
            if e1:
                n1, e1 = mirror()
            if e1:
                rc = 1
                log('ARRAYS MIRROR ERRORS %d: P2_DONE not written; a second Run all mirrors again' % e1)
            else:
                try:
                    s = jload(os.path.join(LOGS, 'COMPARE_P2_SUMMARY.json'))
                    tgt = '%d MATCH, %d DIFFER' % (s['targets_match'], s['targets_differ'])
                except Exception:
                    tgt = '0 MATCH, 0 DIFFER (COMPARE_P2 not made)'
                line = 'P2 DONE %s exit 0; steps %d of %d; COMPARE_P2 targets %s' % (common.now_pair()[0], passed, total, tgt)
                with open(DONE + '.partial', 'w', encoding='ascii', newline='\n') as fh:
                    fh.write(line + '\n')
                os.replace(DONE + '.partial', DONE)
                log(line)
        log('ARRAYS END: exit %d; mirror at the end' % rc)
        n, e = mirror()
        if e:
            n, e = mirror()
        log('ARRAYS MIRROR: %d files copied to EMUL/W_P2 and EMUL/RUN; %d errors' % (n, e))
        if e and rc == 0:
            rc = 1
            log('ARRAYS END: exit 1 (the last mirror failed: P2_DONE may not be on the Drive; a second Run all mirrors again)')
    return rc
if __name__ == '__main__':
    try:
        RC = main(sys.argv[1:])
    except SystemExit:
        raise
    except BaseException:
        import traceback
        sys.stderr.write(common.mask(traceback.format_exc()))
        sys.stderr.flush()
        RC = 1
    sys.exit(RC)
