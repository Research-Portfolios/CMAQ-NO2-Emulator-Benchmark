"""Driver of the fill and statistics stage on Colab: waits for the arrays stage and the year checks, restores files from the Drive, copies and fills the emission channels of 2018, 2015 and 2016 with their checks and uploads, runs the NO2 screen only when needed, then computes the main folds' statistics and writes the end marker."""
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from choi_dl import common
DRIVE = '/content/drive/MyDrive'
EMUL = DRIVE + '/PROJECT_DATA/EMUL'
D_ARR = EMUL + '/W_P2/arrays_p2'
D_CFG = EMUL + '/RUN/config'
D_LOGS = EMUL + '/RUN/logs'
D_EXT = EMUL + '/EXTRACTS'
VM_ROOT = '/content/emul'
PY = VM_ROOT + '/venv/bin/python'
W = common.W
RUN = common.RUN
ARR = os.path.join(W, 'arrays_p2')
WORK = os.path.join(ARR, 'work')
CFG = os.path.join(RUN, 'config')
CFG2 = os.path.join(CFG, 'p2')
LOGS = os.path.join(RUN, 'logs')
STEPS = os.path.join(LOGS, 'S8_STEPS')
CODE = os.path.join(RUN, 'code')
SCREEN_DIR = os.path.join(LOGS, 'S8_NO2_SCREEN')
YEARS = (2018, 2015, 2016)
ALL_YEARS = (2015, 2016, 2017, 2018, 2019)
LOG = os.path.join(LOGS, 'S8_COLAB_DRIVER_ASCII.txt')
EVERY_S = 120
POLL_S = 300
WAIT_MAX_S = 3 * 3600
BUILD = ('copy', 'disk', 'd4', 'pre', 'fill', 'after', 'verify', 'upload')
POST = ('d4record', 'backup', 'cfg')
ORDER = BUILD + POST
SIX = tuple(('target_%d.npy' % y for y in ALL_YEARS)) + ('mask_conus_land.npy',)

def log(text):
    u, _loc = common.now_pair()
    lines = common.wrap('%s ' % u, common.mask(text))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'ab') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii', 'replace'))
    print('\n'.join(lines), flush=True)

def utc():
    return common.now_pair()[0]

def under_emul(p):
    if not (p == EMUL or p.startswith(EMUL + '/')) or '..' in p:
        raise SystemExit(9)
    return p

def jload(p):
    with open(p, 'r', encoding='ascii') as f:
        return json.load(f)

def wjson(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.partial', 'w', encoding='ascii', newline='\n') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(p + '.partial', p)

def copy(src, dst, to_drive):
    if to_drive:
        under_emul(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)

def arr_test(rel):
    n = os.path.basename(rel)
    return not (n.startswith('inputs_f32_') and n.endswith('.npy'))

def logs_test(rel):
    return rel.startswith(('S8_', 'B2C_C7A_COPY_'))
PAIRS = ((ARR, D_ARR, arr_test), (CFG2, D_CFG + '/p2', None), (LOGS, D_LOGS, logs_test))
SEEN = {}
MLOCK = threading.Lock()
RESTORED = set()

def skip(test, rel):
    n = os.path.basename(rel)
    return n.endswith('.partial') or n.endswith('.lock') or (test is not None and (not test(rel)))

def mirror():
    n = 0
    with MLOCK:
        for loc, dd, test in PAIRS:
            if not os.path.isdir(loc):
                continue
            for dp, _dn, fn in os.walk(loc):
                for f in fn:
                    p = os.path.join(dp, f)
                    rel = os.path.relpath(p, loc)
                    if skip(test, rel):
                        continue
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    sig = (st.st_size, st.st_mtime_ns)
                    if SEEN.get(p) == sig:
                        continue
                    try:
                        copy(p, os.path.join(dd, rel), True)
                    except OSError as e:
                        log('MIRROR ERROR %s: %s' % (f, str(e)[:120]))
                        continue
                    SEEN[p] = sig
                    n += 1
    return n

def seen_now(p):
    st = os.stat(p)
    SEEN[p] = (st.st_size, st.st_mtime_ns)

def restore_tree(dd, loc, test):
    n = 0
    if not os.path.isdir(dd):
        return 0
    for dp, _dn, fn in os.walk(dd):
        for f in fn:
            src = os.path.join(dp, f)
            rel = os.path.relpath(src, dd)
            if skip(test, rel):
                continue
            dst = os.path.join(loc, rel)
            if os.path.exists(dst):
                continue
            copy(src, dst, False)
            seen_now(dst)
            n += 1
    return n

def list_rows():
    lst = jload(os.path.join(CFG2, 'S3C_LIST.json'))
    out = {}
    for e in lst['files']:
        out[e['path'].replace('\\', '/').split('/')[-1]] = e
    return out

def restore_array(name, want_bytes, force=False):
    dst = os.path.join(ARR, name)
    if os.path.exists(dst) and (not force):
        return 'on the VM'
    src = os.path.join(D_ARR, name)
    for k in (1, 2):
        copy(src, dst, False)
        if os.stat(dst).st_size == int(want_bytes):
            seen_now(dst)
            RESTORED.add(name)
            return 'copied (size equal, try %d)' % k
    raise RuntimeError("%s: the copy's size differs from %d bytes" % (name, int(want_bytes)))
WORK_NAMES = tuple(('%s_%d.%s' % (s, y, e) for y in ALL_YEARS for s, e in (('daystats', 'npz'), ('smallk', 'npz'), ('build', 'json'))))

def work_test(rel):
    n = os.path.basename(rel)
    return os.path.dirname(rel) == '' and (n in WORK_NAMES or '_s8_' in n or n.startswith('norm_index'))

def restore(years):
    n = restore_tree(D_CFG + '/p2', CFG2, None)
    inv = D_CFG + '/S2C_INVENTORY.json'
    if os.path.isfile(inv) and (not os.path.exists(os.path.join(CFG, 'S2C_INVENTORY.json'))):
        copy(inv, os.path.join(CFG, 'S2C_INVENTORY.json'), False)
        n += 1
    n += restore_tree(D_LOGS, LOGS, logs_test)
    n += restore_tree(D_ARR + '/work', WORK, work_test)
    rows = list_rows()
    log('restore: %d config, log and work files copied from the Drive' % n)
    log('restore mask_conus_land.npy: %s' % restore_array('mask_conus_land.npy', rows['mask_conus_land.npy']['bytes']))
    for y in years:
        name = 'inputs_f32_%d.npy' % y
        if marker(y, 'cfg') and (not os.path.isfile(os.path.join(CFG2, 'LOCKBOX_DAYS.json'))):
            log('restore %s: not needed (year done, no lockbox)' % name)
            continue
        up = marker(y, 'upload')
        want = up['bytes'] if up else rows[name]['bytes']
        log('restore %s (%s): %s' % (name, 'completed' if up else 'partial', restore_array(name, want)))
    if os.path.isfile(os.path.join(CFG2, 'LOCKBOX_DAYS.json')):
        for y in (2017, 2019):
            name = 'inputs_f32_%d.npy' % y
            log("restore %s (the lockbox's smallest-K parts): %s" % (name, restore_array(name, rows[name]['bytes'])))

def marker_path(y, step):
    return os.path.join(STEPS, 'S8_STEP_%s_%02d_%s.json' % (y, ORDER.index(step) + 1 if step in ORDER else 99, step))

def marker(y, step):
    p = marker_path(y, step)
    return jload(p) if os.path.isfile(p) else None

def set_marker(y, step, **extra):
    d = {'step': step, 'year': y, 'utc': utc(), 'exit': 0}
    d.update(extra)
    wjson(marker_path(y, step), d)

def run_cmd(args, tag, quiet=None):
    env = dict(os.environ, HOME=VM_ROOT + '/s8_vm_home')
    os.makedirs(env['HOME'], exist_ok=True)
    cmd = [PY, '-I', '-B'] + [str(a) for a in args]
    log('RUN %s: %s' % (tag, ' '.join((os.path.relpath(a, VM_ROOT) if str(a).startswith(VM_ROOT) else str(a) for a in cmd[3:]))))
    con = os.path.join(LOGS, 'S8_CONSOLE_ASCII.txt')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=VM_ROOT)
    CHILD[0] = p
    with open(con, 'a', encoding='ascii', errors='replace', newline='\n') as cf:
        cf.write('== %s %s\n' % (utc(), tag))
        for line in p.stdout:
            line = common.mask(line.rstrip('\n'))
            cf.write(line + '\n')
            if quiet is None or not quiet(line):
                print('  ' + line, flush=True)
    rc = p.wait()
    CHILD[0] = None
    log('END %s: exit %d' % (tag, rc))
    return rc
CHILD = [None]

def on_term(_s, _f):
    p = CHILD[0]
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    sys.exit(143)

def per_file_line(line):
    return ' sums ' in line and ' manifest ' in line

def step_args(step, y):
    s8 = os.path.join(CODE, 's8')
    if step == 'copy':
        return [os.path.join(CODE, 'b2c', 'copy_c7a_year.py'), y]
    if step in ('disk', 'd4', 'pre', 'fill', 'after', 'verify'):
        return [os.path.join(s8, 's8_run.py'), step, y]
    if step == 'd4record':
        return [os.path.join(s8, 's8_d4_record.py'), y]
    if step == 'backup':
        return [os.path.join(CODE, 'bc', 'backup_beside.py'), 'S8_%d' % y, os.path.join(CFG2, 'S3_ARRAYS.json'), os.path.join(CFG2, 'S3_CHANNELS.json')]
    if step == 'cfg':
        return [os.path.join(s8, 's8_cfg.py'), y]
    raise ValueError(step)

def upload(y):
    name = 'inputs_f32_%d.npy' % y
    src = os.path.join(ARR, name)
    af = jload(os.path.join(WORK, 'after_s8_%d.json' % y))
    ve = jload(os.path.join(WORK, 'verify_s8_%d.json' % y))
    if af.get('verdict') != 'PASS' or ve.get('verdict') != 'PASS':
        log('UPLOAD %s refused: after %s, verify %s' % (name, af.get('verdict'), ve.get('verdict')))
        return 2
    n = os.stat(src).st_size
    if n != int(af['bytes']):
        log('UPLOAD %s refused: %d bytes, after_s8 %s' % (name, n, af['bytes']))
        return 2
    dst = under_emul(os.path.join(D_ARR, name))
    shutil.copyfile(src, dst + '.partial')
    if os.stat(dst + '.partial').st_size != n:
        log("UPLOAD %s: the Drive copy's size differs; not renamed" % name)
        return 2
    os.replace(dst + '.partial', dst)
    set_marker(y, 'upload', bytes=n, sha256=af['sha256'])
    mp = marker_path(y, 'upload')
    copy(mp, D_LOGS + '/S8_STEPS/' + os.path.basename(mp), True)
    seen_now(mp)
    log("UPLOAD %s: %d bytes, sha16 %s (after_s8's), renamed over the partial array on the Drive" % (name, n, af['sha256'][:16]))
    return 0

def run_year(y):
    if marker(y, 'cfg'):
        log('YEAR %d: done earlier (its cfg marker)' % y)
        return 0
    name = 'inputs_f32_%d.npy' % y
    if not marker(y, 'upload'):
        if name not in RESTORED:
            rows = list_rows()
            log('YEAR %d: the partial array copied again from the Drive: %s' % (y, restore_array(name, rows[name]['bytes'], force=True)))
        for step in BUILD:
            if step == 'upload':
                rc = upload(y)
            else:
                rc = run_cmd(step_args(step, y), '%s %d' % (step, y), per_file_line if step == 'copy' else None)
                if rc == 0:
                    set_marker(y, step)
            if rc != 0:
                log('YEAR %d: STOP at %s (exit %d)' % (y, step, rc))
                return rc
    for step in POST:
        if marker(y, step):
            log('YEAR %d: %s done earlier' % (y, step))
            continue
        if step == 'cfg':
            f32 = jload(os.path.join(CFG2, 'S3_ARRAYS.json'))['years'][str(y)]['files']['inputs_f32']
            l8p = os.path.join(CFG2, 'S3C_LIST_8_%d.json' % y)
            up = marker(y, 'upload') or {}
            if f32.get('partial') is False and os.path.isfile(l8p) and (f32.get('sha256') == up.get('sha256')):
                log('YEAR %d: cfg applied earlier (S3_ARRAYS.json)' % y)
                set_marker(y, step, list=jload(l8p)['files'][0])
                continue
        if step == 'cfg':
            with MLOCK:
                rc = run_cmd(step_args(step, y), '%s %d' % (step, y))
        else:
            rc = run_cmd(step_args(step, y), '%s %d' % (step, y))
        if rc != 0:
            log('YEAR %d: STOP at %s (exit %d)' % (y, step, rc))
            return rc
        if step == 'cfg':
            l8 = jload(os.path.join(CFG2, 'S3C_LIST_8_%d.json' % y))
            set_marker(y, step, list=l8['files'][0])
        else:
            set_marker(y, step)
    log('YEAR %d: DONE; mirror %d files' % (y, mirror()))
    return 0

def first_line(p):
    try:
        with open(p, 'r', encoding='ascii', errors='replace') as f:
            return f.readline().strip()
    except OSError:
        return ''

def wait_inputs():
    t0 = time.time()
    while True:
        ln = first_line(D_LOGS + '/P2_DONE_ASCII.txt')
        p2 = ln.startswith('P2 DONE') and 'exit 0' in ln
        ok = {y: first_line('%s/%d/C7A_4_LINE_%d_ASCII.txt' % (D_EXT, y, y)).startswith('C-7a (4) %d: PASS' % y) for y in YEARS}
        if p2 and all(ok.values()):
            log("WAIT: the arrays stage's P2_DONE (exit 0) and the C-7a (4) PASS lines of %s are present" % ', '.join(map(str, YEARS)))
            return (p2, ok)
        if time.time() - t0 + POLL_S > WAIT_MAX_S:
            log('WAIT ENDED after %.0f min: P2_DONE %s; PASS lines %s' % ((time.time() - t0) / 60, 'present' if p2 else 'ABSENT', ', '.join(('%d %s' % (y, 'yes' if v else 'NO') for y, v in sorted(ok.items())))))
            return (p2, ok)
        log('WAIT: %s; %s; next look in %d min' % ("the arrays stage's P2_DONE_ASCII.txt with exit 0 is not on the Drive yet" if not p2 else 'P2_DONE present', 'C-7a (4) PASS line absent for %s' % ', '.join((str(y) for y, v in sorted(ok.items()) if not v)) if not all(ok.values()) else 'PASS lines present', POLL_S // 60))
        time.sleep(POLL_S)

def compare_rows():
    p = D_LOGS + '/COMPARE_P2.csv'
    out = {}
    if not os.path.isfile(p):
        return out
    with open(p, 'r', encoding='ascii', errors='replace', newline='') as f:
        for r in csv.DictReader(f):
            if str(r.get('key') or ''):
                continue
            vals = [str(r.get('name') or '')] if 'name' in r else [str(v or '') for v in r.values()]
            for name in SIX:
                if any((v.replace('\\', '/').split('/')[-1] == name for v in vals)):
                    lower = {str(k).lower(): str(v or '') for k, v in r.items()}
                    v = lower.get('verdict') or next((x for x in vals if x.startswith(('MATCH', 'DIFFER'))), '')
                    out[name] = v
    return out

def screen():
    m = marker('all', 'screen')
    if m:
        log('SCREEN: done earlier: %s' % m['line'])
        return (0, m['line'])
    rows = compare_rows()
    verdicts = ', '.join(('%s %s' % (n, rows.get(n, 'NO ROW')) for n in SIX))
    if all((rows.get(n, '').startswith('MATCH') for n in SIX)):
        line = 'screen not needed'
        log("SCREEN NOT NEEDED: COMPARE_P2.csv: %s; the local machine's empty FAIL list stands (work/FAIL_LIST.csv and run_b\\FAIL_LIST.csv, header only; COMPARE_ASCII.txt EQUAL)" % verdicts)
        set_marker('all', 'screen', line=line, compare=rows)
        return (0, line)
    log('SCREEN NEEDED: COMPARE_P2.csv: %s' % verdicts)
    lr = list_rows()
    for y in ALL_YEARS:
        name = 'target_%d.npy' % y
        log('restore %s: %s' % (name, restore_array(name, lr[name]['bytes'])))
    inp = os.path.join(W, 'tmp', 's8_screen_in')
    if os.path.isdir(inp):
        shutil.rmtree(inp)
    os.makedirs(inp)
    sums, pins = ([], {})
    for name in SIX:
        os.symlink(os.path.join(ARR, name), os.path.join(inp, name))
        sums.append('%s  %s' % (lr[name]['sha256'], name))
        pins[name] = lr[name]['sha256']
    with open(os.path.join(inp, 'SHA256SUMS_P2.txt'), 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(sums) + '\n')
    pp = os.path.join(W, 'tmp', 's8_screen_pins.json')
    wjson(pp, pins)
    if os.path.isdir(SCREEN_DIR):
        k = 1
        while os.path.exists('%s_try%d' % (SCREEN_DIR, k)):
            k += 1
        os.rename(SCREEN_DIR, '%s_try%d' % (SCREEN_DIR, k))
        log('SCREEN: an earlier unfinished try kept as %s_try%d' % (os.path.basename(SCREEN_DIR), k))
    os.makedirs(SCREEN_DIR)
    sc = os.path.join(CODE, 'no2_screen')
    years = ','.join((str(y) for y in ALL_YEARS))
    for tag, runner in (('a', 'code_a/wait_screen_a.py'), ('b', 'code_b/wait_screen_b.py')):
        rc = run_cmd([os.path.join(sc, *runner.split('/')), '--inputs', inp, '--out', os.path.join(SCREEN_DIR, 'run_' + tag), '--years', years, '--pins', pp], 'screen ' + tag)
        if rc != 0:
            return (1, 'screen code %s exit %d' % (tag, rc))
    cp = os.path.join(SCREEN_DIR, 'COMPARE_ASCII.txt')
    rc = run_cmd([os.path.join(sc, 'wait_compare_ab.py'), os.path.join(SCREEN_DIR, 'run_a'), os.path.join(SCREEN_DIR, 'run_b'), cp], 'screen compare')
    if rc != 0:
        log("SCREEN STOP: the two codes' runs are not EQUAL (see RUN/logs/S8_NO2_SCREEN/COMPARE_ASCII.txt): a manual decision is needed before d7")
        return (4, 'screen run, runs not EQUAL')
    with open(os.path.join(SCREEN_DIR, 'run_a', 'FAIL_LIST.csv'), 'r', encoding='ascii') as f:
        nfail = sum((1 for ln in f.read().split('\n')[1:] if ln.strip()))
    if nfail:
        log("SCREEN STOP: the FAIL list is not empty. The frozen d7 rule does not define how a FAIL cell-day leaves the study design's statistics, so a manual decision is needed first; d7 is not run. The list: RUN/logs/S8_NO2_SCREEN/run_a/FAIL_LIST.csv")
        return (4, 'screen run, FAIL list not empty')
    line = 'run, FAIL list empty'
    log('SCREEN: both codes PASS, their runs EQUAL, the FAIL list empty')
    set_marker('all', 'screen', line=line, compare=rows)
    return (0, line)

def lockbox_sha():
    p = os.path.join(CFG2, 'LOCKBOX_DAYS.json')
    if not os.path.isfile(p):
        return None
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def d7():
    m = marker('all', 'd7')
    if m and m.get('lockbox_sha256') == lockbox_sha():
        log('D7: done earlier: %s' % ', '.join(m['written']))
        return (0, len(m['written']))
    if not marker('all', 'selftest'):
        rc = run_cmd([os.path.join(CODE, 's8', 's8_d7_selftest.py')], 'd7 self-test')
        if rc != 0:
            return (1, 0)
        set_marker('all', 'selftest')
    with MLOCK:
        rc = run_cmd([os.path.join(CODE, 's8', 's8_d7.py')], 'd7')
    res = jload(os.path.join(LOGS, 'S8_D7_RESULT.json')) if rc in (0, 3) else {'written': []}
    if rc == 0:
        set_marker('all', 'd7', written=res['written'], lockbox_sha256=lockbox_sha())
    elif rc == 3:
        log('D7: some folds wait: %s' % ', '.join(sorted(res['waiting'])))
    return (rc, len(res['written']))

def main():
    for d in (LOGS, STEPS, ARR, WORK, CFG2):
        os.makedirs(d, exist_ok=True)
    dlog = D_LOGS + '/' + os.path.basename(LOG)
    if not os.path.exists(LOG) and os.path.isfile(dlog):
        copy(dlog, LOG, False)
        seen_now(LOG)
    lk = open(os.path.join(LOGS, 'S8_DRIVER.lock'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ANOTHER DRIVER RUNS ON THIS VM: not started')
        return 10
    for _l, dd, _t in PAIRS:
        under_emul(dd)
    common.set_temp()
    signal.signal(signal.SIGTERM, on_term)
    log('S8 START: VM disk free %.1f GB' % (shutil.disk_usage('/content').free / 1000000000.0))
    p2, ok = wait_inputs()
    if not p2:
        log('S8 END: exit 3: the arrays stage has not finished (no P2_DONE with exit 0 within 3 hours); nothing built. Run all again later.')
        return 3
    years = [y for y in YEARS if ok[y]]
    for y in YEARS:
        if not ok[y]:
            log('YEAR %d: NOT FILLED: its C7A_4_LINE file does not begin "C-7a (4) %d: PASS" on the Drive' % (y, y))
    try:
        restore(years)
    except (RuntimeError, OSError, KeyError) as e:
        log('S8 END: exit 1: the restore from the Drive failed (%s: %s); Run all again' % (type(e).__name__, str(e)[:160]))
        return 1
    stop = threading.Event()

    def loop():
        while not stop.wait(EVERY_S):
            mirror()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc, filled, nd7, sline = (1, [], 0, 'not reached')
    try:
        for y in years:
            r = run_year(y)
            if r != 0:
                rc = 1
                return rc
            filled.append(y)
        r, sline = screen()
        if r != 0:
            rc = r
            return rc
        r, nd7 = d7()
        if r not in (0, 3):
            rc = 1
            return rc
        done = r == 0 and len(filled) == len(YEARS)
        rc = 0 if done else 3
        text = 'S8 %s %s exit %d; years filled %s; d7 written %d files; screen %s' % ('DONE' if done else 'PARTIAL', utc(), rc, ', '.join(map(str, filled)) or 'none', nd7, sline)
        name = 'S8_DONE_ASCII.txt' if done else 'S8_PARTIAL_ASCII.txt'
        with open(os.path.join(LOGS, name), 'w', encoding='ascii', newline='\n') as f:
            f.write(text + '\n')
        res = ['S8 RESULT (the fills stage): ' + text]
        for y in filled:
            m = marker(y, 'cfg')
            if m:
                res.append('inputs_f32_%d.npy completed: %d bytes, sha256 %s' % (y, m['list']['bytes'], m['list']['sha256']))
        with open(os.path.join(LOGS, 'S8_RESULT_ASCII.txt'), 'w', encoding='ascii', newline='\n') as f:
            for t in res:
                f.write('\n'.join(common.wrap('', t)) + '\n')
        log(text)
    finally:
        stop.set()
        th.join(timeout=5)
        log('S8 END: exit %d; mirror at the end' % rc)
        log('S8 MIRROR: %d files copied to EMUL' % mirror())
    return rc
if __name__ == '__main__':
    sys.exit(main())
