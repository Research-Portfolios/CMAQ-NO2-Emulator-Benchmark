"""Driver of the trainings on a Colab GPU runtime: checks the earlier stages' markers, copies the configuration files and arrays with size checks, runs the equivalence test, the timing and the trainings through the trainer shim, records each runtime's use, and mirrors outputs to the Drive; the later drivers import its helpers."""
import fcntl
import glob
import hashlib
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
import tier_a_rules as R
import train_shim
from choi_dl import common
PKG = os.path.dirname(HERE)
CONFIG = os.path.join(PKG, 'config', 'TIER_A_CONFIG.json')
SHIM = os.path.join(HERE, 'train_shim.py')
TRAINER = os.path.join(HERE, 'train', 'trainer_main.py')
BENCH = os.path.join(HERE, 'train', 'bench.py')
EQUIV = os.path.join(HERE, 'train', 'equiv_test.py')
CORE = os.path.join(HERE, 'train', 'emul_core.py')
PY = sys.executable
ROOT = '/content/emul'
W = ROOT + '/work'
RUN = ROOT + '/run'
ARR = W + '/arrays_p2'
CFG = RUN + '/config/p2'
TA_WORK = W + '/tier_a'
TA_LOGS = RUN + '/logs/tier_a'
MARKER = TA_LOGS + '/TIER_A_MARKER_ASCII.txt'
SESS_DIR = TA_LOGS + '/sessions'
ENV_FMT = TA_LOGS + '/SESSION_ENV_%s.json'
LOG_FMT = TA_LOGS + '/driver/%s_%s_ASCII.txt'
LOG_NAME = ['TRAIN_DRIVER']
EQ_DIR = TA_LOGS + '/equiv'
TM_DIR = TA_LOGS + '/timing'
PLAN_DIR = TA_LOGS + '/plan'
UNITS = TA_LOGS + '/UNITS_ASCII.txt'
TYPED_FMT = TA_LOGS + '/TYPED_RATE_%s.json'
CHILD_HOME = ROOT + '/ta_vm_home'
DRIVE = '/content/drive/MyDrive'
EMUL = DRIVE + '/PROJECT_DATA/EMUL'
D_TRAIN = EMUL + '/TRAIN'
LOG_DRIVE = [D_TRAIN + '/logs/driver']
D_ARR = EMUL + '/W_P2/arrays_p2'
D_CFG = EMUL + '/RUN/config/p2'
D_MARKS = ((EMUL + '/RUN/logs/P2_DONE_ASCII.txt', 'P2 DONE'), (EMUL + '/RUN/logs/S8_DONE_ASCII.txt', 'S8 DONE'))
WRITE_OK = (D_TRAIN + '/', EMUL + '/W_P2/results_sealed/')
EVERY_S = 120
NOTEBOOK = 'EMUL_06_TIER_A_TRAIN'

class Stop(Exception):

    def __init__(self, code, text):
        Exception.__init__(self, text)
        self.code = code

def install_overrides():
    table = train_shim.load_overrides(CONFIG)
    if table:
        sys.meta_path.insert(0, train_shim._Finder(table))
    return table

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def log_path(name=None):
    return LOG_FMT % (name or LOG_NAME[0], boot12())

def env_file():
    return ENV_FMT % boot12()

def log(text):
    u = common.now_pair()[0]
    lines = common.wrap('%s ' % u, common.mask(text))
    lp = log_path()
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    with open(lp, 'ab') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii', 'replace'))
    print('\n'.join(lines), flush=True)

def push_log():
    p = log_path()
    if os.path.isfile(p) and os.path.isdir(EMUL):
        try:
            copy_file(p, LOG_DRIVE[0] + '/' + os.path.basename(p), True)
        except (OSError, Stop):
            pass

def under(p):
    if not p.startswith(WRITE_OK) or '..' in p:
        raise Stop(1, 'a Drive write outside EMUL/TRAIN and EMUL/W_P2/results_sealed was refused: %s' % p.replace(EMUL, 'EMUL'))
    return p

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def jload(p):
    with open(p, 'rb') as f:
        return json.loads(f.read().decode('ascii'))

def jsave(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.partial', 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    os.replace(p + '.partial', p)

def copy_file(src, dst, to_drive):
    if to_drive:
        under(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)

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

def gb(n):
    return n / 1000000000.0

def boot12():
    with open('/proc/sys/kernel/random/boot_id') as f:
        return f.read().strip().replace('-', '')[:12]

def uptime_s():
    with open('/proc/uptime') as f:
        return float(f.read().split()[0])

def gpu_name_smi():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
        names = [x.strip() for x in q.stdout.splitlines() if x.strip()]
        return names[0] if q.returncode == 0 and names else None
    except Exception:
        return None

def accelerator(conf):
    name = gpu_name_smi()
    if not name:
        return ('CPU', 'CPU')
    g = R.candidate_for(conf, name)
    return (g['key'] if g else name, name)

def session_update(conf, notebook, step):
    sid = boot12()
    up = uptime_s()
    p = os.path.join(SESS_DIR, sid + '.json')
    d = jload(p) if os.path.isfile(p) else {}
    acc, name = accelerator(conf)
    steps = sorted(set(d.get('steps', [])) | {step})
    d.update({'schema': 'tier-a-session-1', 'session': sid, 'notebook': d.get('notebook', notebook), 'vm_started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - up)), 'updated_utc': utc(), 'hours': round(up / 3600.0, 4), 'accelerator': acc, 'device_name': name, 'steps': steps})
    tp = TYPED_FMT % sid
    if os.path.isfile(tp):
        try:
            t = jload(tp)
            v = t.get('units_per_hour')
            d['units_per_hour_typed'] = float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v > 0) else None
            d['units_per_hour_typed_utc'] = t.get('typed_utc')
            d['runtime_typed'] = t.get('runtime')
        except (OSError, ValueError):
            d['units_per_hour_typed'] = None
            d['runtime_typed'] = None
    jsave(p, d)
    copy_file(p, D_TRAIN + '/logs/sessions/' + sid + '.json', True)
    rebuild_units(conf)
    return d

def read_units_other(conf):
    p = EMUL + '/' + conf['units_other']['file']
    try:
        return jload(p)
    except (OSError, ValueError):
        return None

def this_session():
    p = os.path.join(SESS_DIR, boot12() + '.json')
    try:
        return jload(p)
    except (OSError, ValueError):
        return {}

def rate_stop(conf):
    why = R.rate_refusal(conf, this_session())
    if why:
        raise Stop(14, why)

def drive_sessions():
    out = []
    for p in sorted(glob.glob(D_TRAIN + '/logs/sessions/*.json')):
        try:
            out.append(jload(p))
        except (OSError, ValueError):
            out.append({'session': os.path.basename(p), 'accelerator': 'unreadable record', 'hours': 0})
    return out

def rebuild_units(conf):
    ss = sorted(drive_sessions(), key=lambda s: s.get('vm_started_utc', ''))
    lines = ["UNITS OF THE TRAINING STAGES (the benchmark on Colab). One line a session, rebuilt by train_colab_driver.py at every step from EMUL/TRAIN/logs/sessions/ (hours from the VM's /proc/uptime; units = hours x the rate typed in cell 1 for that session's runtime (the latest of its key only for a session without one))."]
    for s in ss:
        r = R.session_rate(conf, s, ss)
        u = '%.3f' % (float(s.get('hours', 0)) * r) if isinstance(r, (int, float)) else 'unknown (no rate typed)'
        lines.append('%s session %s %s %s: %.3f h x %s = %s units (typed here: %s); updated %s' % (s.get('vm_started_utc'), s.get('session'), s.get('notebook'), s.get('accelerator'), float(s.get('hours', 0)), r if r is not None else 'unknown', u, s.get('units_per_hour_typed'), s.get('updated_utc')))
    spent, miss = R.spent_units(conf, ss)
    other = R.units_other_total(conf)
    lines.append('stages 1 to 5 (EMUL/%s): %s; these sessions and those: %s units%s; cap %g' % (conf['units_other']['file'], 'not stated' if other is None else '%.3f' % other, '%.3f' % spent, ' (units missing: %s)' % ', '.join(miss) if miss else '', conf['unit_cap']))
    os.makedirs(TA_LOGS, exist_ok=True)
    with open(UNITS + '.partial', 'w', encoding='ascii', newline='\n') as f:
        for t in lines:
            f.write('\n'.join(common.wrap('', t)) + '\n')
    os.replace(UNITS + '.partial', UNITS)
    copy_file(UNITS, D_TRAIN + '/logs/UNITS_ASCII.txt', True)

class Mirror:

    def __init__(self, pairs):
        self.pairs = pairs
        self.seen = {}
        self.lock = threading.Lock()

    @staticmethod
    def sig(p):
        st = os.stat(p)
        return (st.st_size, st.st_mtime_ns)

    def run(self):
        n, err = (0, 0)
        with self.lock:
            for loc, dd, test in self.pairs:
                for r, p in walk(loc, test):
                    try:
                        s = self.sig(p)
                    except OSError:
                        continue
                    if self.seen.get(p) == s:
                        continue
                    try:
                        copy_file(p, dd + '/' + r, True)
                    except OSError as e:
                        err += 1
                        log('MIRROR ERROR %s: %s' % (os.path.basename(p), str(e)[:120]))
                        continue
                    self.seen[p] = s
                    n += 1
        return (n, err)

    def restore(self, skip=None):
        todo = []
        for loc, dd, test in self.pairs:
            for r, p in walk(dd, test):
                if skip and skip(r, dd):
                    continue
                dst = os.path.join(loc, *r.split('/'))
                if not os.path.exists(dst):
                    todo.append((p, dst))

        def one(a):
            copy_file(a[0], a[1], False)
            return a[1]
        with ThreadPoolExecutor(4) as ex:
            done = list(ex.map(one, todo))
        for p in done:
            self.seen[p] = self.sig(p)
        return len(done)

def train_pairs(conf):
    pred = conf['places']['pred_sealed' if conf['sealed'] else 'pred_open']
    return [(TA_WORK + '/ckpt', D_TRAIN + '/ckpt', None), (TA_WORK + '/pred', EMUL + '/' + pred, None), (EQ_DIR, D_TRAIN + '/EQUIV', None), (TM_DIR, D_TRAIN + '/TIMING', None), (TA_LOGS, D_TRAIN + '/logs', lambda r: not r.startswith(('equiv/', 'timing/')))]

def step_gates(conf, sha):
    for p, head in D_MARKS:
        first = ''
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                first = f.read().decode('ascii', 'replace').split('\n')[0]
        if not (first.startswith(head) and 'exit 0' in first):
            raise Stop(2, "%s is absent or not '%s ... exit 0' (notebook %s has not finished): run it first" % (p.replace(EMUL, 'EMUL'), head, '4' if head.startswith('P2') else '5'))
        log('GATE %s: %s' % (os.path.basename(p), first[:120]))
    bad = R.refusals(conf, 'train')
    for b in bad:
        log('REFUSED: ' + b)
    if bad:
        raise Stop(3, "the config refuses (%d reasons above): nothing to do in Colab; the user's decision fills TIER_A_CONFIG.json, then the notebook is rebuilt and uploaded" % len(bad))
    rate_stop(conf)
    me = this_session()
    log('CONFIG %s: design %s decided by %s; sealed %s; lockbox %s; c7a_applies %s (%s); priorities %s; first %d trainings by --only%s; reader %s; cap %g units; overrides %d; code changes %d; this runtime %s at %s units per hour (typed); stages 1 to 5: %.2f units (%s)' % (sha[:16], conf['design'], conf['decided_by'], conf['sealed'], conf['lockbox'].get('decided_by') or 'not decided', conf['c7a_applies']['value'], conf['c7a_applies']['decided_by'], conf['priorities'], len(conf['first_only']), ', then stop' if conf['first_only_then_stop'] else '', conf['reader'], conf['unit_cap'], len(conf['trainer_overrides']), len(conf['code_changes']), me.get('accelerator'), me.get('units_per_hour_typed'), R.units_other_total(conf) or 0.0, (conf.get('units_other_doc') or {}).get('decided_by')))
    return 0

def list_rows():
    lists = []
    for p in sorted(glob.glob(CFG + '/S3C_LIST*.json')):
        if '_Backup_PriorTo' in p:
            continue
        d = jload(p)
        rows = list(d.get('files') or []) + list(d.get('configs_by_outbox') or [])
        lists.append((str(d.get('made', '')), os.path.basename(p), rows))
    out = {}
    for made, name, rows in sorted(lists):
        for r in rows:
            b = str(r.get('path', '')).replace('\\', '/').rsplit('/', 1)[-1]
            if b and 'bytes' in r:
                out[b] = (int(r['bytes']), name)
    return out

def copy_configs():
    names = sorted((os.path.basename(p) for p in glob.glob(D_CFG + '/*.json') if '_Backup_PriorTo' not in p and (not os.path.basename(p).startswith('PLAN_RUNTIME'))))
    if 'S3_ARRAYS.json' not in names:
        raise Stop(4, 'EMUL/RUN/config/p2/S3_ARRAYS.json is absent')
    os.makedirs(CFG, exist_ok=True)
    for n in names:
        copy_file(D_CFG + '/' + n, CFG + '/' + n, False)
        if os.stat(CFG + '/' + n).st_size != os.stat(D_CFG + '/' + n).st_size:
            raise Stop(4, "config %s: the VM copy's size differs from the Drive's: choose Run all again" % n)
    rows = list_rows()
    rec, bad = ([], [])
    for n in names:
        b = os.stat(CFG + '/' + n).st_size
        row = rows.get(n)
        if row and row[0] != b:
            bad.append('%s %d bytes, %s %d' % (n, b, row[1], row[0]))
        rec.append({'name': n, 'bytes': b, 'sha256': sha256_file(CFG + '/' + n), 'listed_in': row[1] if row else None, 'listed_bytes': row[0] if row else None})
    cp = TA_LOGS + '/COPY_CONFIGS_%s.json' % boot12()
    jsave(cp, {'made_utc': utc(), 'configs': rec, 'rule': "size equal to the Drive's and to the newest list row naming it); sha256 recorded"})
    copy_file(cp, D_TRAIN + '/logs/' + os.path.basename(cp), True)
    for x in rec:
        log('config %s %d bytes sha16 %s%s' % (x['name'], x['bytes'], x['sha256'][:16], ' = %s' % x['listed_in'] if x['listed_in'] else ' (no list row)'))
    if bad:
        raise Stop(4, 'configs differ from their list rows: %s' % '; '.join(bad))
    return len(names)

def array_rows():
    a = jload(CFG + '/S3_ARRAYS.json')
    ch = jload(CFG + '/S3_CHANNELS.json')
    out = {}
    ents = []
    for y, ent in sorted(a['years'].items()):
        ents += list(ent.get('files', {}).values())
    ents += list(a.get('other', {}).values())
    for e in ents:
        if not isinstance(e, dict) or 'path' not in e:
            continue
        b = str(e['path']).replace('\\', '/').rsplit('/', 1)[-1]
        if b.startswith('S3_') or b in ('SPLITS.json',):
            continue
        out[b] = e.get('bytes')
    m = str(ch['mask']['path']).replace('\\', '/').rsplit('/', 1)[-1]
    out.setdefault(m, None)
    return sorted(out.items())

def copy_arrays():
    rows = array_rows()
    lrows = list_rows()
    todo, need, bad = ([], 0, [])
    for n, b in rows:
        src = D_ARR + '/' + n
        if not os.path.isfile(src):
            bad.append('%s absent on the Drive' % n)
            continue
        ds = os.stat(src).st_size
        lr = lrows.get(n)
        for what, v in (('S3_ARRAYS.json', b), (lr[1] if lr else '', lr[0] if lr else None)):
            if v is not None and int(v) != ds:
                bad.append('%s: Drive %d bytes, %s %d' % (n, ds, what, v))
        dst = ARR + '/' + n
        if not (os.path.isfile(dst) and os.stat(dst).st_size == ds):
            todo.append((src, dst, ds))
            need += ds
    if bad:
        raise Stop(4, 'the arrays on the Drive differ from their records: %s' % '; '.join(bad[:6]))
    os.makedirs(ARR, exist_ok=True)
    free = shutil.disk_usage(ROOT if os.path.isdir(ROOT) else '/').free
    if free < need + 20000000000.0:
        raise Stop(12, 'VM disk free %.1f GB, the arrays need %.1f GB + 20 GB: choose a runtime with a larger disk' % (gb(free), gb(need)))
    t0 = time.time()

    def one(x):
        copy_file(x[0], x[1], False)
        return x
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, todo))
    wrong = [os.path.basename(d) for s, d, ds in todo if os.stat(d).st_size != ds]
    if wrong:
        raise Stop(4, 'copies of a wrong size: %s; choose Run all again' % wrong)
    log("ARRAYS: %d files (%d copied now, %.2f GB in %.0f s); each size equal to S3_ARRAYS.json's, its newest list row's and the Drive's " % (len(rows), len(todo), gb(need), time.time() - t0))
    return len(rows)

def gpu_key(conf):
    env = jload(env_file())
    name = env.get('gpu', {}).get('name') or ''
    g = R.candidate_for(conf, name)
    return (g['key'] if g else None, name, env)

def chosen_on_drive(conf):
    p = D_TRAIN + '/TIMING/CHOICE.json'
    if not os.path.isfile(p):
        return None
    d = jload(p)
    if d.get('rule_sha256') != R.rule_sha256(conf):
        return None
    return d.get('chosen')

def lb1_check(conf):
    lb = conf['lockbox']
    if lb.get('decided_by') is None:
        return None
    p = CFG + '/' + lb['file']
    if not os.path.isfile(p):
        raise Stop(13, 'the lockbox is decided but RUN/config/p2/%s is absent: nothing is trained' % lb['file'])
    with open(p, 'rb') as f:
        raw = f.read()
    errs = R.lb1_errors(conf, raw, jload(CFG + '/S3_ARRAYS.json'))
    if errs:
        for e in errs:
            log('LOCKBOX DATES DIFFER: ' + e)
        raise Stop(13, "the lockbox's dates on this run's S3_ARRAYS.json are not the decided ones (the lines above): nothing is trained")
    log("LOCKBOX_DAYS.json sha256 %s; on S3_ARRAYS.json's dates the lockbox dates hash %s and the buffer dates %s, as the config's" % (lb['sha256'][:16], lb['hashes']['lockbox'][:16], lb['hashes']['buffers'][:16]))
    return 0

def step_copy(conf):
    n = copy_configs()
    log('CONFIGS: %d copied to RUN/config/p2 on the VM' % n)
    lb1_check(conf)
    key = gpu_key(conf)[0]
    if conf['copy_arrays'] == 'always' or (key and chosen_on_drive(conf) == key):
        copy_arrays()
    else:
        log('ARRAYS: not copied in this cell (copy_arrays %s; this GPU is not yet the chosen one): step train copies them on the chosen GPU; a timing session copies none' % conf['copy_arrays'])
    return 0

def next_candidate_line(conf, refused=()):
    for g in R.candidates(conf):
        if g['key'] in refused:
            continue
        eq = equiv_record(conf, g['key'])
        if eq is not None and (not eq.get('pass')):
            continue
        if not os.path.isfile(D_TRAIN + '/TIMING/%s.json' % g['key']):
            return 'time the next GPU: Runtime > Change runtime type > %s, type its rate and its name in cell 1, then Run all' % g['runtime_type']
    return 'no candidate GPU is left to time'

def core16():
    return sha256_file(CORE)[:16]

def five_of(env):
    return {k: env.get(k) for k in ('python', 'torch', 'torch_cuda', 'cudnn', 'numpy')}

def stream(cmd, out_file, env=None, cwd=None):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    e = dict(os.environ, HOME=CHILD_HOME)
    os.makedirs(CHILD_HOME, exist_ok=True)
    e.update(env or {})
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e, cwd=cwd or ROOT, start_new_session=True)
    CHILD[0] = p
    with open(out_file, 'ab') as fo:
        for line in p.stdout:
            t = common.mask(line.rstrip('\n'))
            print(t, flush=True)
            fo.write((t + '\n').encode('ascii', 'replace'))
    rc = p.wait()
    CHILD[0] = None
    return rc

def equiv_record(conf, key):
    p = os.path.join(EQ_DIR, key, 'core_' + core16(), 'EQUIV_RECORD.json')
    dp = D_TRAIN + '/EQUIV/%s/core_%s/EQUIV_RECORD.json' % (key, core16())
    for x in (p, dp):
        if os.path.isfile(x):
            return jload(x)
    return None

def step_equiv(conf):
    key, name, env = gpu_key(conf)
    if key is None or key not in [g['key'] for g in R.candidates(conf)]:
        raise Stop(5, 'this GPU (%s) is not a candidate of the choice rule: %s' % (name, next_candidate_line(conf)))
    c16 = core16()
    vm = os.path.join(EQ_DIR, key, 'core_' + c16)
    dd = D_TRAIN + '/EQUIV/%s/core_%s' % (key, c16)
    m = Mirror([(vm, dd, None)])
    m.restore()
    res_p = os.path.join(vm, 'results.json')
    rec_p = os.path.join(vm, 'EQUIV_RECORD.json')
    res = jload(res_p) if os.path.isfile(res_p) else {}
    reused = bool(res.get('finished_utc') and res.get('gate'))
    if reused:
        log('EQUIV %s: finished in an earlier session (%s); not run again' % (key, res['finished_utc']))
    else:
        rc = stream([PY, '-I', '-B', '-u', SHIM, '--config', CONFIG, EQUIV, '--device', 'cuda', '--out', vm, '--config-dir', CFG, '--ckpt-dir', os.path.join(TA_WORK, 'tmp', 'equiv_ckpt', key, c16)], os.path.join(vm, 'console_%s.txt' % boot12()))
        m.run()
        res = jload(res_p) if os.path.isfile(res_p) else {}
        if rc != 0 or not res.get('finished_utc'):
            raise Stop(4, 'the equivalence test did not finish on %s (exit %d): choose Run all again; if it repeats' % (key, rc))
    from train import plan as P
    probe = res.get('precision_probe') or {}
    try:
        rec = P.equivalence_from_results(res_p)
        count = rec['nondeterministic_count']
    except P.PlanError as e:
        rec, count = (None, None)
        log('EQUIV %s: no record (%s)' % (key, str(e)[:200]))
    ok = bool(probe.get('pass')) and rec is not None and (count == 0)
    same = {n: bool(v.get('pass')) for n, v in (res.get('same_machine_cpu') or {}).items()}
    if reused and os.path.isfile(rec_p):
        out = jload(rec_p)
    else:
        out = {'schema': 'tier-a-equiv-1', 'gpu': key, 'device_name': name, 'pass': ok, 'record': rec, 'gate': res.get('gate'), 'probe': {k: probe.get(k) for k in ('pass', 'matmul_rel_err', 'conv_rel_err', 'tol')}, 'nondeterministic_count': count, 'emul_core_sha256': sha256_file(CORE), 'five': five_of(res.get('env') or {}), 'runtime_tag': env.get('runtime_tag'), 'same_machine_cpu_pass': same, 'session': boot12(), 'made_utc': utc()}
        jsave(rec_p, out)
        m.run()
    ok = bool(out.get('pass'))
    log('EQUIV %s (%s): probe %s (matmul %.3g, conv %.3g, tol %g); nondeterministic_count %s; gate %s; emul_core.py sha256 %s; same-machine CPU (descriptive) %s' % (key, name, 'PASS' if probe.get('pass') else 'FAIL', float(probe.get('matmul_rel_err') or 0), float(probe.get('conv_rel_err') or 0), float(probe.get('tol') or 0), count, str(res.get('gate'))[:60], out['emul_core_sha256'], same))
    if not ok:
        raise Stop(4, 'the equivalence test refuses this GPU: %s' % next_candidate_line(conf, [key]))
    return 0

def timing_summary(conf, key, name, env, vm):
    nets = conf['choice_rule']['networks']
    rows, peak_max, total = ({}, 0, 0)
    for n in nets:
        d = jload(os.path.join(vm, n + '.json'))
        pk = d.get('peak') or {}
        a = int(pk.get('cuda_max_memory_allocated_bytes') or 0)
        t = int(pk.get('cuda_total_memory_bytes') or 0)
        peak_max, total = (max(peak_max, a), max(total, t))
        di = d.get('device_inputs') or {}
        rows[n] = {'median_s': float(d['synthetic']['median_s']), 'cuda_max_memory_allocated_bytes': a, 'cuda_max_memory_reserved_bytes': int(pk.get('cuda_max_memory_reserved_bytes') or 0), 'device_inputs_bytes': di.get('bytes'), 'device_inputs_training': di.get('training')}
    return {'schema': 'tier-a-timing-1', 'gpu': key, 'device_name': name, 'total_memory_bytes': total, 'step_seconds': {n: rows[n]['median_s'] for n in nets}, 'peak_allocated_max_bytes': peak_max, 'networks': rows, 'rule_sha256': R.rule_sha256(conf), 'config_sha256': sha256_file(CONFIG), 'emul_core_sha256': sha256_file(CORE), 'five': five_of(env), 'runtime_tag': env.get('runtime_tag'), 'session': boot12(), 'made_utc': utc()}

def step_timing(conf):
    key, name, env = gpu_key(conf)
    if key is None or key not in [g['key'] for g in R.candidates(conf)]:
        raise Stop(5, 'this GPU (%s) is not a candidate: %s' % (name, next_candidate_line(conf)))
    vm = os.path.join(TM_DIR, key)
    summ = os.path.join(TM_DIR, key + '.json')
    m = Mirror([(TM_DIR, D_TRAIN + '/TIMING', None)])
    m.restore()
    s = jload(summ) if os.path.isfile(summ) else None
    if s and s.get('rule_sha256') == R.rule_sha256(conf) and (s.get('emul_core_sha256') == sha256_file(CORE)):
        log('TIMING %s: made in an earlier session (%s); not run again' % (key, s.get('made_utc')))
    else:
        pri = ','.join((str(p) for p in conf['choice_rule']['priorities']))
        c7a = 'true' if conf['c7a_applies']['value'] else 'false'
        for n in conf['choice_rule']['networks']:
            out = os.path.join(vm, n + '.json')
            cp = os.path.join(vm, 'console_%s_%s.txt' % (boot12(), n))
            rc = stream([PY, '-I', '-B', '-u', SHIM, '--config', CONFIG, BENCH, '--network', n, '--width', '1', '--device', 'cuda', '--synthetic-only', '--config-dir', CFG, '--work-dir', os.path.join(TA_WORK, 'bench'), '--out', out, '--device-inputs', pri, '--c7a-applies', c7a], cp)
            if rc != 0 or not os.path.isfile(out):
                txt = ''
                if os.path.isfile(cp):
                    with open(cp, 'rb') as f:
                        txt = f.read()[-4000:].decode('ascii', 'replace')
                if 'out of memory' in txt.lower():
                    jsave(out, {'network': n, 'synthetic': {'median_s': 1000000000.0}, 'peak': {'cuda_max_memory_allocated_bytes': int(env.get('gpu', {}).get('total_memory_bytes') or 0) + 1, 'cuda_total_memory_bytes': int(env.get('gpu', {}).get('total_memory_bytes') or 0)}, 'oom': 'CUDA out of memory with the device inputs: this GPU does not hold them'})
                    log('TIMING %s %s: out of memory with the device inputs (recorded: the choice rule refuses this GPU)' % (key, n))
                    continue
                raise Stop(4, 'the timing of %s failed on %s (exit %d): choose Run all again; if it repeats' % (n, key, rc))
        s = timing_summary(conf, key, name, env, vm)
        jsave(summ, s)
        m.run()
        log('TIMING %s: %s; peak allocated %.2f GB of %.2f GB' % (key, ', '.join(('%s %.4f s' % (n, v) for n, v in sorted(s['step_seconds'].items()))), gb(s['peak_allocated_max_bytes']), gb(s['total_memory_bytes'])))
    timings, equivs = ({}, {})
    for g in R.candidates(conf):
        k = g['key']
        tp = D_TRAIN + '/TIMING/%s.json' % k
        timings[k] = jload(tp) if os.path.isfile(tp) else None
        equivs[k] = equiv_record(conf, k)
    d = R.choose(conf, timings, equivs, drive_sessions())
    for k, r in sorted((d.get('rows') or {}).items()):
        if r['projected_units'] is None:
            log('CHOICE ROW %s: %s' % (k, '; '.join(r['refused'])))
            continue
        log('CHOICE ROW %s: %.2f projected units (%.2f h x %s), peak %.2f of %.2f GB; %s' % (k, r['projected_units'], r['projected_hours'], r['units_per_hour'], r['peak_gb'], r['memory_gb'], '; '.join(r['refused']) or 'eligible'))
    st = d['status']
    if st == 'untimed':
        raise Stop(5, 'time the next GPU: Runtime > Change runtime type > %s, type its rate and its name in cell 1, then Run all' % d['next']['runtime_type'])
    if st == 'rule_changed':
        raise Stop(6, 'the choice rule changed after the timings of %s were made (rule 15): a decision must say which rule applies' % d['keys'])
    if st == 'rates_missing':
        raise Stop(6, 'the choice rule refuses: no rate typed for %s in any of its sessions: connect that runtime, type its rate and its name in cell 1, then Run all' % ', '.join(d['keys']))
    if st == 'none_eligible':
        raise Stop(6, 'no candidate GPU is eligible (the rows above say why)')
    ch = d['chosen']
    rec = dict(d, made_utc=utc(), session=boot12(), config_sha256=sha256_file(CONFIG))
    jsave(os.path.join(TM_DIR, 'CHOICE.json'), rec)
    m.run()
    rt = [g['runtime_type'] for g in R.candidates(conf) if g['key'] == ch][0]
    log('CHOICE: %s (fewest projected units: %s; the A100 within %d percent is preferred)' % (ch, d['fewest'], round(100 * conf['choice_rule']['prefer_within_fraction'])))
    if ch != key:
        raise Stop(5, 'this GPU is not the chosen one: Runtime > Change runtime type > %s, type its rate and its name in cell 1, then Run all' % rt)
    return 0

def check_five(conf, key, env):
    eq = equiv_record(conf, key)
    if not eq or not eq.get('pass'):
        raise Stop(4, 'no passed equivalence test on %s for this emul_core.py (cell 5): choose Run all again' % key)
    now = five_of(env)
    diff = {k: (eq['five'].get(k), now.get(k)) for k in now if eq['five'].get(k) != now.get(k)}
    log('five versions now %s; at the passed test (%s) %s; runtime tag now %s, then %s' % (now, eq.get('made_utc'), eq['five'], env.get('runtime_tag'), eq.get('runtime_tag')))
    if diff:
        raise Stop(7, "this session's versions differ from the session that passed the equivalence test on %s (%s): it trains nothing" % (key, diff))
    tm = jload(D_TRAIN + '/TIMING/%s.json' % key)
    mm = R.device_mismatch(env.get('gpu', {}), tm)
    if mm:
        raise Stop(7, "this session's GPU (%s, %.1f GB) is not the one timed and tested (%s, %.1f GB): it trains nothing" % (mm['session_name'], gb(mm['session_memory_bytes']), mm['timed_name'], gb(mm['timed_memory_bytes'])))
    log('DEVICE: %s, %.1f GB, as the timing of %s ' % (tm.get('device_name'), gb(int(tm.get('total_memory_bytes') or 0)), key))
    return eq

def step_env(conf):
    key, name, env = gpu_key(conf)
    if key is None or chosen_on_drive(conf) != key:
        raise Stop(5, 'this GPU (%s) is not the chosen one (TIMING/CHOICE.json): run cell 6 first, or change the runtime type as it says' % name)
    check_five(conf, key, env)
    return 0

def norm_file(configs, name):
    fn = 'NORM_%s.json' % name
    if name.startswith('f6a_fold') and (not os.path.isfile(CFG + '/' + fn)):
        alt = 'NORM_f6a_%s.json' % name[len('f6a_fold'):]
        if os.path.isfile(CFG + '/' + alt):
            fn = alt
    return CFG + '/' + fn

def lockbox_check(conf, configs, trainings, norm_trainings=None):
    if conf['lockbox'].get('decided_by') is None:
        log('LOCKBOX: not decided at training time; sealed %s; the departure is in every training log' % conf['sealed'])
        return []
    from train import data
    if data.LOCKBOX_FILE != conf['lockbox']['file']:
        raise Stop(8, "the lockbox is decided but train.data.LOCKBOX_FILE is %r (the shim's entry)" % (data.LOCKBOX_FILE,))
    p = CFG + '/' + conf['lockbox']['file']
    if not os.path.isfile(p):
        raise Stop(8, 'the lockbox is decided but RUN/config/p2/%s is absent' % conf['lockbox']['file'])
    lb = R.lockbox_days(jload(p))
    lsha = sha256_file(p)
    days, norms = ({}, set())
    for t in trainings:
        sp = data.Split(configs, t)
        days[t['name']] = [(y, i) for y, i, _ in list(sp.train) + list(sp.val) + list(sp.held)]
    for t in trainings if norm_trainings is None else norm_trainings:
        norms.add(norm_file(configs, t['norm']))
    hit = R.lockbox_conflicts(days, lb)
    nolock = []
    for n in sorted(norms):
        with open(n, 'rb') as f:
            if lsha not in f.read().decode('ascii', 'replace'):
                nolock.append(os.path.basename(n))
    if hit or nolock:
        raise Stop(8, "the lockbox check fails: %d trainings hold lockbox days (%s); NORM files without the lockbox's sha256: %s" % (len(hit), sorted(hit)[:4], nolock[:4]))
    log("LOCKBOX %s (decided by %s): no lockbox or buffer day in the training, validation or TEST_CV days of %d trainings (the study design's Split); %d NORM files record it" % (lsha[:16], conf['lockbox']['decided_by'], len(trainings), len(norms)))
    return ['lockbox decided by %s: %s sha256 %s held out of every fit, validation, early stop and normalisation (checked by the driver)' % (conf['lockbox']['decided_by'], conf['lockbox']['file'], lsha)]

def real_step_seconds(paths, pl_ids):
    from train import plan as P
    from train import predict
    acc = {}
    for t in P.trainings_for(pl_ids['priorities'], pl_ids['c7a_applies']):
        pid = {'width': pl_ids['width'], 'steps': int(pl_ids[t['steps_key']])}
        ok, _ = predict.done_check(paths, t, pid, check_hash=False)
        if not ok:
            continue
        d = jload(os.path.join(paths.ckpt(t['name']), 'DONE.json'))
        r = d.get('result') or {}
        if r.get('train_wall_s') and r.get('steps_run'):
            acc.setdefault(t['net'], []).append(float(r['train_wall_s']) / float(r['steps_run']))
    return {k: sum(v) / len(v) for k, v in acc.items()}

def write_plan(conf, paths, key, eq, timing, extra_deps, run_only):
    session_update(conf, NOTEBOOK, 'train')
    from train import flags
    from train import plan as P
    from train import trainer_main as TM
    c7a = conf['c7a_applies']['value']
    pl_ids = {'priorities': conf['priorities'], 'c7a_applies': c7a, 'width': float(conf['width']), 'steps_main': int(conf['steps_main']), 'steps_f6': int(conf['steps_f6'])}
    prior_h, counted, partial = TM.prior_hours(paths, pl_ids)
    try:
        b = R.budget(conf, drive_sessions(), key, prior_h)
    except ValueError as e:
        raise Stop(6, 'the budget refuses: %s' % e)
    syn = dict(timing['step_seconds'])
    real = real_step_seconds(paths, pl_ids)
    ss = dict(syn)
    if conf['step_seconds_rule'] == 'slower_of_synthetic_and_real':
        for net, v in real.items():
            base = syn.get(net, syn.get(P.STEP_FALLBACK.get(net, ''), 0.0))
            ss[net] = max(float(base), float(v))
    threads = conf['threads'] or flags.rule_threads()
    deps = list(conf['departures']) + R.auto_departures(conf) + extra_deps
    import psutil
    measured = {'gpu': key, 'device_name': timing.get('device_name'), 'total_memory_bytes': timing.get('total_memory_bytes'), 'synthetic_step_seconds': syn, 'real_step_seconds': real, 'step_seconds_rule': conf['step_seconds_rule'], 'budget': b, 'prior_hours_counted': counted, 'prior_hours_partial': partial, 'equivalence_session': eq.get('session'), 'five': eq.get('five'), 'runtime_tag': eq.get('runtime_tag'), 'config_sha256': sha256_file(CONFIG), 'emul_core_sha256': sha256_file(CORE), 'choice_rule_sha256': R.rule_sha256(conf), 'design': conf['design'], 'decided_by': conf['decided_by'], 'c7a_applies_ruled_by': conf['c7a_applies']['decided_by'], 'session': boot12(), 'run_only': run_only}
    path = CFG + '/PLAN_RUNTIME.json'
    P.write_plan_runtime(path, 'cuda', float(conf['width']), int(threads), conf['priorities'], c7a, steps_main=int(conf['steps_main']), steps_f6=int(conf['steps_f6']), budget_hours=b['budget_hours'], step_seconds=ss, reader=conf['reader'], departures=deps, measured=measured, free_memory_gb=round(psutil.virtual_memory().available / 1000000000.0, 3), made_by='train_colab_driver.py (the benchmark on Colab)', day_sets=conf['day_sets'], f19=bool(conf['f19']), hold_back_p1=bool(conf['hold_back_p1']), equivalence=eq['record'])
    pl = P.load_plan(paths)
    os.makedirs(PLAN_DIR, exist_ok=True)
    shutil.copyfile(path, PLAN_DIR + '/PLAN_RUNTIME_%s.json' % time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    log('PLAN %s: budget %.3f h (%.2f units left at %s units/h + %.3f h of %d done trainings); step seconds %s; %d departures' % (pl['_sha256'][:16], b['budget_hours'], b['left_units'], b['rate'], prior_h, len(counted), ', '.join(('%s %.4f' % kv for kv in sorted(ss.items()))), len(deps)))
    return (pl, b)

def next_training(paths, pl, names):
    from train import loop, predict
    from train import plan as P
    for t in P.trainings_for(pl['priorities'], pl['c7a_applies']):
        if names and t['name'] not in names:
            continue
        pid = {'width': pl['width'], 'steps': int(pl[t['steps_key']])}
        if predict.done_check(paths, t, pid, check_hash=False)[0]:
            continue
        if loop.stopped_record(paths, t['name']) is not None:
            continue
        return t
    return None

def status_line():
    st = TA_LOGS + '/TRAIN_STATUS.json'
    pr = TA_LOGS + '/TRAIN_PROGRESS_ASCII.txt'
    cur, step = (None, None)
    try:
        cur = jload(st).get('current')
    except (OSError, ValueError):
        pass
    try:
        with open(pr, 'rb') as f:
            f.seek(max(0, os.path.getsize(pr) - 20000))
            for ln in f.read().decode('ascii', 'replace').splitlines():
                mm = re.search(' (\\S+) step (\\d+) val ', ln)
                if mm:
                    cur, step = (mm.group(1), mm.group(2))
    except OSError:
        pass
    return 'STATUS %s: training %s, step %s (the values are in the logs)' % (utc(), cur or '-', step or '-')

def cap_line(conf, g):
    return '%.2f of %g units spent, %.2f left, reserve %s%s' % (g['spent'], conf['unit_cap'], g['left'], '%.2f' % g['reserve'] if g['reserve'] is not None else 'unknown', ' (rates missing: %s)' % ', '.join(g['missing']) if g['missing'] else '')

def first_only_line(only, then_stop=False):
    try:
        res = jload(TA_LOGS + '/TRAIN_STATUS.json').get('results') or {}
    except (OSError, ValueError):
        res = {}
    nr = [n for n in only if str(res.get(n, '')).startswith('not reached')]
    later = 'they are tried again in the next run of first_only, under the same hold-back' if then_stop else 'they run later in plan order'
    return "FIRST_ONLY: %d of the %d trainings of first_only were not reached in this run%s (the trainer's hold-back of priority 1,: %s)" % (len(nr), len(only), ': ' + ', '.join(nr) if nr else '', later)
MOUNT = '/content/drive'

def drive_mounted():
    return os.path.ismount(MOUNT)

def guard_period(conf, key, fails, cap_hit, stop_child):
    g = None
    try:
        if not drive_mounted():
            raise OSError('the Drive is not mounted')
        session_update(conf, NOTEBOOK, 'train')
        g = R.cap_guard(conf, drive_sessions(), key, EVERY_S)
        fails[0] = 0
    except Exception as e:
        fails[0] += 1
        print('cap guard: %s (%d)' % (type(e).__name__, fails[0]), flush=True)
    if not cap_hit.is_set() and (g is not None and g['hit'] or fails[0] >= 2):
        log('THE CAP: %s; the trainer is stopped' % (cap_line(conf, g) if g is not None else 'the session records cannot be written or read on the Drive (two mirror periods)'))
        cap_hit.set()
    if cap_hit.is_set():
        stop_child()
        return False
    return True

def step_train(conf):
    key, name, env = gpu_key(conf)
    if key is None or chosen_on_drive(conf) != key:
        raise Stop(5, 'this GPU (%s) is not the chosen one: run cells 5 and 6, or change the runtime type as cell 6 says' % name)
    eq = check_five(conf, key, env)
    timing = jload(D_TRAIN + '/TIMING/%s.json' % key)
    g = R.cap_guard(conf, drive_sessions(), key, EVERY_S)
    if g['hit']:
        raise Stop(6, 'THE CAP: %s; nothing starts' % cap_line(conf, g))
    lk = open(os.path.join(TA_LOGS, 'TRAIN_DRIVER.lock'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise Stop(10, 'another driver runs on this VM: not started')
    lb1_check(conf)
    from train import cfg as tcfg
    from train import plan as P
    paths = tcfg.Paths(CFG, TA_WORK, TA_LOGS, MARKER, ARR)
    configs = tcfg.Configs(paths)
    ts = P.trainings_for(conf['priorities'], conf['c7a_applies']['value'])
    then_stop = bool(conf['first_only_then_stop'])
    run_ts = [t for t in ts if t['name'] in conf['first_only']] if then_stop else ts
    if then_stop and len(run_ts) != len(conf['first_only']):
        raise Stop(4, 'first_only names trainings that are not in the plan (%d of %d found)' % (len(run_ts), len(conf['first_only'])))
    extra = lockbox_check(conf, configs, ts, run_ts)
    copy_arrays()
    m = Mirror(train_pairs(conf))
    MIRROR[0] = m

    def skip(rel, dd):
        if dd == D_TRAIN + '/ckpt' and rel.endswith('/last.pt'):
            return os.path.isfile(dd + '/' + rel.rsplit('/', 1)[0] + '/DONE.json')
        return False
    n = m.restore(skip)
    log("RESTORE: %d files from EMUL/TRAIN and the predictions' place to the VM (resume)" % n)
    stop = threading.Event()
    cap_hit = threading.Event()

    def stop_child():
        p = CHILD[0]
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGINT)
        except OSError:
            return
        for _ in range(120):
            if p.poll() is not None or stop.wait(1):
                return
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
    fails = [0]

    def loop_():
        while not stop.wait(EVERY_S):
            try:
                m.run()
            except Exception as e:
                print('mirror: %s' % type(e).__name__, flush=True)
            if guard_period(conf, key, fails, cap_hit, stop_child):
                print(status_line(), flush=True)
    th = threading.Thread(target=loop_, daemon=True)
    th.start()
    rc = 0
    try:
        runs = []
        if conf['first_only']:
            runs.append(list(conf['first_only']))
        if then_stop:
            log('EARLY START: only the %d trainings of first_only run (%s); nothing else is started early and the whole plan does not follow (first_only_then_stop)' % (len(conf['first_only']), ', '.join(conf['first_only'])))
        else:
            runs.append(None)
        for only in runs:
            pl, b = write_plan(conf, paths, key, eq, timing, extra, only)
            g = R.cap_guard(conf, drive_sessions(), key, EVERY_S)
            if cap_hit.is_set() or g['hit']:
                log('THE CAP: %s; no run starts' % cap_line(conf, g))
                rc = 6
                break
            nt = next_training(paths, pl, only)
            if nt is None:
                log('RUN %s: nothing left to train' % ('first_only' if only else 'whole plan'))
                continue
            proj = P.projected_seconds(nt, pl) or 0.0
            need = 2.0 * proj / 3600.0 * float(b['rate'])
            if need > b['left_units']:
                log("THE CAP: %s's 2 x projected units %.2f pass the %.2f left of %g; no run starts" % (nt['name'], need, b['left_units'], conf['unit_cap']))
                rc = 6
                break
            log('RUN %s: next %s (2 x projected %.2f units of %.2f left)' % ('first_only (%d)' % len(only) if only else 'whole plan', nt['name'], need, b['left_units']))
            os.makedirs(os.path.join(TA_WORK, 'tmp'), exist_ok=True)
            cmd = [PY, '-I', '-B', '-u', SHIM, '--config', CONFIG, TRAINER, '--config-dir', CFG, '--work-dir', TA_WORK, '--logs-dir', TA_LOGS, '--marker', MARKER, '--arrays-dir', ARR]
            if only:
                cmd += ['--only', ','.join(only)]
            if cap_hit.is_set():
                rc = 6
                break
            t_rc = stream(cmd, log_path('TRAINER_STDOUT'), cwd=TA_WORK)
            fs = ''
            try:
                fs = str(jload(TA_LOGS + '/TRAIN_STATUS.json').get('final_status', ''))
            except (OSError, ValueError):
                pass
            log('TRAINER exit %d; status %s' % (t_rc, fs or '-'))
            if only and t_rc != 3:
                log(first_only_line(only, then_stop))
            if cap_hit.is_set():
                rc = 6
                break
            if t_rc == 3:
                rc = 10
                break
            if fs.startswith('failed: ') and (not re.match('failed: [0-9]+ trainings failed$', fs)):
                rc = 11
                break
    finally:
        stop.set()
        th.join(timeout=5)
        n1, e1 = m.run()
        if e1:
            n1, e1 = m.run()
        session_update(conf, NOTEBOOK, 'train')
        log('MIRROR at the end: %d files; %d errors' % (n1, e1))
    if then_stop and rc == 0:
        from train import predict
        nd = sum((1 for t in run_ts if predict.done_check(paths, t, {'width': float(conf['width']), 'steps': int(conf[t['steps_key']])}, check_hash=False)[0]))
        try:
            res = jload(TA_LOGS + '/TRAIN_STATUS.json').get('results') or {}
        except (OSError, ValueError):
            res = {}
        vals = [str(res.get(t['name'], '')) for t in run_ts]
        log('EARLY START: the run of first_only is over: %d of the %d trainings done, %d not reached, %d failed in this run; the whole plan is not started (first_only_then_stop)' % (nd, len(run_ts), sum((1 for v in vals if v.startswith('not reached'))), sum((1 for v in vals if v.startswith('failed')))))
    if rc == 6:
        raise Stop(6, 'THE CAP: what is left of the %g units is used to the margin (the THE CAP line above)' % conf['unit_cap'])
    if rc == 11:
        raise Stop(11, 'the trainer stopped the run (status above; its TRAIN_PROGRESS_ASCII.txt in EMUL/TRAIN/logs says why)')
    if rc == 10:
        raise Stop(10, 'another trainer holds the lock on this VM')
    return 0

def step_end(conf):
    from train import cfg as tcfg
    from train import plan as P
    from train import predict
    st = {}
    try:
        st = jload(TA_LOGS + '/TRAIN_STATUS.json')
    except (OSError, ValueError):
        pass
    res = st.get('results') or {}
    for k in sorted(res):
        log('RESULT %s: %s' % (k, str(res[k])[:100]))
    left = []
    pp = CFG + '/PLAN_RUNTIME.json'
    if os.path.isfile(pp):
        paths = tcfg.Paths(CFG, TA_WORK, TA_LOGS, MARKER, ARR)
        pl = P.load_plan(paths)
        for t in P.trainings_for(pl['priorities'], pl['c7a_applies']):
            pid = {'width': pl['width'], 'steps': int(pl[t['steps_key']])}
            if not predict.done_check(paths, t, pid, check_hash=False)[0]:
                left.append(t['name'])
        log('LEFT: %d of %d trainings of the plan not done: %s' % (len(left), len(P.trainings_for(pl['priorities'], pl['c7a_applies'])), ', '.join(left[:12]) + (' ...' if len(left) > 12 else '')))
    ss = drive_sessions()
    me = [s for s in ss if s.get('session') == boot12()]
    spent, miss = R.spent_units(conf, ss)
    if me:
        r = R.rate_of(conf, me[0].get('accelerator'), ss)
        log('UNITS: this session %.3f h on %s = %s units; spent %.2f of %g (left %.2f)%s' % (float(me[0]['hours']), me[0]['accelerator'], '%.2f' % (float(me[0]['hours']) * r) if isinstance(r, (int, float)) else 'unknown', spent, conf['unit_cap'], conf['unit_cap'] - spent, ' (rates missing: %s)' % ', '.join(miss) if miss else ''))
    m = Mirror(train_pairs(conf))
    n, e = m.run()
    log('END: the last mirror %d files, %d errors' % (n, e))
    return 0
STEPS = {'session': lambda conf: 0, 'gates': step_gates, 'copy': step_copy, 'equiv': step_equiv, 'timing': step_timing, 'env': step_env, 'train': step_train, 'end': step_end}
CHILD = [None]
MIRROR = [None]

def on_term(_s, _f):
    p = CHILD[0]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=120)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    if MIRROR[0] is not None:
        try:
            MIRROR[0].run()
        except Exception:
            pass
    sys.exit(143)

def main(argv):
    if not argv or argv[0] not in STEPS:
        print('usage: train_colab_driver.py %s' % '|'.join(STEPS))
        return 1
    if not os.path.isdir(EMUL):
        print('THE DRIVE IS NOT MOUNTED (PROJECT_DATA/EMUL not found under /content/drive/MyDrive): run cell 1 first')
        return 9
    os.makedirs(TA_LOGS, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term)
    install_overrides()
    conf, sha = R.load(CONFIG)
    conf = R.attach_units_other(conf, read_units_other(conf))
    session_update(conf, NOTEBOOK, argv[0])
    try:
        if argv[0] not in ('session', 'gates', 'end'):
            bad = R.refusals(conf, 'train')
            for b in bad:
                log('REFUSED: ' + b)
            if bad:
                raise Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))
            rate_stop(conf)
        if argv[0] == 'gates':
            return step_gates(conf, sha)
        return STEPS[argv[0]](conf)
    except Stop as s:
        log('STOP (exit %d): %s' % (s.code, s))
        return s.code
    finally:
        push_log()
if __name__ == '__main__':
    try:
        RC = main(sys.argv[1:])
    except SystemExit:
        raise
    except BaseException:
        import traceback
        for ln in traceback.format_exc().splitlines()[-12:]:
            print(common.mask(ln), flush=True)
        RC = 1
    sys.exit(RC)
