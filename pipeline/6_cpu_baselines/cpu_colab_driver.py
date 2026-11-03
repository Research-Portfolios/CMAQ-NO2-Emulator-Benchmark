"""Driver of the CPU baselines stage on Colab: checks the earlier stages' markers and the runtime, copies the configuration files and arrays with size checks, then runs the read-only pass, the statistics rewrite, the climatology tables and the baselines, mirroring results to the Drive."""
import glob
import os
import shutil
import signal
import sys
import threading
import time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tier_a_rules as R
import train_colab_driver as TD
NOTEBOOK = 'EMUL_07C_TIER_A_CPU'
YEARS = (2015, 2016, 2017, 2018, 2019)
C7A_YEARS = (2015, 2016, 2018)
FOLDS = ('2016', '2015', '2017', '2018', '2019', 'f6')
PD17 = os.path.join(HERE, 'cpu', 'pd17_pass.py')
NORMF6 = os.path.join(HERE, 'cpu', 'norm_f6.py')
CTABLES = os.path.join(HERE, 'cpu', 'ctables.py')
RUNP7 = os.path.join(HERE, 'baselines', 'run_p7.py')
CPU_WORK = TD.W + '/cpu'
CPU_LOGS = TD.RUN + '/logs/cpu'
D2828_VM = TD.RUN + '/config/p2_rev'
CT_VM = TD.ARR + '/ctables'
WORK_VM = TD.ARR + '/work'
MAN_VM = CPU_WORK + '/pd17/manifests'
PRED_VM = CPU_WORK + '/pred/baselines'
LB_VM = CPU_WORK + '/lockbox/baselines'
D_WORK = TD.D_ARR + '/work'
D_EXT = TD.EMUL + '/EXTRACTS'
PD17_NAME = 'PD17_RECORD.json'
NORM_LIST = 'S3C_LIST_8_NORM_F6.json'
STAGE1_NAMES = ['NORM_f6a_fold2016', 'NORM_f6a_fold2018'] + ['NORM_f6_train%s_test%s_%s' % (a, b, m) for a, b in (('2017', '2019'), ('2019', '2017')) for m in ('TRUE', 'OTHER', 'OWN')]
WORK_FILES = tuple(('%s_%d.%s' % (s, y, e) for y in YEARS for s, e in (('daystats', 'npz'), ('daystats_c7', 'npz'), ('build', 'json'), ('build_c7', 'json')))) + ('norm_index.json',)

def places(conf):
    p = conf['places']
    e = TD.EMUL + '/'
    return {'d2828': e + p['cpu_config'], 'ct': e + p['ctables'], 'base': e + p['baselines_sealed'], 'lb': e + p['lockbox_sealed'] + '/baselines', 'logs': e + p['cpu_logs'], 'done': e + p['cpu_done']}

class Overwrite(Exception):
    pass

def ok_place(conf, dst):
    pl = places(conf)
    tops = (pl['d2828'], pl['ct'], pl['base'], pl['lb'], pl['logs'])
    if '..' in dst or not (dst == pl['done'] or any((dst.startswith(t + '/') for t in tops))):
        raise TD.Stop(1, "a Drive write outside this package's places was refused: %s" % dst.replace(TD.EMUL, 'EMUL'))
    return dst

def put(conf, src, dst, own=False):
    ok_place(conf, dst)
    if os.path.isfile(dst):
        if os.path.getsize(dst) == os.path.getsize(src) and TD.sha256_file(dst) == TD.sha256_file(src):
            return 'same'
        if not (own and TD.boot12() in os.path.basename(dst)):
            mk = marker_of(conf, dst)
            if mk is None or os.path.isfile(mk):
                raise Overwrite(dst)
            TD.log('REPLACED (incomplete group): EMUL%s; its marker EMUL%s is absent' % (dst[len(TD.EMUL):], mk[len(TD.EMUL):]))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)
    return 'copied'

class Mirror7:

    def __init__(self, conf, pairs):
        self.conf = conf
        self.pairs = pairs
        self.seen = {}
        self.lock = threading.Lock()
        self.refused = []

    def run(self):
        n, err = (0, 0)
        with self.lock:
            items = [(is_marker(os.path.basename(p)), dd, r, p) for loc, dd, test in self.pairs for r, p in TD.walk(loc, test)]
            items.sort(key=lambda t: t[0])
            for last, dd, r, p in items:
                try:
                    s = TD.Mirror.sig(p)
                except OSError:
                    continue
                if self.seen.get(p) == s:
                    continue
                if last:
                    try:
                        whole = group_on_drive(self.conf, p, dd + '/' + r)
                    except (OSError, ValueError, KeyError):
                        whole = False
                    if not whole:
                        err += 1
                        continue
                try:
                    put(self.conf, p, dd + '/' + r, own=dd == places(self.conf)['logs'])
                except Overwrite:
                    err += 1
                    if p not in self.refused:
                        self.refused.append(p)
                        TD.log('MIRROR REFUSED: EMUL%s holds other bytes; nothing is overwritten' % (dd + '/' + r)[len(TD.EMUL):])
                    self.seen[p] = s
                    continue
                except OSError as e:
                    err += 1
                    TD.log('MIRROR ERROR %s: %s' % (os.path.basename(p), str(e)[:120]))
                    continue
                self.seen[p] = s
                n += 1
        return (n, err)

    def restore(self, skip=None):
        todo = []
        for loc, dd, test in self.pairs:
            for r, p in TD.walk(dd, test):
                if skip and skip(r, dd):
                    continue
                dst = os.path.join(loc, *r.split('/'))
                if not os.path.exists(dst):
                    todo.append((p, dst))
        for src, dst in todo:
            TD.copy_file(src, dst, False)
            self.seen[dst] = TD.Mirror.sig(dst)
        return len(todo)

def pairs_of(conf, which):
    pl = places(conf)
    all_ = {'d2828': (D2828_VM, pl['d2828'], None), 'ct': (CT_VM, pl['ct'], lambda r: not r.startswith('_work')), 'base': (PRED_VM, pl['base'], None), 'lb': (LB_VM, pl['lb'], None), 'logs': (CPU_LOGS, pl['logs'], None)}
    return [all_[k] for k in which]
MARKERS = ('DONE.json', 'LOCKBOX_DONE.json', NORM_LIST, PD17_NAME)

def is_marker(name):
    return name in MARKERS or (name.startswith('CTABLES_') and name.endswith('.json'))

def marker_of(conf, dst):
    pl = places(conf)
    b = os.path.basename(dst)
    if is_marker(b):
        return None
    for k, m in (('base', 'DONE.json'), ('lb', 'LOCKBOX_DONE.json')):
        if dst.startswith(pl[k] + '/'):
            return os.path.dirname(dst) + '/' + m
    if dst.startswith(pl['ct'] + '/') and b.startswith('CT_'):
        return pl['d2828'] + '/CTABLES_%s.json' % b.split('_')[1]
    if dst.startswith(pl['d2828'] + '/'):
        return pl['d2828'] + '/' + NORM_LIST
    return None

def group_on_drive(conf, vm_m, d_m):
    b = os.path.basename(vm_m)
    if b == PD17_NAME:
        return True
    d = TD.jload(vm_m)
    if b.startswith('CTABLES_'):
        ct = places(conf)['ct']
        two = [(os.path.join(CT_VM, f[k]['file']), ct + '/' + f[k]['file']) for f in d['fits'].values() for k in ('doy', 'r')]
    else:
        rows = d['files']
        rows = rows.values() if isinstance(rows, dict) else rows
        two = []
        for r in rows:
            n = os.path.basename(str(r['path']).replace('\\', '/'))
            two.append((os.path.join(os.path.dirname(vm_m), n), os.path.dirname(d_m) + '/' + n))
    return all((drive_size_ok(v, x) for v, x in two))

def cpu_check(conf):
    acc, name = TD.accelerator(conf)
    if acc != 'CPU':
        raise TD.Stop(5, "this notebook runs on a CPU runtime and this one has %s: Runtime > Change runtime type > CPU (High-RAM on), type the CPU's rate in cell 1 and choose CPU in RUNTIME_SHOWN, then Run all" % name)

def cap_check(conf):
    why = R.stage9_cap(conf, TD.drive_sessions())
    if why:
        raise TD.Stop(6, why)
    return 0

def jsave(p, d):
    TD.jsave(p, d)
    return {'name': os.path.basename(p), 'bytes': os.path.getsize(p), 'sha256': TD.sha256_file(p)}

def step_gates(conf, sha):
    for p, head in TD.D_MARKS:
        first = ''
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                first = f.read().decode('ascii', 'replace').split('\n')[0]
        if not (first.startswith(head) and 'exit 0' in first):
            raise TD.Stop(2, "%s is absent or not '%s ... exit 0': notebook %s has not finished: run it first" % (p.replace(TD.EMUL, 'EMUL'), head, '4' if head.startswith('P2') else '5'))
        TD.log('GATE %s: %s' % (os.path.basename(p), first[:120]))
    bad = R.refusals(conf, 'stage9')
    for b in bad:
        TD.log('REFUSED: ' + b)
    if bad:
        raise TD.Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))
    TD.rate_stop(conf)
    cpu_check(conf)
    cap_check(conf)
    me = TD.this_session()
    TD.log('CONFIG %s: design %s decided by %s; stage9 %s decided by %s; sealed %s; lockbox %s; c7a_applies %s; this runtime %s at %s units per hour (typed); stages 1 to 5: %.2f units (%s)' % (sha[:16], conf['design'], conf['decided_by'], conf['stage9']['names'], conf['stage9']['decided_by'], conf['sealed'], conf['lockbox'].get('decided_by'), conf['c7a_applies']['value'], me.get('accelerator'), me.get('units_per_hour_typed'), R.units_other_total(conf) or 0.0, (conf.get('units_other_doc') or {}).get('decided_by')))
    return 0

def own_vm_files():
    out = [os.path.join(TD.CFG, n + '_STAGE1.json') for n in STAGE1_NAMES]
    out.append(os.path.join(TD.CFG, NORM_LIST))
    out += glob.glob(os.path.join(WORK_VM, 'norm_index_Backup_PriorToNormF6*.json'))
    return [p for p in out if os.path.isfile(p)]

def copy_configs7():
    for p in own_vm_files():
        os.remove(p)
    names = sorted((os.path.basename(p) for p in glob.glob(TD.D_CFG + '/*.json') if '_Backup_PriorTo' not in p and (not os.path.basename(p).startswith('PLAN_RUNTIME'))))
    if 'S3_ARRAYS.json' not in names:
        raise TD.Stop(4, 'EMUL/RUN/config/p2/S3_ARRAYS.json is absent')
    os.makedirs(TD.CFG, exist_ok=True)
    for n in names:
        TD.copy_file(TD.D_CFG + '/' + n, TD.CFG + '/' + n, False)
        if os.stat(TD.CFG + '/' + n).st_size != os.stat(TD.D_CFG + '/' + n).st_size:
            raise TD.Stop(4, "config %s: the VM copy's size differs from the Drive's: choose Run all again" % n)
    rows = TD.list_rows()
    rec, bad = ([], [])
    for n in names:
        b = os.stat(TD.CFG + '/' + n).st_size
        row = rows.get(n)
        if row and row[0] != b:
            bad.append('%s %d bytes, %s %d' % (n, b, row[1], row[0]))
        rec.append({'name': n, 'bytes': b, 'sha256': TD.sha256_file(TD.CFG + '/' + n), 'listed_in': row[1] if row else None})
    if bad:
        raise TD.Stop(4, 'configs differ from their list rows: %s' % '; '.join(bad))
    return rec

def copy_small(src_dir, names, dst_dir, what):
    os.makedirs(dst_dir, exist_ok=True)
    out, miss = ([], [])
    for n in names:
        s = src_dir + '/' + n
        if not os.path.isfile(s):
            miss.append(n)
            continue
        d = os.path.join(dst_dir, n)
        TD.copy_file(s, d, False)
        if os.path.getsize(d) != os.path.getsize(s):
            raise TD.Stop(4, "%s %s: the VM copy's size differs from the Drive's: choose Run all again" % (what, n))
        out.append({'name': n, 'bytes': os.path.getsize(d), 'sha256': TD.sha256_file(d)})
    if miss:
        raise TD.Stop(4, '%s absent on the Drive: %s' % (what, ', '.join(miss[:6])))
    return out

def step_copy(conf):
    rec = {'schema': 'cpu-copy-1', 'session': TD.boot12(), 'made_utc': TD.utc(), 'rule': "sizes equal to the Drive's and to the newest list rows (the shared package); sha256 recorded"}
    rec['configs'] = copy_configs7()
    TD.log('CONFIGS: %d copied to RUN/config/p2 on the VM' % len(rec['configs']))
    TD.lb1_check(conf)
    TD.copy_arrays()
    rec['work'] = copy_small(D_WORK, WORK_FILES, WORK_VM, 'work file')
    TD.log('WORK: %d files of EMUL/W_P2/arrays_p2/work/ copied' % len(rec['work']))
    rec['manifests'] = []
    for y in C7A_YEARS:
        rec['manifests'] += copy_small(D_EXT + '/%d' % y, ['MANIFEST_%d.csv' % y], MAN_VM, 'manifest')
    TD.log('MANIFESTS: %d C-7a manifests copied' % len(rec['manifests']))
    row = jsave(os.path.join(CPU_LOGS, 'COPY_CPU_%s.json' % TD.boot12()), rec)
    m = Mirror7(conf, pairs_of(conf, ['logs']))
    m.run()
    TD.log('COPY record %s (%d bytes, sha16 %s)' % (row['name'], row['bytes'], row['sha256'][:16]))
    return 0

def need_copy():
    if not os.path.isfile(TD.CFG + '/S3_ARRAYS.json'):
        raise TD.Stop(4, 'the configs are not on this VM: run the copy cell (cell 4) first')

def stop_child():
    p = TD.CHILD[0]
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGINT)
    except OSError:
        return
    for _ in range(120):
        if p.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass

def run_child(conf, step, cmd, mirror):
    TD.MIRROR[0] = mirror
    stop = threading.Event()
    hit = [None]
    fails = [0]

    def loop_():
        while not stop.wait(TD.EVERY_S):
            why = None
            try:
                if not TD.drive_mounted():
                    raise OSError('the Drive is not mounted')
                TD.session_update(conf, NOTEBOOK, step)
                why = R.stage9_cap(conf, TD.drive_sessions())
                fails[0] = 0
            except Exception as e:
                fails[0] += 1
                print('cap test: %s (%d)' % (type(e).__name__, fails[0]), flush=True)
            if why or fails[0] >= 2:
                hit[0] = why or 'the session records cannot be written or read on the Drive (two periods)'
                TD.log('THE CAP: %s; the step is stopped' % hit[0])
                stop_child()
                return
            try:
                mirror.run()
            except Exception as e:
                print('mirror: %s' % type(e).__name__, flush=True)
    th = threading.Thread(target=loop_, daemon=True)
    th.start()
    try:
        os.makedirs(CPU_WORK, exist_ok=True)
        rc = TD.stream(cmd, os.path.join(CPU_LOGS, '%s_STDOUT_%s_ASCII.txt' % (step.upper(), TD.boot12())), cwd=CPU_WORK)
    finally:
        stop.set()
        th.join(timeout=5)
        n1, e1 = mirror.run()
        if e1:
            n1, e1 = mirror.run()
        TD.session_update(conf, NOTEBOOK, step)
        TD.log('MIRROR at the end of %s: %d files; %d errors' % (step, n1, e1))
    if hit[0]:
        raise TD.Stop(6, 'THE CAP: %s' % hit[0])
    if mirror.refused:
        raise TD.Stop(4, "the Drive holds %d of this step's files with other bytes (nothing was overwritten)" % len(mirror.refused))
    return rc

def pd17_state(conf):
    p = places(conf)['d2828'] + '/' + PD17_NAME
    if not os.path.isfile(p):
        return (None, None)
    rec = TD.jload(p)
    if rec.get('verdict') == 'FAIL':
        return ('FAIL', rec)
    if rec.get('S3_ARRAYS_sha256') != TD.sha256_file(TD.CFG + '/S3_ARRAYS.json'):
        return ('OLD', rec)
    return ('PASS' if rec.get('verdict') == 'PASS' else None, rec)

def need_pd17(conf):
    st, _ = pd17_state(conf)
    if st != 'PASS':
        raise TD.Stop(4, "the study design's pass has not passed on these arrays (state %s): nothing is fitted; run the pd17 cell" % st)
FAIL_LINE = "the study design's pass FAILS (the reasons above and in RUN/config/p2_rev/%s): the GBT stage stops here and nothing is fitted; d7's NORM files and every training that read a failed array (the first training stage's included) are void until a manual decision is needed" % PD17_NAME

def need_sklearn_test():
    p = os.path.join(CPU_LOGS, 'SKLEARN_TEST_%s.json' % TD.boot12())
    if not (os.path.isfile(p) and TD.jload(p).get('result') == 'PASS'):
        raise TD.Stop(4, "cell 2's self-test with scikit-learn has not passed in this session: nothing is fitted; run cell 2")

def step_pd17(conf):
    need_copy()
    st, rec = pd17_state(conf)
    if st == 'PASS':
        TD.log('passed before on these arrays (its record on the Drive); skipped')
        return 0
    if st == 'FAIL':
        for r in rec.get('reasons', [])[:12]:
            TD.log('READ-ONLY CHECK REASON: ' + r)
        raise TD.Stop(4, FAIL_LINE)
    if st == 'OLD':
        raise TD.Stop(4, "the study design's record on the Drive is of another S3_ARRAYS.json (nothing is overwritten)")
    m = Mirror7(conf, pairs_of(conf, ['d2828', 'logs']))
    rc = run_child(conf, 'pd17', [TD.PY, '-I', '-B', '-u', PD17, '--config-dir', TD.CFG, '--arrays-dir', TD.ARR, '--manifest-dir', MAN_VM, '--out', os.path.join(D2828_VM, PD17_NAME)], m)
    if rc == 4:
        raise TD.Stop(4, FAIL_LINE)
    if rc != 0:
        raise TD.Stop(4, 'pd17_pass.py ended with exit %d (its lines above): choose Run all again; if it repeats' % rc)
    return 0

def norm_list_ok(conf, d_dir):
    lp = os.path.join(d_dir, NORM_LIST)
    if not os.path.isfile(lp):
        return False
    lst = TD.jload(lp)
    for r in lst.get('files', []):
        n = os.path.basename(str(r['path']).replace('\\', '/'))
        p = os.path.join(d_dir, n)
        if not os.path.isfile(p) or os.path.getsize(p) != int(r['bytes']):
            return False
        if d_dir == D2828_VM and TD.sha256_file(p) != r['sha256']:
            return False
    return len(lst.get('files', [])) == 2 * len(STAGE1_NAMES) + 2

def step_norm_f6(conf):
    need_copy()
    need_pd17(conf)
    pl = places(conf)
    if norm_list_ok(conf, pl['d2828']):
        TD.log('NORM_F6: the rewrites are on the Drive (RUN/config/p2_rev/%s and its files by size); skipped' % NORM_LIST)
        return 0
    for p in own_vm_files():
        os.remove(p)
    names = [n + '.json' for n in STAGE1_NAMES]
    copy_small(TD.D_CFG, names, TD.CFG, 'stage (1) NORM file')
    copy_small(D_WORK, WORK_FILES, WORK_VM, 'work file')
    m = Mirror7(conf, pairs_of(conf, ['d2828', 'logs']))
    rc = run_child(conf, 'norm_f6', [TD.PY, '-I', '-B', '-u', NORMF6, '--config-dir', TD.CFG, '--work-dir', WORK_VM, '--arrays-dir', TD.ARR, '--result', os.path.join(CPU_LOGS, 'NORM_F6_RESULT_%s.json' % TD.boot12())], m)
    if rc != 0:
        raise TD.Stop(4, 'norm_f6.py ended with exit %d (its lines above): choose Run all again; if it repeats' % rc)
    lst = TD.jload(os.path.join(TD.CFG, NORM_LIST))
    os.makedirs(D2828_VM, exist_ok=True)
    for r in lst['files']:
        n = os.path.basename(str(r['path']).replace('\\', '/'))
        src = os.path.join(WORK_VM if n.startswith('norm_index') else TD.CFG, n)
        dst = os.path.join(D2828_VM, n)
        TD.copy_file(src, dst, False)
        if TD.sha256_file(dst) != r['sha256']:
            raise TD.Stop(4, "%s differs from %s's row" % (n, NORM_LIST))
    TD.copy_file(os.path.join(TD.CFG, NORM_LIST), os.path.join(D2828_VM, NORM_LIST), False)
    n, e = m.run()
    if m.refused or e:
        raise TD.Stop(4, 'the rewrites could not all be put on the Drive (%d errors, %d refused): choose Run all again; if it repeats' % (e, len(m.refused)))
    TD.log('NORM_F6: %d files staged and put on the Drive (RUN/config/p2_rev/)' % n)
    return 0

def ct_done_vm():
    from cpu import ctables as CT
    want = CT.want_of(TD.CFG)
    return [f for f in FOLDS if CT.index_ok(os.path.join(D2828_VM, 'CTABLES_%s.json' % f), CT_VM, want)]

def step_ctables(conf):
    need_copy()
    need_pd17(conf)
    m = Mirror7(conf, pairs_of(conf, ['d2828', 'ct', 'logs']))
    n = m.restore()
    TD.log('RESTORE: %d files of RUN/config/p2_rev/ and W_P2/ctables_rev/ to the VM (a fold whose index and tables verify is skipped)' % n)
    done = ct_done_vm()
    if len(done) == len(FOLDS):
        TD.log('CTABLES: every fold done before (indexes and tables verify by sha256); skipped')
        m.run()
        return 0
    rc = run_child(conf, 'ctables', [TD.PY, '-I', '-B', '-u', TD.SHIM, '--config', TD.CONFIG, CTABLES, '--config-dir', TD.CFG, '--arrays-dir', TD.ARR, '--arrays-out', CT_VM, '--index-out', D2828_VM, '--folds', ','.join(FOLDS)], m)
    if rc != 0:
        raise TD.Stop(4, 'ctables.py ended with exit %d (its lines above): choose Run all again (done folds are skipped); if it repeats' % rc)
    return 0

def names_of(conf):
    names = list(conf['stage9']['names'])
    if 'CLIM-EM' in names and (not conf['c7a_applies']['value']):
        names.remove('CLIM-EM')
        TD.log('CLIM-EM left out: C-7a does not apply (c7a_applies false)')
    return names

def step_baselines(conf):
    need_copy()
    need_pd17(conf)
    m = Mirror7(conf, pairs_of(conf, ['d2828', 'ct', 'base', 'lb', 'logs']))
    n = m.restore()
    TD.log('RESTORE: %d files (C tables, sealed predictions) to the VM (a part whose DONE file verifies is skipped)' % n)
    done = ct_done_vm()
    if len([f for f in done if f != 'f6']) != 5:
        raise TD.Stop(4, 'the C tables of folds %s are not done: run the ctables cell first' % sorted(set(FOLDS[:5]) - set(done)))
    names = names_of(conf)
    rc = run_child(conf, 'baselines', [TD.PY, '-I', '-B', '-u', TD.SHIM, '--config', TD.CONFIG, RUNP7, '--names', ','.join(names), '--config-dir', TD.CFG, '--work-dir', CPU_WORK, '--arrays-dir', TD.ARR, '--ctables-index', D2828_VM, '--ctables-arrays', CT_VM, '--log-dir', CPU_LOGS], m)
    if rc != 0:
        raise TD.Stop(4, 'baselines\\run_p7.py ended with exit %d (its lines above): choose Run all again (done parts are skipped); if it repeats' % rc)
    return 0

def drive_size_ok(vm_path, d_path):
    return os.path.isfile(d_path) and os.path.isfile(vm_path) and (os.path.getsize(d_path) == os.path.getsize(vm_path))

def done_rows(conf):
    pl = places(conf)
    rows, left = ([], [])

    def add(kind, vm_path, d_path, want=None, name=None):
        name = name or os.path.basename(vm_path)
        if not os.path.isfile(vm_path):
            left.append('%s %s (not on the VM)' % (kind, name))
            return
        sha = TD.sha256_file(vm_path)
        if want is not None and sha != want:
            left.append('%s %s (sha256 differs from its record)' % (kind, name))
            return
        if not drive_size_ok(vm_path, d_path):
            left.append('%s %s (not on the Drive)' % (kind, name))
            return
        rows.append((kind, name, os.path.getsize(vm_path), sha))
    st, _ = pd17_state(conf)
    if st != 'PASS':
        left.append('the read-only check (state %s)' % st)
    else:
        add('PD17', os.path.join(D2828_VM, PD17_NAME), pl['d2828'] + '/' + PD17_NAME)
    lp = os.path.join(D2828_VM, NORM_LIST)
    if os.path.isfile(lp):
        for r in TD.jload(lp)['files']:
            n = os.path.basename(str(r['path']).replace('\\', '/'))
            add('NORM', os.path.join(D2828_VM, n), pl['d2828'] + '/' + n, r['sha256'])
        add('NORMLIST', lp, pl['d2828'] + '/' + NORM_LIST)
    else:
        left.append('the NORM rewrites')
    for f in FOLDS:
        ip = os.path.join(D2828_VM, 'CTABLES_%s.json' % f)
        if not os.path.isfile(ip):
            left.append('CTABLES_%s.json' % f)
            continue
        add('CTABLES', ip, pl['d2828'] + '/' + os.path.basename(ip))
        for fit in TD.jload(ip)['fits'].values():
            for k in ('doy', 'r'):
                n = fit[k]['file']
                add('CT', os.path.join(CT_VM, n), pl['ct'] + '/' + n, fit[k]['sha256'])
    for name in names_of(conf):
        for y in YEARS:
            for kind, vm, dd, fn in (('CV', PRED_VM, pl['base'], 'DONE.json'), ('LOCKBOX', LB_VM, pl['lb'], 'LOCKBOX_DONE.json')):
                sub = '%s/fold%d' % (name, y)
                dp = os.path.join(vm, name, 'fold%d' % y, fn)
                if not os.path.isfile(dp):
                    left.append('%s %s %s' % (kind, name, y))
                    continue
                for r in TD.jload(dp)['files'].values():
                    b = os.path.basename(str(r['path']).replace('\\', '/'))
                    add(kind, os.path.join(vm, name, 'fold%d' % y, b), dd + '/' + sub + '/' + b, r['sha256'], sub + '/' + b)
                add(kind + '_DONE', dp, dd + '/' + sub + '/' + fn, name=sub + '/' + fn)
    return (rows, left)

def step_end(conf):
    m = Mirror7(conf, pairs_of(conf, ['d2828', 'ct', 'base', 'lb', 'logs']))
    n, e = m.run()
    TD.log('END: the last mirror %d files, %d errors' % (n, e))
    if m.refused or e:
        raise TD.Stop(4, 'the last mirror refused %d files and had %d errors (the lines above): CPU_DONE is not written; choose Run all again; if it repeats' % (len(m.refused), e))
    sealed = []
    for top in (PRED_VM, LB_VM):
        sealed += list(TD.walk(top))
    TD.log("SEALED: %d files under the VM's cpu/pred/baselines and cpu/lockbox/baselines (names and sizes only): %s" % (len(sealed), ', '.join(('%s %d' % (r, os.path.getsize(p)) for r, p in sealed[:12]))))
    ss = [s for s in TD.drive_sessions() if s.get('session') == TD.boot12()]
    if ss:
        TD.log('UNITS: this session %.3f h on %s' % (float(ss[0]['hours']), ss[0]['accelerator']))
    rows, left = done_rows(conf)
    pl = places(conf)
    if left:
        TD.log('LEFT: %d outputs not done or not on the Drive: %s' % (len(left), '; '.join(left[:12])))
        TD.log('CPU_DONE not written: choose Run all again; if it repeats')
        return 0
    if os.path.isfile(pl['done']):
        TD.log('CPU_DONE: EMUL/%s is there from before (not written again)' % conf['places']['cpu_done'])
        return 0
    pc = conf['places']
    lines = ['CPU DONE %s exit 0' % TD.utc(), 'notebook %s, package 6_cpu_baselines' % NOTEBOOK, 'Each file: FILE <kind> <name> <bytes>, then its sha256 line.', 'PLACE PD17 NORM NORMLIST CTABLES %s' % pc['cpu_config'], 'PLACE CT %s' % pc['ctables'], 'PLACE CV CV_DONE %s' % pc['baselines_sealed'], 'PLACE LOCKBOX LOCKBOX_DONE %s/baselines' % pc['lockbox_sealed']]
    for kind, name, size, sha in rows:
        lines.append('FILE %s %s %d' % (kind, name, size))
        lines.append('  sha256 %s' % sha)
    lines.append('END %d files' % len(rows))
    vm = os.path.join(TD.RUN, 'logs', 'CPU_DONE_ASCII.txt')
    with open(vm, 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    put(conf, vm, pl['done'])
    TD.log('CPU_DONE: EMUL/%s written (%d files listed, sha16 %s)' % (conf['places']['cpu_done'], len(rows), TD.sha256_file(vm)[:16]))
    return 0
STEPS = {'session': lambda conf: 0, 'gates': step_gates, 'copy': step_copy, 'pd17': step_pd17, 'norm_f6': step_norm_f6, 'ctables': step_ctables, 'baselines': step_baselines, 'end': step_end}

def on_term7(s, f):
    m = TD.MIRROR[0]
    if m is not None and m.lock.locked():
        TD.MIRROR[0] = None
    TD.on_term(s, f)

def main(argv):
    if not argv or argv[0] not in STEPS:
        print('usage: cpu_colab_driver.py %s' % '|'.join(STEPS))
        return 1
    if not os.path.isdir(TD.EMUL):
        print('THE DRIVE IS NOT MOUNTED: run cell 1 first')
        return 9
    TD.LOG_NAME[0] = 'CPU_DRIVER'
    TD.LOG_FMT = CPU_LOGS + '/%s_%s_ASCII.txt'
    TD.LOG_DRIVE[0] = TD.D_TRAIN + '/logs/cpu'
    os.makedirs(CPU_LOGS, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term7)
    TD.install_overrides()
    conf, sha = R.load(TD.CONFIG)
    conf = R.attach_units_other(conf, TD.read_units_other(conf))
    TD.session_update(conf, NOTEBOOK, argv[0])
    try:
        if argv[0] not in ('session', 'gates', 'end'):
            bad = R.refusals(conf, 'stage9')
            for b in bad:
                TD.log('REFUSED: ' + b)
            if bad:
                raise TD.Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))
            TD.rate_stop(conf)
            cpu_check(conf)
            cap_check(conf)
            if argv[0] in ('norm_f6', 'ctables', 'baselines'):
                need_sklearn_test()
        if argv[0] == 'gates':
            return step_gates(conf, sha)
        return STEPS[argv[0]](conf)
    except TD.Stop as s:
        TD.log('STOP (exit %d): %s' % (s.code, s))
        return s.code
    except Overwrite as o:
        TD.log('STOP (exit 4): the Drive already holds EMUL%s with other bytes; nothing is overwritten' % str(o)[len(TD.EMUL):])
        return 4
    finally:
        TD.push_log()
if __name__ == '__main__':
    try:
        RC = main(sys.argv[1:])
    except SystemExit:
        raise
    except BaseException:
        import traceback
        for ln in traceback.format_exc().splitlines()[-12:]:
            print(TD.common.mask(ln), flush=True)
        RC = 1
    sys.exit(RC)
