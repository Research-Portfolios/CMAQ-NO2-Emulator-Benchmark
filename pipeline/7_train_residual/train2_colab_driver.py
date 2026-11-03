"""Driver of the residual and other design trainings on Colab A100 runtimes in two lanes: loads the shared package, overlays this folder's files, runs each lane's trainings through the trainer, writes the held-back-day predictions, records each runtime's spend, and mirrors outputs to the Drive."""
import calendar
import fcntl
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
NOTEBOOK = 'EMUL_08_TIER_A_TRAIN2'
PKG = TD.PKG
CONFIG = TD.CONFIG
CORE = TD.CORE
TRAINER = TD.TRAINER
PY = TD.PY
Stop = TD.Stop
TA_WORK = TD.TA_WORK
TA2 = TD.RUN + '/logs/tier_a2'
CT_VM = TD.ARR + '/ctables'
LB_VM = TA_WORK + '/lockbox'
D_TRAIN = TD.D_TRAIN
D_TA2 = TD.D_TRAIN + '/logs/tier_a2'
D_SESS = TD.D_TRAIN + '/logs/sessions'
D_SESS8 = TD.D_TRAIN + '/logs/sessions_tier_a2'
D_CPUCFG = TD.EMUL + '/RUN/config/p2_rev'
D_CT = TD.EMUL + '/W_P2/ctables_rev'
D_LB = TD.EMUL + '/W_P2/results_sealed/TIER_A/lockbox'
D_CPU_DONE = TD.EMUL + '/RUN/logs/CPU_DONE_ASCII.txt'
D_CHOICE = TD.D_TRAIN + '/TIMING/CHOICE.json'
LOCK_FRESH_S = 600.0
LOCK_LOST_RC = 10
RERUN_SUFFIX = '_clip1'
CHILD_ENV = {'NVIDIA_TF32_OVERRIDE': '0'}
CPU_KINDS_SHA = ('NORM', 'NORMLIST', 'CTABLES')
CPU_KINDS_SIZE = ('CT',)
GUARD_LINE = R.GUARD_LINE
TRAINER_GUARD_RC = 5
LANE_LINE = "this runtime has no lane (cell 1 of EMUL_08_TIER_A_TRAIN2_LANE1.ipynb or EMUL_08_TIER_A_TRAIN2_LANE2.ipynb writes it): choose Run all in the lane's own notebook file"
WAIT8_LINE = "the first training stage's trainings %s are not done yet: their lockbox files wait; Run all this lane again after the first training stage ends"
T2_NAME = 'fold2019_MC7'
LANE = [None]
TA2L = None
D_TA2L = None
MARKER2 = None
EQ2 = None
TM2 = None
PLAN2 = None
BUDGET_STATE = None
D_LOCK = None
LOCK_VM = None
SESS_VM = None
LANE_FILE = None
LEDGER_CACHE = None

def set_lane(k):
    global TA2L, D_TA2L, MARKER2, EQ2, TM2, PLAN2, BUDGET_STATE, D_LOCK
    global LOCK_VM, SESS_VM, LANE_FILE, LEDGER_CACHE
    k = str(k)
    if k not in R.LANES:
        raise Stop(3, LANE_LINE)
    LANE[0] = k
    TA2L = TA2 + '/lane' + k
    D_TA2L = D_TA2 + '/lane' + k
    MARKER2 = TA2L + '/TIER_A2_MARKER_ASCII.txt'
    EQ2 = TA2L + '/equiv'
    TM2 = TA2L + '/timing'
    PLAN2 = TA2L + '/plan'
    BUDGET_STATE = TA2L + '/BUDGET_STATE.json'
    D_LOCK = TD.D_TRAIN + '/logs/TRAIN_LOCK_LANE%s.json' % k
    LOCK_VM = TA2 + '/locks/TRAIN_LOCK_LANE%s.json' % k
    SESS_VM = TA2 + '/sessions'
    LANE_FILE = TA2 + '/LANE.json'
    LEDGER_CACHE = TA2 + '/LEDGER_CACHE.json'
    TD.SESS_DIR = SESS_VM
    TD.LOG_FMT = TA2L + '/driver/%s_%s_ASCII.txt'
    TD.LOG_NAME[0] = 'TRAIN2_DRIVER'
    TD.LOG_DRIVE[0] = D_TA2L + '/driver'
    TD.UNITS = TA2L + '/UNITS_ASCII.txt'
    return k

def read_lane():
    try:
        d = TD.jload(TA2 + '/LANE.json')
    except (OSError, ValueError):
        raise Stop(3, LANE_LINE)
    k = str((d or {}).get('lane', '')).strip()
    if k not in R.LANES:
        raise Stop(3, LANE_LINE)
    return k
set_lane('1')

def boot12():
    return TD.boot12()

def jload(p):
    return TD.jload(p)

def jsave(p, d):
    return TD.jsave(p, d)

def copy_file(src, dst, to_drive):
    return TD.copy_file(src, dst, to_drive)

def utc():
    return TD.utc()
TD_LOG = TD.log

def log8(text):
    TD_LOG('lane %s: %s' % (LANE[0] or '-', text))

def log(text):
    TD.log(text)

def session_update8(conf, notebook, step):
    sid = TD.boot12()
    up = TD.uptime_s()
    p = os.path.join(SESS_VM, sid + '.json')
    d = TD.jload(p) if os.path.isfile(p) else {}
    acc, name = TD.accelerator(conf)
    steps = sorted(set(d.get('steps', [])) | {step})
    d.update({'schema': 'tier-a-session-1', 'session': sid, 'notebook': d.get('notebook', notebook), 'vm_started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - up)), 'updated_utc': TD.utc(), 'hours': round(up / 3600.0, 4), 'accelerator': acc, 'device_name': name, 'steps': steps})
    tp = TD.TYPED_FMT % sid
    if os.path.isfile(tp):
        try:
            t = TD.jload(tp)
            v = t.get('units_per_hour')
            d['units_per_hour_typed'] = float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v > 0) else None
            d['units_per_hour_typed_utc'] = t.get('typed_utc')
            d['runtime_typed'] = t.get('runtime')
        except (OSError, ValueError):
            d['units_per_hour_typed'] = None
            d['runtime_typed'] = None
    d['lane'] = LANE[0]
    d['lanes'] = sorted(set(d.get('lanes') or []) | ({LANE[0]} if LANE[0] else set()))
    TD.jsave(p, d)
    TD.copy_file(p, D_SESS8 + '/' + sid + '.json', True)
    rebuild_units8(conf)
    return d

def this_session8():
    try:
        return TD.jload(os.path.join(SESS_VM, TD.boot12() + '.json'))
    except (OSError, ValueError):
        return {}

def _ledger_cache():
    try:
        d = TD.jload(LEDGER_CACHE)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}

def drive_sessions8():
    cache = _ledger_cache()
    changed = False
    recs = {}
    for folder, tag in ((D_SESS, 'sessions'), (D_SESS8, 'sessions_tier_a2')):
        for p in sorted(glob.glob(folder + '/*.json')):
            b = os.path.basename(p)
            key = tag + '/' + b
            try:
                d = TD.jload(p)
                if not isinstance(d, dict):
                    raise ValueError('not a record')
                if cache.get(key) != d:
                    cache[key] = d
                    changed = True
                d = dict(d, ledger_folder=tag)
            except (OSError, ValueError):
                if isinstance(cache.get(key), dict):
                    d = dict(cache[key], ledger_folder=tag, unreadable=True)
                else:
                    d = {'session': b, 'accelerator': 'unreadable record', 'hours': 0, 'unreadable': True, 'never_read': True, 'ledger_folder': tag}
            old = recs.get(b)
            if old is None or (old.get('never_read') and (not d.get('never_read'))) or (not d.get('never_read') and float(d.get('hours') or 0) > float(old.get('hours') or 0)):
                recs[b] = d
    if changed:
        try:
            TD.jsave(LEDGER_CACHE, cache)
        except OSError:
            pass
    return [recs[k] for k in sorted(recs)]

def rebuild_units8(conf):
    from choi_dl import common
    ss = sorted(drive_sessions8(), key=lambda s: str(s.get('vm_started_utc', '')))
    lines = ["UNITS OF THE BENCHMARK ON COLAB (the training, CPU and evaluation stages), rebuilt by train2_colab_driver.py (the training stage, lane %s) at every step from EMUL/TRAIN/logs/sessions/ and EMUL/TRAIN/logs/sessions_tier_a2/ (a session in both counted once, at the larger hours; hours from the VM's /proc/uptime; units = hours x the config's rate for that session's runtime, one without a rate at %g)." % (LANE[0], R.missing_rate(conf))]
    for s in ss:
        r, note = R.guard_session_rate(conf, s, ss)
        u = '%.3f' % (float(s.get('hours') or 0) * r) if r is not None else 'not counted'
        lines.append('%s %s/%s %s %s: %.3f h x %s = %s units%s; updated %s' % (s.get('vm_started_utc'), s.get('ledger_folder'), s.get('session'), s.get('notebook'), s.get('accelerator'), float(s.get('hours') or 0), r if r is not None else '-', u, ' ' + note if note else '', s.get('updated_utc')))
    spent, _rows, never = R.guard_spent(conf, ss)
    other = R.units_other_total(conf)
    lines.append("the benchmark's own spend: %.3f units of the %g-unit runaway guard%s; stages 1 to 5 (the accounting file, shown, not counted): %s" % (spent, float(conf['guard']['units']), ' (never read, a failed guard period: %s)' % ', '.join(never) if never else '', 'not stated' if other is None else '%.3f' % other))
    os.makedirs(os.path.dirname(TD.UNITS), exist_ok=True)
    with open(TD.UNITS + '.partial', 'w', encoding='ascii', newline='\n') as f:
        for t in lines:
            f.write('\n'.join(common.wrap('', t)) + '\n')
    os.replace(TD.UNITS + '.partial', TD.UNITS)
    TD.copy_file(TD.UNITS, D_TA2L + '/UNITS_ASCII.txt', True)
TD.NOTEBOOK = NOTEBOOK
TD.session_update = session_update8
TD.drive_sessions = drive_sessions8
TD.this_session = this_session8
TD.rebuild_units = rebuild_units8
TD.log = log8

def plan_mod():
    from train import plan as P
    return P

def design_rows(conf):
    P = plan_mod()
    return P.design_trainings(conf['c7a_applies']['value'], conf['c_channel'])

def base_names():
    P = plan_mod()
    return set((t['name'] for t in P.all_trainings(True))) | set((t['name'] for t in P.all_trainings(False)))

def lane_reruns():
    for p in (TA2L + '/LANE_STATE.json', D_TA2L + '/LANE_STATE.json'):
        try:
            return [str(n) for n in TD.jload(p).get('reruns') or []]
        except (OSError, ValueError, AttributeError):
            continue
    return []

def lane_trainings(conf):
    rr = set(lane_reruns())
    out = []
    for n in R.lane_names(conf, LANE[0]):
        out.append({'name': n})
        if n in rr:
            out.append({'name': n + RERUN_SUFFIX})
    return out

def lane_own(conf):
    names = R.lane_names(conf, LANE[0])
    return set(names) | set((n + RERUN_SUFFIX for n in names))

def lane_design_rows(conf, extra=()):
    by = dict(((t['name'], t) for t in design_rows(conf)))
    names = R.lane_names(conf, LANE[0]) + [n for n in extra]
    return [by[n] for n in names if n in by]

class Overwrite(Exception):
    pass

def _pred_top(conf):
    return TD.EMUL + '/' + conf['places']['pred_sealed' if conf['sealed'] else 'pred_open']

def drive_pairs(conf):
    own = lane_own(conf)
    lb = own | set(R.lane_early(conf, LANE[0]))

    def mine(r):
        return r.split('/', 1)[0] in own

    def mine_lb(r):
        return r.split('/', 1)[0] in lb
    return [(TA_WORK + '/ckpt', D_TRAIN + '/ckpt', mine), (TA_WORK + '/pred', _pred_top(conf), mine), (LB_VM, D_LB, mine_lb), (TA2L, D_TA2L, None)]

def restore_pairs(conf, extra=()):
    names = lane_own(conf) | set(extra)

    def mine(r):
        return r.split('/', 1)[0] in names
    return [(TA_WORK + '/ckpt', D_TRAIN + '/ckpt', mine), (TA_WORK + '/pred', _pred_top(conf), mine), (LB_VM, D_LB, mine), (TA2L, D_TA2L, None)]
MARKERS = ('DONE.json', 'LOCKBOX_DONE.json')

def is_marker(name):
    return name in MARKERS

def marker_of(conf, dst):
    pl = drive_pairs(conf)
    b = os.path.basename(dst)
    if is_marker(b):
        return None
    ck, pr, lb = (pl[0][1], pl[1][1], pl[2][1])
    for top, mk in ((ck, None), (pr, None), (lb, 'LOCKBOX_DONE.json')):
        if dst.startswith(top + '/'):
            name = dst[len(top) + 1:].split('/', 1)[0]
            if mk:
                return top + '/' + name + '/' + mk
            return ck + '/' + name + '/DONE.json'
    return None

def ok_place(conf, dst):
    tops = [d for _l, d, _t in drive_pairs(conf)]
    if '..' in dst or not any((dst.startswith(t + '/') for t in tops)):
        raise Stop(1, "a Drive write outside this package's places was refused: %s" % dst.replace(TD.EMUL, 'EMUL'))
    return dst

def put(conf, src, dst, replace_ok=False):
    ok_place(conf, dst)
    if os.path.isfile(dst):
        if os.path.getsize(dst) == os.path.getsize(src) and TD.sha256_file(dst) == TD.sha256_file(src):
            return 'same'
        if not replace_ok:
            mk = marker_of(conf, dst)
            if mk is None or os.path.isfile(mk):
                raise Overwrite(dst)
            if dst not in REPLACED:
                REPLACED.add(dst)
                log('REPLACED (incomplete group): EMUL%s; its marker EMUL%s is absent' % (dst[len(TD.EMUL):], mk[len(TD.EMUL):]))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)
    return 'copied'
REPLACED = set()

def drive_of(conf, vm_path):
    for loc, dd, _t in drive_pairs(conf):
        if vm_path.startswith(loc + '/'):
            return dd + vm_path[len(loc):]
    return None

def group_on_drive(conf, vm_marker):
    d = TD.jload(vm_marker)
    rows = list((d.get('files') or {}).values())
    if d.get('best_pt'):
        rows.append(d['best_pt'])
    for r in rows:
        p = str(r['path'])
        vm = p if p.startswith('/') else TD.ROOT + '/' + p
        dp = drive_of(conf, vm)
        if dp is None or not os.path.isfile(dp) or os.path.getsize(dp) != int(r['bytes']):
            return False
    return True

class Mirror8:

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
                        whole = group_on_drive(self.conf, p)
                    except (OSError, ValueError, KeyError):
                        whole = False
                    if not whole:
                        err += 1
                        continue
                try:
                    put(self.conf, p, dd + '/' + r, replace_ok=dd == D_TA2L)
                except Overwrite:
                    err += 1
                    if p not in self.refused:
                        self.refused.append(p)
                        log('MIRROR REFUSED: EMUL%s holds other bytes; nothing is overwritten' % (dd + '/' + r)[len(TD.EMUL):])
                    self.seen[p] = s
                    continue
                except OSError as e:
                    err += 1
                    log('MIRROR ERROR %s: %s' % (os.path.basename(p), str(e)[:120]))
                    continue
                self.seen[p] = s
                n += 1
        return (n, err)

    def restore(self, skip=None, pairs=None):
        todo = []
        for loc, dd, test in pairs or self.pairs:
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
MIRROR8 = [None]

def cpu_done_rows(text):
    lines = text.split('\n')
    rows, place = ([], {})
    cur = None
    for ln in lines[1:]:
        if ln.startswith('PLACE '):
            parts = ln.split()
            for k in parts[1:-1]:
                place[k] = parts[-1]
        elif ln.startswith('FILE '):
            parts = ln.split()
            cur = [parts[1], parts[2], int(parts[3]), None]
            rows.append(cur)
        elif ln.startswith('  sha256 ') and cur is not None:
            cur[3] = ln.split()[1]
    return (lines[0] if lines else '', [tuple(r) for r in rows], place)

def cpu_done_drive_path(kind, name, place):
    top = place.get(kind)
    if top is None:
        return None
    return TD.EMUL + '/' + top + '/' + name

def gate_cpu_done():
    if not os.path.isfile(D_CPU_DONE):
        raise Stop(2, 'EMUL/RUN/logs/CPU_DONE_ASCII.txt is absent: the GBT stage has not finished: run it first')
    with open(D_CPU_DONE, 'rb') as f:
        first, rows, place = cpu_done_rows(f.read().decode('ascii', 'replace'))
    if not (first.startswith('CPU DONE') and 'exit 0' in first):
        raise Stop(2, "CPU_DONE_ASCII.txt's first line is not 'CPU DONE ... exit 0'")
    want = [r for r in rows if r[0] in CPU_KINDS_SHA + CPU_KINDS_SIZE]
    kinds = set((r[0] for r in want))
    if not {'NORM', 'NORMLIST', 'CTABLES', 'CT'} <= kinds:
        raise Stop(4, 'CPU_DONE lists no %s' % sorted({'NORM', 'NORMLIST', 'CTABLES', 'CT'} - kinds))
    bad = []
    for kind, name, size, sha in want:
        p = cpu_done_drive_path(kind, name, place)
        if p is None or not os.path.isfile(p):
            bad.append('%s %s absent' % (kind, name))
        elif os.path.getsize(p) != size:
            bad.append('%s %s size' % (kind, name))
        elif kind in CPU_KINDS_SHA and TD.sha256_file(p) != sha:
            bad.append('%s %s sha256' % (kind, name))
    if bad:
        raise Stop(4, "the GBT stage's outputs differ from CPU_DONE: %s" % '; '.join(bad[:6]))
    log('GATE CPU_DONE: %s; %d C tables, %d indexes, %d NORM rewrites and their list present (the JSON by sha256, the tables by size)' % (first[:60], sum((1 for r in want if r[0] == 'CT')), sum((1 for r in want if r[0] == 'CTABLES')), sum((1 for r in want if r[0] == 'NORM'))))
    return (want, place)

def gate_early8(conf):
    P = plan_mod()
    pred = TD.EMUL + '/' + conf['places']['pred_sealed']
    bad = []
    for n in P.EARLY_8:
        p = TD.D_TRAIN + '/ckpt/%s/DONE.json' % n
        try:
            d = TD.jload(p)
        except (OSError, ValueError):
            bad.append('%s: no DONE.json' % n)
            continue
        pid = d.get('plan_id') or {}
        if d.get('status') != 'done' or d.get('training') != n or float(pid.get('width', 0)) != 1.0 or (int(pid.get('steps', 0)) != 6000):
            bad.append("%s: DONE.json not of the 8's plan" % n)
            continue
        rows = list((d.get('files') or {}).values()) + [d.get('best_pt') or {}]
        for r in rows:
            rp = str(r.get('path', ''))
            b = os.path.basename(rp)
            dp = TD.D_TRAIN + '/ckpt/%s/%s' % (n, b) if b.endswith('.pt') else pred + '/%s/%s' % (n, b)
            if not os.path.isfile(dp) or os.path.getsize(dp) != int(r.get('bytes', -1)):
                bad.append('%s: %s' % (n, b))
    if bad:
        raise Stop(2, "the first training stage's 8 early trainings are not all done: %s; run the first training stage first" % '; '.join(bad[:6]))
    log("GATE EARLY 8: the DONE markers of the first training stage's 8 trainings verified on the Drive (status, name, width 1, 6,000 steps, every listed file's size)")

def choice6(conf):
    try:
        d = TD.jload(D_CHOICE)
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict) or d.get('rule_sha256') != R.rule_sha256(conf):
        return {}
    return d

def refused6(conf, ch=None):
    ch = choice6(conf) if ch is None else ch
    out = {}
    for g in R.candidates(conf):
        k = g['key']
        row = (ch.get('rows') or {}).get(k) or {}
        if row.get('refused'):
            out[k] = "the first training stage's CHOICE.json: " + '; '.join((str(x) for x in row['refused']))
            continue
        p = TD.D_TRAIN + '/TIMING/%s.json' % k
        try:
            tm = TD.jload(p) if os.path.isfile(p) else None
        except (OSError, ValueError):
            tm = None
        why = R.type_refusal(conf, tm, None) if tm else None
        if why:
            out[k] = "the first training stage's timing: " + why
    return out

def refused8(conf):
    out = {}
    for g in R.candidates(conf):
        k = g['key']
        eq = equiv8(conf, k)
        if eq is not None and (not eq.get('pass')):
            out[k] = "this package's equivalence test did not pass"
            continue
        tm = timing8(k)
        why = R._peak_refusal(conf, tm) if tm else None
        if why:
            out[k] = "this package's timing: " + why
    return out

def change_stop(conf, key, why):
    refused = set(refused6(conf)) | set(refused8(conf)) | {key}
    line = R.change_line(conf, refused)
    raise Stop(6 if line.startswith('no candidate') else 5, 'this GPU type (%s) is refused (%s): %s' % (key, why, line))

def type_stop8(conf):
    key, name = TD.accelerator(conf)
    if key not in [g['key'] for g in R.candidates(conf)]:
        line = R.change_line(conf, set(refused6(conf)) | set(refused8(conf)))
        raise Stop(6 if line.startswith('no candidate') else 5, 'this runtime (%s) is not a candidate of the GPU rule (the H100, else the A100): %s' % (name, line))
    return (key, name)

def gate_gpu8(conf):
    key, name = type_stop8(conf)
    ch = choice6(conf)
    if not ch or not ch.get('eligible'):
        raise Stop(2, 'the first training stage has not finished its tests (EMUL/TRAIN/TIMING/CHOICE.json is absent, of another rule, or names no eligible GPU type): run it first')
    r6 = refused6(conf, ch)
    if key in r6:
        change_stop(conf, key, r6[key])
    eq = equiv8(conf, key)
    if eq is not None and (not eq.get('pass')):
        raise Stop(4, "the equivalence test refuses this package's emul_core.py on %s in lane %s" % (key, LANE[0]))
    r8 = refused8(conf)
    if key in r8:
        change_stop(conf, key, r8[key])
    log("GATE GPU: %s (%s), a candidate; the first training stage's eligible types %s%s" % (key, name, ', '.join(ch['eligible']), '' if key in ch['eligible'] else "this type is not yet among them: this package's own equivalence test and timing decide"))
    return key

def step_gates(conf, sha):
    type_stop8(conf)
    for p, head in TD.D_MARKS:
        first = ''
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                first = f.read().decode('ascii', 'replace').split('\n')[0]
        if not (first.startswith(head) and 'exit 0' in first):
            raise Stop(2, "%s is absent or not '%s ... exit 0': run its notebook first" % (p.replace(TD.EMUL, 'EMUL'), head))
        log('GATE %s: %s' % (os.path.basename(p), first[:120]))
    bad = R.refusals(conf, 'train')
    P = plan_mod()
    if not bad:
        bad += R.p8_data_errors(conf, P.DROP_ORDER, P.NEVER_DROPPED)
        bad += R.lanes_errors(conf, P.design_lanes(design_rows(conf), P.EARLY_8))
    if conf.get('design') != P.DESIGN:
        bad.append('design %r is not %s' % (conf.get('design'), P.DESIGN))
    for b in bad:
        log('REFUSED: ' + b)
    if bad:
        raise Stop(3, 'the config refuses (%d reasons above)' % len(bad))
    refuse_other_lock()
    gate_cpu_done()
    gate_early8(conf)
    key = gate_gpu8(conf)
    log("CONFIG %s: design %s (%s), decided by %s; sealed %s; lockbox %s; c7a_applies %s; c_channel %s; S_M %d; lane %s: %d trainings, the first training stage's %s; this runtime %s at %s units per hour (the config's); the guard %g units of the benchmark's own spend; stages 1 to 5 (shown, not counted): %.2f units" % (sha[:16], conf['design'], conf['design_sha256'][:16], conf['decided_by'], conf['sealed'], conf['lockbox'].get('decided_by'), conf['c7a_applies']['value'], conf['c_channel'], R.steps_sm_value(conf), LANE[0], len(R.lane_names(conf, LANE[0])), ', '.join(R.lane_early(conf, LANE[0])), key, R.rate_of(conf, key), float(conf['guard']['units']), R.units_other_total(conf) or 0.0))
    return 0

def copy_configs8():
    os.makedirs(TD.CFG, exist_ok=True)
    names = {}
    for top in (TD.D_CFG, D_CPUCFG):
        for p in sorted(glob.glob(top + '/*.json')):
            b = os.path.basename(p)
            if '_Backup_PriorTo' in b or b.startswith('PLAN_RUNTIME'):
                continue
            TD.copy_file(p, TD.CFG + '/' + b, False)
            if os.path.getsize(TD.CFG + '/' + b) != os.path.getsize(p):
                raise Stop(4, "config %s: the VM copy's size differs from the Drive's: choose Run all again" % b)
            names[b] = top
    if 'S3_ARRAYS.json' not in names:
        raise Stop(4, 'EMUL/RUN/config/p2/S3_ARRAYS.json is absent')
    rows = TD.list_rows()
    rec, bad = ([], [])
    for b in sorted(names):
        size = os.path.getsize(TD.CFG + '/' + b)
        row = rows.get(b)
        if row and row[0] != size:
            bad.append('%s %d bytes, %s %d' % (b, size, row[1], row[0]))
        rec.append({'name': b, 'bytes': size, 'from': names[b].replace(TD.EMUL, 'EMUL'), 'sha256': TD.sha256_file(TD.CFG + '/' + b), 'listed_in': row[1] if row else None})
    cp = TA2L + '/COPY_CONFIGS_%s.json' % TD.boot12()
    TD.jsave(cp, {'made_utc': utc(), 'configs': rec, 'rule': "EMUL/RUN/config/p2 then p2_rev over it; size equal to the Drive's and to the newest list row naming it "})
    if bad:
        raise Stop(4, 'configs differ from their list rows: %s' % '; '.join(bad[:6]))
    log("CONFIGS: %d copied to the VM's config\\p2 (%d from p2_rev)" % (len(names), sum((1 for v in names.values() if v == D_CPUCFG))))
    return names

def copy_ctables(want, place):
    os.makedirs(CT_VM, exist_ok=True)
    n = 0
    for kind, name, size, sha in want:
        if kind != 'CT':
            continue
        dst = CT_VM + '/' + name
        if not (os.path.isfile(dst) and os.path.getsize(dst) == size and (TD.sha256_file(dst) == sha)):
            TD.copy_file(cpu_done_drive_path(kind, name, place), dst, False)
            n += 1
    bad, backups = ([], 0)
    for kind, name, size, sha in want:
        if kind != 'CT' and '_Backup_PriorTo' in name:
            backups += 1
            continue
        vm = (CT_VM if kind == 'CT' else TD.CFG) + '/' + name
        if not os.path.isfile(vm) or TD.sha256_file(vm) != sha:
            bad.append('%s %s' % (kind, name))
    if bad:
        raise Stop(4, "the VM copies of the GBT stage's outputs differ from CPU_DONE's sha256: %s; choose Run all again; if it repeats" % '; '.join(bad[:6]))
    log('C TABLES: %d copied now to W/arrays_p2/ctables; %d files of CPU_DONE tested by sha256 on the VM; %d backups (_Backup_PriorTo) left out, never copied to the VM and tested on the Drive by the gate' % (n, len(want) - backups, backups))

def coverage(conf):
    import numpy as np
    from train import cfg as tcfg
    from train import data
    paths = tcfg.Paths(TD.CFG, TA_WORK, TA2L, MARKER2, TD.ARR)
    configs = tcfg.Configs(paths)
    mask = data.load_mask(configs, paths)
    rec, maps = data.coverage_record(mask)
    n_low, wmin = data.blend_coverage(mask)
    os.makedirs(TA2L, exist_ok=True)
    files = {}
    for k, a in sorted(maps.items()):
        p = TA2L + '/COVERAGE_MAP_%s.npy' % k
        np.save(p, a)
        files[k] = {'file': os.path.basename(p), 'sha256': TD.sha256_file(p), 'shape': list(a.shape), 'dtype': 'int32'}
    TD.jsave(TA2L + '/COVERAGE_RECORD.json', {'schema': 'p8-coverage-1', 'made_utc': utc(), 'mask_sha256': configs.channels['mask']['sha256'], 'rules': rec, 'maps': files, 'blend': {'masked_cells_at_or_below_floor': n_low, 'masked_weight_sum_min': wmin, 'floor': 0.001}, 'source': "the mask alone (the study design's assertion)"})
    log('COVERAGE: %s; blend: %d masked cells at or below the floor' % ('; '.join(('%s %d positions, %d masked cells never covered' % (k, v['valid_positions'], v['masked_cells_never_covered']) for k, v in sorted(rec.items()))), n_low))
    if n_low:
        raise Stop(4, "the study design's blend coverage assertion fails: %d masked cells have a summed blend weight at or below the 1e-3 floor (the line above): nothing is trained" % n_low)

def split_stop(conf, key):
    ts = lane_trainings(conf)
    states = training_states(ts)
    sp = R.type_split(conf, key, states, [t['name'] for t in ts])
    for n2, g2 in sorted(sp['elsewhere'].items()):
        log('ONE TYPE: %s was begun on %s; it resumes only there and does not run on %s' % (n2, g2 or 'a GPU with no record', key))
    if sp['elsewhere'] and (not sp['left']):
        elsewhere_stop(conf, sp['elsewhere'])
    return (sp, states)
L4_FILE = 'S3_L4_ARM18.json'

def copy_l4():
    p = TD.CFG + '/' + L4_FILE
    if not os.path.isfile(p):
        return 0
    rows = sorted((TD.jload(p).get('files') or {}).items())
    n, bad = (0, [])
    for _y, r in rows:
        name = str(r['path']).replace('\\', '/').rsplit('/', 1)[-1]
        src, dst = (TD.D_ARR + '/' + name, TD.ARR + '/' + name)
        size = int(r['bytes'])
        if not os.path.isfile(src):
            bad.append('%s absent on the Drive' % name)
            continue
        if os.path.getsize(src) != size:
            bad.append('%s: Drive %d bytes, %s %d' % (name, os.path.getsize(src), L4_FILE, size))
            continue
        if not (os.path.isfile(dst) and os.path.getsize(dst) == size and (TD.sha256_file(dst) == r['sha256'])):
            TD.copy_file(src, dst, False)
            n += 1
            if os.path.getsize(dst) != size or TD.sha256_file(dst) != r['sha256']:
                bad.append("%s: the VM copy's sha256" % name)
    if bad:
        raise Stop(4, "L-4's arrays differ from %s: %s; choose Run all again; if it repeats" % (L4_FILE, '; '.join(bad[:6])))
    log('L-4: %d arrays of %s on the VM, each by size and sha256 (%d copied now)' % (len(rows), L4_FILE, n))
    return n

def step_copy(conf):
    want, place = gate_cpu_done()
    copy_configs8()
    TD.lb1_check(conf)
    key, _name = type_stop8(conf)
    split_stop(conf, key)
    copy_ctables(want, place)
    TD.copy_arrays()
    copy_l4()
    coverage(conf)
    return 0

def core16():
    return TD.sha256_file(CORE)[:16]

def stream(cmd, out):
    return TD.stream(cmd, out, env=dict(CHILD_ENV))

def step_equiv(conf):
    key = gate_gpu8(conf)
    _k, name, env = TD.gpu_key(conf)
    c16 = core16()
    vm = os.path.join(EQ2, key, 'core_' + c16)
    res_p = os.path.join(vm, 'results.json')
    rec_p = os.path.join(vm, 'EQUIV_RECORD.json')
    m = Mirror8(conf, [(os.path.join(EQ2, key), D_TA2L + '/equiv/' + key, None)])
    m.restore()
    res = TD.jload(res_p) if os.path.isfile(res_p) else {}
    if res.get('finished_utc') and os.path.isfile(rec_p):
        log('EQUIV %s core %s: finished in an earlier session (%s)' % (key, c16, res['finished_utc']))
    else:
        rc = stream([PY, '-I', '-B', '-u', TD.SHIM, '--config', CONFIG, TD.EQUIV, '--device', 'cuda', '--out', vm, '--config-dir', TD.CFG, '--ckpt-dir', os.path.join(TA_WORK, 'tmp', 'equiv2_ckpt', key, c16)], os.path.join(vm, 'console_%s.txt' % TD.boot12()))
        res = TD.jload(res_p) if os.path.isfile(res_p) else {}
        if rc != 0 or not res.get('finished_utc'):
            raise Stop(4, "the equivalence test of this package's emul_core.py did not finish on %s (exit %d): choose Run all again; if it repeats" % (key, rc))
        P = plan_mod()
        probe = res.get('precision_probe') or {}
        try:
            rec = P.equivalence_from_results(res_p)
        except P.PlanError:
            rec = None
        ok = bool(probe.get('pass')) and rec is not None and (rec['nondeterministic_count'] == 0)
        TD.jsave(rec_p, {'schema': 'tier-a-equiv-1', 'gpu': key, 'device_name': name, 'pass': ok, 'record': rec, 'probe': {k: probe.get(k) for k in ('pass', 'matmul_rel_err', 'conv_rel_err', 'tol')}, 'emul_core_sha256': TD.sha256_file(CORE), 'five': TD.five_of(res.get('env') or {}), 'runtime_tag': env.get('runtime_tag'), 'session': TD.boot12(), 'made_utc': utc(), 'lane': LANE[0], 'package': '7_train_residual'})
    m.run()
    out = TD.jload(rec_p)
    log("EQUIV %s: this package's emul_core.py %s: %s (nondeterministic_count %s)" % (key, c16, 'PASS' if out.get('pass') else 'FAIL', (out.get('record') or {}).get('nondeterministic_count')))
    if not out.get('pass'):
        raise Stop(4, "the equivalence test refuses this package's emul_core.py on %s in lane %s" % (key, LANE[0]))
    return 0

def equiv8(conf, key):
    p = os.path.join(EQ2, key, 'core_' + core16(), 'EQUIV_RECORD.json')
    dp = D_TA2L + '/equiv/%s/core_%s/EQUIV_RECORD.json' % (key, core16())
    for x in (p, dp):
        if os.path.isfile(x):
            try:
                return TD.jload(x)
            except (OSError, ValueError):
                continue
    return None

def timing8_path(key):
    return os.path.join(TM2, '%s_core_%s.json' % (key, core16()))

def step_timing(conf):
    key = gate_gpu8(conf)
    _k, name, env = TD.gpu_key(conf)
    m = Mirror8(conf, [(TM2, D_TA2L + '/timing', None)])
    m.restore()
    summ = timing8_path(key)
    if os.path.isfile(summ):
        log('TIMING %s core %s: made in an earlier session' % (key, core16()))
        return 0
    vm = os.path.join(TM2, key + '_core_' + core16())
    pri = ','.join((str(p) for p in conf['choice_rule']['priorities']))
    c7a = 'true' if conf['c7a_applies']['value'] else 'false'
    for n in conf['choice_rule']['networks']:
        out = os.path.join(vm, n + '.json')
        cp = os.path.join(vm, 'console_%s_%s.txt' % (TD.boot12(), n))
        rc = stream([PY, '-I', '-B', '-u', TD.SHIM, '--config', CONFIG, TD.BENCH, '--network', n, '--width', '1', '--device', 'cuda', '--synthetic-only', '--config-dir', TD.CFG, '--work-dir', os.path.join(TA_WORK, 'bench2'), '--out', out, '--device-inputs', pri, '--c7a-applies', c7a], cp)
        if rc != 0 or not os.path.isfile(out):
            raise Stop(4, 'the timing of %s failed on %s (exit %d): choose Run all again; if it repeats' % (n, key, rc))
    s = TD.timing_summary(conf, key, name, env, vm)
    s['package'] = '7_train_residual'
    s['lane'] = LANE[0]
    TD.jsave(summ, s)
    m.run()
    log("TIMING %s (this package's core %s): %s; peak allocated %.2f GB of %.2f GB" % (key, core16(), ', '.join(('%s %.4f s' % kv for kv in sorted(s['step_seconds'].items()))), TD.gb(s['peak_allocated_max_bytes']), TD.gb(s['total_memory_bytes'])))
    pk = R._peak_refusal(conf, s)
    if pk:
        change_stop(conf, key, "this package's timing: " + pk)
    return 0

def timing8(key):
    p = timing8_path(key)
    dp = D_TA2L + '/timing/' + os.path.basename(p)
    for x in (p, dp):
        if os.path.isfile(x):
            try:
                return TD.jload(x)
            except (OSError, ValueError):
                continue
    return None

def gpu_dir():
    return TA2L + '/gpu'

def d_gpu():
    return D_TA2L + '/gpu'

def gpu_record(name):
    for p in (gpu_dir() + '/' + name + '.json', d_gpu() + '/' + name + '.json'):
        try:
            d = jload(p)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get('gpu'), str):
            return d
    return None

def training_states(trainings):
    out = {}
    for t in trainings:
        n = t['name']
        dirs = (TA_WORK + '/ckpt/' + n, D_TRAIN + '/ckpt/' + n)

        def has(f):
            return any((os.path.isfile(d + '/' + f) for d in dirs))
        fin = has('DONE.json') or has('STOPPED.json')
        rec = gpu_record(n)
        out[n] = {'begun': has('last.pt') and (not has('DONE.json')), 'finished': fin, 'gpu': rec.get('gpu') if rec else None}
    return out

def claim_gpu(names, key, env, states):
    gpu = env.get('gpu', {}) if isinstance(env, dict) else {}
    n_new = 0
    for n in names:
        s = states.get(n) or {}
        if s.get('finished'):
            continue
        old = gpu_record(n)
        if old and old.get('gpu') == key:
            rec = dict(old)
            if boot12() in (rec.get('sessions') or []):
                continue
            rec['sessions'] = list(rec.get('sessions') or []) + [boot12()]
            rec['devices'] = dict(rec.get('devices') or {})
            rec['devices'][boot12()] = gpu.get('name')
        else:
            rec = {'schema': 'tier-a-gpu-of-training-1', 'training': n, 'gpu': key, 'device_name': gpu.get('name'), 'total_memory_bytes': gpu.get('total_memory_bytes'), 'capability': gpu.get('capability'), 'sessions': [boot12()], 'devices': {boot12(): gpu.get('name')}, 'claimed_utc': utc(), 'replaces': (old or {}).get('gpu'), 'rule': 'a training resumes only on this type (written before the trainer starts)'}
        p = gpu_dir() + '/' + n + '.json'
        jsave(p, rec)
        try:
            copy_file(p, d_gpu() + '/' + n + '.json', True)
        except OSError as e:
            raise Stop(4, 'the GPU record of %s could not be written to the Drive (%s): choose Run all again' % (n, str(e)[:80]))
        n_new += 1
    return n_new

def elsewhere_stop(conf, elsewhere):
    code, text = elsewhere_line(conf, elsewhere)
    raise Stop(code, 'nothing more to train on this GPU type: ' + text)

def next_session_line(conf, key):
    ts = lane_trainings(conf)
    sp = R.type_split(conf, key, training_states(ts), [t['name'] for t in ts])
    if not sp['elsewhere']:
        return None
    return 'NEXT SESSION: this GPU type (%s) has trained what it may in lane %s; %s' % (key, LANE[0], elsewhere_line(conf, sp['elsewhere'])[1])

def elsewhere_line(conf, elsewhere):
    by = {}
    for n, g in sorted(elsewhere.items()):
        by.setdefault(g, []).append(n)
    rt = dict(((g['key'], g['runtime_type']) for g in R.candidates(conf)))
    parts, code = ([], 5)
    for g, ns in sorted(by.items(), key=lambda kv: str(kv[0])):
        if g in rt:
            parts.append('%s %s begun on the %s and resume%s only on it: Runtime > Change runtime type > %s with High-RAM on, keep the LANE in cell 1, then Run all; if Colab does not give it' % (', '.join(ns), 'was' if len(ns) == 1 else 'were', g, 's' if len(ns) == 1 else '', rt[g]))
        else:
            code = 6
            parts.append('%s %s a checkpoint without a GPU record (or of a type that is not a candidate: %s): resumed on no type' % (', '.join(ns), 'has' if len(ns) == 1 else 'have', g))
    return (code, '; '.join(parts))

def write_plan8(conf, key, made_by, need_timing=True):
    from train import flags
    P = plan_mod()
    eq = equiv8(conf, key)
    tm = timing8(key)
    if not (eq and eq.get('pass')) or (need_timing and tm is None):
        raise Stop(4, "the equivalence test and the timing of this package's core on %s in lane %s are needed first (cells 5 and 6)" % (key, LANE[0]))
    threads = conf['threads'] or flags.rule_threads()
    deps = [d for d in conf['departures'] if 'started before' not in d]
    deps += R.auto_departures(conf)
    guard = {'units': float(conf['guard']['units']), 'max_age_s': P.GUARD_MAX_AGE_S, 'missing_rate': R.missing_rate(conf), 'decided_by': list(conf['guard']['decided_by'])}
    path = TD.CFG + '/PLAN_RUNTIME.json'
    P.write_plan_runtime3(path, 'cuda', int(threads), conf['c7a_applies']['value'], R.steps_sm_value(conf), guard, BUDGET_STATE, step_seconds=dict((tm or {}).get('step_seconds') or {}), reader=conf['reader'], departures=deps, measured={'gpu': key, 'device_name': (tm or {}).get('device_name'), 'lane': LANE[0], 'emul_core_sha256': TD.sha256_file(CORE), 'config_sha256': TD.sha256_file(CONFIG)}, made_by=made_by, equivalence=eq['record'], c_channel=conf['c_channel'], day_sets=conf['day_sets'], lane=LANE[0], lane_names=R.lane_names(conf, LANE[0]), profile=LANE[0] == '1')
    os.makedirs(PLAN2, exist_ok=True)
    shutil.copyfile(path, PLAN2 + '/PLAN_RUNTIME_%s.json' % time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    log('PLAN (schema 3): lane %s (%d trainings), S_M %d, on %s; %d departures; the guard %g units' % (LANE[0], len(R.lane_names(conf, LANE[0])), R.steps_sm_value(conf), key, len(deps), guard['units']))

def write_budget_state8(conf, key, g):
    TD.jsave(BUDGET_STATE, {'schema': 'p8-guard-state-1', 'made_utc': utc(), 'made_epoch': time.time(), 'hit': bool(g['hit']), 'spent_units': g['spent'], 'units': g['units'], 'left': g['left'], 'reserve': g['reserve'], 'rate': g['rate'], 'gpu': key, 'counted_high': g.get('counted_high'), 'other_shown': g.get('other_shown'), 'session': TD.boot12(), 'lane': LANE[0], 'rule': "the benchmark's own spend (the session records of TRAIN/logs/sessions/ and TRAIN/logs/sessions_tier_a2/) against the 700-unit runaway guard"})

def cap_line8(conf, g):
    return "the benchmark's own spend %.2f of %g units, %.2f left, reserve %.2f%s; stages 1 to 5 (shown, not counted) %s" % (g['spent'], g['units'], g['left'], g['reserve'], ' (no rate, counted at %g: %s)' % (R.missing_rate(conf), ', '.join(g['counted_high'])) if g.get('counted_high') else '', '%.2f' % g['other_shown'] if g.get('other_shown') is not None else 'not stated')

def guard_now(conf, key, step='train', tries=3, wait_s=5.0):
    last = None
    for k in range(tries):
        try:
            session_update8(conf, NOTEBOOK, step)
            g = R.cap_guard(conf, drive_sessions8(), key, TD.EVERY_S)
            write_budget_state8(conf, key, g)
            return g
        except Exception as e:
            last = e
            if k + 1 < tries:
                time.sleep(wait_s)
    raise Stop(6, "THE GUARD: the benchmark's own spend cannot be read, or this session's record cannot be written (%s: %s; fail closed): wait 10 minutes and choose Run all again; if it repeats" % (type(last).__name__, str(last)[:120]))

def guard_period8(conf, key, fails, cap_hit, stop_child, lost=None):
    g = None
    try:
        if not TD.drive_mounted():
            raise OSError('the Drive is not mounted')
        session_update8(conf, NOTEBOOK, 'train')
        g = R.cap_guard(conf, drive_sessions8(), key, TD.EVERY_S)
        fails[0] = 0
        if lost is None or not lost.is_set():
            write_budget_state8(conf, key, g)
    except Exception as e:
        fails[0] += 1
        print('guard: %s (%d)' % (type(e).__name__, fails[0]), flush=True)
    if not cap_hit.is_set() and (g is not None and g['hit'] or fails[0] >= 2):
        log('%s (%s); the trainer is stopped' % (GUARD_LINE, cap_line8(conf, g) if g is not None else 'the session records cannot be written or read on the Drive, two guard periods in a row'))
        cap_hit.set()
    if cap_hit.is_set():
        stop_child()
        return False
    return True

def lock_state():
    try:
        d = TD.jload(D_LOCK)
    except (OSError, ValueError):
        return None
    if d.get('session') == TD.boot12() or d.get('state') == 'released':
        return None
    try:
        t = calendar.timegm(time.strptime(d['updated_utc'], '%Y-%m-%dT%H:%M:%SZ'))
    except (KeyError, ValueError):
        return None
    age = time.time() - t
    return (d.get('session'), age) if age < LOCK_FRESH_S else None

def write_lock(state='running'):
    TD.jsave(LOCK_VM, {'session': TD.boot12(), 'updated_utc': utc(), 'state': state, 'notebook': NOTEBOOK, 'lane': LANE[0]})
    TD.copy_file(LOCK_VM, D_LOCK, True)

def lock_lost_line():
    return "another session took TRAIN/logs/TRAIN_LOCK_LANE%s.json while this one ran: this session's trainer was stopped and nothing more of it was copied to the Drive; close this tab (the other session goes on)" % LANE[0]

def refuse_other_lock():
    other = lock_state()
    if other:
        raise Stop(10, 'another session (%s) holds TRAIN/logs/TRAIN_LOCK_LANE%s.json, updated %.0f s ago (under 10 minutes): run each lane in one browser tab and one runtime only; wait 10 minutes after the other one stopped, then Run all' % (other[0], LANE[0], other[1]))

def trainer_cmd(extra=()):
    return [PY, '-I', '-B', '-u', TD.SHIM, '--config', CONFIG, TRAINER, '--config-dir', TD.CFG, '--work-dir', TA_WORK, '--logs-dir', TA2L, '--marker', MARKER2, '--arrays-dir', TD.ARR] + list(extra)

def restore_all(conf, m, extra=()):

    def skip(rel, dd):
        if dd == D_TRAIN + '/ckpt' and rel.endswith('/last.pt'):
            return os.path.isfile(dd + '/' + rel.rsplit('/', 1)[0] + '/DONE.json')
        return False
    n = m.restore(skip, restore_pairs(conf, extra))
    log("RESTORE: %d files of lane %s's trainings%s from EMUL/TRAIN/ckpt, the sealed predictions and lockbox places and TRAIN/logs/tier_a2/lane%s to the VM" % (n, LANE[0], "and the first training stage's %s" % ', '.join(extra) if extra else '', LANE[0]))

def run_trainer(conf, key, extra, what):
    m = MIRROR8[0]
    stop = threading.Event()
    cap_hit = threading.Event()
    lost = threading.Event()
    fails = [0]

    def stop_child():
        p = TD.CHILD[0]
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

    def loop_():
        while not stop.wait(TD.EVERY_S):
            if not lost.is_set() and lock_state():
                lost.set()
                print('TRAIN_LOCK: another session took the lock; the trainer is stopped', flush=True)
                stop_child()
            if not lost.is_set():
                try:
                    write_lock()
                except Exception as e:
                    print('lock: %s' % type(e).__name__, flush=True)
                try:
                    m.run()
                except Exception as e:
                    print('mirror: %s' % type(e).__name__, flush=True)
            if guard_period8(conf, key, fails, cap_hit, stop_child, lost) and (not lost.is_set()):
                print(status_line(), flush=True)
    refuse_other_lock()
    g = guard_now(conf, key, what)
    write_lock()
    th = threading.Thread(target=loop_, daemon=True)
    th.start()
    try:
        if g['hit']:
            log('%s (%s); no run starts' % (GUARD_LINE, cap_line8(conf, g)))
            return 6
        t_rc = stream(trainer_cmd(extra), TD.log_path('TRAINER2_STDOUT'))
        log('TRAINER (%s) exit %d' % (what, t_rc))
        if lost.is_set():
            log('TRAIN_LOCK: another session took the lock; the trainer was stopped')
            return LOCK_LOST_RC
        return 6 if cap_hit.is_set() else t_rc
    finally:
        stop.set()
        th.join(timeout=5)
        if lost.is_set():
            log('MIRROR at the end: not run (another session holds the lock; its files are not replaced)')
        else:
            n1, e1 = m.run()
            if e1:
                n1, e1 = m.run()
            log('MIRROR at the end: %d files; %d errors' % (n1, e1))
        try:
            session_update8(conf, NOTEBOOK, 'train')
        except Exception as e:
            print('session record: %s' % type(e).__name__, flush=True)
CUR = {'name': None, 'since': None}

def status_line():
    cur = None
    try:
        cur = TD.jload(TA2L + '/TRAIN_STATUS.json').get('current')
    except (OSError, ValueError, AttributeError):
        pass
    now = time.time()
    if cur != CUR['name']:
        CUR['name'], CUR['since'] = (cur, now)
    mins = int((now - CUR['since']) / 60.0) if cur and CUR['since'] else 0
    return 'STATUS %s: lane %s, training %s, running %d min' % (utc(), LANE[0], cur or '-', mins)

def prepare8(conf, phase, extra=()):
    key = gate_gpu8(conf)
    _k, name, env = TD.gpu_key(conf)
    eq = equiv8(conf, key)
    if not eq or not eq.get('pass'):
        raise Stop(4, "no passed equivalence test of this package's core on %s in lane %s (cell 5): choose Run all again" % (key, LANE[0]))
    if phase == 'train' and timing8(key) is None:
        raise Stop(4, "no timing of this package's core on %s in lane %s (cell 6): choose Run all again" % (key, LANE[0]))
    now = TD.five_of(env)
    diff = {k: (eq['five'].get(k), now.get(k)) for k in now if eq['five'].get(k) != now.get(k)}
    if diff:
        raise Stop(7, "this session's versions differ from the session that passed this package's equivalence test (%s): it trains nothing" % diff)
    os.makedirs(TA2, exist_ok=True)
    lk = open(os.path.join(TA2, 'TRAIN2_DRIVER.lock'), 'w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise Stop(10, 'another driver runs on this VM: not started')
    TD.lb1_check(conf)
    from train import cfg as tcfg
    paths = tcfg.Paths(TD.CFG, TA_WORK, TA2L, MARKER2, TD.ARR)
    configs = tcfg.Configs(paths)
    rows = lane_design_rows(conf, extra)
    ok_rows, wait = ([], [])
    lsha = TD.sha256_file(TD.CFG + '/' + conf['lockbox']['file'])
    for t in rows:
        try:
            with open(TD.norm_file(configs, t['norm']), 'rb') as f:
                has = lsha in f.read().decode('ascii', 'replace')
        except OSError:
            has = False
        (ok_rows if has else wait).append(t)
    bad = [t['name'] for t in wait if t.get('committed')]
    if bad:
        raise Stop(8, "the lockbox check fails: the NORM files of the committed trainings %s are absent or do not record the lockbox's sha256" % bad[:4])
    for t in wait:
        log("LOCKBOX: %s waits (tail; its NORM file %s is absent or does not record the lockbox's sha256: not reached; reading)" % (t['name'], os.path.basename(TD.norm_file(configs, t['norm']))))
    TD.lockbox_check(conf, configs, ok_rows)
    m = Mirror8(conf, drive_pairs(conf))
    MIRROR8[0] = m
    TD.MIRROR[0] = m
    restore_all(conf, m, extra)
    return (key, lk, env)

def map_rc(rc, what):
    if rc == LOCK_LOST_RC:
        raise Stop(10, lock_lost_line())
    if rc in (6, TRAINER_GUARD_RC):
        raise Stop(6, GUARD_LINE)
    if rc == 3:
        raise Stop(10, 'another trainer holds the lock on this VM')
    if rc != 0:
        raise Stop(11, 'the %s stopped (exit %d; TRAIN/logs/tier_a2/lane%s/TRAIN_PROGRESS_ASCII.txt says why): choose Run all again; if it repeats' % (what, rc, LANE[0]))

def early_on_drive(conf):
    have, wait = ([], [])
    for n in R.lane_early(conf, LANE[0]):
        try:
            ok = TD.jload(D_TRAIN + '/ckpt/%s/DONE.json' % n).get('status') == 'done'
        except (OSError, ValueError, AttributeError):
            ok = False
        (have if ok else wait).append(n)
    return (have, wait)

def step_lockbox_pred(conf):
    refuse_other_lock()
    have, wait = early_on_drive(conf)
    key, lk, env = prepare8(conf, 'lockbox_pred', have)
    write_plan8(conf, key, 'train2_colab_driver.py lockbox_pred', need_timing=False)
    only = [t['name'] for t in lane_trainings(conf)] + have
    rc = run_trainer(conf, key, ['--lockbox-only', '--only', ','.join(only)], 'lockbox_pred')
    map_rc(rc, 'LOCKBOX predictions')
    if wait:
        log(WAIT8_LINE % ', '.join(wait))
    return 0

def step_train(conf):
    refuse_other_lock()
    key, lk, env = prepare8(conf, 'train')
    sp, states = split_stop(conf, key)
    g = guard_now(conf, key)
    if g['hit']:
        raise Stop(6, '%s (%s)' % (GUARD_LINE, cap_line8(conf, g)))
    only = list(sp['allowed'])
    if not [n for n in only if n in sp['left']]:
        log('TRAIN: nothing left to train in lane %s on %s' % (LANE[0], key))
        return 0
    write_plan8(conf, key, 'train2_colab_driver.py train')
    refuse_other_lock()
    write_lock()
    k2 = claim_gpu(only, key, env, states)
    log('GPU RECORDS: %d written for lane %s on %s (%s), each on the Drive before the trainer starts' % (k2, LANE[0], key, (env.get('gpu') or {}).get('name')))
    rc = run_trainer(conf, key, ['--only', ','.join(only)], 'train')
    map_rc(rc, 'trainer')
    while True:
        ts = lane_trainings(conf)
        states = training_states(ts)
        sp = R.type_split(conf, key, states, [t['name'] for t in ts])
        new = [n for n in sp['left'] if n not in only]
        if not new:
            break
        only = list(sp['allowed'])
        claim_gpu(new, key, env, states)
        rc = run_trainer(conf, key, ['--only', ','.join(only)], 'train')
        map_rc(rc, 'trainer')
    ln = next_session_line(conf, key)
    if ln:
        log(ln)
    return 0

def step_end(conf):
    from train import cfg as tcfg
    from train import predict
    try:
        st = TD.jload(TA2L + '/TRAIN_STATUS.json')
    except (OSError, ValueError):
        st = {}
    res = st.get('results') or {}
    for k in sorted(res):
        log('RESULT %s: %s' % (k, str(res[k])[:100]))
    rows = lane_design_rows(conf)
    share = lane_design_rows(conf, R.lane_early(conf, LANE[0]))[len(rows):]
    paths = tcfg.Paths(TD.CFG, TA_WORK, TA2L, MARKER2, TD.ARR)

    def done_(t):
        steps = R.steps_sm_value(conf) if t['steps_key'] == 'steps_sm' else R.steps_value(conf)
        return predict.done_check(paths, t, {'width': 1.0, 'steps': steps}, check_hash=False)[0]
    com = [t for t in rows if t.get('committed')]
    tail = [t for t in rows if not t.get('committed')]
    left = [t['name'] for t in rows if not done_(t)]
    log('LEFT: lane %s: %d of its %d committed and %d of its %d tail trainings not done: %s' % (LANE[0], sum((1 for t in com if t['name'] in left)), len(com), sum((1 for t in tail if t['name'] in left)), len(tail), ', '.join(left[:12]) + (' ...' if len(left) > 12 else '') or 'none'))
    if T2_NAME in left:
        log('T2 (%s, M-C7 on 2019) waits for NORM_f6a_fold2019 rewritten with the lockbox (a decision; reading): not reached' % T2_NAME)
    lack = [t['name'] for t in rows + share if done_(t) and (not predict.lockbox_done_check(paths, t)[0])]
    log('LOCKBOX: %d done trainings of lane %s lack a verified LOCKBOX file: %s' % (len(lack), LANE[0], ', '.join(lack[:12]) + (' ...' if len(lack) > 12 else '') or 'none'))
    by = {}
    for t in lane_trainings(conf):
        r = gpu_record(t['name'])
        if r:
            by.setdefault(r.get('gpu'), []).append(t['name'])
    for k in sorted(by, key=str):
        log('GPU OF LANE %s: %s: %s' % (LANE[0], k, ', '.join(by[k])))
    ss = drive_sessions8()
    me = [s for s in ss if s.get('session') == TD.boot12()]
    spent, _rows, never = R.guard_spent(conf, ss)
    if me:
        r, note = R.guard_session_rate(conf, me[0], ss)
        log("UNITS: this session %.3f h on %s at %s units per hour%s; the benchmark's own spend %.2f of the %g-unit guard%s; stages 1 to 5 (shown, not counted): %s" % (float(me[0].get('hours') or 0), me[0].get('accelerator'), r, ' ' + note if note else '', spent, float(conf['guard']['units']), ' (never read: %s)' % ', '.join(never) if never else '', '%.2f' % (R.units_other_total(conf) or 0.0)))
    if lock_state() is not None:
        log('END: the last mirror is not run: another session holds TRAIN/logs/TRAIN_LOCK_LANE%s.json; its files are not replaced' % LANE[0])
    else:
        m = Mirror8(conf, drive_pairs(conf))
        n, e = m.run()
        log('END: the last mirror %d files, %d errors' % (n, e))
        try:
            if lock_state() is None:
                write_lock('released')
        except (OSError, Stop):
            pass
    try:
        key, _name = TD.accelerator(conf)
        nxt = next_session_line(conf, key) if key in [g['key'] for g in R.candidates(conf)] else None
    except (OSError, ValueError):
        nxt = None
    if nxt:
        log(nxt)
    return 0
STEPS = {'session': lambda conf: 0, 'gates': step_gates, 'copy': step_copy, 'equiv': step_equiv, 'timing': step_timing, 'train': step_train, 'lockbox_pred': step_lockbox_pred, 'end': step_end}

def on_term8(_s, _f):
    p = TD.CHILD[0]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=120)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    m = MIRROR8[0]
    if m is not None and lock_state() is None and m.lock.acquire(blocking=False):
        m.lock.release()
        try:
            m.run()
        except Exception:
            pass
    sys.exit(143)

def main(argv):
    if not argv or argv[0] not in STEPS:
        print('usage: train2_colab_driver.py %s' % '|'.join(STEPS))
        return 1
    if not os.path.isdir(TD.EMUL):
        print('THE DRIVE IS NOT MOUNTED (PROJECT_DATA/EMUL not found under /content/drive/MyDrive): run cell 1 first')
        return 9
    os.makedirs(TA2, exist_ok=True)
    try:
        set_lane(read_lane())
    except Stop as s:
        print('STOP (exit %d): %s' % (s.code, s), flush=True)
        return s.code
    signal.signal(signal.SIGTERM, on_term8)
    TD.install_overrides()
    conf, sha = R.load(CONFIG)
    conf = R.attach_units_other(conf, TD.read_units_other(conf))
    session_update8(conf, NOTEBOOK, argv[0])
    try:
        if argv[0] not in ('session', 'gates', 'end'):
            bad = R.refusals(conf, 'train')
            for b in bad:
                log('REFUSED: ' + b)
            if bad:
                raise Stop(3, 'the config refuses (%d reasons above)' % len(bad))
            type_stop8(conf)
        if argv[0] == 'gates':
            return step_gates(conf, sha)
        return STEPS[argv[0]](conf)
    except Stop as s:
        log('STOP (exit %d): %s' % (s.code, s))
        return s.code
    finally:
        TD.push_log()
if __name__ == '__main__':
    try:
        RC = main(sys.argv[1:])
    except SystemExit:
        raise
    except BaseException:
        import traceback
        from choi_dl import common
        for ln in traceback.format_exc().splitlines()[-12:]:
            print(common.mask(ln), flush=True)
        RC = 1
    sys.exit(RC)
