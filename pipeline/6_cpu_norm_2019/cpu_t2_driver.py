"""Driver on Colab that rewrites the statistics of the climatological-emission fold with 2019 held out: checks that the file is not yet written and that the CPU baselines stage has finished, copies the inputs, runs the rewrite, and puts the new file and its index on the Drive."""
import glob
import os
import re
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
import t2_guard as G
NOTEBOOK = 'EMUL_07T2_NORM_F6A_2019'
YEARS = (2015, 2016, 2017, 2018, 2019)
FOLD = '2019'
NORMT2 = os.path.join(HERE, 'cpu', 'norm_f6a_2019.py')
NORM_NAME = 'NORM_f6a_fold2019'
NORM_FILE = NORM_NAME + '.json'
STAGE1_FILE = NORM_NAME + '_STAGE1.json'
LIST_NAME = 'S3C_LIST_9_NORM_F6A_2019.json'
LIST_7C = 'S3C_LIST_8_NORM_F6.json'
INDEX = 'norm_index.json'
BACKUP_PREFIX = 'norm_index_Backup_PriorToNormF6aT2'
BACKUP_RE = re.compile('^norm_index_Backup_PriorToNormF6aT2(_[0-9]+)?\\.json$')
CPU_WORK = TD.W + '/cpu_t2'
CPU_LOGS = TD.RUN + '/logs/cpu_t2'
STAGE_D2828 = TD.RUN + '/config/t2_stage/p2_rev'
STAGE_D2888 = TD.RUN + '/config/t2_stage/p2_rev2'
WORK_VM = TD.ARR + '/work'
D_WORK = TD.D_ARR + '/work'
LEDGER_CACHE = CPU_WORK + '/LEDGER_CACHE.json'
WORK_FILES = tuple(('%s_%d.%s' % (s, y, e) for y in YEARS for s, e in (('daystats', 'npz'), ('daystats_c7', 'npz'), ('build', 'json'), ('build_c7', 'json'))))
END_S = 600.0
GUARD_LINE = "THE GUARD: the benchmark's own spend reached the 700-unit runaway guard"
DONE_LINE = 'already written; nothing to do: NORM_f6a_fold2019 with the lockbox and its whole group are on the Drive (the 2019 normalization stage finished before)'
PART_LINE = 'written in part; do not run this notebook again'

def places(conf):
    p = conf['places']
    e = TD.EMUL + '/'
    return {'d2828': e + p['cpu_config'], 'd2888': e + p['t2_index'], 'logs': e + p['t2_logs'], 'done': e + p['cpu_done'], 'sess': TD.D_TRAIN + '/logs/sessions', 'sess8': TD.D_TRAIN + '/logs/sessions_tier_a2'}

class Overwrite(Exception):
    pass

class AlreadyDone(Exception):
    pass

class GuardUnreadable(Exception):
    pass

def ok_place(conf, dst):
    pl = places(conf)
    d, b = (os.path.dirname(dst), os.path.basename(dst))
    ok = False
    if '..' not in dst and '\\' not in dst:
        if d == pl['d2828']:
            ok = b in (NORM_FILE, STAGE1_FILE, LIST_NAME)
        elif d == pl['d2888']:
            ok = b == INDEX or bool(BACKUP_RE.match(b))
        elif dst.startswith(pl['logs'] + '/'):
            ok = bool(b) and (not b.endswith('.partial'))
    if not ok:
        raise TD.Stop(1, "a Drive write outside the 2019 normalization stage's places was refused: %s" % dst.replace(TD.EMUL, 'EMUL'))
    return dst

def put(conf, src, dst, own=False, replace_ok=False):
    ok_place(conf, dst)
    if os.path.isfile(dst):
        if os.path.getsize(dst) == os.path.getsize(src) and TD.sha256_file(dst) == TD.sha256_file(src):
            return 'same'
        if not (own and TD.boot12() in os.path.basename(dst)):
            if not replace_ok or os.path.basename(dst) in (NORM_FILE, LIST_NAME):
                raise Overwrite(dst)
            TD.log('REPLACED (incomplete group): EMUL%s; the list EMUL%s/%s is absent' % (dst[len(TD.EMUL):], places(conf)['d2828'][len(TD.EMUL):], LIST_NAME))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + '.partial')
    os.replace(dst + '.partial', dst)
    return 'copied'

class LogMirror:

    def __init__(self, conf):
        self.conf = conf
        self.seen = {}
        self.lock = threading.Lock()
        self.refused = []

    def run(self):
        n, err = (0, 0)
        dd = places(self.conf)['logs']
        with self.lock:
            for r, p in TD.walk(CPU_LOGS):
                try:
                    s = TD.Mirror.sig(p)
                except OSError:
                    continue
                if self.seen.get(p) == s:
                    continue
                try:
                    put(self.conf, p, dd + '/' + r, own=True)
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

def push_log(conf):
    p = TD.log_path()
    if os.path.isfile(p) and os.path.isdir(TD.EMUL):
        try:
            put(conf, p, places(conf)['logs'] + '/' + os.path.basename(p), own=True)
        except (OSError, Overwrite, TD.Stop):
            pass

def _ledger_cache():
    try:
        d = TD.jload(LEDGER_CACHE)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}

def drive_sessions_t2(conf):
    pl = places(conf)
    cache = _ledger_cache()
    changed = False
    recs = {}
    for folder, tag in ((pl['sess'], 'sessions'), (pl['sess8'], 'sessions_tier_a2')):
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

def guard_test(conf, sessions):
    spent, rows, never = G.guard_spent(conf, sessions)
    if never:
        raise GuardUnreadable('session records unreadable and never read: %s' % ', '.join(never[:6]))
    rate = R._rate_field(conf, 'CPU')
    if not (R._num(rate) and rate > 0):
        rate = G.missing_rate(conf)
    reserve = float(rate) * (3.0 * float(TD.EVERY_S) + END_S) / 3600.0
    units = float(conf['guard']['units'])
    left = units - spent
    high = sorted(set((str(s.get('accelerator')) for s, r, note in rows if note)))
    return {'hit': left <= reserve, 'spent': spent, 'left': left, 'reserve': reserve, 'units': units, 'rate': float(rate), 'counted_high': high, 'other_shown': R.units_other_total(conf)}

def guard_line(g):
    return "the benchmark's own spend %.2f of %g units (left %.2f; reserve %.3f at the CPU's %g per hour)%s; stages 1 to 5 shown, not counted: %s" % (g['spent'], g['units'], g['left'], g['reserve'], g['rate'], ' (counted at the missing rate: %s)' % ', '.join(g['counted_high']) if g['counted_high'] else '', 'not stated' if g['other_shown'] is None else '%.2f' % g['other_shown'])

def guard_check(conf, tries=2, wait_s=5.0):
    g = None
    for k in range(tries):
        try:
            g = guard_test(conf, drive_sessions_t2(conf))
            break
        except GuardUnreadable as e:
            if k + 1 == tries:
                raise TD.Stop(6, 'THE GUARD fails closed: %s' % e)
            time.sleep(wait_s)
    TD.log('GUARD: ' + guard_line(g))
    if g['hit']:
        raise TD.Stop(6, GUARD_LINE)
    return g

def cpu_check(conf):
    acc, name = TD.accelerator(conf)
    if acc != 'CPU':
        raise TD.Stop(5, 'this notebook runs on a CPU runtime and this one has %s: Runtime > Change runtime type > CPU (High-RAM on), then Run all' % name)

def jsave(p, d):
    TD.jsave(p, d)
    return {'name': os.path.basename(p), 'bytes': os.path.getsize(p), 'sha256': TD.sha256_file(p)}

def drive_group(conf):
    pl = places(conf)
    lp = pl['d2828'] + '/' + LIST_NAME
    if not os.path.isfile(lp):
        return (None, [])
    lst = TD.jload(lp)
    rows = []
    for key, top in (('files', pl['d2828']), ('index_files', pl['d2888'])):
        for r in lst.get(key) or []:
            n = os.path.basename(str(r['path']).replace('\\', '/'))
            rows.append((key, top + '/' + n, int(r['bytes']), r['sha256']))
    return (lst, rows)

def group_whole(conf):
    try:
        lst, rows = drive_group(conf)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if lst is None:
        return False
    names = sorted((os.path.basename(p) for k, p, _b, _s in rows if k == 'files'))
    idx = sorted((os.path.basename(p) for k, p, _b, _s in rows if k == 'index_files'))
    if names != sorted([NORM_FILE, STAGE1_FILE]) or len(idx) != 2 or INDEX not in idx:
        return False
    for _k, p, b, s in rows:
        if not (os.path.isfile(p) and os.path.getsize(p) == b and (TD.sha256_file(p) == s)):
            return False
    return True

def written_state(conf):
    pl = places(conf)
    nf = pl['d2828'] + '/' + NORM_FILE
    if not os.path.isfile(nf):
        if os.path.isfile(pl['d2828'] + '/' + LIST_NAME):
            return 'list_only'
        return None
    try:
        d = TD.jload(nf)
    except (OSError, ValueError):
        d = None
    if not isinstance(d, dict) or 'lockbox' not in d:
        return 'nolockbox'
    return 'done' if group_whole(conf) else 'part'

def gate_written(conf):
    st = written_state(conf)
    if st is None:
        TD.log('GATE NORM: EMUL/RUN/config/p2_rev/%s is not written yet' % NORM_FILE)
        return None
    if st == 'done':
        raise AlreadyDone(DONE_LINE)
    if st == 'part':
        raise TD.Stop(4, 'EMUL/RUN/config/p2_rev/%s holds the lockbox but its group is not whole on the Drive: %s' % (NORM_FILE, PART_LINE))
    if st == 'list_only':
        raise TD.Stop(4, 'EMUL/RUN/config/p2_rev/%s is there without %s: %s' % (LIST_NAME, NORM_FILE, PART_LINE))
    raise TD.Stop(4, "EMUL/RUN/config/p2_rev/%s is there without a lockbox key (not this notebook's file); do not run this notebook again" % NORM_FILE)

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

def gate_cpu_done(conf):
    p = places(conf)['done']
    if not os.path.isfile(p):
        raise TD.Stop(2, 'EMUL/RUN/logs/CPU_DONE_ASCII.txt is absent: the CPU stage has not finished')
    with open(p, 'rb') as f:
        first, rows, place = cpu_done_rows(f.read().decode('ascii', 'replace'))
    if not (first.startswith('CPU DONE') and 'exit 0' in first):
        raise TD.Stop(2, "CPU_DONE_ASCII.txt's first line is not 'CPU DONE ... exit 0'")
    want = {('NORMLIST', LIST_7C): None, ('NORM', INDEX): None}
    for kind, name, size, sha in rows:
        if (kind, name) in want:
            want[kind, name] = (size, sha)
    bad, out = ([], {})
    for (kind, name), v in sorted(want.items()):
        top = place.get(kind)
        if v is None or top is None:
            bad.append('%s %s not listed' % (kind, name))
            continue
        dp = TD.EMUL + '/' + top + '/' + name
        if not os.path.isfile(dp) or os.path.getsize(dp) != v[0] or TD.sha256_file(dp) != v[1]:
            bad.append('%s %s differs from its Drive file' % (kind, name))
            continue
        out[name] = v[1]
    if bad:
        raise TD.Stop(2, "the CPU stage's CPU_DONE does not hold what the 2019 normalization stage needs (%s)" % '; '.join(bad))
    TD.log('GATE CPU_DONE: %s; %s sha16 %s and %s sha16 %s equal to their Drive files' % (first[:40], LIST_7C, out[LIST_7C][:16], INDEX, out[INDEX][:16]))
    return out

def step_gates(conf, sha):
    gate_written(conf)
    gate_cpu_done(conf)
    bad = R.refusals(conf, 'stage9')
    for b in bad:
        TD.log('REFUSED: ' + b)
    if bad:
        raise TD.Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))
    cpu_check(conf)
    guard_check(conf)
    me = TD.this_session()
    TD.log("CONFIG %s: design %s decided by %s; sealed %s; lockbox %s; this runtime %s at %s units per hour (the config's number); guard %g units; stages 1 to 5: %.2f units (%s)" % (sha[:16], conf['design'], conf['decided_by'], conf['sealed'], conf['lockbox'].get('decided_by'), me.get('accelerator'), conf['cpu_units_per_hour'], conf['guard']['units'], R.units_other_total(conf) or 0.0, (conf.get('units_other_doc') or {}).get('decided_by')))
    return 0

def own_vm_files():
    out = [os.path.join(TD.CFG, STAGE1_FILE), os.path.join(TD.CFG, LIST_NAME)]
    out += glob.glob(os.path.join(WORK_VM, BACKUP_PREFIX + '*.json'))
    for top in (STAGE_D2828, STAGE_D2888):
        if os.path.isdir(top):
            out += [os.path.join(top, n) for n in sorted(os.listdir(top))]
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

def fold_years():
    sp = TD.jload(TD.CFG + '/SPLITS.json')
    f = (sp.get('f6a') or {}).get(FOLD)
    if not isinstance(f, dict):
        raise TD.Stop(4, 'SPLITS.json has no "f6a" "%s" entry' % FOLD)
    ys = sorted((int(y) for y in f['train']['index_by_year']))
    if not ys or int(FOLD) in ys:
        raise TD.Stop(4, 'SPLITS.json "f6a" "%s": its training years are not a fold\'s' % FOLD)
    return ys

def array_names(years):
    return sorted(['inputs_f32_%d.npy' % y for y in years] + ['inputs_c7_%d.npy' % y for y in years])

def copy_arrays_t2(years):
    want = array_names(years)
    base = TD.array_rows

    def narrowed():
        rows = [r for r in base() if r[0] in want]
        if sorted((r[0] for r in rows)) != want:
            raise TD.Stop(4, 'S3_ARRAYS.json does not list every array the rewrite reads (%s)' % ', '.join(sorted(set(want) - set((r[0] for r in rows)))))
        return rows
    TD.array_rows = narrowed
    try:
        return TD.copy_arrays()
    finally:
        TD.array_rows = base

def copy_index(conf, want_sha):
    row = copy_small(places(conf)['d2828'], [INDEX], WORK_VM, "the CPU stage's index")[0]
    if row['sha256'] != want_sha:
        raise TD.Stop(4, "EMUL/RUN/config/p2_rev/%s differs from CPU_DONE's row (sha256)" % INDEX)
    return row

def step_copy(conf):
    want = gate_cpu_done(conf)
    rec = {'schema': 'cpu-t2-copy-1', 'session': TD.boot12(), 'made_utc': TD.utc(), 'rule': "sizes equal to the Drive's and to the newest list rows (the shared package); sha256 recorded"}
    rec['configs'] = copy_configs7()
    TD.log('CONFIGS: %d copied to RUN/config/p2 on the VM' % len(rec['configs']))
    TD.lb1_check(conf)
    years = fold_years()
    rec['arrays'] = array_names(years)
    copy_arrays_t2(years)
    rec['work'] = copy_small(D_WORK, WORK_FILES, WORK_VM, 'work file')
    rec['index'] = copy_index(conf, want[INDEX])
    TD.log("WORK: %d files of EMUL/W_P2/arrays_p2/work/ and the CPU stage's %s (sha16 %s) copied; arrays of the training years %s only" % (len(rec['work']), INDEX, rec['index']['sha256'][:16], ', '.join((str(y) for y in years))))
    row = jsave(os.path.join(CPU_LOGS, 'COPY_T2_%s.json' % TD.boot12()), rec)
    m = LogMirror(conf)
    m.run()
    TD.log('COPY record %s (%d bytes, sha16 %s)' % (row['name'], row['bytes'], row['sha256'][:16]))
    return 0

def need_copy():
    if not os.path.isfile(TD.CFG + '/S3_ARRAYS.json'):
        raise TD.Stop(4, 'the configs are not on this VM: run the copy cell (cell 4) first')

def need_selftest():
    p = os.path.join(CPU_LOGS, 'SELFTEST_T2_%s.json' % TD.boot12())
    try:
        ok = TD.jload(p).get('result') == 'PASS'
    except (OSError, ValueError):
        ok = False
    if not ok:
        raise TD.Stop(4, "cell 2's self-test has not passed in this session: nothing is written; choose Run all (cell 2 runs first)")

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
            g = None
            try:
                if not TD.drive_mounted():
                    raise OSError('the Drive is not mounted')
                TD.session_update(conf, NOTEBOOK, step)
                g = guard_test(conf, drive_sessions_t2(conf))
                fails[0] = 0
            except Exception as e:
                fails[0] += 1
                print('guard: %s (%d)' % (type(e).__name__, fails[0]), flush=True)
            if g is not None and g['hit'] or fails[0] >= 2:
                hit[0] = GUARD_LINE if g is not None and g['hit'] else 'THE GUARD fails closed: the session records cannot be written or read on the Drive (two periods)'
                TD.log('%s; the step is stopped' % hit[0])
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
        raise TD.Stop(6, hit[0])
    if mirror.refused:
        raise TD.Stop(4, "the Drive holds %d of this step's logs with other bytes (nothing was overwritten)" % len(mirror.refused))
    return rc

def stage_outputs():
    lp = os.path.join(TD.CFG, LIST_NAME)
    lst = TD.jload(lp)
    files = lst.get('files') or []
    idx = lst.get('index_files') or []
    names = sorted((os.path.basename(str(r['path']).replace('\\', '/')) for r in files))
    if names != sorted([NORM_FILE, STAGE1_FILE]) or len(idx) != 2 or 'configs_by_outbox' in lst:
        raise TD.Stop(4, "%s's rows are not the NORM file and its STAGE1 copy" % LIST_NAME)
    for top in (STAGE_D2828, STAGE_D2888):
        os.makedirs(top, exist_ok=True)
    out = []
    for key, rows, src_dir, top in (('d2888', idx, WORK_VM, STAGE_D2888), ('d2828', files, TD.CFG, STAGE_D2828)):
        for r in rows:
            n = os.path.basename(str(r['path']).replace('\\', '/'))
            if key == 'd2888' and (not (n == INDEX or BACKUP_RE.match(n))):
                raise TD.Stop(4, 'an index row of %s names %s' % (LIST_NAME, n))
            dst = os.path.join(top, n)
            shutil.copyfile(os.path.join(src_dir, n), dst)
            if TD.sha256_file(dst) != r['sha256'] or os.path.getsize(dst) != int(r['bytes']):
                raise TD.Stop(4, "%s differs from %s's row" % (n, LIST_NAME))
            out.append((dst, key, n))
    shutil.copyfile(lp, os.path.join(STAGE_D2828, LIST_NAME))
    order = {INDEX: 1, STAGE1_FILE: 2, NORM_FILE: 3}
    out.sort(key=lambda t: order.get(t[2], 0))
    return (TD.sha256_file(lp), out)

def newest_rows_ok():
    rows = TD.list_rows()
    lst = TD.jload(os.path.join(TD.CFG, LIST_NAME))
    want = {os.path.basename(str(r['path']).replace('\\', '/')): int(r['bytes']) for r in lst['files']}
    ok = all((rows.get(n) == (b, LIST_NAME) for n, b in want.items()))
    return ok and INDEX not in want

def put_group(conf, staged):
    pl = places(conf)
    lst_d = pl['d2828'] + '/' + LIST_NAME
    nf_d = pl['d2828'] + '/' + NORM_FILE
    if os.path.isfile(lst_d):
        raise TD.Stop(4, 'EMUL/RUN/config/p2_rev/%s is there already (the list is never replaced): %s' % (LIST_NAME, PART_LINE))
    done = []
    for path, key, n in staged:
        if n in (NORM_FILE, LIST_NAME):
            continue
        dst = pl[key] + '/' + n
        how = put(conf, path, dst, replace_ok=True)
        done.append((n, how))
    if os.path.isfile(nf_d) or os.path.isfile(lst_d):
        raise TD.Stop(4, 'EMUL/RUN/config/p2_rev/%s (or the list) appeared during this step; nothing is overwritten: %s' % (NORM_FILE, PART_LINE))
    src_nf = os.path.join(STAGE_D2828, NORM_FILE)
    src_l = os.path.join(STAGE_D2828, LIST_NAME)
    ok_place(conf, nf_d)
    ok_place(conf, lst_d)
    shutil.copyfile(src_nf, nf_d + '.partial')
    shutil.copyfile(src_l, lst_d + '.partial')
    os.replace(nf_d + '.partial', nf_d)
    os.replace(lst_d + '.partial', lst_d)
    done += [(NORM_FILE, 'copied'), (LIST_NAME, 'copied')]
    return done

def step_norm(conf):
    need_copy()
    need_selftest()
    want = gate_cpu_done(conf)
    for p in own_vm_files():
        os.remove(p)
    copy_small(TD.D_CFG, [NORM_FILE], TD.CFG, 'stage (1) NORM file')
    copy_small(D_WORK, WORK_FILES, WORK_VM, 'work file')
    copy_index(conf, want[INDEX])
    m = LogMirror(conf)
    rc = run_child(conf, 'norm', [TD.PY, '-I', '-B', '-u', NORMT2, '--config-dir', TD.CFG, '--work-dir', WORK_VM, '--arrays-dir', TD.ARR, '--result', os.path.join(CPU_LOGS, 'NORM_T2_RESULT_%s.json' % TD.boot12())], m)
    if rc != 0:
        raise TD.Stop(4, 'norm_f6a_2019.py ended with exit %d (its lines above): choose Run all again; if it repeats' % rc)
    lsha, staged = stage_outputs()
    if not newest_rows_ok():
        raise TD.Stop(4, '%s does not give the newest list rows of the NORM file and its STAGE1 copy on this VM: nothing is put' % LIST_NAME)
    done = put_group(conf, staged)
    n, e = m.run()
    TD.log('NORM: %s; the list sha16 %s (logs mirrored: %d files, %d errors)' % ('; '.join(('%s %s' % (a, b) for a, b in done)), lsha[:16], n, e))
    return 0

def drive_size_ok(vm_path, d_path):
    return os.path.isfile(d_path) and os.path.isfile(vm_path) and (os.path.getsize(d_path) == os.path.getsize(vm_path))

def step_end(conf):
    m = LogMirror(conf)
    n, e = m.run()
    TD.log('END: the last mirror %d files, %d errors' % (n, e))
    if m.refused or e:
        raise TD.Stop(4, 'the last mirror refused %d files and had %d errors (the lines above): choose Run all again; if it repeats' % (len(m.refused), e))
    ss = [s for s in TD.drive_sessions() if s.get('session') == TD.boot12()]
    if ss:
        TD.log('UNITS: this session %.3f h on %s' % (float(ss[0]['hours']), ss[0]['accelerator']))
    pl = places(conf)
    lp = os.path.join(STAGE_D2828, LIST_NAME)
    left, rows = ([], [])
    if not os.path.isfile(lp):
        left.append('the staged list (the norm step has not ended here)')
    else:
        lst = TD.jload(lp)
        for key, top, vm in (('files', pl['d2828'], STAGE_D2828), ('index_files', pl['d2888'], STAGE_D2888)):
            for r in lst.get(key) or []:
                nm = os.path.basename(str(r['path']).replace('\\', '/'))
                vp = os.path.join(vm, nm)
                if not (os.path.isfile(vp) and TD.sha256_file(vp) == r['sha256']):
                    left.append("%s (the VM's staged copy)" % nm)
                elif not drive_size_ok(vp, top + '/' + nm):
                    left.append('%s (not on the Drive)' % nm)
                else:
                    rows.append((nm, os.path.getsize(vp), r['sha256']))
        if not drive_size_ok(lp, pl['d2828'] + '/' + LIST_NAME):
            left.append('%s (not on the Drive)' % LIST_NAME)
        else:
            rows.append((LIST_NAME, os.path.getsize(lp), TD.sha256_file(lp)))
    if left:
        TD.log('LEFT: %d outputs not done or not on the Drive: %s' % (len(left), '; '.join(left[:8])))
        raise TD.Stop(4, 'the outputs are not all on the Drive (the LEFT line above); do not run this notebook again until told')
    TD.log('T2 NORM DONE: %d files on the Drive (RUN/config/p2_rev/ and RUN/config/p2_rev2/): %s' % (len(rows), '; '.join(('%s %d sha16 %s' % (a, b, c[:16]) for a, b, c in rows))))
    return 0
STEPS = {'session': lambda conf: 0, 'gates': step_gates, 'copy': step_copy, 'norm': step_norm, 'end': step_end}

def on_term7(s, f):
    m = TD.MIRROR[0]
    if m is not None and m.lock.locked():
        TD.MIRROR[0] = None
    TD.on_term(s, f)

def main(argv):
    if not argv or argv[0] not in STEPS:
        print('usage: cpu_t2_driver.py %s' % '|'.join(STEPS))
        return 1
    if not os.path.isdir(TD.EMUL):
        print('THE DRIVE IS NOT MOUNTED: run cell 1 first')
        return 9
    TD.LOG_NAME[0] = 'T2_DRIVER'
    TD.LOG_FMT = CPU_LOGS + '/%s_%s_ASCII.txt'
    TD.LOG_DRIVE[0] = TD.D_TRAIN + '/logs/cpu_t2'
    os.makedirs(CPU_LOGS, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term7)
    TD.install_overrides()
    conf, sha = R.load(TD.CONFIG)
    conf = R.attach_units_other(conf, TD.read_units_other(conf))
    TD.session_update(conf, NOTEBOOK, argv[0])
    flag = os.path.join(CPU_LOGS, 'ALREADY_WRITTEN_%s.txt' % TD.boot12())
    try:
        if argv[0] in ('copy', 'norm'):
            gate_written(conf)
            bad = R.refusals(conf, 'stage9')
            for b in bad:
                TD.log('REFUSED: ' + b)
            if bad:
                raise TD.Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))
            cpu_check(conf)
            guard_check(conf)
        if argv[0] == 'gates':
            return step_gates(conf, sha)
        return STEPS[argv[0]](conf)
    except AlreadyDone as d:
        TD.log('ALREADY WRITTEN (exit 0): %s' % d)
        with open(flag, 'w', encoding='ascii', newline='\n') as f:
            f.write('%s\n' % DONE_LINE)
        return 0
    except TD.Stop as s:
        TD.log('STOP (exit %d): %s' % (s.code, s))
        return s.code
    except Overwrite as o:
        TD.log('STOP (exit 4): the Drive already holds EMUL%s with other bytes; nothing is overwritten' % str(o)[len(TD.EMUL):])
        return 4
    finally:
        push_log(conf)
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
