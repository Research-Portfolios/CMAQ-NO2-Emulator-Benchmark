"""Driver of the evaluation on a high-memory Colab CPU runtime: checks the package files against their sums, copies the configuration files and arrays, evaluates the test days of the five folds and then, behind its gate, the held-back days, and records the compute units used."""
import os
import signal
import subprocess
import sys
import threading
import time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import nb9_gate as G
import nb9_ledger as L
import tier_a_rules as R
import train_colab_driver as TD
NOTEBOOK = 'EMUL_09B_TIER_A_EVAL'
SRC = ['/content/emul/src/work']
PE_ENTRY = os.path.join(HERE, 'eval', 'run_eval.py')
SUMS = 'CODE_SHA256SUMS_ASCII.txt'
PE_PLACES = {'pred_sealed': 'W_P2/results_sealed/TIER_A/pred', 'ckpt': 'TRAIN/ckpt', 'baselines_sealed': 'W_P2/results_sealed/TIER_A/baselines', 'lockbox_sealed': 'W_P2/results_sealed/TIER_A/lockbox', 'cpu_config': 'RUN/config/p2_rev', 'eval_sealed': 'W_P2/results_sealed/TIER_A/eval', 'train_logs': 'TRAIN/logs/train', 'train_logs_lane1': 'TRAIN/logs/tier_a2/lane1/train', 'train_logs_lane2': 'TRAIN/logs/tier_a2/lane2/train'}
RETRY_S = 30
CHILD = [None]
STATE = {}

def vm_logs():
    return TD.RUN + '/logs/eval'

def d_logs(conf):
    return TD.EMUL + '/' + conf['nb9']['logs']

def places(conf):
    out = dict(PE_PLACES)
    for k, v in (conf.get('places') or {}).items():
        if k in out and isinstance(v, str) and v:
            out[k] = v
    for k, v in out.items():
        if '..' in v or v.startswith('/'):
            raise TD.Stop(3, "the config's place %s is not below EMUL" % k)
    return out

def stamp():
    return time.strftime('%Y%m%d_%H%M%S', time.gmtime())

def sums_rows(path):
    with open(path, 'rb') as f:
        lines = f.read().decode('ascii').splitlines()
    out = {}
    for ln in lines[2:]:
        h, _n, rel = ln.split(' ', 2)
        out[rel] = h
    return out

def verify_package(conf):
    cr, pe = (conf['nb9']['code_root'], conf['nb9']['pe'])
    trees = (('BASE', os.path.join(SRC[0], '6_shared', SUMS), cr['base_sums_sha256']), ('P7', os.path.join(SRC[0], '6_cpu_baselines', SUMS), cr['p7_sums_sha256']), ('PE2', os.path.join(SRC[0], pe['tree'], SUMS), pe['sums_sha256']))
    want = {}
    for what, p, sha in trees:
        if not os.path.isfile(p) or G.sha256_file(p) != sha:
            raise TD.Stop(4, "%s's sums file on the VM is absent or not the pinned one (sha256 %s...): choose Run all again; if it repeats" % (what, sha[:16]))
        for rel, h in sums_rows(p).items():
            if what == 'PE2' and (not rel.startswith('code/eval/')):
                continue
            want[rel] = h
    own_p = os.path.join(TD.PKG, SUMS)
    if not os.path.isfile(own_p):
        raise TD.Stop(4, "this package's sums file is absent on the VM: choose Run all again")
    own_sha = G.sha256_file(own_p)
    want.update(sums_rows(own_p))
    want.pop(SUMS, None)
    bad = []
    for rel, h in sorted(want.items()):
        p = os.path.join(TD.PKG, *rel.split('/'))
        if not os.path.isfile(p) or G.sha256_file(p) != h:
            bad.append(rel)
    extra = []
    for r, _p in TD.walk(os.path.join(TD.PKG, 'code')):
        if r.endswith('.py') and 'code/' + r not in want:
            extra.append('code/' + r)
    if bad or extra:
        raise TD.Stop(4, 'the package on the VM differs from its sums (%d files differ or are absent, %d not listed; first: %s): choose Run all again; if it repeats' % (len(bad), len(extra), ', '.join((bad + extra)[:4])))
    TD.log("PACKAGE: %d files equal to the sums of the shared package %s, the baselines' package %s, the evaluation package %s and this package %s" % (len(want), cr['base_sums_sha256'][:16], cr['p7_sums_sha256'][:16], pe['sums_sha256'][:16], own_sha[:16]))
    return own_sha

def mem_total_gb():
    try:
        with open('/proc/meminfo') as f:
            for ln in f:
                if ln.startswith('MemTotal:'):
                    return int(ln.split()[1]) * 1024 / 1000000000.0
    except (OSError, ValueError, IndexError):
        pass
    return None

def runtime_check(conf):
    rt = conf['nb9']['runtime']
    acc, name = TD.accelerator(conf)
    if acc != rt['accelerator']:
        raise TD.Stop(5, 'this notebook runs on a CPU runtime with High-RAM and this one has %s: Runtime > Change runtime type > CPU, High-RAM on, then Run all' % name)
    mem = mem_total_gb()
    if mem is None or mem < float(rt['mem_total_min_gb']):
        raise TD.Stop(5, 'this CPU runtime has %s GB of memory, under the %g GB this notebook needs: Runtime > Change runtime type > CPU with High-RAM on, then Run all' % ('unknown' if mem is None else '%.1f' % mem, rt['mem_total_min_gb']))
    return (acc, mem)

def cache_dir():
    return vm_logs() + '/ledger_seen'

def this_rate(conf):
    return L.rate_for(conf, TD.this_session() or {'accelerator': TD.accelerator(conf)[0]})[0]

def guard_line(conf):
    g = L.guard(TD.EMUL, conf, cache_dir(), this_rate(conf), TD.EVERY_S)
    if g['hit']:
        return L.guard_text(g)
    return None

def guard_check(conf):
    try:
        why = guard_line(conf)
    except L.Unreadable as e:
        TD.log('GUARD: a session record cannot be read (%s); testing again in %d s' % (e, RETRY_S))
        time.sleep(RETRY_S)
        try:
            why = guard_line(conf)
        except L.Unreadable as e2:
            raise TD.Stop(6, 'THE GUARD cannot be tested: a session record cannot be read (EMUL/%s) twice in a row, and it fails closed; choose Run all again in a few minutes; if it repeats' % e2)
    if why:
        raise TD.Stop(6, why)

def rebuild_units9(conf):
    led = L.read_ledger(TD.EMUL, conf, cache_dir())
    lines = L.units_lines(conf, led, R.units_other_total(conf), NOTEBOOK)
    vm = vm_logs() + '/' + os.path.basename(conf['nb9']['guard']['units_file'])
    os.makedirs(os.path.dirname(vm), exist_ok=True)
    with open(vm + '.partial', 'w', encoding='ascii', newline='\n') as f:
        for t in lines:
            f.write('\n'.join(TD.common.wrap('', t)) + '\n')
    os.replace(vm + '.partial', vm)
    TD.copy_file(vm, TD.EMUL + '/' + conf['nb9']['guard']['units_file'], True)
    return led
TD.rebuild_units = rebuild_units9

def refusals(conf):
    bad = R.refusals(conf, 'stage9')
    for b in bad:
        TD.log('REFUSED: ' + b)
    if bad:
        raise TD.Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))

def gate_cv(conf, own_sha):
    st, miss = G.check_cv(TD.EMUL, conf, places(conf), own_sha)
    if miss:
        for m in miss:
            TD.log('GATE CLOSED: ' + m)
        raise TD.Stop(2, "the evaluation stage's gate is closed (%d items missing, the lines above): nothing sealed was opened or read; do not run it again until the opening file is on the Drive" % len(miss))
    o = st['opening']
    TD.log('GATE CV OPEN: EMUL/%s sha256 %s, decided by %s; models table EMUL/%s, %d rows (%d networks, %d baselines), sha256 %s; packages %s' % (conf['nb9']['gate_cv']['opening_file'], st['opening_sha256'][:16], o['decided_by'], o['models'], len(st['models']), sum((1 for r in st['models'] if r['kind'] == 'network')), sum((1 for r in st['models'] if r['kind'] == 'baseline')), st['models_sha256'][:16], ', '.join(('%s %s' % (n, s[:16]) for n, s in st['packages']))))
    return st

def preflight(conf):
    own = verify_package(conf)
    refusals(conf)
    st = gate_cv(conf, own)
    runtime_check(conf)
    guard_check(conf)
    STATE['cv'] = st
    return st

def need_selftest():
    p = vm_logs() + '/PE_SELFTEST_%s.json' % TD.boot12()
    try:
        d = TD.jload(p)
    except (OSError, ValueError):
        d = {}
    if d.get('result') != 'PASS' or d.get('session') != TD.boot12():
        raise TD.Stop(4, "PE's self-tests have not passed in this session (cell 2): choose Run all; if this line comes again")

def put_once(conf, vm_path, rel):
    dst = TD.EMUL + '/' + rel
    if os.path.isfile(dst):
        if G.sha256_file(dst) == G.sha256_file(vm_path):
            return 'left'
        raise TD.Stop(4, 'EMUL/%s exists with other bytes: nothing is overwritten' % rel)
    TD.copy_file(vm_path, dst, True)
    return 'written'

def write_text(p, lines):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')

def models_on_vm(conf, st):
    rel = st['opening']['models']
    vm = vm_logs() + '/MODELS_%s.json' % st['models_sha256'][:16]
    if not (os.path.isfile(vm) and G.sha256_file(vm) == st['models_sha256']):
        TD.copy_file(TD.EMUL + '/' + rel, vm, False)
    if G.sha256_file(vm) != st['models_sha256']:
        raise TD.Stop(4, "the models table's VM copy is not the file the opening names: choose Run all again")
    return vm

def step_gates(conf, sha):
    st = preflight(conf)
    mk = G.read_marker(TD.EMUL + '/' + G.cv_marker_rel(conf, st['opening_sha256']), 'NB9 CV DONE')
    TD.log('CV STAGE: %s' % ('done before (run id %s)' % mk.get('run_id') if mk else 'to run in this session'))
    lp = TD.EMUL + '/' + conf['nb9']['gate_lockbox']['opening_file']
    lb = G.read_marker(TD.EMUL + '/' + G.lockbox_marker_rel(conf), 'NB9 LOCKBOX DONE')
    TD.log('LOCKBOX STAGE: %s' % ('done before (run id %s)' % lb.get('run_id') if lb else 'its file is on the Drive (tested at its step)' if os.path.isfile(lp) else "waits for the user's file EMUL/%s" % conf['nb9']['gate_lockbox']['opening_file']))
    acc, mem = runtime_check(conf)
    TD.log("CONFIG %s: design %s decided by %s; sealed %s; lockbox %s; c7a_applies %s; runtime %s with %.1f GB at %g units per hour (the config's; nothing typed); guard %g units of the benchmark's own spend" % (sha[:16], conf['design'], conf['decided_by'], conf['sealed'], conf['lockbox'].get('decided_by'), conf['c7a_applies']['value'], acc, mem, this_rate(conf), conf['guard']['units']))
    return 0

def step_copy(conf):
    st = STATE['cv']
    n = TD.copy_configs()
    TD.log('CONFIGS: %d copied to RUN/config/p2 on the VM' % n)
    TD.lb1_check(conf)
    TD.copy_arrays()
    vm = models_on_vm(conf, st)
    TD.log('MODELS: %s on the VM, sha256 %s' % (os.path.basename(vm), st['models_sha256'][:16]))
    return 0

def need_copied():
    if not os.path.isfile(TD.CFG + '/S3_ARRAYS.json') or not os.path.isdir(TD.ARR):
        raise TD.Stop(4, 'the configs and arrays are not on the VM: choose Run all (cell 4 copies them)')

def stop_child():
    p = CHILD[0]
    if p is None or p.poll() is not None:
        return
    try:
        if hasattr(os, 'killpg'):
            os.killpg(p.pid, signal.SIGINT)
        else:
            p.terminate()
        p.wait(timeout=60)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass

def run_pe(conf, stage, args):
    cmd = [sys.executable, '-I', '-B', '-u', PE_ENTRY] + list(args)
    TD.log('PE %s: run_eval.py %s' % (stage, ' '.join((os.path.basename(a) if a.startswith('/') else a for a in args))))
    kw = {'start_new_session': True} if hasattr(os, 'killpg') else {}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=TD.ROOT if os.path.isdir(TD.ROOT) else None, **kw)
    CHILD[0] = p
    hit, done = ([], threading.Event())

    def watch():
        fails = 0
        while not done.wait(TD.EVERY_S):
            try:
                TD.session_update(conf, NOTEBOOK, stage)
                why = guard_line(conf)
                TD.push_log()
                fails = 0
            except Exception as e:
                fails += 1
                why = 'THE GUARD could not be tested in two periods in a row (%s)' % type(e).__name__ if fails >= 2 else None
            if why:
                hit.append(why)
                stop_child()
                return
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        for line in p.stdout:
            TD.log(line.rstrip('\n'))
        rc = p.wait()
    finally:
        done.set()
        t.join(timeout=30)
        CHILD[0] = None
    if hit:
        raise TD.Stop(6, hit[0])
    return rc
PE_MEANING = {3: 'PE refused (its REFUSED line above)', 4: "PE refused: the study design's record is absent or not PASS", 6: 'PE refused a lockbox read without an opening', 1: 'PE ended with an error (the lines above)'}

def pe_args(conf, st, run_id):
    return ['--config', TD.CONFIG, '--emul', TD.EMUL, '--config-dir', TD.CFG, '--arrays-dir', TD.ARR, '--models', models_on_vm(conf, st), '--run-id', run_id]

def eval_done(conf, run_id):
    p = '%s/%s/%s/EVAL_DONE.json' % (TD.EMUL, places(conf)['eval_sealed'], run_id)
    if not os.path.isfile(p):
        raise TD.Stop(4, 'PE ended with exit 0 but %s/EVAL_DONE.json is absent' % run_id)
    return G.sha256_file(p)

def step_cv(conf):
    st = STATE['cv']
    rel = G.cv_marker_rel(conf, st['opening_sha256'])
    mk = G.read_marker(TD.EMUL + '/' + rel, 'NB9 CV DONE')
    if mk:
        TD.log('CV STAGE: done before (run id %s, EVAL_DONE.json sha256 %s); not run again' % (mk.get('run_id'), str(mk.get('eval_done_sha256'))[:16]))
        return 0
    need_selftest()
    need_copied()
    run_id = 'eval_cv_' + stamp()
    rc = run_pe(conf, 'cv', pe_args(conf, st, run_id))
    if rc != 0:
        raise TD.Stop(4, 'the cv stage stopped: %s (PE exit %d)' % (PE_MEANING.get(rc, 'PE failed'), rc))
    sha = eval_done(conf, run_id)
    vm = vm_logs() + '/' + os.path.basename(rel)
    write_text(vm, ['NB9 CV DONE %s exit 0' % TD.utc(), 'run_id: %s' % run_id, 'eval_done_sha256: %s' % sha, 'opening_sha256: %s' % st['opening_sha256'], 'opening_ruled_by: %s' % st['opening']['decided_by'], 'models_sha256: %s' % st['models_sha256'], 'session: %s' % TD.boot12(), 'END'])
    put_once(conf, vm, rel)
    TD.log('CV STAGE DONE: run id %s; EVAL_DONE.json sha256 %s; marker EMUL/%s' % (run_id, sha[:16], rel))
    return 0

def step_lockbox(conf):
    st = STATE['cv']
    g = conf['nb9']['gate_lockbox']
    lrel = G.lockbox_marker_rel(conf)
    lb = G.read_marker(TD.EMUL + '/' + lrel, 'NB9 LOCKBOX DONE')
    if lb:
        TD.log('LOCKBOX STAGE: done before (run id %s); not run again' % lb.get('run_id'))
        return 0
    fields, miss = G.check_lockbox(TD.EMUL, conf, places(conf), st)
    if fields is None:
        for m in miss:
            TD.log('LOCKBOX GATE CLOSED: ' + m)
        if not os.path.isfile(TD.EMUL + '/' + g['opening_file']):
            TD.log('LOCKBOX STAGE NOT RUN: its gate is closed; nothing of the lockbox was opened or read')
            return 0
        raise TD.Stop(3, "the lockbox opening file does not match this notebook's cv stage (the lines above): nothing of the lockbox was opened")
    need_selftest()
    need_copied()
    op = vm_logs() + '/LOCKBOX_OPENING_%s.json' % TD.boot12()
    TD.jsave(op, G.pe_opening_record(conf, fields))
    run_id = 'eval_lockbox_' + stamp()
    TD.log('GATE LOCKBOX OPEN: EMUL/%s, decided by %s, after cv run %s' % (g['opening_file'], fields['decided_by'], fields['cv_run_id']))
    rc = run_pe(conf, 'lockbox', pe_args(conf, st, run_id) + ['--opening', op])
    if rc != 0:
        raise TD.Stop(4, 'the lockbox stage stopped: %s (PE exit %d); do not run it again' % (PE_MEANING.get(rc, 'PE failed'), rc))
    sha = eval_done(conf, run_id)
    vm = vm_logs() + '/' + os.path.basename(lrel)
    write_text(vm, ['NB9 LOCKBOX DONE %s exit 0' % TD.utc(), 'run_id: %s' % run_id, 'eval_done_sha256: %s' % sha, 'lockbox_ruled_by: %s' % fields['decided_by'], 'cv_run_id: %s' % fields['cv_run_id'], 'opening_sha256: %s' % st['opening_sha256'], 'session: %s' % TD.boot12(), 'END'])
    put_once(conf, vm, lrel)
    TD.log('LOCKBOX STAGE DONE: run id %s; EVAL_DONE.json sha256 %s' % (run_id, sha[:16]))
    return 0

def step_end(conf):
    led = L.read_ledger(TD.EMUL, conf, cache_dir())
    ss = [r for r in led['rows'] if r['sid'] == TD.boot12()]
    if ss:
        TD.log("UNITS: this session %.3f h on %s; the benchmark's own spend %.2f of the %g-unit guard (%d records marked at %g%s)" % (ss[0]['hours'], ss[0]['rec'].get('accelerator'), L.spent(led), conf['guard']['units'], sum((1 for r in led['rows'] if r['how'] in ('missing', 'unreadable'))), conf['guard']['missing_rate'], '; %d unreadable' % len(led['never_read']) if led['never_read'] else ''))
    TD.log("COMPUTE UNITS: read 'Available: N compute units' under Runtime > View resources now and compare it with the reading taken before Run all")
    try:
        st, gm = G.check_cv(TD.EMUL, conf, places(conf), G.sha256_file(os.path.join(TD.PKG, SUMS)))
    except Exception:
        st, gm = (None, ['not readable'])
    if gm:
        TD.log('END: gate cv closed; nothing was evaluated')
        return 0
    mk = G.read_marker(TD.EMUL + '/' + G.cv_marker_rel(conf, st['opening_sha256']), 'NB9 CV DONE')
    lb = G.read_marker(TD.EMUL + '/' + G.lockbox_marker_rel(conf), 'NB9 LOCKBOX DONE')
    TD.log('END: cv stage %s; lockbox stage %s' % ('done (run id %s)' % mk.get('run_id') if mk else 'not done', 'done (run id %s)' % lb.get('run_id') if lb else "not run (it waits for the user's file EMUL/%s)" % conf['nb9']['gate_lockbox']['opening_file']))
    return 0
STEPS = {'session': lambda conf: 0, 'gates': step_gates, 'copy': step_copy, 'cv': step_cv, 'lockbox': step_lockbox, 'end': step_end}

def on_term9(_s, _f):
    stop_child()
    try:
        TD.push_log()
    except Exception:
        pass
    sys.exit(143)

def main(argv):
    if not argv or argv[0] not in STEPS:
        print('usage: eval_colab_driver.py %s' % '|'.join(STEPS))
        return 1
    if not os.path.isdir(TD.EMUL):
        print('THE DRIVE IS NOT MOUNTED: run cell 1 first')
        return 9
    TD.LOG_NAME[0] = 'EVAL_DRIVER'
    TD.LOG_FMT = vm_logs() + '/%s_%s_ASCII.txt'
    os.makedirs(vm_logs(), exist_ok=True)
    signal.signal(signal.SIGTERM, on_term9)
    TD.install_overrides()
    conf, sha = R.load(TD.CONFIG)
    TD.LOG_DRIVE[0] = d_logs(conf)
    conf = R.attach_units_other(conf, TD.read_units_other(conf))
    TD.session_update(conf, NOTEBOOK, argv[0])
    try:
        if argv[0] in ('copy', 'cv', 'lockbox'):
            preflight(conf)
        if argv[0] == 'gates':
            return step_gates(conf, sha)
        return STEPS[argv[0]](conf)
    except TD.Stop as s:
        TD.log('STOP (exit %d): %s' % (s.code, s))
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
        for ln in traceback.format_exc().splitlines()[-12:]:
            print(TD.common.mask(ln), flush=True)
        RC = 1
    sys.exit(RC)
