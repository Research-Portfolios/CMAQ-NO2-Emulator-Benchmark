"""Driver that fits the gradient-boosted trees and their linear model on a CPU runtime through the trainer shim, reading the same arrays and configuration files as the trainings and writing sealed predictions and metrics."""
import os
import sys
import threading
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tier_a_rules as R
import train_colab_driver as TD
NOTEBOOK = 'EMUL_07_STAGE9_GBT'
RUNPY = os.path.join(HERE, 'baselines', 'run.py')
S9_WORK = TD.W + '/stage9'
S9_LOGS = TD.RUN + '/logs/stage9'

def s9_pairs(conf):
    pred = conf['places']['stage9_pred_sealed' if conf['sealed'] else 'stage9_pred_open']
    return [(S9_WORK + '/pred/baselines', TD.EMUL + '/' + pred, None), (S9_WORK + '/results_sealed/BASE2', TD.EMUL + '/' + conf['places']['stage9_sealed'], None), (S9_LOGS, TD.D_TRAIN + '/logs/stage9', None)]

def cap_check(conf):
    why = R.stage9_cap(conf, TD.drive_sessions())
    if why:
        raise TD.Stop(6, why)
    return 0

def step_gates(conf, sha):
    for p, head in TD.D_MARKS:
        first = ''
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                first = f.read().decode('ascii', 'replace').split('\n')[0]
        if not (first.startswith(head) and 'exit 0' in first):
            raise TD.Stop(2, "%s is absent or not '%s ... exit 0': run notebook %s first" % (p.replace(TD.EMUL, 'EMUL'), head, '4' if head.startswith('P2') else '5'))
        TD.log('GATE %s: %s' % (os.path.basename(p), first[:120]))
    bad = R.refusals(conf, 'stage9')
    for b in bad:
        TD.log('REFUSED: ' + b)
    if bad:
        raise TD.Stop(3, 'the config refuses (%d reasons above): nothing to do in Colab' % len(bad))
    TD.rate_stop(conf)
    cap_check(conf)
    TD.log('CONFIG %s: stage (9) names %s; sealed %s; lockbox %s' % (sha[:16], conf['stage9']['names'], conf['sealed'], conf['lockbox'].get('decided_by') or 'not decided'))
    return 0

def step_copy(conf):
    n = TD.copy_configs()
    TD.log('CONFIGS: %d copied' % n)
    TD.lb1_check(conf)
    TD.copy_arrays()
    return 0

def lockbox_fit_check(conf, configs):
    rec = {'lockbox': 'not decided at fit time: no lockbox day is held out of these fits'}
    if conf['lockbox'].get('decided_by') is None:
        TD.log('LOCKBOX: ' + rec['lockbox'])
        return rec
    from baselines import common
    from train import data
    p = TD.CFG + '/' + conf['lockbox']['file']
    if not os.path.isfile(p):
        raise TD.Stop(8, 'the lockbox is decided but its file is absent')
    lb = R.lockbox_days(TD.jload(p))
    days = {}
    for y in (2016, 2015, 2017, 2018, 2019):
        sp = data.Split(configs, common.fold_training(y))
        days['fold%d' % y] = [(a, i) for a, i, _ in sp.train]
    hit = R.lockbox_conflicts(days, lb)
    if hit:
        raise TD.Stop(8, "lockbox days in the GBT's training days of %s" % sorted(hit))
    rec['lockbox'] = "decided by %s: %s sha256 %s; no lockbox day in the fits' training days" % (conf['lockbox']['decided_by'], conf['lockbox']['file'], TD.sha256_file(p))
    TD.log('LOCKBOX: ' + rec['lockbox'])
    return rec

def step_run(conf):
    TD.lb1_check(conf)
    TD.copy_arrays()
    from train import cfg as tcfg
    paths = tcfg.Paths(TD.CFG, S9_WORK, S9_WORK, os.path.join(S9_WORK, 'tmp', 'unused_marker'), TD.ARR)
    configs = tcfg.Configs(paths)
    rec = lockbox_fit_check(conf, configs)
    m = TD.Mirror(s9_pairs(conf))
    TD.MIRROR[0] = m
    n = m.restore()
    TD.log('RESTORE: %d files to the VM (a baseline and fold whose DONE.json verifies and whose sealed file exists is skipped by run.py)' % n)
    rec.update({'schema': 'tier-a-stage9-1', 'session': TD.boot12(), 'names': conf['stage9']['names'], 'made_utc': TD.utc(), 'config_sha256': TD.sha256_file(TD.CONFIG), 'sealed_to': conf['places']['stage9_sealed']})
    TD.jsave(os.path.join(S9_LOGS, 'STAGE9_RECORD_%s.json' % TD.boot12()), rec)
    stop = threading.Event()

    def loop_():
        while not stop.wait(TD.EVERY_S):
            try:
                m.run()
                TD.session_update(conf, NOTEBOOK, 'run')
            except Exception as e:
                print('mirror: %s' % type(e).__name__, flush=True)
    th = threading.Thread(target=loop_, daemon=True)
    th.start()
    try:
        os.makedirs(os.path.join(S9_WORK, 'tmp'), exist_ok=True)
        rc = TD.stream([TD.PY, '-I', '-B', '-u', TD.SHIM, '--config', TD.CONFIG, RUNPY, '--names', ','.join(conf['stage9']['names']), '--config-dir', TD.CFG, '--work-dir', S9_WORK, '--arrays-dir', TD.ARR], os.path.join(S9_LOGS, 'STAGE9_STDOUT_%s_ASCII.txt' % TD.boot12()), cwd=S9_WORK)
    finally:
        stop.set()
        th.join(timeout=5)
        n1, e1 = m.run()
        if e1:
            n1, e1 = m.run()
        TD.session_update(conf, NOTEBOOK, 'run')
        TD.log('MIRROR at the end: %d files; %d errors' % (n1, e1))
    if rc != 0:
        raise TD.Stop(4, 'baselines\\run.py ended with exit %d (its lines above): choose Run all again (done folds are skipped); if it repeats' % rc)
    return 0

def step_end(conf):
    sealed = S9_WORK + '/results_sealed/BASE2'
    files = list(TD.walk(sealed))
    TD.log('SEALED: %d files under W/stage9/results_sealed/BASE2 (names and sizes only): %s' % (len(files), ', '.join(('%s %d' % (r, os.path.getsize(p)) for r, p in files[:12]))))
    m = TD.Mirror(s9_pairs(conf))
    n, e = m.run()
    ss = [s for s in TD.drive_sessions() if s.get('session') == TD.boot12()]
    if ss:
        TD.log('UNITS: this session %.3f h on %s' % (float(ss[0]['hours']), ss[0]['accelerator']))
    TD.log('END: the last mirror %d files, %d errors' % (n, e))
    return 0
STEPS = {'session': lambda conf: 0, 'gates': step_gates, 'copy': step_copy, 'run': step_run, 'end': step_end}

def main(argv):
    if not argv or argv[0] not in STEPS:
        print('usage: stage9_colab_driver.py %s' % '|'.join(STEPS))
        return 1
    if not os.path.isdir(TD.EMUL):
        print('THE DRIVE IS NOT MOUNTED: run cell 1 first')
        return 9
    TD.LOG_NAME[0] = 'STAGE9_DRIVER'
    TD.LOG_FMT = S9_LOGS + '/%s_%s_ASCII.txt'
    TD.LOG_DRIVE[0] = TD.D_TRAIN + '/logs/stage9'
    os.makedirs(S9_LOGS, exist_ok=True)
    import signal
    signal.signal(signal.SIGTERM, TD.on_term)
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
            cap_check(conf)
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
