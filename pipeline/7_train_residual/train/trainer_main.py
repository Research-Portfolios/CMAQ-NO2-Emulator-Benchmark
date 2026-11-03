"""The detached trainer of the design's trainings: runs its rows in two lanes with a budget stop, the climatology tables checked first, and, on request, the held-back-day predictions of every finished training."""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from train import cfg
USAGE = 'usage: trainer_main.py --live [--dry-run] | --config-dir D --work-dir D --logs-dir D --marker F [--arrays-dir D] [--test-values] [--only A,B] [--lockbox-only] [--dry-run]'

class DiskStop(Exception):
    pass

class Opts:
    pass

def parse(argv):
    o = Opts()
    o.live = False
    o.dry = False
    o.test_values = False
    o.lockbox_only = False
    o.only = None
    o.config_dir = o.work_dir = o.logs_dir = o.marker = o.arrays_dir = None
    keys = {'--config-dir': 'config_dir', '--work-dir': 'work_dir', '--logs-dir': 'logs_dir', '--marker': 'marker', '--arrays-dir': 'arrays_dir', '--only': 'only'}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--live':
            o.live = True
        elif a == '--dry-run':
            o.dry = True
        elif a == '--lockbox-only':
            o.lockbox_only = True
        elif a == '--test-values':
            o.test_values = True
        elif a in keys and i + 1 < len(argv):
            setattr(o, keys[a], argv[i + 1])
            i += 1
        else:
            raise SystemExit('unknown or incomplete option %r\n%s' % (a, USAGE))
        i += 1
    named = [o.config_dir, o.work_dir, o.logs_dir, o.marker]
    if o.live:
        if any(named) or o.arrays_dir or o.test_values or o.only:
            raise SystemExit('--live takes no other option but --dry-run\n' + USAGE)
    elif not all(named):
        raise SystemExit('name all four places, or --live\n' + USAGE)
    if o.only:
        o.only = [s for s in o.only.split(',') if s]
    return o

def make_paths(o):
    if o.live:
        return cfg.live_paths()
    return cfg.Paths(o.config_dir, o.work_dir, o.logs_dir, o.marker, o.arrays_dir)

def set_env(paths):
    t = paths.tmp_dir
    os.makedirs(t, exist_ok=True)
    os.environ['TMP'] = t
    os.environ['TEMP'] = t
    tempfile.tempdir = t
    for k, sub in (('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('MPLCONFIGDIR', 'mplconfig')):
        p = os.path.join(t, sub)
        os.makedirs(p, exist_ok=True)
        os.environ[k] = p
    from train import flags
    flags.prepare_cuda_env()

def _mask(s):
    from choi_dl import common
    return common.mask(s)

def _now():
    from choi_dl import common
    return common.now_pair()[1]

def marker_line(paths, text):
    from choi_dl.progress import append_bytes
    line = _mask(text)
    print(line, flush=True)
    os.makedirs(os.path.dirname(paths.marker), exist_ok=True)
    return append_bytes(paths.marker, (line + '\n').encode('ascii', 'replace'))

def _pid_alive(pid):
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        cmd = ' '.join(psutil.Process(pid).cmdline()).lower()
        return 'trainer_main.py' in cmd
    except Exception:
        return False

def take_lock(paths):
    if os.path.exists(paths.lock):
        try:
            with open(paths.lock, 'rb') as f:
                old = int(f.read().decode('ascii').split()[0])
        except Exception:
            old = None
        if old and old != os.getpid() and _pid_alive(old):
            return old
    with open(paths.lock, 'wb') as f:
        f.write(('%d %s\n' % (os.getpid(), _now())).encode('ascii'))
    return None

def release_lock(paths):
    try:
        with open(paths.lock, 'rb') as f:
            mine = f.read().decode('ascii').split()[0] == str(os.getpid())
        if mine:
            os.remove(paths.lock)
    except Exception:
        pass

def counts_text(results, total):
    c = {'done': 0, 'skipped': 0, 'failed': 0, 'not reached': 0}
    for v in results.values():
        for k in c:
            if str(v).startswith(k):
                c[k] += 1
    return ('trainings done %d skipped %d failed %d not reached %d of %d' % (c['done'], c['skipped'], c['failed'], c['not reached'], total), c)

def dry_run(paths, o):
    from train import plan as plan_mod
    configs = cfg.Configs(paths)
    try:
        pl = plan_mod.load_plan(paths, o.test_values)
        pri = pl['priorities']
        c7a = pl['c7a_applies']
        note = 'PLAN_RUNTIME.json read'
    except plan_mod.PlanError as e:
        pl, pri, c7a = (None, list(plan_mod.PRIORITIES), True)
        note = 'no valid PLAN_RUNTIME.json (%s): all priorities listed, the climatological-emission folds as if C-7a applies' % e
    ts = plan_mod.trainings_for(pri, c7a)
    if o.only:
        ts = [t for t in ts if t['name'] in o.only]
    rows = []
    for t in ts:
        net, chset, logs = plan_mod.network_spec(t['net'])
        try:
            nch = '%s (%d)' % (chset, len(configs.channel_set(chset)))
        except cfg.ConfigError:
            nch = '%s (absent from S3_CHANNELS.json)' % chset
        rows.append({'name': t['name'], 'priority': t['priority'], 'network': net, 'channels': nch, 'log_scale': logs, 'model_id': t['model_id'], 'seed': t['seed'], 'norm': 'NORM_%s.json' % t['norm'], 'split': '.'.join(t['split']), 'arm': t['arm'], 'label': t.get('label'), 'lr0': t.get('lr0'), 'steps': pl[t['steps_key']] if pl else t['steps_key'], 'full_grid_2016': t['full_grid'], 'day_sets': plan_mod.day_set_wanted(t, pl) if pl else t.get('day_sets'), 'block': t['block']})
    print(_mask(json.dumps({'note': note, 'trainings': rows}, indent=1)))
    return 0

def preflight(configs, paths, ts, departures=None):
    from train import data
    from train import plan as plan_mod
    out = {}
    for t in ts:
        if t.get('net') == 's18' and (not data.s18_l4_ok(configs, departures)):
            out[t['name']] = 'not reached: ' + data.S18_L4
            continue
        try:
            sp = data.Split(configs, t)
            net, chset, logs = plan_mod.network_spec(t['net'])
            norm = configs.norm(t['norm'])
            data.check_norm(norm, t, sp)
            cp = data.ChannelPlan(configs, chset, logs, norm, data.emission_order_for(configs, t))
            data.check_built(configs, cp, sp.train_years + [sp.held_year], t.get('arm'))
            if t.get('arm') in cfg.F6_ARMS:
                data.check_arm_text(configs, t['split'][1], t['arm'], int(t['train_year']), sp.held_year, norm)
                data.check_arm_channels(cp)
            elif t.get('arm') == 'F6A':
                configs.f6a()
                data.check_arm_channels(cp)
            out[t['name']] = 'ok'
        except (cfg.ConfigError, data.HeldOutError, KeyError, ValueError) as e:
            out[t['name']] = '%s: %s' % (type(e).__name__, _mask(str(e)))
    return out

def prior_hours(paths, pl):
    from train import predict
    from train import plan as plan_mod
    total_s = 0.0
    counted, partial = ([], [])
    for t in plan_mod.trainings_for(pl['priorities'], pl['c7a_applies']):
        pid = {'width': pl['width'], 'steps': int(pl[t['steps_key']])}
        ok, _ = predict.done_check(paths, t, pid)
        if not ok:
            continue
        d = cfg.read_json(os.path.join(paths.ckpt(t['name']), 'DONE.json'))
        s = float((d.get('result') or {}).get('train_wall_s') or 0.0)
        try:
            pi = cfg.read_json(os.path.join(paths.pred(t['name']), 'pred_info.json'))
            s += float(pi.get('seconds') or 0.0)
        except (cfg.ConfigError, TypeError, ValueError):
            partial.append(t['name'])
        total_s += s
        counted.append(t['name'])
    return (total_s / 3600.0, counted, partial)

class ProbeStop(Exception):
    pass

def budget_check(t, pl, prior_h, t_run0, results):
    from train import plan as plan_mod
    if not pl.get('budget_hours'):
        return None
    hold = bool(pl.get('hold_back_p1'))
    if t['priority'] < 3 and (not (hold and t['priority'] >= 2)):
        return None
    proj = plan_mod.projected_seconds(t, pl)
    used_h = prior_h + (time.monotonic() - t_run0) / 3600.0
    if proj is None:
        return 'not reached: no step time in the plan'
    held_h = 0.0
    if hold:
        for u in plan_mod.trainings_for([1], pl['c7a_applies']):
            r = results.get(u['name'], '')
            if r.startswith('done') or r.startswith('skipped'):
                continue
            pu = plan_mod.projected_seconds(u, pl)
            held_h += plan_mod.BUDGET_MARGIN * (pu or 0.0) / 3600.0
    if used_h + held_h + plan_mod.BUDGET_MARGIN * proj / 3600.0 > pl['budget_hours']:
        return 'not reached: budget (%.2f h used, %.2f h held back for priority 1, %.2f h projected x %g margin, %.2f h budget)' % (used_h, held_h, proj / 3600.0, plan_mod.BUDGET_MARGIN, pl['budget_hours'])
    return None

def run_one(t, configs, paths, pl, values, device, prog, setup_rows, t_run0, prior_h=0.0, results=None, profile=None):
    import torch
    from train import data, emul_core, flags, loop, predict
    from train import plan as plan_mod
    name = t['name']
    steps = int(pl[t['steps_key']])
    pid = {'width': pl['width'], 'steps': steps}
    design = t.get('schedule') == 'design'
    lr_every = cfg.lr_every_for(steps) if design else cfg.LR_EVERY
    if design and 'patience' not in (pl.get('test_values') or {}):
        values = dict(values, patience=cfg.patience_for(steps))
    ok, why = predict.done_check(paths, t, pid)
    if ok:
        prog.training(name, status='skipped (done marker verified)')
        prog.note('%s skipped: DONE.json verified' % name)
        return 'skipped'
    if why.startswith('another plan'):
        prog.training(name, status='failed: done marker of another plan')
        prog.note('%s: DONE.json is of %s; left as it is' % (name, why))
        return 'failed: DONE.json of %s' % why
    st = loop.stopped_record(paths, name)
    if st is not None:
        prog.training(name, status='failed: stopped earlier')
        prog.note('%s not run again: %s at step %s (STOPPED.json)' % (name, st.get('status'), st.get('step')))
        return 'failed: stopped earlier by a non-finite value'
    if t.get('net') == 's18' and (not data.s18_l4_ok(configs, pl.get('departures'))):
        prog.training(name, status='not reached: L-4 not built')
        prog.note('%s not reached: %s' % (name, data.S18_L4))
        return 'not reached: ' + data.S18_L4
    why_b = None if design else budget_check(t, pl, prior_h, t_run0, results or {})
    if why_b:
        prog.training(name, status='not reached: budget')
        return why_b
    cell = None
    if t.get('block'):
        cc = configs.city_cells()
        city = t['block']['city']
        if not cc or city not in cc.get('cells', {}):
            prog.training(name, status='not reached: no city cell')
            prog.note('%s not reached: %s lacks %s' % (name, cfg.CITY_CELLS, city))
            return 'not reached: no %s cell in %s' % (city, cfg.CITY_CELLS)
        cell = (int(cc['cells'][city]['row']), int(cc['cells'][city]['col']))
    os.makedirs(paths.work_dir, exist_ok=True)
    free = shutil.disk_usage(paths.work_dir).free / 1000000000.0
    if free < cfg.DISK_FLOOR_GB:
        raise DiskStop('free disk %.2f GB under %.1f GB before %s' % (free, cfg.DISK_FLOOR_GB, name))
    probe = emul_core.precision_probe(device)
    if not probe['pass']:
        prog.note('%s: precision probe FAILED on %s: matmul %.3g, conv %.3g (tolerance %g); no training starts, for a decision' % (name, device, probe['matmul_rel_err'], probe['conv_rel_err'], probe['tol']))
        raise ProbeStop('precision probe above %g on %s' % (probe['tol'], device))
    prog.training(name, status='running', steps=steps, started=_now())
    prog.note('%s start: seed %d, %d steps, %s, w %s%s' % (name, t['seed'], steps, device, pl['width'], ', label %s' % t['label'] if t.get('label') else ''))
    td = data.TrainingData(configs, paths, t, device=device, reader=pl['reader'], val_tiles=values['val_tiles'], block_cell=cell, train_stat_tiles=values['train_stat_tiles'], departures=pl.get('departures'))
    flags.seed_everything(t['seed'])
    model = loop.build_model(t['net'], td.plan.n, pl['width'], zero_head=t.get('target') == 'residual')
    minfo = loop.model_info(model, t['net'], td.plan.n, pl['width'])
    model = model.to(device)
    opt = loop.make_optimizer(model, float(t.get('lr0', cfg.LR)))
    head = {'schema': 'choi-train-log-2', 'run': 'emul', 'label': cfg.LABEL, 'training': name, 'spec': dict(t), 'seed': t['seed'], 'seeds': {'python random.seed': t['seed'], 'numpy legacy np.random.seed': t['seed'] % 2 ** 32, 'torch.manual_seed': t['seed'], 'training tiles': 'PCG64(SeedSequence(%d).spawn(3)[0])' % t['seed'], 'validation tiles': 'PCG64(SeedSequence(%d).spawn(3)[1])' % t['seed'], 'peak tiles': 'PCG64(SeedSequence(%d).spawn(3)[2])' % t['seed'], 'training-statistic tiles': 'PCG64(SeedSequence(%d))' % (t['seed'] + cfg.STAT_SEED_OFFSET)}, 'plan': {k: pl.get(k) for k in ('device', 'width', 'steps_main', 'steps_f6', 'threads', 'reader', 'priorities', 'c7a_applies', 'day_sets', 'f19', 'budget_hours', 'hold_back_p1', 'departures', '_sha256')}, 'test_values': pl.get('test_values') or None, 'values': values, 'steps': steps, 'hyperparameters': loop.hyperparameters(t, values, steps), 'flags': flags.flag_report(device, setup_rows), 'precision_probe': probe, 'data': td.describe(), 'configs_sha256': dict(configs.sha256), 'model': minfo, 'optimizer': {'name': 'Adam', 'betas': list(cfg.BETAS), 'eps': cfg.ADAM_EPS, 'weight_decay': cfg.WEIGHT_DECAY, 'lr': float(t.get('lr0', cfg.LR)), 'gamma': cfg.LR_GAMMA, 'every': lr_every, 'defaults': {k: str(v) for k, v in opt.defaults.items()}}, 'paths': paths.as_dict(), 'started': _now(), 'pid': os.getpid()}
    log = loop.TrainLog(paths, name, head)
    log.write()
    try:
        res = loop.train(td, model, opt, steps, values, device, paths, log, note=prog.note, plan_id=pid, lr_every=lr_every, clip=t.get('clip'), profile=profile)
    except loop.NonFiniteStop as e:
        prog.training(name, status='failed: non-finite', stopped=e.record)
        prog.note('%s STOPPED: %s; last finite checkpoint at step %s; NOT DONE' % (name, e.record['status'], e.record.get('last_finite_checkpoint_step')))
        return 'failed: non-finite value at step %d' % e.record['step']
    prog.training(name, status='predicting', result={k: res.get(k) for k in ('steps_planned',)} if design else res)
    del td, model, opt
    pmodel, best = loop.fresh_model_from_best(t['net'], minfo['in_ch'], pl['width'], os.path.join(paths.ckpt(name), 'best.pt'), device)
    pdat = data.PredictData(configs, paths, t, device=device)
    info = predict.predict_heldout(pdat, pmodel, best, paths, t, device, note=prog.note)
    del pdat
    if plan_mod.day_set_wanted(t, pl):
        predict.predict_day_sets(lambda ds: data.PredictData(configs, paths, t, device=device, day_set=ds), pmodel, paths, t, device, info, note=prog.note)
    predict.write_info(paths, t, info)
    if design:
        try:
            lockbox_one(t, configs, paths, device, prog, pmodel, cfg.sha256_file(os.path.join(paths.ckpt(name), 'best.pt')), pl)
        except Exception as e:
            prog.note('%s LOCKBOX NOT WRITTEN %s: %s (the lockbox_pred step writes it)' % (name, type(e).__name__, _mask(str(e))[:200]))
        try:
            pt = data.PredictData(configs, paths, t, device=device, day_set='TILE_CHECK')
            predict.tile_check(pt, pmodel, paths, t, device)
            del pt
            prog.note('%s: tile-consistency check written (sealed, not printed)' % name)
        except Exception as e:
            prog.note('%s: tile-consistency check NOT WRITTEN %s' % (name, type(e).__name__))
    log.d['prediction'] = {k: info[k] for k in ('n_days', 'n_cells', 'clipped_total', 'clipped_masked_total', 'files', 'seconds', 'best_step', 'predicted_by')}
    log.d['prediction']['day_sets'] = {k: {kk: v[kk] for kk in ('n_days', 'clipped_total', 'clipped_masked_total')} for k, v in info['day_sets'].items()}
    log.d['ended'] = _now()
    log.write()
    lp = os.path.join(paths.train_logs, name + '.json')
    predict.write_done(paths, t, res, info, lp, plan_id=dict(pid, device=device))
    if design:
        prog.training(name, status='done', ended=_now())
        prog.note('%s done: %d steps planned, clipped %d (masked %d)' % (name, res.get('steps_planned'), info['clipped_total'], info['clipped_masked_total']))
    else:
        prog.training(name, status='done', ended=_now(), best_step=res.get('best_step'), steps_run=res.get('steps_run'))
        prog.note('%s done: %d steps, best step %s, clipped %d (masked %d)%s' % (name, res.get('steps_run'), res.get('best_step'), info['clipped_total'], info['clipped_masked_total'], '; day sets TRAIN-S and VAL' if info['day_sets'] else ''))
    del pmodel
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'xpu':
        torch.xpu.empty_cache()
    return 'done'

def f19_phase(configs, paths, pl, device, prog, ts, results):
    try:
        from train import f19
    except ImportError as e:
        prog.note('train.f19 not importable (%s): NOT DONE' % type(e).__name__)
        return
    for t in ts:
        if not t.get('f19'):
            continue
        r = results.get(t['name'], '')
        if not (r.startswith('done') or r.startswith('skipped')):
            prog.note('emission-scaling check %s: not run (the training is %s)' % (t['name'], r or 'not listed'))
            continue
        try:
            out = f19.run_training(configs, paths, t, pl, device, note=prog.note)
            prog.note('emission-scaling check %s: %s' % (t['name'], out['status']))
        except Exception as e:
            prog.note('emission-scaling check %s FAILED %s: %s' % (t['name'], type(e).__name__, _mask(str(e))[:200]))
LANE_STATE = 'LANE_STATE.json'
PROFILE_DONE = 'PROFILE_DONE.json'
STATE_MAX_AGE_S = 600.0
RERUN_SUFFIX = '_clip1'
RERUN_DEPARTURE = 'the one pre-declared rerun with gradient clipping at 1.0 after a non-finite stop (a departure)'
GUARD_RC = 5

class BudgetStop(Exception):
    pass

def lane_state(paths):
    p = os.path.join(paths.logs_dir, LANE_STATE)
    try:
        d = cfg.read_json(p)
    except cfg.ConfigError:
        d = {}
    d.setdefault('schema', 'p8-lane-state-1')
    d.setdefault('reruns', [])
    return d

def lane_state_save(paths, d):
    from train.loop import _write_json_atomic
    _write_json_atomic(d, os.path.join(paths.logs_dir, LANE_STATE))

def rerun_row(t):
    r = dict(t)
    r.update({'name': t['name'] + RERUN_SUFFIX, 'clip': 1.0, 'rerun_of': t['name'], 'label': '%s rerun clip 1.0' % (t.get('label') or ''), 'departure': RERUN_DEPARTURE})
    return r

def design_rows(pl, state):
    from train import plan as plan_mod
    rows = plan_mod.plan_trainings(pl)
    for nm in state.get('reruns', []):
        i = [k for k, r in enumerate(rows) if r['name'] == nm]
        if not i or any((r['name'] == nm + RERUN_SUFFIX for r in rows)):
            continue
        rr = rerun_row(rows[i[0]])
        if rr.get('committed') and rr.get('never_dropped'):
            rows.insert(0, rr)
        else:
            rows.insert(i[0] + 1, rr)
    return rows

def lane_rows(pl, state, only=None):
    rows = design_rows(pl, state)
    names = list(pl.get('lane_names') or [t['name'] for t in rows if not t.get('rerun_of')])
    by = dict(((t['name'], t) for t in rows))
    out = [by[n] for n in names if n in by]
    first = []
    for t in rows:
        b = t.get('rerun_of')
        if b is None or b not in names:
            continue
        if t.get('committed') and t.get('never_dropped'):
            first.append(t)
        else:
            k = [i for i, r in enumerate(out) if r['name'] == b][0]
            out.insert(k + 1, t)
    out = first + out
    if only:
        out = [t for t in out if t['name'] in only]
    return out

def row_status(paths, pl, rows, results):
    from train import loop, predict
    out = {}
    for t in rows:
        n = t['name']
        r = str(results.get(n, ''))
        pid = {'width': pl['width'], 'steps': int(pl[t['steps_key']])}
        if r.startswith('done') or r.startswith('skipped') or predict.done_check(paths, t, pid, check_hash=False)[0]:
            out[n] = 'done'
        elif r.startswith('failed') or r.startswith('not reached') or loop.stopped_record(paths, n) is not None:
            out[n] = 'failed'
        elif os.path.isfile(os.path.join(paths.ckpt(n), 'last.pt')):
            out[n] = 'started'
        else:
            out[n] = 'pending'
    return out

def next_row(rows, status):
    for want in ('started', 'pending'):
        for t in rows:
            if status.get(t['name']) == want:
                return t
    return None

def guard_state(pl):
    p = pl.get('budget_state')
    try:
        d = cfg.read_json(p)
    except (cfg.ConfigError, TypeError) as e:
        raise BudgetStop('the spend file %s cannot be read (%s)' % (p, type(e).__name__))
    age = time.time() - float(d.get('made_epoch') or 0.0)
    mx = float((pl.get('guard') or {}).get('max_age_s') or STATE_MAX_AGE_S)
    if age > mx or age < -60.0:
        raise BudgetStop("the spend file is %.0f s old (more than %.0f s): the driver's mirror loop is not writing it" % (age, mx))
    if not isinstance(d.get('hit'), bool):
        raise BudgetStop('the spend file holds no reading of the guard')
    return {'hit': d['hit'], 'spent': d.get('spent_units'), 'units': d.get('units'), 'left': d.get('left'), 'age_s': round(age, 1)}

def _device_name(device):
    if device != 'cuda':
        return str(device)
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return None

def lockbox_one(t, configs, paths, device, prog, pmodel, best_sha, pl=None):
    from train import data, predict
    gpu = ((pl or {}).get('measured') or {}).get('gpu')
    pdl = data.PredictData(configs, paths, t, device=device, day_set='LOCKBOX')
    info = predict.predict_lockbox(pdl, pmodel, best_sha, paths, t, device, note=prog.note, gpu=gpu, device_name=_device_name(device))
    del pdl
    return info

def lockbox_phase(configs, paths, pl, device, prog, rows, first=()):
    from train import loop, predict
    out = {}
    by = {t['name']: t for t in rows}
    order = [by[n] for n in first if n in by] + [t for t in rows if t['name'] not in set(first)]
    for t in order:
        pid = {'width': pl['width'], 'steps': int(pl[t['steps_key']])}
        ok, why = predict.done_check(paths, t, pid)
        if not ok:
            if t['name'] in set(first):
                out[t['name']] = 'failed: DONE.json %s' % why
                prog.note('LOCKBOX %s FAILED: its DONE.json %s on the VM (no LOCKBOX file written)' % (t['name'], _mask(why)[:160]))
            continue
        if predict.lockbox_done_check(paths, t)[0]:
            out[t['name']] = 'skipped'
            continue
        try:
            bp = os.path.join(paths.ckpt(t['name']), 'best.pt')
            m, _best = loop.fresh_model_from_best(t['net'], _n_in(configs, t), pl['width'], bp, device)
            lockbox_one(t, configs, paths, device, prog, m, cfg.sha256_file(bp), pl)
            out[t['name']] = 'done'
            del m
        except Exception as e:
            out[t['name']] = 'failed: %s' % type(e).__name__
            prog.note('LOCKBOX %s FAILED %s: %s' % (t['name'], type(e).__name__, _mask(str(e))[:200]))
    return out

def _n_in(configs, t):
    from train import plan as plan_mod
    _, chset, _ = plan_mod.network_spec(t['net'])
    return len(configs.channel_set(chset)) + (1 if t.get('c_channel') else 0)

def preflight_design(configs, paths, rows, departures):
    from train import data
    pre = preflight(configs, paths, rows, departures)
    lbx = data.lockbox_maps(configs)
    for t in rows:
        if pre.get(t['name']) != 'ok' or lbx is None:
            continue
        np_ = configs.paths.config('NORM_%s.json' % t['norm'])
        if t['norm'].startswith('f6a_fold') and (not os.path.isfile(np_)):
            np_ = configs.paths.config('NORM_f6a_%s.json' % t['norm'][len('f6a_fold'):])
        try:
            with open(np_, 'rb') as f:
                has = lbx[2] in f.read().decode('ascii', 'replace')
        except OSError:
            has = False
        if not has:
            pre[t['name']] = "ConfigError: %s lacks the lockbox's sha256 (a NORM file of the lockbox is needed)" % os.path.basename(np_)
    for t in rows:
        if pre.get(t['name']) != 'ok' or not data.needs_c(t):
            continue
        try:
            fold = str(t.get('ctables_fold') or t['held_out_year'])
            idx = cfg.ctables_index(paths, fold)
            sp = data.Split(configs, t)
            for f in sorted(set((data.c_fit_of(t, sp, y) for y in list(sp.train_years) + [sp.held_year]))):
                ent = idx['fits'][f]
                for part in ('doy', 'r'):
                    p = cfg.ctable_path(paths, ent[part]['file'])
                    if not os.path.isfile(p):
                        raise cfg.ConfigError('C table %s absent' % ent[part]['file'])
        except (cfg.ConfigError, KeyError, ValueError) as e:
            pre[t['name']] = '%s: %s' % (type(e).__name__, _mask(str(e)))
    for t in rows:
        v = pre.get(t['name'], 'ok')
        if v != 'ok' and (not v.startswith('not reached')) and (not t.get('committed')):
            pre[t['name']] = 'not reached: ' + v
    return pre

def design_main(o, configs, paths, pl, values, prog, t_run0, results):
    from train import flags
    from train import plan as plan_mod
    rows = design_rows(pl, lane_state(paths))
    if o.only:
        unknown = [n for n in o.only if n not in [t['name'] for t in rows]]
        if unknown:
            raise plan_mod.PlanError('--only names %s not in the plan' % unknown)
        rows = [t for t in rows if t['name'] in o.only]
    device = pl['device']
    pre = preflight_design(configs, paths, rows, pl.get('departures'))
    bad = {k: v for k, v in pre.items() if v != 'ok' and (not v.startswith('not reached'))}
    for k, v in pre.items():
        if v != 'ok':
            prog.note('PREFLIGHT %s: %s' % (k, v[:240]))
    prog.set(preflight=pre)
    if bad:
        raise cfg.ConfigError("preflight: %d of %d committed trainings' configs failed (TRAIN_STATUS.json preflight)" % (len(bad), len(rows)))
    for k, v in pre.items():
        if v.startswith('not reached'):
            results[k] = v
    setup_rows = flags.setup(device, pl['threads'])
    prog.set(device=device, width=pl['width'], plan_sha256=pl['_sha256'], test_values=bool(o.test_values), lane=pl.get('lane'), order=[t['name'] for t in lane_rows(pl, lane_state(paths), o.only)])
    prog.note('plan (schema 3, %s): device %s, w %s, steps %d and S_M %d, threads %d, lane %s, %d trainings (%d committed)%s' % (pl['design'], device, pl['width'], pl['steps_main'], pl['steps_sm'], pl['threads'], pl.get('lane') or '-', len(rows), sum((1 for t in rows if t.get('committed'))), '; TEST VALUES' if o.test_values else ''))
    if o.lockbox_only:
        out = lockbox_phase(configs, paths, pl, device, prog, rows, plan_mod.EARLY_8)
        for k, v in out.items():
            results['lockbox ' + k] = v
        prog.note('LOCKBOX phase: %d done, %d skipped (sealed file verified), %d failed' % (sum((1 for v in out.values() if v == 'done')), sum((1 for v in out.values() if v == 'skipped')), sum((1 for v in out.values() if v.startswith('failed')))))
        return 1 if any((v.startswith('failed') for v in out.values())) else 0
    return design_run(o, configs, paths, pl, values, device, prog, setup_rows, t_run0, results)

def design_run(o, configs, paths, pl, values, device, prog, setup_rows, t_run0, results):
    from train import loop
    state = lane_state(paths)
    prof = None if not pl.get('profile') or os.path.isfile(os.path.join(paths.logs_dir, PROFILE_DONE)) else (1, 2) if o.test_values else (10, 20)
    while True:
        rows = lane_rows(pl, state, o.only)
        status = row_status(paths, pl, rows, results)
        g = guard_state(pl)
        if g['hit']:
            prog.note("THE GUARD: the spend file says the benchmark's own spend reached the %s-unit runaway guard: no training starts" % g.get('units'))
            return GUARD_RC
        t = next_row(rows, status)
        if t is None:
            prog.note("the lane's trainings: none left to start (%d done, %d failed or not reached, of %d)" % (sum((1 for v in status.values() if v == 'done')), sum((1 for v in status.values() if v == 'failed')), len(rows)))
            return 0
        plt = dict(pl, departures=list(pl.get('departures') or []) + [t['departure']]) if t.get('departure') else pl
        try:
            r = run_one(t, configs, paths, plt, values, device, prog, setup_rows, t_run0, 0.0, results, profile=prof)
        except (loop.MemStop, DiskStop, cfg.ConfigError, ProbeStop) as e:
            results[t['name']] = 'failed: %s (run stopped)' % type(e).__name__
            raise
        except Exception as e:
            from train import data
            if isinstance(e, data.HeldOutError):
                results[t['name']] = 'failed: HeldOutError (run stopped)'
                raise
            r = 'failed: %s' % type(e).__name__
            prog.training(t['name'], status=r, error=_mask(str(e))[:300])
            prog.note('%s FAILED %s: %s' % (t['name'], type(e).__name__, _mask(str(e))[:300]))
            for ln in traceback.format_exc().splitlines()[-6:]:
                prog.note('  ' + _mask(ln.strip())[:200])
        results[t['name']] = r
        if prof is not None and os.path.isfile(os.path.join(paths.train_logs, t['name'] + '_profile_ASCII.txt')):
            from train.loop import _write_json_atomic
            _write_json_atomic({'training': t['name'], 'utc': _now()}, os.path.join(paths.logs_dir, PROFILE_DONE))
            prof = None
        if r.startswith('failed: non-finite') and (not t.get('rerun_of')) and (t['name'] not in state['reruns']):
            state['reruns'].append(t['name'])
            lane_state_save(paths, state)
            prog.note('%s: its one clipped rerun %s enters the lane' % (t['name'], t['name'] + RERUN_SUFFIX))

def main(argv):
    o = parse(argv)
    paths = make_paths(o)
    if o.dry:
        return dry_run(paths, o)
    set_env(paths)
    os.makedirs(paths.logs_dir, exist_ok=True)
    other = take_lock(paths)
    if other:
        marker_line(paths, 'run: %s not started %s; another instance is live (pid %d)' % (cfg.LABEL, _now(), other))
        return 3
    marker_line(paths, 'run: %s started %s; pid %d' % (cfg.LABEL, _now(), os.getpid()))
    status = 'failed: did not finish'
    results = {}
    total = 0
    try:
        from train.progress import TrainProgress
        prog = TrainProgress(paths.progress, paths.status)
        t_run0 = time.monotonic()
        from train import data, flags, loop
        from train import plan as plan_mod
        prog.note('START %s pid %d; places: %s' % (cfg.LABEL, os.getpid(), '; '.join(('%s %s' % (k, v) for k, v in paths.as_dict().items()))))
        configs = cfg.Configs(paths)
        pl = plan_mod.load_plan(paths, test_values=o.test_values)
        values = plan_mod.run_values(pl)
        if pl.get('schema') == plan_mod.SCHEMA3:
            rc = design_main(o, configs, paths, pl, values, prog, t_run0, results)
            txt, c = counts_text(results, len(results))
            total = len(results)
            status = 'ok' if c['failed'] == 0 else 'failed: %d trainings failed' % c['failed']
            if rc == GUARD_RC and (not o.lockbox_only):
                status = 'stopped: THE GUARD'
            elif rc:
                status = 'failed: %s' % rc
            return rc if rc else 0 if c['failed'] == 0 else 1
        ts = plan_mod.trainings_for(pl['priorities'], pl['c7a_applies'])
        if o.only:
            unknown = [n for n in o.only if n not in [t['name'] for t in ts]]
            if unknown:
                raise plan_mod.PlanError('--only names %s not in the plan' % unknown)
            ts = [t for t in ts if t['name'] in o.only]
        total = len(ts)
        device = pl['device']
        pre = preflight(configs, paths, ts, pl.get('departures'))
        bad = {k: v for k, v in pre.items() if v != 'ok' and (not v.startswith('not reached'))}
        for k, v in pre.items():
            if v.startswith('not reached'):
                prog.note('PREFLIGHT %s: %s' % (k, v[:240]))
        prog.set(preflight=pre)
        if bad:
            for k, v in bad.items():
                prog.note('PREFLIGHT %s: %s' % (k, v[:240]))
            raise cfg.ConfigError("preflight: %d of %d trainings' configs failed (TRAIN_STATUS.json preflight)" % (len(bad), len(ts)))
        prog.note('preflight: the configs of %d trainings checked' % len(ts))
        prior_h = 0.0
        if pl.get('budget_hours') and any((t['priority'] >= 2 for t in ts)):
            prior_h, counted, partial = prior_hours(paths, pl)
            prog.note('budget: %.2f h of %.2f h used before this start by %d trainings with verified done markers%s' % (prior_h, pl['budget_hours'], len(counted), '; pred_info.json seconds not read for %s' % ', '.join(partial) if partial else ''))
        setup_rows = flags.setup(device, pl['threads'])
        prog.set(device=device, width=pl['width'], plan_sha256=pl['_sha256'], test_values=bool(o.test_values), order=[t['name'] for t in ts])
        prog.note('plan: device %s, w %s, steps %d and %d, threads %d, priorities %s, %d trainings%s' % (device, pl['width'], pl['steps_main'], pl['steps_f6'], pl['threads'], pl['priorities'], total, '; TEST VALUES' if o.test_values else ''))
        for t in ts:
            try:
                r = run_one(t, configs, paths, pl, values, device, prog, setup_rows, t_run0, prior_h, results)
            except (loop.MemStop, data.HeldOutError, DiskStop, cfg.ConfigError, ProbeStop) as e:
                results[t['name']] = 'failed: %s (run stopped)' % type(e).__name__
                prog.training(t['name'], status='stopped: %s' % type(e).__name__, error=_mask(str(e))[:300])
                raise
            except Exception as e:
                r = 'failed: %s' % type(e).__name__
                prog.training(t['name'], status=r, error=_mask(str(e))[:300])
                prog.note('%s FAILED %s: %s' % (t['name'], type(e).__name__, _mask(str(e))[:300]))
                for ln in traceback.format_exc().splitlines()[-6:]:
                    prog.note('  ' + _mask(ln.strip())[:200])
            results[t['name']] = r
            if r.startswith('not reached'):
                prog.note('%s %s' % (t['name'], r))
        if pl.get('f19') and 1 in pl['priorities']:
            f19_phase(configs, paths, pl, device, prog, ts, results)
        txt, c = counts_text(results, total)
        status = 'ok' if c['failed'] == 0 else 'failed: %d trainings failed' % c['failed']
        return 0 if c['failed'] == 0 else 1
    except BaseException as e:
        kind = type(e).__name__
        reasons = {'MemStop': 'memory', 'HeldOutError': 'held-out guard', 'DiskStop': 'disk', 'ConfigError': 'config', 'PlanError': 'plan', 'KeyboardInterrupt': 'interrupted', 'SystemExit': 'exit', 'ProbeStop': 'probe'}
        status = 'failed: %s' % reasons.get(kind, kind)
        try:
            prog.note('ERROR %s: %s' % (kind, _mask(str(e))[:300]))
            for ln in traceback.format_exc().splitlines()[-6:]:
                prog.note('  ' + _mask(ln.strip())[:200])
        except Exception:
            pass
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return 1
    finally:
        txt, c = counts_text(results, total)
        try:
            prog.set(final_status=status, results=results)
            prog.note('END %s status %s; %s' % (cfg.LABEL, status, txt))
        except Exception:
            pass
        marker_line(paths, 'run: %s finished; status %s; %s; ended %s' % (cfg.LABEL, status, txt, _now()))
        release_lock(paths)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
