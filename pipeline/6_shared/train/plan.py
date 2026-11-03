"""The list of trainings in their order, with model ids, seeds, labels and network settings, the checks of the runtime plan, and the projected run time of each training."""
import datetime
import json
import os
import re
from train import cfg
MODEL_IDS = {'baseline': 0, 'advanced': 1, 'f6_TRUE': 2, 'f6_OTHER': 3, 'f6_OWN': 4, 'pixmlp': 6, 'e1x': 7, 'e2mx': 8, 'e6': 15}
NETWORKS = {'baseline': ('BaselineUNet', 'baseline', False), 'advanced': ('AdvancedResUNet', 'advanced', True), 'pixmlp': ('PixMLP', 'advanced', False), 'e1x': ('BaselineUNet', 'advanced', False), 'e2mx': ('AdvancedResUNet', 'baseline', True), 'e6': ('BaselineUNet', 'baseline', False), 's18': ('BaselineUNet', 'arm18', False)}
PROJECTION_MARGIN = 1.07
BUDGET_MARGIN = 2.0
STEP_FALLBACK = {'e1x': 'baseline', 'e2mx': 'advanced', 'e6': 'baseline', 's18': 'baseline'}
SCHEMA = 'choi-plan-runtime-2'
OLD_SCHEMA = 'choi-plan-runtime-1'
PRIORITIES = (1, 2, 3, 4, 5, 6, 7, 8)
DAY_SETS = ('all', '2016_2019', 'none')
LR_SENSITIVITY = (('LR-3e-4', 0.0003), ('LR-3e-3', 0.003))
FOLD_ORDER = (2016, 2015, 2017, 2018, 2019)
LAPTOP_DEVICES = ('xpu', 'cpu')
WIDTH_DEPARTURE = re.compile('width|\\bw\\s*=\\s*(?:1/2|0?\\.5)', re.I)
EQUIV_REQUIRED = "the equivalence test's record is required"

class PlanError(Exception):
    pass

def names_width(text):
    return bool(WIDTH_DEPARTURE.search(str(text)))

def _is_int(v):
    return isinstance(v, int) and (not isinstance(v, bool))

def equivalence_from_results(path):
    try:
        d = cfg.read_json(path)
    except cfg.ConfigError as e:
        raise PlanError('%s (%s)' % (EQUIV_REQUIRED, e))
    nd = d.get('nondeterministic_count')
    if not _is_int(nd):
        raise PlanError('%s: %s holds nondeterministic_count %r, not an integer' % (EQUIV_REQUIRED, cfg.rel(path), nd))
    return {'nondeterministic_count': nd, 'results': cfg.rel(path), 'results_sha256': cfg.sha256_file(path)}

def seed_of(test_year, model_id, offset=0):
    return cfg.SEED_BASE + 100 * cfg.fold_index(test_year) + model_id + offset

def _fold(year, net, priority, model_id, offset=0, suffix='', block=False, label=None, arm=None, norm=None, lr0=None, day_sets=False, name_net=None):
    name = 'fold%d_%s%s' % (year, name_net or net, suffix)
    return {'name': name, 'priority': priority, 'kind': 'fold', 'split': ('folds', str(year)), 'held_out_year': year, 'norm': norm or 'fold%d' % year, 'net': net, 'model_id': model_id, 'seed_offset': offset, 'seed': seed_of(year, model_id, offset), 'arm': arm, 'label': label, 'lr0': float(lr0 if lr0 else cfg.LR), 'steps_key': 'steps_main', 'block': {'city': cfg.BLOCK_CITY, 'size': cfg.BLOCK_SIZE} if block else None, 'full_grid': year == 2016, 'day_sets': bool(day_sets), 'f19': False}

def _f6(train_year, test_year, arm):
    key = 'train%d_test%d' % (train_year, test_year)
    mid = MODEL_IDS['f6_' + arm]
    return {'name': 'f6_%s_%s' % (key, arm), 'priority': 2, 'kind': 'f6', 'split': ('f6', key), 'held_out_year': test_year, 'train_year': train_year, 'norm': 'f6_%s_%s' % (key, arm), 'net': 'baseline', 'model_id': mid, 'seed_offset': 0, 'seed': seed_of(test_year, mid), 'arm': arm, 'label': None, 'lr0': float(cfg.LR), 'steps_key': 'steps_f6', 'block': None, 'full_grid': test_year == 2016, 'day_sets': False, 'f19': False}

def all_trainings(c7a_applies=True):
    out = []
    for y in FOLD_ORDER:
        for net in ('baseline', 'advanced'):
            t = _fold(y, net, 1, MODEL_IDS[net], day_sets=True)
            t['f19'] = True
            out.append(t)
    for y in FOLD_ORDER:
        t = _fold(y, 'pixmlp', 1, MODEL_IDS['pixmlp'], day_sets=True)
        t['f19'] = True
        out.append(t)
    for tr, te in cfg.F6_PAIRS:
        for arm in cfg.F6_ARMS:
            out.append(_f6(tr, te, arm))
    if c7a_applies:
        for y in FOLD_ORDER:
            t = _fold(y, 'baseline', 2, MODEL_IDS['baseline'], suffix='_F6A', label='F6A', arm='F6A', norm='f6a_fold%d' % y)
            t['split'] = ('f6a', str(y))
            out.append(t)
    for y, off in ((2016, 1000), (2016, 2000), (2019, 1000)):
        for net in ('baseline', 'advanced'):
            out.append(_fold(y, net, 3, MODEL_IDS[net], off, '_seed%d' % off))
    out.append(_fold(2016, 'e1x', 4, MODEL_IDS['e1x']))
    out.append(_fold(2016, 'e2mx', 4, MODEL_IDS['e2mx']))
    out.append(_fold(2016, 'baseline', 5, MODEL_IDS['baseline'], suffix='_block', block=True))
    out.append(_fold(2016, 'e6', 6, MODEL_IDS['e6']))
    for lab, lr in LR_SENSITIVITY:
        for net in ('baseline', 'advanced'):
            out.append(_fold(2016, net, 7, MODEL_IDS[net], suffix='_' + lab, label=lab, lr0=lr))
    for y in FOLD_ORDER:
        out.append(_fold(y, 's18', 8, MODEL_IDS['baseline'], suffix='_S18', label='S18', name_net='baseline'))
    names = [t['name'] for t in out]
    if len(set(names)) != len(names):
        raise AssertionError('duplicate training names')
    return out

def trainings_for(priorities, c7a_applies=True):
    ps = set((int(p) for p in priorities))
    return [t for t in all_trainings(c7a_applies) if t['priority'] in ps]

def network_spec(net):
    return NETWORKS[net]

def day_set_wanted(training, plan):
    if not training.get('day_sets'):
        return False
    ds = plan.get('day_sets', 'all')
    if ds == 'all':
        return True
    if ds == '2016_2019':
        return int(training['held_out_year']) in (2016, 2019)
    return False

def validate_plan(plan, test_values=False):
    if not isinstance(plan, dict):
        raise PlanError('the plan is not a JSON object')
    if plan.get('schema') == OLD_SCHEMA:
        raise PlanError("schema %r is the earlier version's; this code needs %r (c7a_applies, priorities 1 to 8)" % (OLD_SCHEMA, SCHEMA))
    if plan.get('schema') != SCHEMA:
        raise PlanError('schema %r, expected %r' % (plan.get('schema'), SCHEMA))
    p = dict(plan)
    deps = list(p.get('departures') or [])
    dev = p.get('device')
    if dev not in cfg.DEVICES:
        raise PlanError('device %r not in %s' % (dev, cfg.DEVICES))
    try:
        w = float(p.get('width'))
    except (TypeError, ValueError):
        raise PlanError('width %r' % (p.get('width'),))
    if not test_values and (not (w == 1.0 or (w == 0.5 and dev in LAPTOP_DEVICES and any((names_width(d) for d in deps))))):
        raise PlanError('width %r: w = 1, or 1/2 on the local machine fall-back with a recorded departure' % (w,))
    p['width'] = w
    try:
        sm = int(p.get('steps_main'))
        sf = int(p.get('steps_f6', cfg.STEPS_F6))
        th = int(p.get('threads'))
    except (TypeError, ValueError):
        raise PlanError('steps_main, steps_f6 and threads must be integers')
    if th < 1:
        raise PlanError('threads %d' % th)
    if not test_values:
        if sm != cfg.STEPS_MAIN:
            if not cfg.STEPS_MIN_DEPARTURE <= sm < cfg.STEPS_MAIN:
                raise PlanError('steps_main %d outside 3000 to 6000' % sm)
            if not any(('steps' in str(d) for d in deps)):
                raise PlanError('steps_main %d without a recorded departure' % sm)
        if sf != cfg.STEPS_F6:
            raise PlanError('steps_f6 %d, the two-year emission test fixes 3000' % sf)
        if p.get('test_values'):
            raise PlanError('test_values present without --test-values')
    elif sm < 1 or sf < 1:
        raise PlanError('steps must be at least 1')
    p['steps_main'], p['steps_f6'], p['threads'] = (sm, sf, th)
    if not isinstance(p.get('c7a_applies'), bool):
        raise PlanError('c7a_applies must be true or false')
    ds = p.get('day_sets', 'all')
    if ds not in DAY_SETS:
        raise PlanError('day_sets %r not in %s' % (ds, DAY_SETS))
    if ds != 'all' and (not test_values) and (not any(('day_sets' in str(d) or 'F-3 ALSO' in str(d) for d in deps))):
        raise PlanError("day_sets %r without a recorded departure (F-3 ALSO: 'said so')" % ds)
    p['day_sets'] = ds
    f19 = p.get('f19', True)
    if not isinstance(f19, bool):
        raise PlanError('f19 must be true or false')
    if not f19 and (not test_values) and (not any(('F-19' in str(d) for d in deps))):
        raise PlanError('f19 false without a recorded departure')
    p['f19'] = f19
    pr = p.get('priorities')
    if not isinstance(pr, list) or not pr or any((int(x) not in PRIORITIES for x in pr)):
        raise PlanError('priorities %r' % (pr,))
    p['priorities'] = sorted(set((int(x) for x in pr)))
    bh = p.get('budget_hours')
    if bh is not None:
        try:
            bh = float(bh)
        except (TypeError, ValueError):
            raise PlanError('budget_hours %r' % (bh,))
        if bh <= 0:
            raise PlanError('budget_hours %r' % (bh,))
    p['budget_hours'] = bh
    p['hold_back_p1'] = bool(p.get('hold_back_p1', False))
    reader = p.get('reader', 'memmap')
    if reader not in ('memmap', 'device'):
        raise PlanError('reader %r' % (reader,))
    p['reader'] = reader
    ss = p.get('step_seconds') or {}
    if not isinstance(ss, dict):
        raise PlanError('step_seconds is not an object')
    for k, v in ss.items():
        if k not in NETWORKS:
            raise PlanError('step_seconds: unknown network %r' % k)
        try:
            if float(v) <= 0:
                raise PlanError('step_seconds[%s] %r' % (k, v))
        except (TypeError, ValueError):
            raise PlanError('step_seconds[%s] %r' % (k, v))
    p['step_seconds'] = {k: float(v) for k, v in ss.items()}
    eq = p.get('equivalence')
    if not test_values and (not isinstance(eq, dict) or not _is_int(eq.get('nondeterministic_count'))):
        raise PlanError('%s (equivalence %s)' % (EQUIV_REQUIRED, 'absent' if eq is None else 'without an integer nondeterministic_count'))
    if eq is not None:
        if not isinstance(eq, dict):
            raise PlanError('equivalence is not an object')
        try:
            nd = int(eq.get('nondeterministic_count', 0))
        except (TypeError, ValueError):
            raise PlanError('equivalence.nondeterministic_count %r' % (eq.get('nondeterministic_count'),))
        if nd > 0:
            raise PlanError('the equivalence test saw %d warnings of an operation without a deterministic implementation: the main runs stop for a code change or a decision' % nd)
    names = [t['name'] for t in trainings_for(p['priorities'], p['c7a_applies'])]
    if p.get('order') is not None and list(p['order']) != names:
        raise PlanError('order differs from the order of priorities %s (c7a_applies %s)' % (p['priorities'], p['c7a_applies']))
    tv = dict(p.get('test_values') or {})
    for k in tv:
        if k not in cfg.TEST_VALUE_KEYS:
            raise PlanError('test_values: unknown key %r' % k)
    p['test_values'] = tv
    p['departures'] = deps
    return p

def run_values(plan):
    tv = plan.get('test_values') or {}
    return {'batch': int(tv.get('batch', cfg.BATCH)), 'eval_every': int(tv.get('eval_every', cfg.EVAL_EVERY)), 'ckpt_every': int(tv.get('ckpt_every', cfg.CKPT_EVERY)), 'mem_every': int(tv.get('mem_every', cfg.MEM_EVERY)), 'val_tiles': int(tv.get('val_tiles', cfg.VAL_TILES)), 'train_stat_tiles': int(tv.get('train_stat_tiles', cfg.TRAIN_STAT_TILES)), 'patience': int(tv.get('patience', cfg.PATIENCE))}

def load_plan(paths, test_values=False):
    p = paths.config('PLAN_RUNTIME.json')
    try:
        d = cfg.read_json(p)
    except cfg.ConfigError as e:
        raise PlanError(str(e))
    out = validate_plan(d, test_values=test_values)
    out['_sha256'] = cfg.sha256_file(p)
    return out

def projected_seconds(training, plan):
    ss = plan.get('step_seconds') or {}
    net = training['net']
    t = ss.get(net)
    if t is None and net in STEP_FALLBACK:
        t = ss.get(STEP_FALLBACK[net])
    if t is None:
        return None
    return int(plan[training['steps_key']]) * float(t) * PROJECTION_MARGIN

def write_plan_runtime(path, device, width, threads, priorities, c7a_applies, steps_main=cfg.STEPS_MAIN, steps_f6=cfg.STEPS_F6, budget_hours=None, step_seconds=None, reader='memmap', departures=None, measured=None, free_memory_gb=None, made_by='the driver', test_values=None, day_sets='all', f19=True, hold_back_p1=False, equivalence=None):
    d = {'schema': SCHEMA, 'run': 'emul', 'made': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'made_by': made_by, 'device': device, 'width': float(width), 'steps_main': int(steps_main), 'steps_f6': int(steps_f6), 'threads': int(threads), 'priorities': sorted(set((int(x) for x in priorities))), 'c7a_applies': bool(c7a_applies), 'day_sets': day_sets, 'f19': bool(f19), 'budget_hours': budget_hours, 'hold_back_p1': bool(hold_back_p1), 'step_seconds': dict(step_seconds or {}), 'reader': reader, 'departures': list(departures or []), 'measured': measured or {}, 'free_memory_gb': free_memory_gb}
    if equivalence is not None:
        d['equivalence'] = dict(equivalence)
    if test_values:
        d['test_values'] = dict(test_values)
    d['order'] = [t['name'] for t in trainings_for(d['priorities'], d['c7a_applies'])]
    validate_plan(d, test_values=bool(test_values))
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)
    return d
