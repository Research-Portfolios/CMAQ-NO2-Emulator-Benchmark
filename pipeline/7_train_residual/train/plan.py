"""The list of trainings: the shared plan plus the design's rows (the residual U-Net, the steps from the reference U-Net, the climatological-emission and 18-input variants, the per-cell network and the extra seeds) with their seeds and settings."""
import datetime
import json
import os
import re
from train import cfg
MODEL_IDS = {'baseline': 0, 'advanced': 1, 'f6_TRUE': 2, 'f6_OTHER': 3, 'f6_OWN': 4, 'pixmlp': 6, 'e1x': 7, 'e2mx': 8, 'e6': 15}
NETWORKS = {'baseline': ('BaselineUNet', 'baseline', False), 'advanced': ('AdvancedResUNet', 'advanced', True), 'pixmlp': ('PixMLP', 'advanced', False), 'e1x': ('BaselineUNet', 'advanced', False), 'e2mx': ('AdvancedResUNet', 'baseline', True), 'e6': ('BaselineUNet', 'baseline', False), 's18': ('BaselineUNet', 'arm18', False), 'pixr': ('PixMLP', 'advanced', True)}
PROJECTION_MARGIN = 1.07
BUDGET_MARGIN = 2.0
STEP_FALLBACK = {'e1x': 'baseline', 'e2mx': 'advanced', 'e6': 'baseline', 's18': 'baseline', 'pixr': 'pixmlp'}
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
    if plan.get('schema') == SCHEMA3:
        return validate_plan3(plan, test_values=test_values)
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
SCHEMA3 = 'choi-plan-runtime-3'
DESIGN = 'STUDY_DESIGN'
DESIGN_IDS = {'P': 0, 'M': 20, 'A1': 21, 'A2': 22, 'A3': 23, 'A4': 24, 'M-C7': 25, 'A3BL': 26, 'PIX-R': 27, 'A3OFF': 28}
CONFIRMATORY = ('M', 'P')
ONE = cfg.MIN_MASKED_FRACTION_ONE
QUARTER = cfg.MIN_MASKED_FRACTION
ARMS = {'P': ('baseline', 'ioa', 'direct', False, QUARTER, 'steps_main', 'baseline', None, False), 'A1': ('baseline', 'mse', 'direct', False, QUARTER, 'steps_main', 'A1', None, False), 'A2': ('advanced', 'mse', 'direct', False, ONE, 'steps_main', 'A2', None, False), 'A3': ('advanced', 'mse', 'residual', True, ONE, 'steps_main', 'A3', None, False), 'A4': ('advanced', 'mse', 'direct', False, ONE, 'steps_sm', 'A4', None, False), 'M': ('advanced', 'mse', 'residual', True, ONE, 'steps_sm', 'M', None, False), 'M-C7': ('advanced', 'mse', 'residual', True, ONE, 'steps_sm', 'MC7', 'F6A', False), 'A3BL': ('advanced', 'mse', 'residual', True, ONE, 'steps_main', 'A3BL', None, True), 'A3OFF': ('advanced', 'mse', 'residual', False, ONE, 'steps_main', 'A3OFF', None, False), 'PIX-R': ('pixr', 'mse', 'residual', True, ONE, 'steps_main', 'PIXR', None, False)}
CORE_FOLDS = (2019, 2016, 2015, 2017, 2018)
DROP_ORDER = ((1, 'PIX-R on 2019', ('fold2019_PIXR',)), (2, 'A3BL', ('fold2016_A3BL',)), (3, 'A1, A2 and A3 on 2019', ('fold2019_A1', 'fold2019_A2', 'fold2019_A3')), (4, 'M s2 on 2019', ('fold2019_M_seed2000',)), (5, 'P s3, s4 on 2016', ('fold2016_baseline_seed3000', 'fold2016_baseline_seed4000')), (6, 'M s3, s4 on 2016', ('fold2016_M_seed3000', 'fold2016_M_seed4000')), (7, 'M-C7 on 2018', ('fold2018_MC7',)), (8, 'PIX-R on 2016', ('fold2016_PIXR',)))
NEVER_DROPPED = tuple(['fold%d_M' % y for y in cfg.FOLD_YEARS] + ['fold%d_baseline' % y for y in cfg.FOLD_YEARS] + ['fold2016_M_seed1000', 'fold2016_M_seed2000', 'fold2016_baseline_seed1000', 'fold2016_baseline_seed2000', 'fold2019_M_seed1000', 'fold2019_baseline_seed1000', 'fold2016_A1', 'fold2016_A2', 'fold2016_A3', 'fold2016_A4', 'fold2016_A3OFF', 'fold2016_MC7'])
EARLY_8 = ('fold2016_baseline', 'fold2015_baseline', 'fold2017_baseline', 'fold2018_baseline', 'fold2019_baseline', 'fold2016_baseline_seed1000', 'fold2016_baseline_seed2000', 'fold2019_baseline_seed1000')
TAIL_ITEMS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7')
STEPS_SM = 16000
GUARD_UNITS = 700.0
GUARD_MAX_AGE_S = 600.0
LANES = ('1', '2')
C_CHANNEL_FORMS = (True, 'offset', 'direct')
F6_FOLD = 'f6'

def seed_k(test_year, k):
    return cfg.SEED_BASE + 100 * cfg.fold_index(test_year) + 1000 * int(k)

def _seed_suffix(k):
    return '_seed%d' % (1000 * int(k)) if int(k) else ''

def _row(model, year, k=0, tset='', lr=None, lr_label=None):
    net, loss, target, cch, tile, skey, tag, arm, block = ARMS[model]
    name = 'fold%d_%s%s' % (year, tag, _seed_suffix(k))
    if lr_label:
        name += '_' + lr_label
    t = {'name': name, 'priority': 1, 'kind': 'fold', 'split': ('folds', str(year)), 'held_out_year': int(year), 'norm': 'fold%d' % year, 'net': net, 'model_id': DESIGN_IDS[model], 'seed_offset': 1000 * int(k), 'seed': seed_k(year, k), 'arm': arm, 'label': model if not lr_label else '%s %s' % (model, lr_label), 'lr0': float(lr if lr else cfg.LR), 'steps_key': skey, 'block': {'city': cfg.BLOCK_CITY, 'size': cfg.BLOCK_SIZE} if block else None, 'full_grid': int(year) == 2016, 'day_sets': False, 'f19': False, 'design': DESIGN, 'model': model, 'k': int(k), 'design_label': 'C' if model in CONFIRMATORY else 'X', 'loss': loss, 'target': target, 'c_channel': bool(cch), 'min_masked_fraction': float(tile), 'schedule': 'design', 'ctables_fold': str(year), 'set': tset, 'tail': None, 'committed': True, 'never_dropped': False, 'drop': None}
    if arm == 'F6A':
        t['split'] = ('f6a', str(year))
        t['norm'] = 'f6a_fold%d' % year
    return t

def _t4(train_year, test_year, arm):
    t = _row('A3', test_year, 0, 'tail')
    key = 'train%d_test%d' % (train_year, test_year)
    t.update({'name': 't4_%s_%s' % (key, arm), 'kind': 'f6', 'split': ('f6', key), 'train_year': int(train_year), 'norm': 'f6_%s_%s' % (key, arm), 'arm': arm, 'label': 'A3, two-year emission test, %s' % arm, 'full_grid': False, 'ctables_fold': F6_FOLD, 'c_fit': 'f6_train%d' % train_year})
    return t

def _p18():
    t = _row('P', 2016, 0, 'tail')
    t.update({'name': 'fold2016_P18', 'net': 's18', 'label': 'P18'})
    return t

def _apply_c_channel(rows, form):
    if form is True:
        return rows
    out = []
    for t in rows:
        t = dict(t)
        if form == 'offset':
            if t['model'] == 'A3OFF':
                continue
            if t['target'] == 'residual':
                t['c_channel'] = False
        elif form == 'direct':
            if t['model'] in ('A3', 'A3BL', 'A3OFF') and t['kind'] != 'f6':
                continue
            if t['target'] == 'residual':
                t['target'] = 'direct'
                t['c_channel'] = False
        else:
            raise PlanError('c_channel %r (one of %s)' % (form, C_CHANNEL_FORMS))
        out.append(t)
    return out

def design_trainings(c7a_applies=True, c_channel=True):
    core = []
    for y in CORE_FOLDS:
        core += [_row('M', y, 0, 'core'), _row('P', y, 0, 'core')]
    t4 = [_t4(tr, te, arm) for tr, te in cfg.F6_PAIRS for arm in cfg.F6_ARMS]
    s16 = [_row('M', 2016, k, '2016') for k in (1, 2, 3, 4)] + [_row('P', 2016, k, '2016') for k in (1, 2, 3, 4)] + [_row(m, 2016, 0, '2016') for m in ('A1', 'A2', 'A3', 'A4')]
    if c7a_applies:
        s16 += [_row('M-C7', 2016, 0, '2016'), _row('M-C7', 2018, 0, '2016')]
    else:
        for t in t4:
            t.update({'set': '2016', 'tail': None})
        s16 += t4
    s16 += [_row(m, 2016, 0, '2016') for m in ('A3BL', 'A3OFF', 'PIX-R')]
    s19 = [_row('M', 2019, k, '2019') for k in (1, 2)] + [_row('P', 2019, 1, '2019')] + [_row(m, 2019, 0, '2019') for m in ('A1', 'A2', 'A3', 'PIX-R')]
    committed = core + s16 + s19
    tail = [dict(_row('A4', 2019, 0, 'tail'), tail='T1')]
    if c7a_applies:
        tail.append(dict(_row('M-C7', 2019, 0, 'tail'), tail='T2'))
    tail += [dict(_row('M', 2019, k, 'tail'), tail='T3') for k in (3, 4)]
    tail.append(dict(_row('P', 2019, 2, 'tail'), tail='T3'))
    if c7a_applies:
        tail += [dict(t, tail='T4') for t in t4]
    for lab, lr in LR_SENSITIVITY:
        tail.append(dict(_row('A3', 2016, 0, 'tail', lr=lr, lr_label=lab), tail='T5'))
    tail += [dict(_row('PIX-R', y, 0, 'tail'), tail='T6') for y in (2015, 2017, 2018)]
    tail.append(dict(_p18(), tail='T7'))
    for t in tail:
        t['committed'] = False
        t['priority'] = 9
        t['design_label'] = 'X'
    drops = {n: d for d, _, names in DROP_ORDER for n in names}
    for t in committed:
        t['never_dropped'] = t['name'] in NEVER_DROPPED
        t['drop'] = drops.get(t['name'])
    out = _apply_c_channel(committed + tail, c_channel)
    names = [t['name'] for t in out]
    if len(set(names)) != len(names):
        raise AssertionError('duplicate training names')
    return out

def design_counts(rows):
    c = [t for t in rows if t['committed']]
    return {'committed': len(c), 'at_sm': sum((1 for t in c if t['steps_key'] == 'steps_sm')), 'e1_6000': sum((1 for t in c if t['net'] in ('baseline', 's18') and t['steps_key'] == 'steps_main')), 'e2_6000': sum((1 for t in c if t['net'] == 'advanced' and t['steps_key'] == 'steps_main')), 'pix': sum((1 for t in c if t['net'] == 'pixr')), 'tail': sum((1 for t in rows if not t['committed'])), 'configurations': len(set((t['model'] for t in c)))}

def plan_trainings(plan):
    if plan.get('schema') == SCHEMA3:
        return design_trainings(plan['c7a_applies'], plan['c_channel'])
    return trainings_for(plan['priorities'], plan['c7a_applies'])

def drop_items(rows):
    have = set((t['name'] for t in rows))
    return [(d, txt, [n for n in names if n in have]) for d, txt, names in DROP_ORDER]

def lane_of(t):
    if t.get('kind') == 'f6' or int(t['held_out_year']) == 2016:
        return '1'
    return '2'

def design_lanes(rows, early8=EARLY_8):
    out = {'1': [], '2': []}
    last = {'1': [], '2': []}
    by = {t['name']: t for t in rows}
    for t in rows:
        if t['name'] in early8:
            continue
        k = lane_of(t)
        (last if t.get('tail') == 'T2' else out)[k].append(t['name'])
    for k in LANES:
        out[k] += last[k]
    out['early_8'] = {k: [n for n in early8 if n in by and lane_of(by[n]) == k] for k in LANES}
    return out

def validate_plan3(plan, test_values=False):
    p = dict(plan)
    deps = list(p.get('departures') or [])
    if p.get('design') != DESIGN:
        raise PlanError('design %r, expected %r' % (p.get('design'), DESIGN))
    dev = p.get('device')
    if dev not in cfg.DEVICES:
        raise PlanError('device %r not in %s' % (dev, cfg.DEVICES))
    try:
        w = float(p.get('width'))
        sm = int(p.get('steps_main'))
        ssm = int(p.get('steps_sm'))
        th = int(p.get('threads'))
    except (TypeError, ValueError):
        raise PlanError('width, steps_main, steps_sm and threads')
    if not test_values:
        if w != 1.0:
            raise PlanError('width %r: w = 1' % (w,))
        if sm != cfg.STEPS_DESIGN:
            raise PlanError('steps_main %d, the design fixes 6000' % sm)
        if ssm != STEPS_SM:
            raise PlanError('steps_sm %d is not 16000' % ssm)
        if p.get('test_values'):
            raise PlanError('test_values present without --test-values')
    elif sm < 1 or ssm < 1:
        raise PlanError('steps must be at least 1')
    if th < 1:
        raise PlanError('threads %d' % th)
    p.update({'width': w, 'steps_main': sm, 'steps_sm': ssm, 'threads': th, 'steps_f6': int(p.get('steps_f6', cfg.STEPS_F6))})
    if not isinstance(p.get('c7a_applies'), bool):
        raise PlanError('c7a_applies must be true or false')
    cc = p.get('c_channel', True)
    if cc not in C_CHANNEL_FORMS:
        raise PlanError('c_channel %r not in %s' % (cc, C_CHANNEL_FORMS))
    if cc is not True and (not any(('PD-23' in str(d) for d in deps))):
        raise PlanError('c_channel %r without a departure that names PD-23' % (cc,))
    p['c_channel'] = cc
    f19 = p.get('f19', False)
    if f19 is not False:
        raise PlanError('f19 must be false')
    if not test_values and (not any(('F-19' in str(d) for d in deps))):
        raise PlanError('f19 false without a recorded departure')
    p['f19'] = False
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
        if not (isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v > 0)):
            raise PlanError('step_seconds[%s] %r' % (k, v))
    p['step_seconds'] = {k: float(v) for k, v in ss.items()}
    eq = p.get('equivalence')
    if not test_values and (not isinstance(eq, dict) or not _is_int(eq.get('nondeterministic_count'))):
        raise PlanError('%s (equivalence %s)' % (EQUIV_REQUIRED, 'absent' if eq is None else 'without an integer nondeterministic_count'))
    if isinstance(eq, dict) and int(eq.get('nondeterministic_count', 0) or 0) > 0:
        raise PlanError('the equivalence test saw warnings of an operation without a deterministic implementation')
    if 'budget' in p:
        raise PlanError("budget is the study design's, which no longer decides: the plan holds the guard")
    gd = p.get('guard')
    if not isinstance(gd, dict):
        raise PlanError('guard is not an object')
    for k in ('units', 'max_age_s'):
        v = gd.get(k)
        if not (isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v > 0)):
            raise PlanError('guard.%s %r' % (k, v))
    if not test_values and (float(gd['units']) != GUARD_UNITS or float(gd['max_age_s']) != GUARD_MAX_AGE_S):
        raise PlanError('guard units and max_age_s must be 700 and 600')
    p['guard'] = {k: float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) else v for k, v in gd.items()}
    if not isinstance(p.get('budget_state'), str):
        raise PlanError("budget_state (the driver's spend file) absent")
    names = [t['name'] for t in design_trainings(p['c7a_applies'], cc)]
    if p.get('order') is not None and list(p['order']) != names:
        raise PlanError("order differs from the design's order")
    lane = p.get('lane')
    ln = p.get('lane_names')
    if lane is not None or ln is not None:
        if lane not in LANES:
            raise PlanError('lane %r is not 1 or 2' % (lane,))
        if not (isinstance(ln, list) and ln and (len(set(ln)) == len(ln)) and (set(ln) <= set(names))):
            raise PlanError("lane_names is not a list of the design's trainings, each once")
        p['lane_names'] = list(ln)
    if not isinstance(p.get('profile', False), bool):
        raise PlanError('profile is not true or false')
    p['profile'] = bool(p.get('profile', False))
    ds = p.get('day_sets', 'all')
    if ds not in DAY_SETS:
        raise PlanError('day_sets %r not in %s' % (ds, DAY_SETS))
    p['day_sets'] = ds
    tv = dict(p.get('test_values') or {})
    for k in tv:
        if k not in cfg.TEST_VALUE_KEYS:
            raise PlanError('test_values: unknown key %r' % k)
    p['test_values'] = tv
    p['departures'] = deps
    p['priorities'] = [1]
    p['budget_hours'] = None
    p['hold_back_p1'] = False
    return p

def write_plan_runtime3(path, device, threads, c7a_applies, steps_sm, guard, budget_state, step_seconds=None, reader='device', departures=None, measured=None, made_by='the driver', test_values=None, equivalence=None, width=1.0, steps_main=cfg.STEPS_DESIGN, c_channel=True, day_sets='all', lane=None, lane_names=None, profile=False):
    d = {'schema': SCHEMA3, 'design': DESIGN, 'run': 'emul', 'made': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'made_by': made_by, 'device': device, 'width': float(width), 'steps_main': int(steps_main), 'steps_sm': int(steps_sm), 'steps_f6': cfg.STEPS_F6, 'threads': int(threads), 'c7a_applies': bool(c7a_applies), 'c_channel': c_channel, 'day_sets': day_sets, 'f19': False, 'step_seconds': dict(step_seconds or {}), 'reader': reader, 'guard': dict(guard), 'budget_state': str(budget_state), 'departures': list(departures or []), 'measured': measured or {}, 'profile': bool(profile)}
    if lane is not None:
        d['lane'] = str(lane)
        d['lane_names'] = list(lane_names or [])
    if equivalence is not None:
        d['equivalence'] = dict(equivalence)
    if test_values:
        d['test_values'] = dict(test_values)
    d['order'] = [t['name'] for t in design_trainings(d['c7a_applies'], c_channel)]
    validate_plan(d, test_values=bool(test_values))
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)
    return d
