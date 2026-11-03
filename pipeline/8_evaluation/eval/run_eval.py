"""The entry of the evaluation: per fold it reads the target, the classes and each model's sealed predictions and computes the metrics by stratum, the diagnostics, the per-day sums for the comparisons and the figures' data; then the comparisons, the exploratory analyses and the five-fold summaries, all written to a new sealed folder."""
import argparse
import datetime
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np
from eval import an02, an05, d8in, gbt_es, labels
from eval import family as FM
from eval import io as EIO
from eval import metrics as MT
SCHEMA = 'tier-a-eval-1'
FIG_MODELS = ('M', 'P')
F5_FOLD = 2016
ENS_FOLD = 2016
ENS_N = 5
SEED_FOLDS = (2016, 2019)
S_M_MODELS = ('M', 'A4', 'M-C7')
LADDER = ('P', 'A1', 'A2', 'A3', 'M')
EXPLORATORY_PAIRS = (('M-C7 stand-in', 'M-C7', 'M', (2016, 2018, 2019), "RMSE(M-C7) - RMSE(M), ppb (the stand-in's cost; the study design row 11, T2)"), ('A3OFF C input', 'A3OFF', 'A3', (2016,), 'RMSE(A3OFF) - RMSE(A3), ppb (the C channel; the study design row 13)'), ('PIX-R context', 'PIX-R', 'M', (2016, 2019, 2015, 2017, 2018), 'RMSE(PIX-R) - RMSE(M), ppb (spatial context; the study design row 14, T6)'), ('CLIM-EM reference', 'CLIM-EM', 'M', (2015, 2016, 2017, 2018, 2019), 'RMSE(CLIM-EM) - RMSE(M), ppb (the second reference)'), ('A4 on 2019', 'A4', 'M', (2019,), 'RMSE(A4) - RMSE(M), ppb (T1, the budget at the direct target)'))
STRATA_MODELS = ('M', 'P', 'GBT', 'CLIM-WD')
STRATA_REFS = ('P', 'GBT', 'CLIM-WD')
STRATA_LABEL = {'season': "a season's days, all masked cells", 'urban': "D4's urban class (all test days)", 'coast': "D5's coast/edge class (all test days)", 'season_x_urban': 'D4, season x urban class', 'tail': "D1, the tail: CMAQ's top decile of the pooled cell-days"}
NO_FIG_FILES = ('GBT-ES',)
GBT_FIT_MODELS = ('GBT', 'GBT-RIDGE', 'GBT-ES')
GBT_FIT_DIFFERS = "GBT's lockbox predictions come from another fit: not read until a manual decision is needed"
GBT_FIT_ABSENT = "GBT's fit record ('s gbt_fit) is absent from a DONE file: not read until a manual decision is needed"

def parse(argv):
    ap = argparse.ArgumentParser(prog='run_eval.py')
    ap.add_argument('--config', required=True)
    ap.add_argument('--emul', required=True)
    ap.add_argument('--config-dir', required=True)
    ap.add_argument('--arrays-dir', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--opening', default=None)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--folds', default=','.join((str(y) for y in EIO.FOLDS)))
    ap.add_argument('--test-resamples', type=int, default=None)
    o = ap.parse_args(argv)
    o.folds = [int(s) for s in o.folds.split(',') if s]
    for y in o.folds:
        if y not in EIO.FOLDS:
            ap.error('unknown fold %r' % y)
    if o.run_id is None:
        now = datetime.datetime.now(datetime.timezone.utc)
        o.run_id = 'eval_' + now.strftime('%Y%m%d_%H%M%S')
    if '/' in o.run_id or '\\' in o.run_id or '..' in o.run_id:
        ap.error('--run-id is a plain name')
    if o.test_resamples is not None and o.test_resamples < 10:
        ap.error('--test-resamples below 10')
    return o

def key_of(row):
    return (row['model'], row['seed'])

def s0_of(rows, model):
    for r in rows:
        if r['model'] == model and r['seed'] in (0, None):
            return r
    return None

class Ctx:

    def __init__(self, sources, fold, test_days):
        self.fold = fold
        self.days = test_days
        self.dates = [d for _y, _i, d in test_days]
        self.n_days = len(test_days)
        self.n_cells = len(sources.cells)

def run(o):
    t0 = time.monotonic()
    conf = EIO.load_config(o.config)
    pl = EIO.places(conf)
    sources = EIO.Sources(o.config_dir, o.arrays_dir, conf)
    pd17 = EIO.gate_pd17(o.emul, pl, sources)
    EIO.say('GATE read-only check: %s PASS (sha256 %s)' % (pd17['file'], pd17['sha256']))
    models, models_sha = EIO.read_models(o.models)
    opening = EIO.read_opening(o.opening)
    eval_dir = os.path.join(o.emul, *pl['eval_sealed'].split('/'))
    out_dir = os.path.join(eval_dir, o.run_id)
    if os.path.exists(out_dir):
        raise EIO.Refused('the eval folder %s exists: a run writes a new one' % o.run_id)
    precheck(o, pl, models, opening)
    if opening:
        EIO.claim_opening(eval_dir, opening, o.run_id)
        EIO.say('LOCKBOX opening claimed by %s (decided_by %s)' % (o.run_id, opening['decided_by']))
    out = EIO.Sealed(out_dir)
    EIO.say('EVAL folder %s/%s (new)' % (pl['eval_sealed'], o.run_id))
    c7a = bool((conf.get('c7a_applies') or {}).get('value'))
    b_fam = o.test_resamples or FM.B_FAMILY
    b_exp = o.test_resamples or FM.B_EXPLORATORY
    R = Results()
    R.notrun.append({'item': 'strata by climate region', 'reason': "NOT RUN: EPA's 12US1 region masks are not held"})
    for y in o.folds:
        rows = [r for r in models if r['fold'] == y]
        if not rows:
            EIO.say('fold %d: no model in the table; skipped' % y)
            continue
        eval_fold(o, conf, pl, sources, y, rows, R, opening, c7a, b_exp)
    write_all(out, R, o, conf, sources, models, models_sha, opening, c7a, b_fam, b_exp, pd17, t0)
    EIO.say('EVAL done in %.0f s; %d files sealed' % (time.monotonic() - t0, len(out.rows) + 1))
    return 0

def precheck(o, pl, models, opening):
    missing = []
    for r in models:
        if r['fold'] not in o.folds or r['model'] == 'GBT-ES':
            continue
        if r['kind'] == 'network':
            p = os.path.join(o.emul, *pl['ckpt'].split('/'), r['name'], 'DONE.json')
        else:
            p = os.path.join(o.emul, *pl['baselines_sealed'].split('/'), r['name'], 'fold%d' % r['fold'], 'DONE.json')
        if not os.path.isfile(p):
            missing.append('%s fold %d' % (r['name'], r['fold']))
        if opening and r['model'] in FM.FAMILY_MODELS and (r['seed'] in (0, None)):
            if r['kind'] == 'network':
                q = os.path.join(o.emul, *pl['lockbox_sealed'].split('/'), r['name'], 'LOCKBOX_DONE.json')
            else:
                q = os.path.join(o.emul, *pl['lockbox_sealed'].split('/'), 'baselines', r['name'], 'fold%d' % r['fold'], 'LOCKBOX_DONE.json')
            if not os.path.isfile(q):
                missing.append('%s fold %d LOCKBOX' % (r['name'], r['fold']))
    if missing:
        raise EIO.Refused('%d DONE files the run needs are absent (first: %s): nothing was read' % (len(missing), ', '.join(missing[:5])))

class Results:

    def __init__(self):
        self.metrics = []
        self.diag = []
        self.exceed = []
        self.decile = []
        self.bands = []
        self.acf = []
        self.dtab = []
        self.f12 = []
        self.an02 = []
        self.an02_maps = []
        self.an05 = {}
        self.gbt_es = []
        self.years = {}
        self.years_lb = {}
        self.seeds = {}
        self.acc_med = {}
        self.sse = {}
        self.block_sse = {}
        self.explor = []
        self.figs = []
        self.city = {}
        self.read = []
        self.notrun = []
        self.n_days = {}
        self.n_cells = {}
        self.n_block = {}
        self.strata = {}
        self.season_idx = {}
        self.d11 = []
        self.d8_ranges = []
        self.gbt_fit = []
        self.mc7 = []
        self.classes = []

    def n_cells_of(self, y, block):
        return self.n_block[y] if block else self.n_cells[y]

def _tag(row):
    return {'model': row['model'], 'seed': row['seed'], 'name': row['name'], 'kind': row['kind']}

def read_cv(o, pl, sources, row, y, R):
    if row['kind'] == 'network':
        e, rec = EIO.read_network_cv(o.emul, pl, row['name'], y, sources)
        curve = rec.pop('d11_curve', None)
        tl = rec.get('train_log') or {}
        R.d11.append(dict(_tag(row), fold=y, status=tl.get('status'), reason=tl.get('reason'), log_file=tl.get('file'), log_sha256=tl.get('sha256'), evaluations=curve or []))
        return (e, rec)
    return EIO.read_baseline_cv(o.emul, pl, row['name'], y, sources)

def mc7_note(R, y, r, rec):
    chk = rec['mc7_check']
    n = int(chk.get('n_cell_days', 0))
    EIO.say('fold %d: AN-02 check of %s (%s): %s, %d cell-days flagged' % (y, r['model'], rec['part'], chk['status'], n))
    R.mc7.append({'fold': y, 'model': r['model'], 'seed': r['seed'], 'name': r['name'], 'part': rec['part'], 'status': chk['status'], 'n_cell_days': n})

def gbt_fit_row(R, r, y, rec):
    cv = [x for x in R.read if x.get('kind') == 'baseline' and x.get('name') == r['name'] and (x.get('fold') == y) and (x.get('part') is None)]
    a = cv[-1].get('gbt_fit') if cv else None
    b = rec.get('gbt_fit')
    same = None if a is None or b is None else bool(a == b)
    return {'fold': y, 'model': r['model'], 'baseline': r['name'], 'gbt_fit_same': same, 'rule': "'s record equal in the TEST_CV DONE.json and the LOCKBOX_DONE.json; null: absent from either"}

def eval_fold(o, conf, pl, sources, y, rows, R, opening, c7a, b_exp):
    t0 = time.monotonic()
    test = sources.test_cv(y)
    cx = Ctx(sources, y, test)
    R.n_days[y] = cx.n_days
    obs = sources.target(y, test, 'predict_cv')
    EIO.say('fold %d: target on %d TEST_CV days x %d cells read' % (y, cx.n_days, cx.n_cells))
    R.n_cells[y] = cx.n_cells
    urban = MT.urban_class(sources.static(y, 'L-3'))
    ctests = {}
    coast = MT.coast_class(sources.mask, sources.static_grid(y, 'L-2'), tests=ctests)
    R.classes.append({'fold': y, 'urban_cells': {u: int(np.sum(urban == k)) for k, u in enumerate(MT.URBAN)}, 'coast_edge_cells': ctests, 'rule': 'class_<name>: the exclusive classes (edge, coast, border, interior, the first that applies; READING); edge, coast, border: the cells meeting each test on its own (a cell may be in more than one)'})
    R.season_idx[y] = MT.season_index(cx.dates)
    R.strata[y] = {}
    top = MT.top_decile_mask(obs)
    if 'M-8' in sources.by_name:
        pblh = sources.channel_rows(y, test, 'M-8', 'predict_cv')
    else:
        pblh = None
        R.notrun.append({'fold': y, 'item': 'D6', 'reason': 'S3_CHANNELS.json has no M-8'})
    tmax, tdom, n_train = sources.target_max(y)
    EIO.say('fold %d: classes, M-8 and the training maxima over %d TRAIN days read' % (y, n_train))
    corners, widx = MT.windows(sources.mask)
    spec_ok = len(corners) >= MT.MIN_WINDOWS
    cmaq_bands = MT.band_power(*MT.spectrum_power(obs, widx)) if spec_ok else None
    if not spec_ok:
        R.notrun.append({'fold': y, 'item': 'D3 spectra and band ratios', 'reason': '%d windows wholly inside the mask, %d needed' % (len(corners), MT.MIN_WINDOWS)})
    w_exp, w_rec = FM.weights_for(y, cx.n_days, FM.BLOCK, b_exp, 'cv')
    tf, su = MT.f12_masks(cx.dates)
    co = MT.f12_map(obs, cx.dates)
    groups = {u: urban == k for k, u in enumerate(MT.URBAN)}
    ys = FM.YearSums(y, cx.n_cells, cx.n_days)
    R.sse[y] = {}
    R.seeds[y] = {}
    blk = block_cells(sources)
    R.n_block[y] = None if blk is None else int(blk.size)
    R.block_sse[y] = {}
    an_groups = {}
    for r in rows:
        if r['model'] in ('CLIM-C', 'CLIM-WD', 'MET-REG', 'GBT', 'GBT-RIDGE', 'GBT-ES', 'CLIM-EM'):
            continue
        src, ch = an02.channels_of(sources, r['model'])
        an_groups.setdefault((src, tuple(ch)), []).append(r)
    an_flags = {}
    for (src, ch), members in sorted(an_groups.items()):
        mn, mx, nd = an02.ranges(sources, y, src, list(ch))
        cnt = {}
        fl = an02.flags(sources, y, test, 'predict_cv', src, list(ch), mn, mx, counts=cnt)
        an_flags[src, ch] = fl
        tag = 'an02_fold%d_%s_%s' % (y, src, '-'.join((c.replace('-', '') for c in ch)))
        R.an02_maps.append((tag + '_min.npy', mn))
        R.an02_maps.append((tag + '_max.npy', mx))
        sh = an02.share(fl, urban, MT.URBAN)
        for r in members:
            rec = {'fold': y, 'model': r['model'], 'seed': r['seed'], 'name': r['name'], 'source': src, 'channels': list(ch), 'train_days': nd, 'maps': [tag + '_min.npy', tag + '_max.npy'], 'part': 'TEST_CV', 'outside_by_channel': cnt}
            rec.update(sh)
            if r['model'] in an02.C7_MODELS:
                rec['mc7_check'] = an02.mc7_check(fl, cx.dates, sources.cells)
                mc7_note(R, y, r, rec)
            R.an02.append(rec)
        EIO.say('fold %d: AN-02 ranges of %s (%d channels) over %d TRAIN days; %d trainings labelled' % (y, src, len(ch), nd, len(members)))
    d8_groups, d8_key = ({}, {})
    for r in rows:
        if r['kind'] != 'network':
            continue
        src, held, miss = d8in.channels_of(sources, r['model'])
        k = (src, tuple(held), tuple(miss))
        d8_groups.setdefault(k, []).append(r)
        d8_key[r['model'], r['seed'], r['name']] = k
    d8_flags = {}
    for k, members in sorted(d8_groups.items()):
        src, ch, miss = k
        lo, hi, nd = d8in.ranges(sources, y, src, list(ch))
        cnt = {}
        fl = d8in.flags(sources, y, test, 'predict_cv', src, list(ch), lo, hi, counts=cnt)
        d8_flags[k] = (fl, {'source': src, 'channels_tested': list(ch), 'channels_not_held': list(miss), 'train_days': nd, 'outside_by_channel': cnt})
        R.d8_ranges.append({'fold': y, 'source': src, 'channels': list(ch), 'min': lo, 'max': hi, 'train_days': nd, 'channels_not_held': list(miss), 'trainings': [m['name'] for m in members]})
        EIO.say('fold %d: D8 input ranges of %s (%d channels, %d not held) over %d TRAIN days; %d trainings labelled' % (y, src, len(ch), len(miss), nd, len(members)))
    ref_row = s0_of(rows, 'CLIM-WD')
    ref = ref_rec = None
    if ref_row is not None:
        ref, ref_rec = read_cv(o, pl, sources, ref_row, y, R)
        R.read.append(ref_rec)
        ys.add('CLIM-WD', ref, obs, keep_diff=True)
    else:
        R.notrun.append({'fold': y, 'item': 'SS, ACC and the family', 'reason': 'CLIM-WD has no row on this fold'})
    es_print = None
    if any((r['model'] == 'GBT-ES' for r in rows)):
        try:
            curve, crow = EIO.read_val_curve(o.emul, pl, y)
            rule = gbt_es.rule(curve)
            rule.update({'fold': y, 'curve_sha256': crow['sha256']})
        except (EIO.Refused, gbt_es.CurveError) as e:
            rule = {'fold': y, 'status': 'NOT RUN', 'reason': str(e), 'printed': False}
        R.gbt_es.append(rule)
        es_print = bool(rule.get('printed'))
        EIO.say("fold %d: GBT-ES's printing rule applied by code (the result sealed)" % y)
    ens = {}
    city = sources.city_blocks_flat() if y == F5_FOLD else None
    for r in sorted(rows, key=lambda r: (r['model'], -1 if r['seed'] is None else r['seed'])):
        if r['model'] == 'GBT-ES' and (not es_print):
            R.notrun.append({'fold': y, 'item': 'GBT-ES', 'reason': 'not printed by its rule; not read'})
            continue
        if r is ref_row:
            e, rec = (ref, ref_rec)
        else:
            e, rec = read_cv(o, pl, sources, r, y, R)
            R.read.append(rec)
        one_model(cx, r, rec, e, obs, ref, urban, coast, pblh, tmax, tdom, an_flags, an_groups, widx if spec_ok else None, cmaq_bands, w_exp, tf, su, co, groups, R, c7a, blk, d8_flags.get(d8_key.get((r['model'], r['seed'], r['name']))), top)
        if r['seed'] in (0, None) and r['model'] in FM.FAMILY_MODELS and (r['model'] != 'CLIM-WD'):
            ys.add(r['model'], e, obs, keep_diff=r['model'] == 'M')
        if y == ENS_FOLD and r['model'] in FIG_MODELS and (r['seed'] is not None):
            s = ens.setdefault(r['model'], [np.zeros_like(obs), 0])
            s[0] += e
            s[1] += 1
        if r['model'] in FIG_MODELS and r['seed'] in (0, None):
            figures(cx, r, e, obs, R, city, sources)
        del e
    for m, (s, n) in sorted(ens.items()):
        if n == ENS_N:
            er = MT.basic(s / n, obs, ref)
            er.update({'model': '%s ens%d' % (m, n), 'seed': None, 'fold': y, 'stratum_kind': 'all', 'stratum': 'all', 'status': 'RUN', 'label': 'exploratory (the %d-member mean)' % n, 'name': '', 'kind': 'ensemble'})
            R.metrics.append(er)
            R.sse[y]['%s ens%d' % (m, n), None] = MT.sse_per_day(s / n, obs)
        else:
            R.notrun.append({'fold': y, 'item': '%s ens5' % m, 'reason': '%d seeds on the table, 5 needed' % n})
    if ref is not None:
        R.years[y] = ys
    if opening:
        lockbox_fold(o, pl, sources, y, rows, R, opening, an_groups, urban)
    EIO.say('fold %d: %d models evaluated in %.0f s' % (y, len(rows), time.monotonic() - t0))

def block_cells(sources):
    from train import cfg
    from train import data as TD
    cc = sources.city_cells()
    if not cc or cfg.BLOCK_CITY not in cc.get('cells', {}):
        return None
    rc = cc['cells'][cfg.BLOCK_CITY]
    b = TD.block_mask(sources.grid, int(rc['row']), int(rc['col']))
    flat = set(np.flatnonzero(b.ravel()).tolist())
    pos = [k for k, c in enumerate(sources.cells) if int(c) in flat]
    return np.array(pos, dtype=np.int64) if pos else None

def one_model(cx, r, rec, e, obs, ref, urban, coast, pblh, tmax, tdom, an_flags, an_groups, widx, cmaq_bands, w_exp, tf, su, co, groups, R, c7a, blk, d8n=None, top=None):
    y = cx.fold
    tag = _tag(r)
    for row in MT.strata_rows(e, obs, ref, cx.dates, urban, coast):
        row.update(tag)
        row['fold'] = y
        R.metrics.append(row)
    sse = MT.sse_per_day(e, obs)
    R.sse[y][key_of(r)] = sse
    if r['model'] in STRATA_MODELS and r['seed'] in (0, None):
        R.strata[y][r['model']] = MT.class_day_sums(e, obs, urban, coast, top)
    if blk is not None:
        d = e[:, blk] - obs[:, blk]
        R.block_sse[y][key_of(r)] = np.einsum('ij,ij->i', d, d)
    rmse = float(np.sqrt(sse.sum() / (cx.n_days * cx.n_cells)))
    d = dict(tag, fold=y, RMSE=rmse, MAPRMSE=MT.maprmse(e, obs), R_space=MT.r_space(e, obs), R_time=MT.summary(MT.r_time_cells(e, obs)), spread=MT.spread(e, obs), above30=MT.above_level(e, obs), own_top_decile=MT.own_top_decile(e, obs), D10_clipped_masked=rec.get('clipped_masked_total'), D11_still_improving=MT.still_improving(rec.get('result')) if r['kind'] == 'network' else None)
    if ref is not None:
        acc = MT.acc_cells(e, obs, ref)
        d['ACC'] = MT.summary(acc)
        d['SS'] = MT.basic(e, obs, ref)['SS']
        d['share_SS_le_0'] = MT.share_le0(MT.skill_cells(e, obs, ref))
        d['share_ACC_le_0'] = MT.share_le0(acc)
        d['murphy_eq15'] = MT.murphy_terms(e, obs, ref)
        if r['model'] == 'M' and r['seed'] == 0:
            R.acc_med[y] = d['ACC']['median']
    nz, nz_map = MT.nearzero(e, obs, coast)
    d['D2_near_zero'] = nz
    if r['model'] not in NO_FIG_FILES:
        R.figs.append(('D2_nearzero_fold%d_%s_s%s.npy' % (y, r['model'], r['seed']), nz_map))
    if r['model'] in ('CLIM-C', 'CLIM-WD', 'MET-REG', 'GBT', 'GBT-RIDGE', 'GBT-ES', 'CLIM-EM'):
        flagged = None
    else:
        src, ch = (None, None)
        for (s, c), members in an_groups.items():
            if any((m is r for m in members)):
                src, ch = (s, c)
        flagged = an_flags.get((src, ch))
    fl_n, info_n = d8n if d8n is not None else (None, None)
    d['D8'] = MT.d8_rows(e, obs, tmax, tdom, flagged, fl_n, info_n)
    if r['kind'] != 'network':
        d['D8']['inputs_outside_norm'] = {'status': 'NOT RUN', 'reason': 'a baseline has no channel set (networks only; READING)'}
    if r['kind'] == 'network':
        inf = rec.get('inference') or {}
        d['G02_inference'] = inf
        d['G02_inference_seconds'] = inf.get('seconds_test_cv')
    R.diag.append(d)
    for lv in MT.LEVELS:
        x = MT.exceedance(e, obs, lv)
        x.update(tag)
        x['fold'] = y
        R.exceed.append(x)
    dec = MT.deciles(e, obs)
    for x in dec['rows']:
        x.update(tag)
        x['fold'] = y
        R.decile.append(x)
    if widx is not None:
        mb = MT.band_power(*MT.spectrum_power(e, widx))
        for x in MT.band_ratios(mb, cmaq_bands):
            x.update(tag)
            x['fold'] = y
            x['windows'] = int(widx.shape[0])
            x['expected'] = 'below 1 for all; lower for the MSE networks than for P at equal network (A1 against P) (the study design D3)'
            R.bands.append(x)
    si = MT.season_index(cx.dates)
    for k, s in enumerate(('year',) + MT.SEASONS):
        sel = None if s == 'year' else si == k - 1
        a = MT.residual_acf(e, obs, cx.dates, sel)
        for x in a['lags']:
            x.update(tag)
            x.update({'fold': y, 'season': s, 'first_below_0_2': a['first_below_0_2'], 'n_days': a['n_days']})
            R.acf.append(x)
    dt = MT.d_rows(e, obs, cx.dates, urban, coast, pblh)
    for k in ('D4', 'D5', 'D6', 'D7'):
        rows = dt[k] if isinstance(dt[k], list) else [dt[k]]
        for x in rows:
            x = dict(x)
            x.update(tag)
            x.update({'fold': y, 'table': k})
            R.dtab.append(x)
    if r['model'] not in NO_FIG_FILES:
        R.figs.append(('D7_maps_fold%d_%s_s%s.npy' % (y, r['model'], r['seed']), MT.d7_maps(e, obs, cx.dates)))
    cm = MT.f12_map(e, cx.dates)
    pt = {'overall': MT.agreement(cm, co)}
    for u, sel in groups.items():
        pt[u] = MT.agreement(cm[sel], co[sel])
    reps = FM.f12_boot(w_exp, e, obs, tf, su, groups)
    words = labels.wording(r['model'], c7a)
    for g in pt:
        for s in ('R', 'slope', 'RMSE'):
            lo, hi, nok = FM.interval_nan(reps[g][s])
            R.f12.append(dict(tag, fold=y, group=g, stat=s, point=pt[g][s], lo=lo, hi=hi, finite_resamples=nok, n_resamples=int(w_exp.shape[0]), words=words['words'], label='exploratory'))
    if r['model'] in an05.ARMS and r['seed'] in (0, None) and (y in an05.FOLDS):
        sh = None
        for x in R.an02:
            if x['fold'] == y and x['model'] == r['model'] and (x['seed'] == r['seed']):
                sh = x['share']
        R.an05.setdefault(y, {})[r['model']] = {'share': sh, 'slope': pt['overall']['slope'], 'n_cell_days': int(cx.n_days * cx.n_cells)}
    if r['model'] in ('M', 'P') and r['seed'] is not None:
        R.seeds[y][key_of(r)] = {'RMSE': rmse, 'MAPRMSE': d['MAPRMSE'], 'SS': d.get('SS')}

def figures(cx, r, e, obs, R, city, sources):
    y = cx.fold
    h, n_out = MT.hist2d(e, obs)
    R.figs.append(('F3_hist_fold%d_%s.npy' % (y, r['model']), h))
    R.figs.append(('F4_mean_fold%d_%s.npy' % (y, r['model']), np.vstack([obs.mean(axis=0), e.mean(axis=0), e.mean(axis=0) - obs.mean(axis=0)]).astype(np.float32)))
    R.figs.append(('F5_domain_fold%d_%s.npy' % (y, r['model']), np.vstack([obs.mean(axis=1), e.mean(axis=1)])))
    R.diag.append({'model': r['model'], 'seed': r['seed'], 'fold': y, 'name': r['name'], 'kind': r['kind'], 'F3_outside_0_100': n_out, 'figure_rows': 'F3 rows CMAQ, columns the model, 0.5 ppb bins; F4 rows CMAQ mean, model mean, model - CMAQ; F5 rows CMAQ, model'})
    if city:
        rows, series = MT.city_series(e, obs, city, sources.cells)
        for c, x in rows.items():
            x = dict(x)
            x.update({'model': r['model'], 'seed': r['seed'], 'fold': y, 'city': c})
            R.city.setdefault('rows', []).append(x)
        for c, (se, so) in series.items():
            R.figs.append(('F5_city_fold%d_%s_%s.npy' % (y, r['model'], ''.join((ch for ch in c if ch.isalnum()))), np.vstack([so, se])))

def lockbox_fold(o, pl, sources, y, rows, R, opening, an_groups, urban):
    lb = sources.lockbox_days(y, opening)
    if not lb:
        R.notrun.append({'fold': y, 'item': 'lockbox', 'reason': 'no lockbox day in this fold'})
        return
    obs = sources.target(y, lb, 'predict_lockbox', opening)
    ys = FM.YearSums(y, len(sources.cells), len(lb))
    for m in FM.FAMILY_MODELS:
        r = s0_of(rows, m)
        if r is None:
            continue
        e, rec = EIO.read_lockbox(o.emul, pl, r['kind'], r['name'], y, sources, opening)
        R.read.append(rec)
        if r['kind'] == 'baseline' and r['model'] in GBT_FIT_MODELS:
            R.gbt_fit.append(gbt_fit_row(R, r, y, rec))
        ys.add(m, e, obs, keep_diff=m in ('M', 'CLIM-WD'))
    if ys.has(FM.FAMILY_MODELS):
        R.years_lb[y] = ys
    for (src, ch), members in sorted(an_groups.items()):
        mn = mx = None
        for name, arr in R.an02_maps:
            tag = 'an02_fold%d_%s_%s' % (y, src, '-'.join((c.replace('-', '') for c in ch)))
            if name == tag + '_min.npy':
                mn = arr
            if name == tag + '_max.npy':
                mx = arr
        fl = an02.flags(sources, y, lb, 'predict_lockbox', src, list(ch), mn, mx, opening)
        sh = an02.share(fl, urban, MT.URBAN)
        for r in members:
            rec = {'fold': y, 'model': r['model'], 'seed': r['seed'], 'name': r['name'], 'source': src, 'channels': list(ch), 'part': 'LOCKBOX (apart)'}
            rec.update(sh)
            if r['model'] in an02.C7_MODELS:
                rec['mc7_check'] = an02.mc7_check(fl, [d for _a, _b, d in lb], sources.cells)
                mc7_note(R, y, r, rec)
            R.an02.append(rec)
    EIO.say('fold %d: lockbox read on %d days under the opening' % (y, len(lb)))

def seed_ranges(R, y, contrast):
    s = R.seeds.get(y, {})

    def vals(model, metric):
        return [v[metric] for (m, k), v in sorted(s.items(), key=lambda kv: kv[0][1]) if m == model and v.get(metric) is not None]
    if contrast == 'C-1':
        return {'M': FM.seed_range(vals('M', 'SS'))}
    if contrast == 'C-2':
        return {'M': FM.seed_range(vals('M', 'RMSE'))}
    if contrast == 'C-3':
        return {'M': FM.seed_range(vals('M', 'RMSE')), 'P': FM.seed_range(vals('P', 'RMSE'))}
    return {'M': FM.seed_range(vals('M', 'MAPRMSE'))}

def family_part(R, b_fam, b_exp):
    out = {'definitions': FM.DEFINITIONS, 'margins': FM.MARGINS, 'covers': FM.COVERS, 'seed': FM.SEED, 'm': len(FM.FAMILY)}
    years = [R.years[y] for y in sorted(R.years) if R.years[y].has(FM.FAMILY_MODELS)]
    folds = [ys.year for ys in years]
    out['folds'] = folds
    if folds != list(EIO.FOLDS):
        out['five_fold'] = {'status': 'NOT RUN', 'reason': 'the family needs M, P, GBT and CLIM-WD (s0) on all five folds; folds with all four: %s' % folds}
    else:
        main = FM.family_reps(years, FM.BLOCK, b_fam)
        hm = FM.holm(main['reps'], main['point'])
        accm = [R.acc_med.get(y) for y in folds]
        acc5 = float(np.mean(accm)) if all((a is not None for a in accm)) else None
        hm['C-1']['acc_median_five_fold'] = acc5
        hm['C-1']['holds'] = FM.c1_holds(hm['C-1'], acc5)
        sens = FM.family_reps(years, FM.BLOCK_SENSITIVITY, b_fam)
        hs = FM.holm(sens['reps'], sens['point'])
        flag = FM.sensitivity(hm, hs)
        for c in FM.FAMILY:
            hm[c]['block_14'] = {k: hs[c][k] for k in ('lo', 'hi', 'level', 'reading', 'passes_holm_step')}
            hm[c]['block_length'] = flag[c]
            hm[c]['seed_2016_ranges_quoted'] = seed_ranges(R, 2016, c)
        out['five_fold'] = {'status': 'RUN', 'rows': hm, 'by_year': main['by_year'], 'draws': main['draws'], 'draws_14': sens['draws'], 'count_of_intervals_read': 2 * len(FM.FAMILY)}
    per_fold = []
    for ys in years:
        pt = FM.family_point(ys)
        for c in FM.FAMILY:
            row = {'fold': ys.year, 'contrast': c, 'point': pt[c]}
            if ys.year in SEED_FOLDS:
                row['seed_rule'] = FM.seed_rule(pt[c], seed_ranges(R, ys.year, c))
            else:
                row['seed_2016_ranges_quoted'] = seed_ranges(R, 2016, c)
            per_fold.append(row)
    out['per_fold'] = per_fold
    s16 = R.seeds.get(2016, {})
    pm = [v['RMSE'] for (m, k), v in sorted(s16.items()) if m == 'P']
    mm = [v['RMSE'] for (m, k), v in sorted(s16.items()) if m == 'M']
    out['c3_seeds_2016'] = {'n_seeds_M': len(mm), 'n_seeds_P': len(pm), 'mean_difference': float(np.mean(pm) - np.mean(mm)) if pm and mm else None, 'sd_M': FM.sd_interval(mm), 'sd_P': FM.sd_interval(pm), 'rule': "mean over seeds of RMSE(P) - mean over seeds of RMSE(M); each model's seed SD (ddof 1) with its 95 percent chi-square interval"}
    fwd = None
    if 2019 in R.years and R.years[2019].has(FM.FAMILY_MODELS):
        f = FM.family_reps([R.years[2019]], FM.BLOCK, b_exp)
        fwd = FM.replication_rows(f['reps'], f['point'])
        for c in fwd:
            fwd[c]['seed_rule'] = FM.seed_rule(f['point'][c], seed_ranges(R, 2019, c))
            fwd[c]['label'] = 'exploratory (2019 alone)'
        out['forward_2019'] = {'rows': fwd, 'draws': f['draws'], 'count_of_intervals_read': len(fwd)}
    else:
        out['forward_2019'] = {'status': 'NOT RUN', 'reason': '2019 lacks a family model'}
    lbx = None
    if R.years_lb:
        lyears = [R.years_lb[y] for y in sorted(R.years_lb)]
        lf = FM.family_reps(lyears, FM.BLOCK, b_exp, part='lockbox')
        lbx = FM.replication_rows(lf['reps'], lf['point'])
        held = gbt_fit_held(R)
        if held and 'C-2' in lbx:
            lbx['C-2']['word'] = held['text']
            lbx['C-2']['gbt_fit_not_equal_folds'] = held['folds']
        out['lockbox'] = {'rows': lbx, 'draws': lf['draws'], 'folds': sorted(R.years_lb), 'count_of_intervals_read': len(lbx), 'gbt_fit_checks': R.gbt_fit}
    else:
        out['lockbox'] = {'status': 'NOT READ', 'reason': 'no opening record (the lockbox stays sealed)'}
    if fwd is not None and lbx is not None:
        out['S6'] = {c: FM.s6(fwd[c]['word'], lbx[c]['word']) for c in FM.FAMILY if c in fwd and c in lbx}
        if gbt_fit_held(R) and 'C-2' in out['S6']:
            out['S6']['C-2'] = "not read (C-2's lockbox word held: GBT's fit record)"
    else:
        out['S6'] = 'not read (2019 or the lockbox missing)'
    return out

def gbt_fit_held(R):
    bad = [g for g in R.gbt_fit if g['model'] == 'GBT' and g['gbt_fit_same'] is not True]
    if not bad:
        return None
    differs = any((g['gbt_fit_same'] is False for g in bad))
    return {'text': GBT_FIT_DIFFERS if differs else GBT_FIT_ABSENT, 'folds': sorted((g['fold'] for g in bad))}

def _closer(a, b):
    return ('%s closer to CMAQ' % b, '%s closer to CMAQ' % a)

def exploratory_part(R, b_exp):
    rows = []
    wcache = {}

    def weights(y, n):
        if (y, n) not in wcache:
            wcache[y, n] = FM.weights_for(y, n, FM.BLOCK, b_exp, 'cv')[0]
        return wcache[y, n]

    def add(name, a, b, folds, definition, block=False):
        for y in folds:
            s = (R.block_sse if block else R.sse).get(y, {})
            ka = [k for k in s if k[0] == a and k[1] in (0, None)]
            kb = [k for k in s if k[0] == b and k[1] in (0, None)]
            if not ka or not kb:
                continue
            sa, sb = (s[ka[0]], s[kb[0]])
            n = np.full(len(sa), float(R.n_cells_of(y, block)))
            w = weights(y, len(sa))
            rep = FM.rmse_contrast(w, sa, sb, n)
            pt = float(FM.rmse_contrast(np.ones((1, len(sa))), sa, sb, n)[0])
            x = FM.exploratory(rep, pt, name, definition, margin=FM.EXPLORATORY_MARGIN, closer=_closer(a, b))
            x['fold'] = y
            rows.append(x)
    for y in (2016, 2019):
        for a, b in zip(LADDER[:-1], LADDER[1:]):
            add('ladder %s -> %s' % (a, b), a, b, (y,), 'RMSE(%s) - RMSE(%s), ppb (the study design the ladder; positive: %s closer)' % (a, b, b))
    for name, a, b, folds, definition in EXPLORATORY_PAIRS:
        add(name, a, b, folds, definition)
    add('A3BL unseen place', 'A3BL', 'A3', (2016,), "RMSE(A3BL) - RMSE(A3) on the block's cells, ppb (the study design)", block=True)
    for y in (2016, 2019):
        s = R.sse.get(y, {})
        need = [('A4', 0), ('M', 0), ('A2', 0), ('A3', 0)]
        if not all((k in s for k in need)):
            continue
        n = np.full(len(s['M', 0]), float(R.n_cells_of(y, False)))
        w = weights(y, len(n))

        def inter(ww):
            return FM.rmse_contrast(ww, s['A4', 0], s['M', 0], n) - FM.rmse_contrast(ww, s['A2', 0], s['A3', 0], n)
        x = FM.exploratory(inter(w), float(inter(np.ones((1, len(n))))[0]), '2 x 2 interaction', '(M - A4) - (A3 - A2) with each letter minus its RMSE, i.e. [RMSE(A4) - RMSE(M)] - [RMSE(A2) - RMSE(A3)], ppb (READING; expected negative, the study design)', margin=0.0, closer=('positive interaction', 'negative interaction (the expected sign, the study design)'))
        x['fold'] = y
        rows.append(x)
    es = [g for g in R.gbt_es if g.get('printed')]
    for g in es:
        add('GBT-ES beside GBT', 'GBT-ES', 'M', (g['fold'],), 'RMSE(GBT-ES) - RMSE(M), ppb (exploratory sensitivity of C-2, printed by its rule)')
    add('ensemble P ens5 - M ens5', 'P ens5', 'M ens5', (ENS_FOLD,), 'RMSE(P ens5) - RMSE(M ens5), ppb (the 5-member means)')
    add('ensemble M s0 - M ens5', 'M', 'M ens5', (ENS_FOLD,), 'RMSE(M s0) - RMSE(M ens5), ppb (one member against the 5-member mean)')
    table = FM.table(rows, 'exploratory (95 percent, %d resamples, 7-day blocks)' % b_exp)
    table['margin'] = FM.EXPLORATORY_MARGIN
    return (table, strata_table(R, b_exp, weights))

def strata_table(R, b_exp, weights):
    srows, snot = ([], [])
    for y in sorted(R.strata):
        st = R.strata[y]
        sse = R.sse.get(y, {})
        if 'M' not in st or ('M', 0) not in sse:
            continue
        si = R.season_idx[y]
        nd = len(si)
        w = weights(y, nd)
        one = np.ones((1, nd))
        allc = np.full(nd, float(R.n_cells_of(y, False)))
        m_all = sse['M', 0]
        for ref in STRATA_REFS:
            kr = [k for k in sse if k[0] == ref and k[1] in (0, None)]
            if ref not in st or not kr:
                continue
            cases = []
            for k, s in enumerate(MT.SEASONS):
                cases.append(('season', s, sse[kr[0]], m_all, allc, si == k))
            for key in sorted(st[ref]):
                cases.append((key[0], key[1], st[ref][key][0], st['M'][key][0], st[ref][key][1], None))
            for k, s in enumerate(MT.SEASONS):
                for u in MT.URBAN:
                    sa, cnt = st[ref]['urban', u]
                    cases.append(('season_x_urban', '%s|%s' % (s, u), sa, st['M']['urban', u][0], cnt, si == k))
            for kind, name, sa, sb, cnt, dsel in cases:
                tot = cnt if dsel is None else cnt * dsel
                if not float(np.sum(tot)) > 0:
                    snot.append({'fold': y, 'reference': ref, 'stratum_kind': kind, 'stratum': name, 'reason': 'no cell-day in the stratum'})
                    continue
                rep = FM.rmse_contrast_sub(w, sa, sb, cnt, dsel)
                pt = float(FM.rmse_contrast_sub(one, sa, sb, cnt, dsel)[0])
                x = FM.exploratory(rep, pt, '%s - M (%s %s)' % (ref, kind, name), "RMSE(%s) - RMSE(M) over the stratum's cell-days, ppb (%s; the study design)" % (ref, STRATA_LABEL[kind]), margin=FM.EXPLORATORY_MARGIN, closer=_closer(ref, 'M'))
                x.update({'fold': y, 'reference': ref, 'stratum_kind': kind, 'stratum': name})
                srows.append(x)
    t = FM.table(srows, 'exploratory strata (95 percent, %d resamples, 7-day blocks)' % b_exp)
    t['margin'] = FM.EXPLORATORY_MARGIN
    t['not_run'] = snot
    return t

def five_fold_part(R):
    by = {}
    for x in R.metrics:
        if x.get('stratum_kind') != 'all' or x.get('status') != 'RUN' or x.get('kind') == 'ensemble':
            continue
        by.setdefault((x['model'], x['seed']), {})[x['fold']] = x
    rows, d9 = ([], [])
    for (m, s), f in sorted(by.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        d9.append({'model': m, 'seed': s, 'rel_change_rmse_2019': MT.rel_change({y: f[y]['RMSE'] for y in f}), 'MB_by_year': {y: f[y]['MB'] for y in sorted(f)}, 'label': 'descriptive (D9)'})
        if sorted(f) != list(EIO.FOLDS):
            continue
        r = {'model': m, 'seed': s, 'folds': 5}
        for k in ('RMSE', 'MB', 'MAE', 'NMB', 'NME', 'R', 'R2', 'IOA', 'SS'):
            v = [f[y].get(k) for y in EIO.FOLDS]
            r[k] = float(np.mean(v)) if all((x is not None for x in v)) else None
        rows.append(r)
    return (rows, d9)

def write_all(out, R, o, conf, sources, models, models_sha, opening, c7a, b_fam, b_exp, pd17, t0):
    mfields = ['model', 'seed', 'name', 'kind', 'fold', 'stratum_kind', 'stratum', 'status', 'n', 'cells', 'RMSE', 'MB', 'MAE', 'NMB', 'NME', 'R', 'R2', 'r_squared', 'IOA', 'SS']
    out.json('metrics.json', {'rows': R.metrics, 'definitions': MT.__doc__.split('Nothing here')[0]})
    out.csv('metrics.csv', R.metrics, mfields)
    d11 = {'S_M': [0, 0], '6000': [0, 0]}
    for d in R.diag:
        v = d.get('D11_still_improving')
        if v is None:
            continue
        k = 'S_M' if d['model'] in S_M_MODELS else '6000'
        d11[k][0] += int(v)
        d11[k][1] += 1
    out.json('diagnostics.json', {'rows': R.diag, 'D11_summary': {k: {'still_improving': v[0], 'networks': v[1]} for k, v in d11.items()}, 'D11_rule': 'S_M arms M, A4, M-C7; expected: the 6,000-step arms still improving at their end more often than the S_M arms (the study design D11)', 'D10_rule': "count of outputs below 0 ppb before the clip (the prediction files' own counts)", 'D8_input_ranges': R.d8_ranges, 'D8_rule': "inputs_outside_norm: any channel of the network's set outside its [min, max] over all cells of the fold's TRAIN days (PROP 4.3; eval\\d8in.py); inputs_outside: an emission channel outside its cell's range (AN-02 (i)); both READINGS ", 'G02_rule': "The inference time (the study design): pred_info.json's seconds less its day sets' (the TEST_CV prediction), seconds"})
    flat = []
    for d in R.diag:
        if 'RMSE' not in d:
            continue
        flat.append({'model': d['model'], 'seed': d['seed'], 'fold': d['fold'], 'RMSE': d['RMSE'], 'MAPRMSE': d['MAPRMSE'], 'R_space': d['R_space'], 'R_time_median': d['R_time'].get('median'), 'ACC_median': (d.get('ACC') or {}).get('median'), 'ACC_p10': (d.get('ACC') or {}).get('p10'), 'ACC_p90': (d.get('ACC') or {}).get('p90'), 'SS': d.get('SS'), 'share_SS_le_0': (d.get('share_SS_le_0') or {}).get('share'), 'share_ACC_le_0': (d.get('share_ACC_le_0') or {}).get('share'), 'spread_median': d['spread'].get('median'), 'above30_R': d['above30']['R'], 'above30_IOA': d['above30']['IOA'], 'near_zero': d['D2_near_zero']['count'], 'G02_inference_seconds': d.get('G02_inference_seconds')})
    out.csv('diagnostics.csv', flat, list(flat[0].keys()) if flat else ['model'])
    for name, rows in (('exceedance', R.exceed), ('deciles', R.decile), ('band_ratios', R.bands), ('residual_acf', R.acf), ('d4_to_d7', R.dtab)):
        out.json(name + '.json', {'rows': rows})
        keys = sorted(set((k for x in rows for k in x if not isinstance(x[k], (dict, list)))))
        out.csv(name + '.csv', rows, keys or ['model'])
    fam = family_part(R, b_fam, b_exp)
    out.json('family.json', fam)
    frows = []
    for c, x in sorted(((fam.get('five_fold') or {}).get('rows') or {}).items()):
        frows.append({'part': 'five-fold 7-day', 'contrast': c, 'point': x['point'], 'lo': x['lo'], 'hi': x['hi'], 'level': x['level'], 'half_width': x['half_width'], 'reading': x['reading'], 'passes': x['passes_holm_step'], 'block_length': x['block_length']})
    for part in ('forward_2019', 'lockbox'):
        for c, x in sorted(((fam.get(part) or {}).get('rows') or {}).items()):
            frows.append({'part': part, 'contrast': c, 'point': x['point'], 'lo': x['lo'], 'hi': x['hi'], 'level': x['level'], 'reading': x['word']})
    out.csv('family.csv', frows, ['part', 'contrast', 'point', 'lo', 'hi', 'level', 'half_width', 'reading', 'passes', 'block_length'])
    ex, exs = exploratory_part(R, b_exp)
    out.json('exploratory.json', ex)
    out.csv('exploratory.csv', ex['rows'], ['contrast', 'fold', 'point', 'lo', 'hi', 'level', 'half_width', 'margin', 'reading', 'label', 'finite_resamples', 'n_resamples'])
    out.json('exploratory_strata.json', exs)
    out.csv('exploratory_strata.csv', exs['rows'], ['fold', 'reference', 'stratum_kind', 'stratum', 'point', 'lo', 'hi', 'level', 'half_width', 'margin', 'reading', 'label', 'finite_resamples', 'n_resamples'])
    out.json('d11_curves.json', {'rows': R.d11, 'source': "the training log DONE.json names (the shared package's predict.py write_done; loop.py TrainLog evaluations: step, train_stat, val_loss, best_step); the study design D11", 'label': 'D11, training-day against validation curves'})
    out.csv('d11_curves.csv', [dict(x, model=d['model'], seed=d['seed'], fold=d['fold'], name=d['name']) for d in R.d11 for x in d['evaluations']], ['model', 'seed', 'name', 'fold', 'step', 'train_stat', 'val_loss', 'best_step'])
    f12 = {'rows': R.f12, 'count_of_intervals_read': sum((1 for x in R.f12 if x['lo'] is not None)), 'label': 'exploratory', 'rule': 'the labels of F-12', 'c7a_applies': c7a}
    out.json('f12.json', f12)
    out.csv('f12.csv', R.f12, ['model', 'seed', 'fold', 'group', 'stat', 'point', 'lo', 'hi', 'finite_resamples', 'words'])
    out.json('an02.json', {'rows': R.an02, 'source': 'the study design'})
    out.csv('an02.csv', R.an02, ['fold', 'model', 'seed', 'part', 'source', 'n_cell_days', 'n_outside', 'share'])
    for name, arr in R.an02_maps:
        out.npy(name, arr)
    per = {}
    for y in an05.FOLDS:
        got = R.an05.get(y, {})
        per[y] = {'M': got.get('M'), 'M-C7': got.get('M-C7')}
    a5 = an05.rows(per, c7a)
    out.json('an05.json', a5)
    out.csv('an05.csv', a5['rows'], ['fold', 'row', 'status', 'n_cell_days', 'an02_share', 'f12_slope', 'reason', 'words'])
    out.json('gbt_es.json', {'rows': R.gbt_es})
    out.csv('gbt_es.csv', R.gbt_es, ['fold', 'status', 'printed', 'worse_fraction', 'min_round', 'round', 'c2_model', 'reason'])
    ff, d9 = five_fold_part(R)
    out.json('five_fold.json', {'rows': ff, 'd9': d9})
    out.csv('five_fold.csv', ff, ['model', 'seed', 'folds', 'RMSE', 'MB', 'MAE', 'NMB', 'NME', 'R', 'R2', 'IOA', 'SS'])
    out.json('city_series.json', R.city)
    out.csv('city_series.csv', R.city.get('rows', []), ['model', 'seed', 'fold', 'city', 'status', 'cells', 'n', 'RMSE', 'MB', 'MAE', 'R', 'R2', 'IOA'])
    for name, arr in R.figs:
        out.npy(name, arr)
    now = datetime.datetime.now(datetime.timezone.utc)
    rec = {'schema': SCHEMA, 'run_id': o.run_id, 'made_utc': now.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.monotonic() - t0, 1), 'design': conf.get('design'), 'design_sha256': conf.get('design_sha256'), 'configs_sha256': sources.sha256, 'models_sha256': models_sha, 'models': models, 'pd17': pd17, 'opening': opening, 'c7a_applies': c7a, 'resampling_seed': FM.SEED, 'resamples': {'family': b_fam, 'exploratory': b_exp}, 'test_resamples': o.test_resamples, 'files_read': R.read, 'not_run': R.notrun, 'guard_py': EIO.guard_record(), 'gbt_fit_checks': R.gbt_fit, 'mc7_checks': R.mc7, 'classes': R.classes, 'exploratory_margin_ppb': FM.EXPLORATORY_MARGIN, 'seal': 'nothing printed but names, paths, sha256, counts and times'}
    out.json('eval_run.json', rec)
    out.done({'schema': SCHEMA, 'run_id': o.run_id, 'made_utc': rec['made_utc'], 'test_resamples': o.test_resamples, 'mc7_checks': R.mc7})

def main(argv=None):
    o = parse(sys.argv[1:] if argv is None else argv)
    from train import cfg
    from train import guard as GD
    try:
        return run(o)
    except EIO.Refused as e:
        EIO.say('REFUSED: %s' % e)
        return e.code
    except GD.GuardError as e:
        EIO.say('REFUSED (a day of another role): %s' % e)
        return EIO.EXIT_REFUSED
    except cfg.ConfigError as e:
        EIO.say('REFUSED (a config or array): %s' % e)
        return EIO.EXIT_REFUSED
if __name__ == '__main__':
    sys.exit(main())
