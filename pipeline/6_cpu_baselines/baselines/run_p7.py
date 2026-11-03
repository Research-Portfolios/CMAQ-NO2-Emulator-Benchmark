"""Refits the baselines CLIM-C, CLIM-WD, MET-REG, GBT, GBT-RIDGE, CLIM-EM and GBT-ES (the trees stopped at their best validation round) on each fold's training days with the cross-fitted weekday climatology, and writes their predictions for the test days and the held-back days to sealed files."""
import json
import os
import sys
import tempfile
import time
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from train import cfg
NAMES = ('CLIM-C', 'CLIM-WD', 'MET-REG', 'GBT', 'GBT-RIDGE', 'GBT-ES', 'CLIM-EM')
ORDER = (2016, 2015, 2017, 2018, 2019)
DESIGN = 'the study design'
CURVE = 'val_curve.json'

def parse(argv):
    o = {'names': list(NAMES), 'folds': list(ORDER), 'config_dir': None, 'work_dir': None, 'arrays_dir': None, 'ct_index': None, 'ct_arrays': None, 'log_dir': None, 'cells_per_day': None, 'gbt_iter': None}
    keys = {'--names': 'names', '--folds': 'folds', '--config-dir': 'config_dir', '--work-dir': 'work_dir', '--arrays-dir': 'arrays_dir', '--ctables-index': 'ct_index', '--ctables-arrays': 'ct_arrays', '--log-dir': 'log_dir', '--test-cells-per-day': 'cells_per_day', '--test-gbt-iter': 'gbt_iter'}
    i = 0
    while i < len(argv):
        a = argv[i]
        v = argv[i + 1] if i + 1 < len(argv) else None
        if a not in keys or v is None:
            raise SystemExit('unknown or incomplete option %r' % a)
        k = keys[a]
        if k == 'names':
            o[k] = [s for s in v.split(',') if s]
        elif k == 'folds':
            o[k] = [int(s) for s in v.split(',') if s]
        elif k in ('cells_per_day', 'gbt_iter'):
            o[k] = int(v)
        else:
            o[k] = v
        i += 2
    for n in o['names']:
        if n not in NAMES:
            raise SystemExit('unknown baseline %r' % n)
    for y in o['folds']:
        if y not in cfg.FOLD_YEARS:
            raise SystemExit('unknown fold %r' % y)
    for k in ('config_dir', 'work_dir', 'arrays_dir', 'ct_index', 'ct_arrays', 'log_dir'):
        if not o[k]:
            raise SystemExit('--%s is needed' % k.replace('_', '-'))
    return o

def say(text):
    from choi_dl import common as dlc
    print(dlc.mask(text), flush=True)

def needs(work, names, y):
    from baselines import common, p7base
    out = {}
    for n in names:
        p = {'cv': not common.done_ok(work, n, y), 'lockbox': not p7base.lockbox_done_ok(work, n, y)}
        if p['cv'] or p['lockbox']:
            out[n] = p
    return out

def predict_members(fold, work, part, members, prep, extra_of, rows, after_close=None, done_extra=None):
    from baselines import common, p7base
    from train import guard as GD
    if part == 'cv':
        days, mode = (fold.held_days, 'predict')
        GD.roles_guard(days, 'predict_cv', fold.roles)
        ws = {n: common.PredWriter(work, n, fold) for n in members}
    else:
        days, mode = (fold.lockbox_days, 'predict_lockbox')
        GD.roles_guard(days, 'predict_lockbox', fold.roles)
        ws = {n: p7base.LockboxWriter(work, n, fold) for n in members}
    t0 = time.monotonic()
    for k, (y, i, d) in enumerate(days):
        ctx = prep(y, i, d, mode)
        for n, fn in members.items():
            ws[n].write(k, fn(ctx))
    for n, w in ws.items():
        info = w.close(extra_of(n))
        if after_close:
            after_close(n, part, info)
        if part == 'cv':
            common.write_done(work, n, fold.year, info, extra=done_extra)
            fr = info['files']['pred_masked.npy']
            say('%s fold %d TEST_CV: %s (%d bytes, sha16 %s); %d days x %d cells' % (n, fold.year, fr['path'], fr['bytes'], fr['sha256'][:16], info['n_days'], info['n_cells']))
        else:
            p7base.lockbox_write_done(work, n, fold.year, info, extra=done_extra)
            fr = info['files']['pred_lockbox.npy']
            say('%s fold %d LOCKBOX: %s (%d bytes, sha256 %s); %d days x %d cells; dates sha16 %s' % (n, fold.year, fr['path'], fr['bytes'], fr['sha256'], info['n_days'], info['n_cells'], info['dates_sha256'][:16]))
        rows.append({'baseline': n, 'fold': fold.year, 'part': part, 'file': fr, 'days': info['n_days'], 'cells': info['n_cells'], 'seconds': round(time.monotonic() - t0, 1)})

def run_fold(o, configs, paths, y, need, rows):
    import numpy as np
    from baselines import clim, clim_em, gbt, gbt_curve, metreg, p7base
    work = o['work_dir']
    fold = p7base.FoldP7(configs, paths, y)
    cross = p7base.CrossC(fold, os.path.join(o['ct_index'], 'CTABLES_%d.json' % y), o['ct_arrays'])
    say('fold %d: %d TRAIN days, %d VAL days, %d TEST_CV days, %d LOCKBOX days, %d masked cells; C tables %s sha16 %s; to do: %s' % (y, len(fold.train_days), len(fold.val_days), len(fold.held_days), len(fold.lockbox_days), len(fold.cells), os.path.basename(cross.index_path), cross.index_sha256[:16], ', '.join(('%s %s' % (n, '+'.join((k for k in ('cv', 'lockbox') if p[k]))) for n, p in sorted(need.items())))))
    base = {'design': DESIGN, 'c_tables': cross.describe(), 'training_days_rule': "the study design's TRAIN days (Split less LOCKBOX_DAYS.json days), guarded"}

    def parts(group):
        return [(part, {n: f for n, f in group.items() if n in need and need[n][part]}) for part in ('cv', 'lockbox')]
    clim_fit = None
    if any((n in need for n in ('CLIM-C', 'CLIM-WD', 'CLIM-EM'))):
        clim_fit = clim.ClimFit(fold, note=say)
        if not p7base.same_as_all(clim_fit, cross):
            raise ValueError("fold %d: ClimFit on the TRAIN days differs from the C tables' all fit" % y)
        say("fold %d: ClimFit on the TRAIN days equals the C tables' all fit (bit for bit)" % y)
        group = {'CLIM-C': lambda ctx: clim_fit.clim_c, 'CLIM-WD': lambda ctx: clim_fit.clim_wd(ctx)}
        for part, ms in parts(group):
            if ms:
                predict_members(fold, work, part, ms, lambda yy, i, d, mode: d, lambda n: dict(base, fit=clim_fit.describe()), rows)
    if 'MET-REG' in need:
        mf = metreg.MetRegFit(fold, cross, note=say)

        def prep_m(yy, i, d, mode):
            x = metreg.predictors(fold.channels_cells(yy, i, metreg.CHANNELS, mode))
            if not np.all(np.isfinite(x)):
                raise metreg.NonFinitePredictor('MET-REG fold %d: non-finite predictors on a %s day; nothing substituted' % (y, mode))
            return (x, cross.clim_wd(d))
        for part, ms in parts({'MET-REG': lambda ctx: mf.predict(*ctx)}):
            if ms:
                predict_members(fold, work, part, ms, prep_m, lambda n: dict(base, fit=mf.describe()), rows)
        del mf
    gm = [n for n in ('GBT', 'GBT-RIDGE', 'GBT-ES') if n in need]
    if gm:
        opts = {'max_iter': o['gbt_iter']} if o['gbt_iter'] else None
        gf = gbt.GBTFit(fold, cross, note=say, cells_per_day=o['cells_per_day'] or gbt.CELLS_PER_DAY, options=opts)
        gfp = {'gbt_fit': gbt_curve.fit_print(fold, gf)}
        kstar, curve_row = (None, None)
        if 'GBT-ES' in need:
            cdir = os.path.join(work, 'pred', 'baselines', 'GBT-ES', 'fold%d' % y)
            cp = os.path.join(cdir, CURVE)
            if need['GBT-ES']['cv']:
                curve = gbt_curve.val_curve(fold, gf)
                curve_row = p7base.write_json(cp, curve)
                say('GBT-ES fold %d: the staged validation curve sealed in %s (%d bytes, sha16 %s; %d VAL days)' % (y, curve_row['path'], curve_row['bytes'], curve_row['sha256'][:16], curve['val_days']))
            else:
                curve = cfg.read_json(cp)
            kstar = int(curve['minimum_round'])

        def prep_g(yy, i, d, mode):
            fx, cw = gf.feat.day(yy, i, d, mode)
            if not np.all(np.isfinite(fx)):
                raise gbt.NonFiniteFeature('GBT fold %d: non-finite features on a %s day' % (y, mode))
            return (fx, cw)
        group = {'GBT': lambda ctx: ctx[1] + gf.gbt.predict(ctx[0]), 'GBT-RIDGE': lambda ctx: ctx[1] + gf.ridge.predict(gf.scaler.transform(ctx[0].copy())), 'GBT-ES': lambda ctx: ctx[1] + gbt_curve.staged_at(gf.gbt, ctx[0], kstar)}

        def extra_g(n):
            e = dict(base, fit=gf.describe())
            if n == 'GBT-ES':
                e['stopped_at'] = 'the first round of the least validation RMSE (sealed in %s)' % CURVE
            return e

        def add_curve(n, part, info):
            if n == 'GBT-ES' and part == 'cv':
                info['files'][CURVE] = curve_row
        for part, ms in parts(group):
            if ms:
                predict_members(fold, work, part, ms, prep_g, extra_g, rows, after_close=add_curve, done_extra=gfp)
        del gf
    if 'CLIM-EM' in need:
        em = clim_em.ClimEM(fold, clim_fit, note=say)

        def prep_e(yy, i, d, mode):
            return (d, clim_em.e9_cells(fold, yy, i, mode, em.parts))
        for part, ms in parts({'CLIM-EM': lambda ctx: em.predict(*ctx)}):
            if ms:
                em.ratio_one = 0
                predict_members(fold, work, part, ms, prep_e, lambda n: dict(base, fit=em.describe()), rows)
        del em
    del fold, cross, clim_fit

def main(argv):
    o = parse(argv)
    t = os.path.join(o['work_dir'], 'tmp')
    os.makedirs(t, exist_ok=True)
    os.environ['TMP'] = t
    os.environ['TEMP'] = t
    tempfile.tempdir = t
    paths = cfg.Paths(o['config_dir'], o['work_dir'], o['work_dir'], os.path.join(t, 'baselines_unused_marker'), o['arrays_dir'])
    configs = cfg.Configs(paths)
    rows = []
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    t0 = time.monotonic()
    for y in o['folds']:
        need = needs(o['work_dir'], o['names'], y)
        if not need:
            say('fold %d: every named baseline done before (TEST_CV and LOCKBOX DONE files verified); skipped' % y)
            continue
        run_fold(o, configs, paths, y, need, rows)
    rec = {'schema': 'tier-a-baselines-p7-run-1', 'started': started, 'ended': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.monotonic() - t0, 1), 'names': o['names'], 'folds': o['folds'], 'configs_sha256': dict(configs.sha256), 'rows': rows, 'metrics': 'none (the evaluation computes every metric)', 'printed': 'paths, bytes, sha256, counts and times only'}
    os.makedirs(o['log_dir'], exist_ok=True)
    p = os.path.join(o['log_dir'], 'BASELINES_P7_RUN_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write('\n')
    say('run record %s; %d (baseline, fold, part) rows written' % (os.path.basename(p), len(rows)))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
