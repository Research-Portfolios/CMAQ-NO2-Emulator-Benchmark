"""Entry of the baselines: fits each named baseline on each fold's training days, predicts the held-out year, writes the predictions with done markers and the held-out metrics to sealed files, and skips any baseline and fold already done."""
import json
import os
import sys
import tempfile
import time
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from train import cfg
ORDER = (2016, 2015, 2017, 2018, 2019)

def parse(argv):
    o = {'names': ['CLIM-C', 'CLIM-WD', 'MET-REG'], 'folds': list(ORDER), 'config_dir': cfg.LIVE_CONFIG, 'work_dir': cfg.W, 'arrays_dir': None, 'cells_per_day': None, 'gbt_iter': None}
    i = 0
    while i < len(argv):
        a = argv[i]
        v = argv[i + 1] if i + 1 < len(argv) else None
        if a == '--names' and v:
            o['names'] = [s for s in v.split(',') if s]
        elif a == '--folds' and v:
            o['folds'] = [int(s) for s in v.split(',') if s]
        elif a == '--config-dir' and v:
            o['config_dir'] = v
        elif a == '--work-dir' and v:
            o['work_dir'] = v
        elif a == '--arrays-dir' and v:
            o['arrays_dir'] = v
        elif a == '--test-cells-per-day' and v:
            o['cells_per_day'] = int(v)
        elif a == '--test-gbt-iter' and v:
            o['gbt_iter'] = int(v)
        else:
            raise SystemExit('unknown or incomplete option %r' % a)
        i += 2
    for n in o['names']:
        if n not in ('CLIM-C', 'CLIM-WD', 'MET-REG', 'GBT'):
            raise SystemExit('unknown baseline %r' % n)
    for y in o['folds']:
        if y not in cfg.FOLD_YEARS:
            raise SystemExit('unknown fold %r' % y)
    return o

def say(text):
    from choi_dl import common as dlc
    print(dlc.mask(text), flush=True)

def main(argv):
    o = parse(argv)
    t = os.path.join(o['work_dir'], 'tmp')
    os.makedirs(t, exist_ok=True)
    os.environ['TMP'] = t
    os.environ['TEMP'] = t
    tempfile.tempdir = t
    from baselines import clim, common, gbt, metreg, sealed
    paths = cfg.Paths(o['config_dir'], o['work_dir'], o['work_dir'], os.path.join(t, 'baselines_unused_marker'), o['arrays_dir'])
    configs = cfg.Configs(paths)
    rows = []
    started = common.now()
    t0 = time.monotonic()
    for y in o['folds']:
        fold = common.Fold(configs, paths, y)
        todo = []
        for n in o['names']:
            outs = ['GBT', 'GBT-RIDGE'] if n == 'GBT' else [n]
            if all((common.done_ok(o['work_dir'], m, y) and os.path.isfile(os.path.join(sealed.sealed_dir(o['work_dir'], m, y), 'metrics.json')) for m in outs)):
                say('%s fold %d: done before (DONE.json verified); skipped' % (n, y))
                continue
            todo.append(n)
        if not todo:
            continue
        say('fold %d: %d training days, %d held-out days, %d masked cells; fitting %s' % (y, len(fold.train_days), len(fold.held_days), len(fold.cells), ','.join(todo)))
        fit = clim.ClimFit(fold, note=say)
        for n in todo:
            ts = time.monotonic()
            if n in ('CLIM-C', 'CLIM-WD'):
                infos = [(n, clim.predict_clim(fold, fit, o['work_dir'], n))]
            elif n == 'MET-REG':
                mf = metreg.MetRegFit(fold, fit, note=say)
                infos = [(n, metreg.predict_metreg(fold, fit, mf, o['work_dir']))]
            else:
                opts = {'max_iter': o['gbt_iter']} if o['gbt_iter'] else None
                gf = gbt.GBTFit(fold, fit, note=say, cells_per_day=o['cells_per_day'] or gbt.CELLS_PER_DAY, options=opts)
                gi, ri = gbt.predict_gbt(fold, gf, o['work_dir'])
                infos = [('GBT', gi), ('GBT-RIDGE', ri)]
            for m, info in infos:
                done = common.write_done(o['work_dir'], m, y, info)
                srow, cnt = sealed.write_fold_metrics(fold, paths, o['work_dir'], m)
                pr = info['files']['pred_masked.npy']
                rows.append({'baseline': m, 'fold': y, 'prediction': pr, 'sealed_metrics': srow, 'done': {k: v for k, v in done.items() if k in ('status', 'ended')}, 'counts': {'days': info['n_days'], 'cells': info['n_cells'], 'seasons': cnt['seasons'], 'cities': cnt['cities']}, 'seconds': round(time.monotonic() - ts, 1)})
                say('%s fold %d: prediction %s (%d bytes, sha16 %s); sealed %s (%d bytes, sha16 %s); %d days x %d cells' % (m, y, pr['path'], pr['bytes'], pr['sha256'][:16], srow['path'], srow['bytes'], srow['sha256'][:16], info['n_days'], info['n_cells']))
        del fit, fold
    five = {}
    for n in o['names']:
        for m in ['GBT', 'GBT-RIDGE'] if n == 'GBT' else [n]:
            r = sealed.write_five_fold(o['work_dir'], m)
            if r:
                five[m] = r
                say('%s five-fold means: %s (%d bytes, sha16 %s)' % (m, r['path'], r['bytes'], r['sha256'][:16]))
    rec = {'schema': 'choi-baselines-run-1', 'started': started, 'ended': common.now(), 'seconds': round(time.monotonic() - t0, 1), 'names': o['names'], 'folds': o['folds'], 'config_dir': cfg.rel(o['config_dir']), 'configs_sha256': dict(configs.sha256), 'rows': rows, 'five_fold': five, 'printed': 'paths, bytes, sha256, counts and times only'}
    d = os.path.join(o['work_dir'], 'pred', 'baselines')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'RUN_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
    with open(p, 'w', encoding='ascii', newline='\n') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write('\n')
    say('run record %s; %d (baseline, fold) rows written' % (cfg.rel(p), len(rows)))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
