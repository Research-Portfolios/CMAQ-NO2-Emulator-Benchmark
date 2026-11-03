"""Writes the baselines' held-out metrics (n, R, IOA, MAE, mean error and RMSE per year and season, and the series of the ten cities for 2016) to sealed files, and returns only paths, sizes and hashes."""
import json
import os
import numpy as np
from baselines import common
from train import cfg
METRICS = ('n', 'R', 'IOA', 'MAE', 'ME', 'RMSE', 'NMB', 'NME', 'R2')
SEASONS = ('winter', 'spring', 'summer', 'fall')
SEALED_SUB = 'BASE2'

def sealed_dir(work_dir, name, year=None):
    d = os.path.join(work_dir, 'results_sealed', SEALED_SUB, name)
    return d if year is None else os.path.join(d, 'fold%d' % int(year))

def _write(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)
    return {'path': cfg.rel(path), 'bytes': os.stat(path).st_size, 'sha256': common.sha256_file(path)}

def _all(e, o):
    from choi import metrics
    return metrics.all_metrics(e, o)
CITY_RULE = "the mean over the masked cells of the 3 x 3 block centred on the city's cell (clipped to the grid), of the stored predictions and, apart, of the held-out target (float64); the metrics on the two daily series (n = the days); a city with no masked block cell is listed, not computed"

def city_blocks(grid, mask, cc):
    h, w = (int(grid[0]), int(grid[1]))
    m = np.asarray(mask, dtype=bool)
    if tuple(m.shape) != (h, w):
        raise ValueError('mask %s, grid %s' % (list(m.shape), [h, w]))
    given = cc.get('block_masked') or {}
    out = {}
    for city, rc in sorted(cc.get('cells', {}).items()):
        r0, c0 = (int(rc['row']), int(rc['col']))
        cells = [(r, c) for r in range(r0 - 1, r0 + 2) for c in range(c0 - 1, c0 + 2) if 0 <= r < h and 0 <= c < w and bool(m[r, c])]
        if city in given:
            want = sorted(((int(a), int(b)) for a, b in given[city]))
            if want != sorted(cells):
                raise ValueError("%s: the masked block cells %s differ from CITY_CELLS.json's block_masked %s" % (city, cells, want))
        out[city] = cells
    return out

def write_fold_metrics(fold, paths, work_dir, name):
    pred, info = common.load_pred(work_dir, name, fold.year)
    days = info['days']
    if days != [d for _, _, d in fold.held_days] or int(info['n_cells']) != len(fold.cells) or info['mask_sha256'] != fold.mask_sha256:
        raise ValueError("%s fold %d: the stored predictions do not match the fold's days, cells or mask" % (name, fold.year))
    tgt = fold.held_out_target(paths)
    e = np.asarray(pred, dtype=np.float64)
    out = {'schema': 'choi-baseline-metrics-1', 'baseline': name, 'fold': fold.year, 'held_out_year': fold.year, 'made': common.now(), 'units': {'MAE': 'ppb', 'ME': 'ppb', 'RMSE': 'ppb', 'R': '1', 'IOA': '1', 'n': 'pairs', 'NMB': '1', 'NME': '1', 'R2': '1'}, 'predictions': {'path': info['files']['pred_masked.npy']['path'], 'sha256': info['files']['pred_masked.npy']['sha256']}, 'pooled': _all(e, tgt), 'by_season': {}, 'city_series': {}}
    seas = np.array([common.season(d) for d in days])
    for s in SEASONS:
        k = seas == s
        if k.any():
            out['by_season'][s] = _all(e[k], tgt[k])
    if fold.year == 2016:
        cc = fold.configs.city_cells()
        if cc:
            h, w = fold.grid
            pos = {int(c): j for j, c in enumerate(fold.cells)}
            out['city_rule'] = CITY_RULE
            for city, cells in sorted(city_blocks(fold.grid, fold.mask, cc).items()):
                rc = cc['cells'][city]
                if not cells:
                    out['city_series'][city] = 'no masked cell in the 3 x 3 block centred on row %d, col %d: listed, not computed' % (int(rc['row']), int(rc['col']))
                    continue
                js = [pos[r * w + c] for r, c in cells]
                se = e[:, js].mean(axis=1)
                so = tgt[:, js].mean(axis=1)
                out['city_series'][city] = {'metrics': _all(se, so), 'centre': [int(rc['row']), int(rc['col'])], 'cells_used': [[int(r), int(c)] for r, c in cells], 'n_cells_used': len(cells), 'n_days': int(se.size)}
        else:
            out['city_series'] = 'CITY_CELLS.json absent: not computed'
    row = _write(os.path.join(sealed_dir(work_dir, name, fold.year), 'metrics.json'), out)
    return (row, {'seasons': len(out['by_season']), 'cities': len(out['city_series']) if isinstance(out['city_series'], dict) else 0})

def write_five_fold(work_dir, name):
    vals = {}
    for y in cfg.FOLD_YEARS:
        p = os.path.join(sealed_dir(work_dir, name, y), 'metrics.json')
        if not os.path.isfile(p):
            return None
        vals[y] = cfg.read_json(p)['pooled']
    mean = {m: float(np.mean([vals[y][m] for y in cfg.FOLD_YEARS])) for m in METRICS if m != 'n'}
    mean['n_folds'] = len(vals)
    mean['n_pairs_total'] = int(sum((vals[y]['n'] for y in cfg.FOLD_YEARS)))
    return _write(os.path.join(sealed_dir(work_dir, name), 'five_fold.json'), {'schema': 'choi-baseline-five-fold-1', 'baseline': name, 'made': common.now(), 'five_fold_mean': mean, 'rule': 'the mean of the five yearly pooled values'})
