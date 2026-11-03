"""AN-05: on the 2016 and 2018 folds, the residual U-Net (M) against the same network with climatological emissions (M-C7) and their difference, each with the AN-02 share, the F-12 weekday slope (the slope of the model's Tuesday-to-Friday minus Sunday map on CMAQ's) and its count of cell-days."""
from eval import labels
FOLDS = (2016, 2018)
ARMS = ('M', 'M-C7')

def rows(per_fold, c7a_applies):
    out = []
    for y in FOLDS:
        f = per_fold.get(y) or {}
        got = {a: f.get(a) for a in ARMS}
        for a in ARMS:
            r = got[a]
            if r is None:
                out.append({'fold': y, 'row': a, 'status': 'NOT RUN', 'reason': f.get('reason') or '%s has no TEST_CV predictions on fold %d' % (a, y), 'n_cell_days': None, 'an02_share': None, 'f12_slope': None, 'words': labels.wording(a, c7a_applies)['words']})
            else:
                out.append({'fold': y, 'row': a, 'status': 'RUN', 'n_cell_days': r['n_cell_days'], 'an02_share': r['share'], 'f12_slope': r['slope'], 'words': labels.wording(a, c7a_applies)['words']})
        if got['M'] is None or got['M-C7'] is None:
            out.append({'fold': y, 'row': 'M less M-C7', 'status': 'NOT RUN', 'reason': f.get('reason') or 'an arm is missing on fold %d' % y, 'n_cell_days': None, 'an02_share': None, 'f12_slope': None})
            continue
        m, c = (got['M'], got['M-C7'])
        out.append({'fold': y, 'row': 'M less M-C7', 'status': 'RUN', 'n_cell_days': m['n_cell_days'], 'an02_share': None if m['share'] is None or c['share'] is None else m['share'] - c['share'], 'f12_slope': None if m['slope'] is None or c['slope'] is None else m['slope'] - c['slope']})
    return {'rows': out, 'threshold': 'none (descriptive, exploratory)', 'source': 'the study design'}
