"""Records the validation RMSE of the gradient-boosted trees every 25 rounds on the validation days with the round of its minimum, and fits GBT-ES (stopped at that round) for the evaluation's printing rule; the curve goes to a sealed file only."""
import hashlib
import itertools
import numpy as np
from train import guard as GD
EVERY = 25
WORSE = 0.01

def rounds_of(n_iter, every=EVERY):
    r = list(range(every, int(n_iter) + 1, every))
    if not r or r[-1] != int(n_iter):
        r.append(int(n_iter))
    return r

def staged_at(gbt, x, k):
    if k < 1 or k > int(gbt.n_iter_):
        raise ValueError('round %d outside 1 to %d' % (k, gbt.n_iter_))
    return next(itertools.islice(gbt.staged_predict(x), k - 1, None))

def fit_print(fold, gfit):
    days = list(fold.val_days)
    if not days:
        raise ValueError('fold %d has no VAL day' % fold.year)
    y, i, d = days[0]
    GD.roles_guard([(y, i)], 'validate', fold.roles)
    fx, _cw = gfit.feat.day(y, i, d, 'validate')
    out = np.ascontiguousarray(gfit.gbt.predict(fx), dtype=np.float64)
    return {'n_iter': int(gfit.gbt.n_iter_), 'val_day': str(d), 'val_day_output_sha256': hashlib.sha256(out.tobytes()).hexdigest()}

class NonFiniteFeature(Exception):
    pass

def val_curve(fold, gfit, every=EVERY):
    days = list(fold.val_days)
    if not days:
        raise ValueError('fold %d has no VAL day' % fold.year)
    GD.roles_guard(days, 'validate', fold.roles)
    n_iter = int(gfit.gbt.n_iter_)
    rounds = rounds_of(n_iter, every)
    pos = {k: j for j, k in enumerate(rounds)}
    sse = np.zeros(len(rounds), dtype=np.float64)
    n = 0
    for y, i, d in days:
        fx, cw = gfit.feat.day(y, i, d, 'validate')
        if not np.all(np.isfinite(fx)):
            raise NonFiniteFeature('GBT fold %d: non-finite features on a VAL day' % fold.year)
        yv = fold.target_cells(y, i, purpose='validate')
        seen = 0
        for k, raw in enumerate(gfit.gbt.staged_predict(fx), 1):
            j = pos.get(k)
            if j is not None:
                e = np.maximum(cw + raw, 0.0) - yv
                sse[j] += float(np.dot(e, e))
                seen += 1
        if seen != len(rounds):
            raise ValueError('staged_predict gave %d of %d rounds' % (seen, len(rounds)))
        n += int(yv.size)
    rmse = np.sqrt(sse / n)
    j = int(np.argmin(rmse))
    return {'schema': 'gbt-val-curve-1', 'fold': fold.year, 'rounds': rounds, 'val_rmse_ppb': [float(v) for v in rmse], 'n_pairs': n, 'val_days': len(days), 'val_dates': [d for _y, _i, d in days], 'minimum_round': rounds[j], 'n_iter': n_iter, 'rule': 'Validation RMSE (ppb) every %d rounds on the VAL days only, every masked cell, prediction C + output clipped at 0; GBT-ES = the GBT stopped at the first round of the least RMSE' % every, 'printing_rule': 'GBT-ES is printed beside GBT only if round %d is more than %g percent worse than the minimum, applied by the evaluation (PE), never by a person; this file is sealed' % (n_iter, 100 * WORSE)}
