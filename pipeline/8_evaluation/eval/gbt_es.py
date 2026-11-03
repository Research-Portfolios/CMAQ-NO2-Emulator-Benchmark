"""Applies the printing rule of GBT-ES, the gradient-boosted trees stopped at the round of least validation RMSE: it is reported beside the trees only when the validation RMSE at round 500 is more than 1 percent above the curve's minimum."""
import math
ROUND = 500
WORSE = 0.01
C2_MODEL = 'GBT'
SCHEMA = 'gbt-val-curve-1'

class CurveError(ValueError):
    pass

def check_curve(curve):
    if not isinstance(curve, dict) or curve.get('schema') != SCHEMA:
        raise CurveError('val_curve.json: schema is not %s' % SCHEMA)
    r = curve.get('rounds')
    v = curve.get('val_rmse_ppb')
    if not isinstance(r, list) or not isinstance(v, list) or len(r) != len(v) or (not r):
        raise CurveError('val_curve.json: rounds and val_rmse_ppb differ')
    if any((not isinstance(k, int) for k in r)) or r != sorted(set(r)):
        raise CurveError('val_curve.json: rounds not increasing integers')
    if any((not (isinstance(x, (int, float)) and math.isfinite(x) and (x >= 0)) for x in v)):
        raise CurveError('val_curve.json: a value is not a finite RMSE')
    if curve.get('minimum_round') not in r:
        raise CurveError('val_curve.json: minimum_round not among rounds')
    return (r, [float(x) for x in v])

def rule(curve):
    rounds, rmse = check_curve(curve)
    lo = min(rmse)
    j_min = rmse.index(lo)
    out = {'rule': "GBT-ES is printed beside GBT only if the validation RMSE at round %d is more than %g percent worse than the curve's minimum" % (ROUND, 100 * WORSE), 'c2_model': C2_MODEL, 'round': ROUND, 'n_iter': curve.get('n_iter'), 'fold': curve.get('fold'), 'min_round': rounds[j_min], 'min_round_recorded': curve.get('minimum_round')}
    if rounds[j_min] != curve.get('minimum_round'):
        raise CurveError('val_curve.json: minimum_round is not the first round of the least RMSE')
    if ROUND not in rounds:
        out.update({'printed': False, 'status': 'NOT RUN', 'reason': 'the curve has no round %d (n_iter %s)' % (ROUND, curve.get('n_iter'))})
        return out
    v500 = rmse[rounds.index(ROUND)]
    if lo > 0:
        worse = v500 / lo - 1.0
    else:
        worse = float('inf') if v500 > 0 else 0.0
    out.update({'status': 'RUN', 'round500_rmse': v500, 'min_rmse': lo, 'worse_fraction': worse, 'printed': bool(v500 > lo * (1.0 + WORSE))})
    return out
