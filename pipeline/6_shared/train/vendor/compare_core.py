"""Compares the device test and the synthetic training between two devices: relative differences of outputs, losses and gradients against fixed tolerances (0.001 for the device test, 2 percent for the final training loss)."""
TOL_DEVICE = 0.001
TOL_TRAIN = 0.02
FINAL_LAST = 10

def _rel(a, b):
    a, b = (float(a), float(b))
    if abs(b) < 1e-30:
        return 0.0 if abs(a) < 1e-30 else float('inf')
    return abs(a - b) / abs(b)

def rel_inf(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    diff = float(np.max(np.abs(a - b))) if b.size else 0.0
    if scale < 1e-30:
        return 0.0 if diff < 1e-30 else float('inf')
    return diff / scale

def compare_grads(ga, gb):
    worst_est, worst_norm, at_est, at_norm = (0.0, 0.0, '', '')
    missing = sorted(set(ga) ^ set(gb))
    for name in sorted(set(ga) & set(gb)):
        a, b = (ga[name], gb[name])
        if a['n'] != b['n'] or len(a['proj']) != len(b['proj']):
            return {'ok_shapes': False, 'param': name}
        d = np.asarray(a['proj']) - np.asarray(b['proj'])
        rms = float(np.sqrt(np.mean(d * d)))
        est = _rel(rms + b['l2'], b['l2']) if b['l2'] >= 1e-30 else 0.0 if rms < 1e-30 else float('inf')
        nrm = _rel(a['l2'], b['l2'])
        if est > worst_est:
            worst_est, at_est = (est, name)
        if nrm > worst_norm:
            worst_norm, at_norm = (nrm, name)
    return {'ok_shapes': not missing, 'missing': missing, 'worst_rel_l2_est': worst_est, 'worst_rel_l2_est_param': at_est, 'worst_rel_norm': worst_norm, 'worst_rel_norm_param': at_norm}

def compare_device_results(ra, rb, out_a, out_b, tol=TOL_DEVICE):
    g = compare_grads(ra['grads'], rb['grads'])
    res = {'weights_same': ra['weights_sha256'] == rb['weights_sha256'], 'inputs_same': ra['inputs_sha256'] == rb['inputs_sha256'], 'output_rel_inf': rel_inf(out_a, out_b), 'loss_rel': _rel(ra['loss'], rb['loss']), 'grads': g, 'tol': tol}
    res['pass'] = bool(res['weights_same'] and res['inputs_same'] and g['ok_shapes'] and (res['output_rel_inf'] <= tol) and (res['loss_rel'] <= tol) and (g['worst_rel_l2_est'] <= tol) and (g['worst_rel_norm'] <= tol))
    return res

def compare_train(ta, tb, tol=TOL_TRAIN):
    la = np.asarray(ta['losses'], dtype=np.float64)
    lb = np.asarray(tb['losses'], dtype=np.float64)
    same_cfg = ta['steps'] == tb['steps'] and ta['batch'] == tb['batch'] and (ta['weights_sha256_start'] == tb['weights_sha256_start']) and (la.size == lb.size == ta['steps'])
    fa = float(np.mean(la[-FINAL_LAST:])) if la.size else float('nan')
    fb = float(np.mean(lb[-FINAL_LAST:])) if lb.size else float('nan')
    gap = float(np.max(np.abs(la - lb) / np.maximum(np.abs(lb), 1e-30))) if same_cfg and la.size else float('nan')
    res = {'same_config': bool(same_cfg), 'final_a': fa, 'final_b': fb, 'final_rel': _rel(fa, fb), 'max_stepwise_rel': gap, 'tol': tol}
    res['pass'] = bool(same_cfg and res['final_rel'] <= tol)
    return res
