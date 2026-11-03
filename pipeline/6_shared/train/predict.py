"""Predicts a training's day sets with the best saved network by whole-grid inference, writes the land-cell predictions in ppb (and the full grid for 2016) with an information file, and writes and checks the training's done marker."""
import datetime
import os
import numpy as np
from train import cfg

def _now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _file_row(path, shape):
    return {'path': cfg.rel(path), 'bytes': os.stat(path).st_size, 'sha256': cfg.sha256_file(path), 'shape': list(shape), 'dtype': 'float32'}

def predict_set(pd, model, paths, training, device, out_name, full_grid=False, batch=cfg.INFER_BATCH, note=None, scale=None):
    name = training['name']
    days = pd.days()
    lb = getattr(pd.split, 'lockbox_set', None) or set()
    hit = [d for y, i, d in days if (int(y), int(i)) in lb]
    if hit:
        from train.data import HeldOutError
        raise HeldOutError('%s %s: %d lockbox or buffer days in the day set' % (name, out_name, len(hit)))
    from choi import infer
    if model.training:
        raise ValueError('predict_set: the model is not in eval mode')
    h, w = pd.ya.grid
    cells = pd.cells
    out_dir = paths.pred(name)
    os.makedirs(out_dir, exist_ok=True)
    pm = os.path.join(out_dir, out_name + '.npy')
    pg = os.path.join(out_dir, 'pred_grid_2016.npy')
    pm_tmp = pm + '.partial'
    pg_tmp = pg + '.partial'
    if full_grid and pd.split.held_year != 2016:
        raise ValueError('full grid asked for a fold other than 2016')
    a_m = np.lib.format.open_memmap(pm_tmp, mode='w+', dtype=np.float32, shape=(len(days), len(cells)))
    a_g = np.lib.format.open_memmap(pg_tmp, mode='w+', dtype=np.float32, shape=(len(days), h, w)) if full_grid else None
    clipped = []
    n_win = None
    wmin = None
    t0 = datetime.datetime.now()
    try:
        for k, (y, i, d) in enumerate(days):
            x = pd.scaled_day(y, i, scale)
            r = infer.predict_grid(model, x, device=device, batch=batch, to_ppb=pd.plan.to_ppb, clip=True, mask=pd.mask)
            f = r['field']
            if f.shape != (h, w) or not np.all(np.isfinite(f)):
                raise ValueError('day %s: field %s, finite %s' % (d, f.shape, bool(np.all(np.isfinite(f)))))
            a_m[k] = f.ravel()[cells]
            if a_g is not None:
                a_g[k] = f
            clipped.append([d, int(r['n_clipped']), int(r['n_clipped_masked'])])
            n_win = r['n_windows']
            wmin = r['weight_sum_min'] if wmin is None else min(wmin, r['weight_sum_min'])
            if note and (k + 1) % 61 == 0:
                note('%s %s: %d of %d days' % (name, out_name, k + 1, len(days)))
        a_m.flush()
        if a_g is not None:
            a_g.flush()
    finally:
        del a_m
        if a_g is not None:
            del a_g
    os.replace(pm_tmp, pm)
    files = {out_name + '.npy': _file_row(pm, (len(days), len(cells)))}
    if full_grid:
        os.replace(pg_tmp, pg)
        files['pred_grid_2016.npy'] = _file_row(pg, (len(days), h, w))
    return {'day_set': pd.day_set, 'days': [d for _, _, d in days], 'n_days': len(days), 'n_cells': int(len(cells)), 'windows_per_day': n_win, 'weight_sum_min': wmin, 'clipped_per_day': clipped, 'clipped_total': int(sum((c[1] for c in clipped))), 'clipped_masked_total': int(sum((c[2] for c in clipped))), 'files': files, 'seconds': round((datetime.datetime.now() - t0).total_seconds(), 1)}

def predict_heldout(pd, model, best, paths, training, device, batch=cfg.INFER_BATCH, note=None):
    s = predict_set(pd, model, paths, training, device, 'pred_masked', full_grid=bool(training.get('full_grid')), batch=batch, note=note)
    info = {'schema': 'choi-pred-info-2', 'training': training['name'], 'made': _now(), 'units': 'ppb', 'held_out_year': pd.split.held_year, 'cell_order': 'row-major flat indices of the mask (numpy.flatnonzero over 299 x 459 as stored)', 'mask_sha256': pd.mask_sha256, 'best_step': best.get('step'), 'best_val_loss': best.get('val_loss'), 'predicted_by': 'a freshly built network that loaded best.pt, eval mode', 'infer_batch': batch, 'day_sets': {}, 'lockbox': {'held_out_is': 'TEST_CV', 'sha256': pd.split.lockbox_sha256, 'days_left_out': pd.split.removed.get('held', 0)} if getattr(pd.split, 'lockbox_sha256', None) else None}
    info.update({k: s[k] for k in ('days', 'n_days', 'n_cells', 'windows_per_day', 'weight_sum_min', 'clipped_per_day', 'clipped_total', 'clipped_masked_total', 'files', 'seconds')})
    return info

def predict_day_sets(make_pd, model, paths, training, device, info, batch=cfg.INFER_BATCH, note=None):
    for ds in ('TRAIN-S', 'VAL'):
        pd = make_pd(ds)
        s = predict_set(pd, model, paths, training, device, 'pred_masked_%s' % ds, full_grid=False, batch=batch, note=note)
        info['day_sets'][ds] = {k: s[k] for k in ('days', 'n_days', 'n_cells', 'clipped_total', 'clipped_masked_total', 'seconds')}
        info['files'].update(s['files'])
        info['seconds'] = round(float(info['seconds']) + s['seconds'], 1)
    return info

def write_done(paths, training, result, info, log_path, plan_id=None):
    from train.loop import _write_json_atomic
    cdir = paths.ckpt(training['name'])
    best_p = os.path.join(cdir, 'best.pt')
    d = {'schema': 'choi-train-done-2', 'training': training['name'], 'status': 'done', 'ended': _now(), 'result': result, 'seed': training.get('seed'), 'label': training.get('label'), 'labels': {'fold': training.get('held_out_year'), 'network': training.get('net'), 'model_id': training.get('model_id'), 'seed_offset': training.get('seed_offset')}, 'plan_id': plan_id or {}, 'files': dict(info['files']), 'best_pt': {'path': cfg.rel(best_p), 'bytes': os.stat(best_p).st_size, 'sha256': cfg.sha256_file(best_p)}, 'pred_info': cfg.rel(os.path.join(paths.pred(training['name']), 'pred_info.json')), 'train_log': cfg.rel(log_path)}
    _write_json_atomic(d, os.path.join(cdir, 'DONE.json'))
    return d

def write_info(paths, training, info):
    from train.loop import _write_json_atomic
    _write_json_atomic(info, os.path.join(paths.pred(training['name']), 'pred_info.json'))

def done_check(paths, training, expect=None, check_hash=True):
    p = os.path.join(paths.ckpt(training['name']), 'DONE.json')
    if not os.path.isfile(p):
        return (False, 'absent')
    try:
        d = cfg.read_json(p)
    except cfg.ConfigError:
        return (False, 'unverified: DONE.json unreadable')
    if d.get('status') != 'done' or d.get('training') != training['name']:
        return (False, 'unverified: status or name')
    if expect:
        pid = d.get('plan_id') or {}
        for k, v in expect.items():
            if pid.get(k) != v:
                return (False, 'another plan (%s %r, now %r)' % (k, pid.get(k), v))
    rows = list(d.get('files', {}).values()) + [d.get('best_pt') or {}]
    if not rows[-1]:
        return (False, 'unverified: no best.pt row')
    for r in rows:
        fp = os.path.join(paths.root, r['path']) if not os.path.isabs(r['path']) else r['path']
        if not os.path.isfile(fp) or os.stat(fp).st_size != r['bytes']:
            return (False, 'unverified: %s' % os.path.basename(fp))
        if check_hash and cfg.sha256_file(fp) != r['sha256']:
            return (False, 'unverified: %s sha256' % os.path.basename(fp))
    return (True, 'verified')

def done_ok(paths, training, check_hash=True, expect=None):
    return done_check(paths, training, expect, check_hash)[0]
