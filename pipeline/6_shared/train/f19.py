"""An exploratory emission-scaling check of trained networks: predicts the held-out year again with the NO, NO2, HONO and point and marine NOx inputs scaled by 0.9, and computes at 1,000 fixed cells the derivative of the emulated NO2 with respect to those inputs."""
import datetime
import os
import numpy as np
from train import cfg

def _now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _row(path):
    return {'path': cfg.rel(path), 'bytes': os.stat(path).st_size, 'sha256': cfg.sha256_file(path)}

def done_ok(paths, name):
    p = os.path.join(paths.pred(name), 'F19_DONE.json')
    if not os.path.isfile(p):
        return False
    try:
        d = cfg.read_json(p)
    except cfg.ConfigError:
        return False
    for r in d.get('files', {}).values():
        fp = os.path.join(paths.root, r['path'])
        if not os.path.isfile(fp) or os.stat(fp).st_size != r['bytes'] or cfg.sha256_file(fp) != r['sha256']:
            return False
    return True

def f19_cells(n_masked, n=cfg.F19_CELLS, seed=cfg.F19_SEED):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    return np.sort(rng.choice(n_masked, size=min(int(n), int(n_masked)), replace=False))

def f19_days(days):
    return [(y, i, d) for y, i, d in days if int(d[6:8]) in cfg.F19_DAYS_OF_MONTH]

def local_form(model, pd, device, cells_pick, days, batch=32):
    import torch
    import torch.nn.functional as F
    p = pd.plan
    names = [n for n in ('E-1', 'E-2', 'M-8', 'M-6', 'X-6') if n in p.names]
    ch = [p.names.index(n) for n in names]
    h, w = pd.ya.grid
    flat = pd.cells[cells_pick]
    rows, cols = (flat // w, flat % w)
    der = np.zeros((len(days), len(flat), len(names)), dtype=np.float32)
    outv = np.zeros((len(days), len(flat)), dtype=np.float32)
    for q in model.parameters():
        q.requires_grad_(False)
    for kd, (y, i, _) in enumerate(days):
        x = pd.scaled_day(y, i)
        xp = F.pad(x[None], (32, 31, 32, 31), mode='replicate')[0]
        for a in range(0, len(flat), batch):
            r_, c_ = (rows[a:a + batch], cols[a:a + batch])
            win = torch.stack([xp[:, int(r):int(r) + 64, int(c):int(c) + 64] for r, c in zip(r_, c_)]).detach()
            win.requires_grad_(True)
            out = model(win)[:, 0, 32, 32]
            out.sum().backward()
            g = win.grad[:, :, 32, 32].detach().to('cpu', torch.float64)
            g = g.numpy() * p.t_den
            der[kd, a:a + batch] = g[:, ch].astype(np.float32)
            outv[kd, a:a + batch] = (out.detach().to('cpu', torch.float64).numpy() * p.t_den + p.t_lo).astype(np.float32)
    return (der, outv, names)

def run_training(configs, paths, t, pl, device, note=None):
    from train import data, loop, predict
    name = t['name']
    if done_ok(paths, name):
        return {'status': 'skipped (F19_DONE.json verified)'}
    best_p = os.path.join(paths.ckpt(name), 'best.pt')
    pd = data.PredictData(configs, paths, t, device=device)
    model, best = loop.fresh_model_from_best(t['net'], pd.plan.n, pl['width'], best_p, device)
    s = predict.predict_set(pd, model, paths, t, device, 'pred_masked_F19S', full_grid=False, note=note, scale=(cfg.F19_SCALED, cfg.F19_SCALE))
    pick = f19_cells(len(pd.cells))
    days = f19_days(pd.days())
    der, outv, names = local_form(model, pd, device, pick, days)
    od = paths.pred(name)
    files = dict(s['files'])
    for fn, arr in (('f19_derivs.npy', der), ('f19_outputs.npy', outv)):
        pth = os.path.join(od, fn)
        np.save(pth + '.partial.npy', arr)
        os.replace(pth + '.partial.npy', pth)
        files[fn] = _row(pth)
    dens = {n: float(pd.plan.den[pd.plan.names.index(n)]) for n in names}
    mix = der[:, :, names.index('M-8')].astype(np.float64) / dens['M-8']
    if t['net'] != 'baseline' and 'X-6' in names:
        mix = mix + der[:, :, names.index('X-6')].astype(np.float64) / dens['X-6']
    pth = os.path.join(od, 'f19_mixing_source.npy')
    np.save(pth + '.partial.npy', mix.astype(np.float32))
    os.replace(pth + '.partial.npy', pth)
    files['f19_mixing_source.npy'] = _row(pth)
    info = {'schema': 'choi-f19-1', 'training': name, 'made': _now(), 'design': 'exploratory', 'scaled_channels': [n for n in cfg.F19_SCALED if n in pd.plan.names], 'scale': cfg.F19_SCALE, 'cells_seed': cfg.F19_SEED, 'n_cells': int(len(pick)), 'cells_index_into_mask': [int(x) for x in pick], 'days': [d for _, _, d in days], 'deriv_channels': names, 'deriv_units': 'ppb per scaled unit, at the cell; window row 32 and column 32', 'd7_ranges': dens, 'log_scaled': [n for n, g in zip(pd.plan.names, pd.plan.is_log) if g and n in names], 'mixing_source': 'M-8' + (' + X-6' if t['net'] != 'baseline' else '') + ' in source units (ppb per m)', 'best_step': best.get('step'), 'scaled_year_clipped_total': s['clipped_total'], 'files': files}
    from train.loop import _write_json_atomic
    _write_json_atomic(info, os.path.join(od, 'f19_info.json'))
    files['f19_info.json'] = _row(os.path.join(od, 'f19_info.json'))
    _write_json_atomic({'schema': 'choi-f19-done-1', 'training': name, 'status': 'done', 'ended': _now(), 'files': files}, os.path.join(od, 'F19_DONE.json'))
    del model
    return {'status': 'done: scaled year %d days, local form %d days x %d cells' % (s['n_days'], len(days), len(pick))}
