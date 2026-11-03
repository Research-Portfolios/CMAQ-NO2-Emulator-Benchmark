"""Runs a 200-step synthetic training with one seed on a given device, from identical starting weights and batches, and writes every step's loss, so that devices can be compared."""
import datetime
import json
import os
import sys
import tempfile
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from train import cfg
POOL = 256

def synthetic_pool(n_in, seed, pool=POOL):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    t = cfg.TILE
    x = rng.random((pool, n_in, t, t), dtype=np.float64)
    k = 5
    pad = np.pad(x[:, :3], ((0, 0), (0, 0), (k // 2, k // 2), (k // 2, k // 2)), mode='edge')
    sm = np.zeros((pool, 3, t, t))
    for a in range(k):
        for b in range(k):
            sm += pad[:, :, a:a + t, b:b + t]
    sm /= k * k
    y = (0.5 * sm[:, 0] + 0.3 * sm[:, 1] * sm[:, 2] + 0.2 * x[:, 0])[:, None]
    m = rng.random((pool, 1, t, t)) < 0.6
    return (x.astype(np.float32), y.astype(np.float32), m)

def main(argv):
    o = {'--network': None, '--width': None, '--device': None, '--out': None, '--steps': '200', '--seed': str(cfg.SEED_BASE), '--threads': None, '--batch': str(cfg.BATCH), '--work-dir': cfg.W, '--config-dir': cfg.LIVE_CONFIG}
    off = False
    i = 0
    while i < len(argv):
        if argv[i] == '--dropout-off':
            off = True
        elif argv[i] in o and i + 1 < len(argv):
            o[argv[i]] = argv[i + 1]
            i += 1
        else:
            raise SystemExit('unknown or incomplete option %r' % argv[i])
        i += 1
    for k in ('--network', '--width', '--device', '--out'):
        if not o[k]:
            raise SystemExit('%s is required' % k)
    t = os.path.join(o['--work-dir'], 'tmp')
    os.makedirs(t, exist_ok=True)
    os.environ['TMP'] = t
    os.environ['TEMP'] = t
    tempfile.tempdir = t
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(t, 'torchinductor')
    os.environ['MPLCONFIGDIR'] = os.path.join(t, 'mplconfig')
    import numpy as np
    import torch
    from train import emul_core as nets
    from train.emul_core import ioa_loss
    from train import flags, loop
    from train import plan as plan_mod
    dev = o['--device']
    seed = int(o['--seed'])
    steps = int(o['--steps'])
    batch = int(o['--batch'])
    threads = int(o['--threads']) if o['--threads'] else flags.rule_threads()
    started = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = flags.setup(dev, threads)
    net = o['--network']
    _, chset, _ = plan_mod.network_spec(net)
    chans = cfg.read_json(os.path.join(o['--config-dir'], 'S3_CHANNELS.json'))
    n_in = len(chans[chset + '_channels'])
    x, y, m = synthetic_pool(n_in, seed)
    flags.seed_everything(seed)
    model = loop.build_model(net, n_in, float(o['--width']))
    if off:
        nets.set_dropout(model, 0.0)
    model = model.to(dev)
    model.train()
    opt = loop.make_optimizer(model)
    rng = np.random.Generator(np.random.PCG64(seed + 1))
    losses = []
    for s in range(steps):
        idx = rng.integers(0, POOL, size=batch)
        xb = torch.from_numpy(x[idx]).to(dev)
        yb = torch.from_numpy(y[idx]).to(dev)
        mb = torch.from_numpy(m[idx]).to(dev)
        for g in opt.param_groups:
            g['lr'] = loop.lr_at(s)
        opt.zero_grad(set_to_none=True)
        loss = ioa_loss(model(xb), yb, mb)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    out = {'schema': 'choi-equiv-1', 'started': started, 'ended': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'network': net, 'in_ch': n_in, 'width': float(o['--width']), 'device': dev, 'seed': seed, 'steps': steps, 'batch': batch, 'dropout': 'off (p = 0)' if off else 'as designed', 'data': 'synthetic pool of %d tiles from the seed; no run data' % POOL, 'losses': losses, 'final_loss': losses[-1] if losses else None, 'mean_last10': sum(losses[-10:]) / len(losses[-10:]) if losses else None, 'flags': flags.flag_report(dev, rows)}
    tmp = o['--out'] + '.partial'
    os.makedirs(os.path.dirname(os.path.abspath(o['--out'])), exist_ok=True)
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    os.replace(tmp, o['--out'])
    print('equiv %s w=%s %s seed %d: %d steps, final loss %.6f' % (net, o['--width'], dev, seed, steps, out['final_loss']))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
