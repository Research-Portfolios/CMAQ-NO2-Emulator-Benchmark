"""A short fit before the main trainings: each network trains 300 steps on 8 fixed tiles, and the loss at steps 0, 100, 200 and 300 is recorded as a check that the code learns."""
import json
import os
import sys
import time
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from train import cfg
NETS = ('baseline', 'advanced', 'pixmlp')
TILES = 8
STEPS = 300
RECORD_AT = (0, 100, 200, 300)
TILE_SEED = cfg.SEED_BASE + 700000
YEARS = (2017, 2019)

def parse(argv):
    o = {'device': 'cpu', 'out': None, 'config_dir': cfg.LIVE_CONFIG, 'arrays_dir': None, 'test_values': False, 'width': 1.0, 'steps': STEPS, 'threads': None, 'norm': 'fold2016'}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--test-values':
            o['test_values'] = True
            i += 1
            continue
        if i + 1 >= len(argv):
            raise SystemExit('incomplete option %r' % a)
        v = argv[i + 1]
        if a == '--device':
            o['device'] = v
        elif a == '--out':
            o['out'] = v
        elif a == '--config-dir':
            o['config_dir'] = v
        elif a == '--arrays-dir':
            o['arrays_dir'] = v
        elif a == '--width':
            o['width'] = float(v)
        elif a == '--steps':
            o['steps'] = int(v)
        elif a == '--threads':
            o['threads'] = int(v)
        elif a == '--norm':
            o['norm'] = v
        else:
            raise SystemExit('unknown option %r' % a)
        i += 2
    if not o['out']:
        raise SystemExit('--out is required')
    if not o['test_values'] and (o['width'] != 1.0 or o['steps'] != STEPS):
        raise SystemExit('w = 1 and 300 steps are fixed without --test-values')
    return o

def main(argv):
    o = parse(argv)
    t = os.path.join(os.path.dirname(os.path.abspath(o['out'])), 'tmp')
    os.makedirs(t, exist_ok=True)
    os.environ['TMP'] = t
    os.environ['TEMP'] = t
    import numpy as np
    from train import data, emul_core, flags, loop
    from train import plan as plan_mod
    dev = o['device']
    rows = flags.setup(dev, o['threads'] or flags.rule_threads())
    paths = cfg.Paths(o['config_dir'], t, t, os.path.join(t, 'unused'), o['arrays_dir'])
    configs = cfg.Configs(paths)
    out = {'schema': 'choi-tinyfit-1', 'made': time.strftime('%Y-%m-%d %H:%M:%S'), 'design': "8 tiles of the 2016 fold's training days of 2017 and 2019 (a labelled departure)", 'device': dev, 'width': o['width'], 'steps': o['steps'], 'batch': TILES, 'lr': 0.001, 'decay': 'none', 'early_stop': 'none', 'test_values': o['test_values'], 'tile_seed': TILE_SEED, 'flags': flags.flag_report(dev, rows), 'years_restriction': {'years': list(YEARS), 'rule': "the reader checks, opens and samples only the 2016 fold's training years 2017 and 2019 (train.data.TrainingData years, a labelled departure)"}, 'networks': {}}
    probe = emul_core.precision_probe(dev)
    out['precision_probe'] = probe
    tiles = None
    for net in NETS:
        tr = dict([x for x in plan_mod.all_trainings(True) if x['name'] == 'fold2016_%s' % net][0])
        tr['norm'] = o['norm']
        td = data.TrainingData(configs, paths, tr, device=dev, reader='memmap', val_tiles=1, train_stat_tiles=1, norm_check=o['norm'] == 'fold2016', years=YEARS)
        out['norm'] = 'NORM_%s.json' % o['norm']
        if tiles is None:
            days = [(y, i) for y, i, _ in td.split.train if y in YEARS]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(TILE_SEED)))
            di = rng.integers(0, len(days), size=TILES)
            pi = rng.integers(0, len(td.rows), size=TILES)
            tiles = [(days[a][0], days[a][1], int(td.rows[b]), int(td.cols[b])) for a, b in zip(di, pi)]
            out['tiles'] = [list(x) for x in tiles]
            out['tile_days'] = [td.years[y].dates[i] for y, i, _, _ in tiles]
        x, y, m = td.batch(tiles, 'train')
        flags.seed_everything(tr['seed'])
        model = loop.build_model(net, td.plan.n, o['width']).to(dev)
        model.train()
        opt = emul_core.make_adam(model, 0.001)
        rec = {}
        t0 = time.monotonic()
        for s in range(o['steps'] + 1):
            if s == o['steps']:
                import torch
                with torch.no_grad():
                    lv = float(emul_core.ioa_loss(model(x), y, m))
            else:
                lv = float(emul_core.train_step(model, opt, x, y, m, 0.001))
            if s in RECORD_AT or s == o['steps']:
                rec[str(s)] = lv
        l0, ln = (rec['0'], rec[str(o['steps'])])
        if net == 'pixmlp':
            flag = ln > 0.5 * l0
            rule = 'loss at the last step above half its loss at step 0'
        else:
            flag = ln > 0.1
            rule = '1 - IOA above 0.1 at the last step'
        out['networks'][net] = {'training_seed': tr['seed'], 'channels': td.plan.n, 'channel_set': td.plan.channel_set, 'years_read': list(td.read_years), 'losses': rec, 'rule': rule, 'flagged': bool(flag), 'verdict': 'sends the code to a fresh process before the trainings' if flag else 'within the rule', 'seconds': round(time.monotonic() - t0, 1)}
        print('tinyfit %s: %d steps, flagged %s' % (net, o['steps'], flag), flush=True)
        del model, opt, td
    out['any_flagged'] = any((v['flagged'] for v in out['networks'].values()))
    out['reads_held_out'] = False
    tmp = o['out'] + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    os.replace(tmp, o['out'])
    print('tinyfit written: %s' % cfg.rel(o['out']), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
