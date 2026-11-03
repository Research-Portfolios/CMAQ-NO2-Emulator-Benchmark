"""Times training steps of one network, width and device: warm-up and timed steps on synthetic tensors and on real batches, the median seconds per step, and the peak memory, written as JSON; no loss or weight is kept."""
import datetime
import json
import os
import statistics
import sys
import tempfile
import time
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from train import cfg
WARMUP = 10
TIMED = 30

def _now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def parse(argv):
    o = {'network': None, 'width': None, 'device': None, 'out': None, 'threads': None, 'config_dir': cfg.LIVE_CONFIG, 'arrays_dir': None, 'synthetic_only': False, 'test_values': False, 'batch': cfg.BATCH, 'warmup': WARMUP, 'timed': TIMED, 'work_dir': cfg.W, 'norm': None, 'years': None, 'departures': [], 'device_inputs': None, 'c7a_applies': 'true'}
    keys = {'--network': 'network', '--width': 'width', '--device': 'device', '--out': 'out', '--threads': 'threads', '--config-dir': 'config_dir', '--arrays-dir': 'arrays_dir', '--batch': 'batch', '--warmup': 'warmup', '--timed': 'timed', '--work-dir': 'work_dir', '--norm': 'norm', '--years': 'years', '--device-inputs': 'device_inputs', '--c7a-applies': 'c7a_applies'}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--synthetic-only':
            o['synthetic_only'] = True
        elif a == '--test-values':
            o['test_values'] = True
        elif a == '--departure' and i + 1 < len(argv):
            o['departures'].append(argv[i + 1])
            i += 1
        elif a in keys and i + 1 < len(argv):
            o[keys[a]] = argv[i + 1]
            i += 1
        else:
            raise SystemExit('unknown or incomplete option %r' % a)
        i += 1
    for k in ('network', 'width', 'device', 'out'):
        if not o[k]:
            raise SystemExit('--%s is required' % k)
    o['width'] = float(o['width'])
    for k in ('batch', 'warmup', 'timed'):
        o[k] = int(o[k])
    if o['years'] is not None:
        o['years'] = [int(s) for s in str(o['years']).split(',') if s]
    if o['device_inputs'] is not None:
        o['device_inputs'] = [int(s) for s in str(o['device_inputs']).split(',') if s]
    if str(o['c7a_applies']).lower() not in ('true', 'false'):
        raise SystemExit('--c7a-applies takes true or false')
    o['c7a_applies'] = str(o['c7a_applies']).lower() == 'true'
    if not o['test_values']:
        if (o['batch'], o['warmup'], o['timed']) != (cfg.BATCH, WARMUP, TIMED):
            raise SystemExit('batch, warmup and timed are fixed (64, 10, 30) without --test-values')
        if o['width'] not in cfg.WIDTHS:
            raise SystemExit('width %r not in %s' % (o['width'], cfg.WIDTHS))
    return o

def set_env(work_dir):
    t = os.path.join(work_dir, 'tmp')
    os.makedirs(t, exist_ok=True)
    os.environ['TMP'] = t
    os.environ['TEMP'] = t
    tempfile.tempdir = t
    for k, sub in (('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('MPLCONFIGDIR', 'mplconfig')):
        p = os.path.join(t, sub)
        os.makedirs(p, exist_ok=True)
        os.environ[k] = p

class Peak:

    def __init__(self, device):
        import psutil
        self.proc = psutil.Process()
        self.device = device
        self.rss_max = 0

    def sample(self):
        r = self.proc.memory_info().rss
        if r > self.rss_max:
            self.rss_max = r

    def report(self):
        import torch
        mi = self.proc.memory_info()
        d = {'rss_max_bytes': int(self.rss_max), 'peak_wset_bytes': int(getattr(mi, 'peak_wset', 0) or 0), 'peak_pagefile_bytes': int(getattr(mi, 'peak_pagefile', 0) or 0)}
        if self.device == 'xpu':
            d['xpu_max_memory_reserved_bytes'] = int(torch.xpu.max_memory_reserved())
        elif self.device == 'cuda':
            d['cuda_max_memory_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
            d['cuda_max_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
            d['cuda_total_memory_bytes'] = int(torch.cuda.get_device_properties(0).total_memory)
        return d

def device_inputs_plan(configs, net, priorities, c7a_applies):
    from train import data
    from train import plan as plan_mod
    best = None
    for t in plan_mod.trainings_for(priorities, c7a_applies):
        if t['net'] != net:
            continue
        sp = data.Split(configs, t)
        _, chset, logs = plan_mod.network_spec(t['net'])
        cp = data.ChannelPlan(configs, chset, logs, configs.norm(t['norm']), data.emission_order_for(configs, t))
        need = {}
        for yr, i, _ in list(sp.train) + list(sp.val):
            need.setdefault(int(yr), set()).add(int(i))
        tens, grid = ([], None)
        for yr in sorted(need):
            n = len(need[yr])
            shp = configs.arrays['years'][str(yr)]['files']['inputs_f16']['shape']
            h, w = (int(shp[2]), int(shp[3]))
            grid = (h, w)
            if cp.pos16:
                tens.append(([n, len(cp.pos16), h, w], 'float16'))
            if cp.pos32:
                tens.append(([n, len(cp.pos32), h, w], 'float32'))
            if t.get('arm'):
                tens.append(([n, len(cp.emis_k), h, w], 'float32'))
            tens.append(([n, h, w], 'float32'))
        if grid is not None:
            tens.append(([grid[0], grid[1]], 'bool'))
        size = {'float16': 2, 'float32': 4, 'bool': 1}
        nbytes = 0
        for shape, dt in tens:
            k = size[dt]
            for s in shape:
                k *= int(s)
            nbytes += k
        if best is None or nbytes > best['bytes']:
            best = {'training': t['name'], 'priority': t['priority'], 'bytes': int(nbytes), 'tensors': tens, 'days_by_year': {str(y): len(v) for y, v in sorted(need.items())}}
    if best is None:
        raise SystemExit('--device-inputs: no training of network %r in priorities %s' % (net, priorities))
    best['reader_peak_bytes'] = int(reader_peak_bytes(best['tensors']))
    return best
_DT_BYTES = {'float16': 2, 'float32': 4, 'bool': 1}

def _nbytes(shape, dt):
    k = _DT_BYTES[dt]
    for s in shape:
        k *= int(s)
    return k

def year_groups(tensors):
    groups, cur, mask = ([], [], None)
    for k, (shape, dt) in enumerate(tensors):
        if dt == 'bool':
            mask = k
            continue
        cur.append(k)
        if len(shape) == 3:
            groups.append(cur)
            cur = []
    if cur:
        raise ValueError("device inputs: a year's tensors without its target")
    return (groups, mask)

def reader_peak_bytes(tensors):
    groups, mask = year_groups(tensors)
    done, peak, last = (0, 0, 0)
    for grp in groups:
        yb = sum((_nbytes(*tensors[k]) for k in grp))
        peak = max(peak, done + 2 * yb)
        done += yb
        last = yb
    mb = _nbytes(*tensors[mask]) if mask is not None else 0
    return max(peak, done + last + mb)

def hold_device_inputs(plan_d, device):
    import torch
    dts = {'float16': torch.float16, 'float32': torch.float32, 'bool': torch.bool}
    tens = plan_d['tensors']
    groups, mask = year_groups(tens)

    def make(k):
        shape, dt = tens[k]
        return torch.empty(tuple(shape), dtype=dts[dt], device=device)
    held = [None] * len(tens)
    parts = None
    for grp in groups:
        parts = None
        parts = [make(k) for k in grp]
        for k in grp:
            held[k] = make(k)
    if mask is not None:
        held[mask] = make(mask)
    del parts
    return held

def _steps(model, opt, get_batch, n, device, peak, loss_fn):
    from train import flags
    tot, dat = ([], [])
    for _ in range(n):
        flags.sync(device)
        t0 = time.perf_counter()
        x, y, m = get_batch()
        flags.sync(device)
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y, m)
        loss.backward()
        opt.step()
        flags.sync(device)
        t2 = time.perf_counter()
        del loss
        tot.append(t2 - t0)
        dat.append(t1 - t0)
        peak.sample()
    return (tot, dat)

def main(argv):
    o = parse(argv)
    set_env(o['work_dir'])
    import psutil
    import torch
    from train.emul_core import ioa_loss
    from train import data, flags, loop
    from train import plan as plan_mod
    started = _now()
    free0 = psutil.virtual_memory().available / 1000000000.0
    dev = o['device']
    threads = int(o['threads']) if o['threads'] else flags.rule_threads()
    setup_rows = flags.setup(dev, threads)
    if dev == 'xpu':
        torch.xpu.reset_peak_memory_stats()
    elif dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    peak = Peak(dev)
    net = o['network']
    tr = [t for t in plan_mod.all_trainings(True) if t['held_out_year'] == 2016 and t['net'] == net and (t['kind'] == 'fold')]
    if not tr:
        raise SystemExit('no fold-2016 training for network %r' % net)
    tr = tr[0]
    _, chset, logs = plan_mod.network_spec(net)
    paths = cfg.Paths(o['config_dir'], o['work_dir'], o['work_dir'], os.path.join(o['work_dir'], 'tmp', 'bench_unused'), o['arrays_dir'])
    configs = cfg.Configs(paths)
    n_in = len(configs.channels[chset + '_channels'])
    flags.seed_everything(tr['seed'])
    model = loop.build_model(net, n_in, o['width'])
    minfo = loop.model_info(model, net, n_in, o['width'])
    model = model.to(dev)
    model.train()
    opt = loop.make_optimizer(model)
    b = o['batch']
    g = torch.Generator().manual_seed(cfg.SEED_BASE)
    xs = torch.rand(b, n_in, cfg.TILE, cfg.TILE, generator=g).to(dev)
    ys = torch.rand(b, 1, cfg.TILE, cfg.TILE, generator=g).to(dev)
    ms = (torch.rand(b, 1, cfg.TILE, cfg.TILE, generator=g) < 0.6).to(dev)
    out = {'schema': 'choi-bench-1', 'started': started, 'network': net, 'model': minfo, 'width': o['width'], 'device': dev, 'threads': threads, 'batch': b, 'warmup': o['warmup'], 'timed': o['timed'], 'test_values': o['test_values'], 'departures': list(o['departures']), 'kept': 'step times and memory only; no loss, metric or weight', 'free_memory_gb_start': round(free0, 3)}
    held = None
    if o['device_inputs']:
        di = device_inputs_plan(configs, net, o['device_inputs'], o['c7a_applies'])
        held = hold_device_inputs(di, dev)
        out['device_inputs'] = dict(di, priorities=list(o['device_inputs']), c7a_applies=o['c7a_applies'], rule='synthetic tensors of the shapes train.data.DeviceReader holds for the named training, on the device from before the first warm-up step to the end (no array read)')
    tot, dat = _steps(model, opt, lambda: (xs, ys, ms), o['warmup'], dev, peak, ioa_loss)
    tot, dat = _steps(model, opt, lambda: (xs, ys, ms), o['timed'], dev, peak, ioa_loss)
    out['synthetic'] = {'step_seconds': [round(t, 6) for t in tot], 'median_s': statistics.median(tot)}
    out['peak_after_synthetic'] = peak.report()
    if not o['synthetic_only']:
        own_norm = tr['norm']
        tr = dict(tr, norm=o['norm'] or own_norm)
        td = data.TrainingData(configs, paths, tr, device=dev, reader='memmap', val_tiles=1, norm_check=tr['norm'] == own_norm, years=o['years'], departures=o['departures'])
        drawn = set()
        out['real_data'] = {'training': tr['name'], 'seed': tr['seed'], 'split': td.split.describe(), 'channels': td.plan.describe()['channels'], 'norm': 'NORM_%s.json' % tr['norm'], 'years_read': list(td.read_years), 'years_restriction': td.years_restriction, 'reader': 'memmap', 'reader_staging': td.reader.staging, 'reader_pinned': td.reader.pinned}
        if o['norm'] or o['years'] is not None:
            out['departure'] = "the real batches scaled by NORM_%s.json and drawn from the 2016 fold's training days of %s only (a labelled departure for fall-back (3)'s record; it chooses nothing)" % (tr['norm'], ', '.join((str(y) for y in td.read_years)))

        def real_batch():
            dr = td.draw_batch(b)
            drawn.update((int(x[0]) for x in dr))
            return td.batch(dr, 'train')
        _steps(model, opt, real_batch, o['warmup'], dev, peak, ioa_loss)
        tot2, dat2 = _steps(model, opt, real_batch, o['timed'], dev, peak, ioa_loss)
        out['real_data']['drawn_years'] = sorted(drawn)
        out['real'] = {'step_seconds': [round(t, 6) for t in tot2], 'data_seconds': [round(t, 6) for t in dat2], 'median_s': statistics.median(tot2), 'median_data_s': statistics.median(dat2)}
        out['slower_median_s'] = max(out['synthetic']['median_s'], out['real']['median_s'])
        del td
    else:
        out['real'] = None
        out['slower_median_s'] = out['synthetic']['median_s']
    out['peak'] = peak.report()
    del model, opt, xs, ys, ms, held
    out['free_memory_gb_end'] = round(psutil.virtual_memory().available / 1000000000.0, 3)
    out['flags'] = flags.flag_report(dev, setup_rows)
    out['configs_sha256'] = dict(configs.sha256)
    out['ended'] = _now()
    tmp = o['out'] + '.partial'
    os.makedirs(os.path.dirname(os.path.abspath(o['out'])), exist_ok=True)
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    os.replace(tmp, o['out'])
    print('bench %s w=%s %s: synthetic %.4f s, real %s s, slower %.4f s; peak wset %.2f GB%s' % (net, o['width'], dev, out['synthetic']['median_s'], '%.4f' % out['real']['median_s'] if out['real'] else '-', out['slower_median_s'], out['peak']['peak_wset_bytes'] / 1000000000.0, '; cuda allocated peak %.2f GB of %.2f GB%s' % (out['peak']['cuda_max_memory_allocated_bytes'] / 1000000000.0, out['peak']['cuda_total_memory_bytes'] / 1000000000.0, '; device inputs %.2f GB (%s)' % (out['device_inputs']['bytes'] / 1000000000.0, out['device_inputs']['training']) if out.get('device_inputs') else '') if dev == 'cuda' else ''))
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
