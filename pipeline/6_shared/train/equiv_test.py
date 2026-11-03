"""The equivalence test of the training code: one forward and backward pass on fixed tensors and weights and a 200-step synthetic training, for each network on one device, compared with the CPU where the device differs."""
import os
import sys
import time
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import numpy as np
from train import emul_core as ec
EQUIV_NETS = ('baseline', 'advanced')
DEV_BATCH = 8
TRAIN_STEPS = 200
TRAIN_BATCH = 64
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor', 'compare_core.py')
VENDOR_SHA256 = '413064f1d5b4377aae9bbad3d6ba9001109205e661625d61d764f672138d6787'

def load_compare():
    import hashlib
    with open(VENDOR, 'rb') as f:
        b = f.read()
    if hashlib.sha256(b).hexdigest() != VENDOR_SHA256:
        raise SystemExit("train\\vendor\\compare_core.py is not the kernels' compare part")
    ns = {'np': np, '__name__': 'compare_core'}
    exec(compile(b.decode('ascii'), VENDOR, 'exec'), ns)
    return ns

def mask_provider(key, p=0.1):
    calls = [0]

    def provide(shape):
        n = int(np.prod(shape))
        u = ec.uniform(tuple(key) + (calls[0],), n)
        calls[0] += 1
        return (u >= p).astype(np.float32).reshape(shape)
    return provide

def device_test(name, device, width, batch, out_dir, tag):
    import torch
    mid = ec.MODEL_IDS[name]
    ec.seed_all(ec.training_seed(name))
    net = ec.build_net(name, width)
    ec.fill_params(net, (ec.DATA_SEED, 1, mid))
    wsha = ec.params_sha(net)
    net = net.to(device).train()
    if hasattr(net, 'drop'):
        net.drop.mask_provider = mask_provider((ec.DATA_SEED, 3, mid), net.drop.p)
    x, y, m = ec.synth_batch((ec.DATA_SEED, 2, mid), batch, ec.N_INPUTS[name])
    isha = ec.sha_arrays(x, y, m)
    xt, yt, mt = [torch.from_numpy(a).to(device) for a in (x, y, m)]
    net.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    pred = net(xt)
    loss = ec.ioa_loss(pred, yt, mt)
    loss.backward()
    ec.sync(device)
    secs = time.perf_counter() - t0
    outp = pred.detach().to('cpu').numpy().astype(np.float32)
    fname = 'device_test_%s_%s_output.npy' % (name, tag)
    np.save(os.path.join(out_dir, fname), outp)
    res = {'weights_sha256': wsha, 'inputs_sha256': isha, 'loss': float(loss.item()), 'output_file': fname, 'output_sha256': ec.sha_arrays(outp), 'seconds': secs, 'batch': batch, 'width': width, 'params': ec.count_params(net), 'n_inputs': ec.N_INPUTS[name], 'grads': ec.grad_summary(net, (ec.PROJ_SEED, mid))}
    del net, pred, loss, xt, yt, mt
    return (res, outp)

def train200(name, device, width, steps, batch, ckpt_dir, tag, final_last):
    import torch
    mid = ec.MODEL_IDS[name]
    ec.seed_all(ec.training_seed(name))
    net = ec.build_net(name, width)
    ec.fill_params(net, (ec.DATA_SEED, 4, mid))
    wsha = ec.params_sha(net)
    net = net.to(device).train()
    opt = ec.make_adam(net)
    losses, start = ([], 0)
    ck = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        ck = os.path.join(ckpt_dir, 'train200_%s_%s.pt' % (name, tag))
        if os.path.isfile(ck):
            state = ec.torch_load(ck, device)
            net.load_state_dict(state['model'])
            opt.load_state_dict(state['opt'])
            losses, start = (list(state['losses']), int(state['step']))
    t0 = time.perf_counter()
    for step in range(start, steps):
        x, y, m = ec.synth_batch((ec.DATA_SEED, 5, step), batch, ec.N_INPUTS[name])
        if hasattr(net, 'drop'):
            net.drop.mask_provider = mask_provider((ec.DATA_SEED, 6, step), net.drop.p)
        xt, yt, mt = [torch.from_numpy(a).to(device) for a in (x, y, m)]
        loss = ec.train_step(net, opt, xt, yt, mt, ec.lr_at(step))
        losses.append(float(loss.item()))
        if ck and ((step + 1) % ec.CKPT_EVERY == 0 or step + 1 == steps):
            ec.save_checkpoint(ck, step + 1, net, opt, losses)
    ec.sync(device)
    secs = time.perf_counter() - t0
    del net, opt
    return {'weights_sha256_start': wsha, 'losses': losses, 'final_mean_last10': float(np.mean(losses[-final_last:])), 'loss_last': losses[-1], 'seconds': secs, 'steps': steps, 'batch': batch, 'width': width, 'resumed_from': start, 'n_inputs': ec.N_INPUTS[name]}

def run_equivalence(device_name='auto', out_dir=None, width=1.0, steps=TRAIN_STEPS, batch=TRAIN_BATCH, dev_batch=DEV_BATCH, nets=EQUIV_NETS, ckpt_dir=None, cpu_check=True, smoke=False, threads=None):
    import torch
    cc = load_compare()
    if smoke:
        width, steps, batch, dev_batch = (0.125, 3, 2, 2)
    device = ec.pick_device(device_name)
    out_dir = out_dir or ec.default_out('choi_emul_equiv')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'results.json')
    config = {'width': width, 'steps': steps, 'batch': batch, 'dev_batch': dev_batch, 'nets': list(nets), 'tol_device': cc['TOL_DEVICE'], 'tol_train': cc['TOL_TRAIN'], 'final_last': cc['FINAL_LAST'], 'seeds': {n: ec.training_seed(n) for n in nets}, 'data_seed': ec.DATA_SEED, 'proj_seed': ec.PROJ_SEED, 'n_proj': ec.N_PROJ, 'n_inputs': {n: ec.N_INPUTS[n] for n in nets}, 'core_version': ec.CORE_VERSION}
    res = {}
    if os.path.isfile(path):
        with open(path, encoding='ascii') as fh:
            res = json.load(fh)
        if res.get('config') != config or res.get('device_type') != device.type:
            raise SystemExit('results.json in this folder holds another configuration or device: use a new folder')
    flags = ec.set_flags(device, threads=threads)
    res.update({'test': 'equivalence', 'format': 1, 'smoke': bool(smoke), 'successor_of': 'work/choi_emul_equiv_test.py', 'config': config, 'flags': flags, 'env': ec.env_report(device), 'device_type': device.type})
    res['pip_freeze_sha256'] = ec.write_pip_freeze(os.path.join(out_dir, 'pip_freeze.txt'))
    res.setdefault('device_test', {})
    res.setdefault('same_machine_cpu', {})
    res.setdefault('train200', {})
    res['started_utc'] = res.get('started_utc') or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    probe = ec.precision_probe(device)
    res['precision_probe'] = probe
    ec.save_json(path, res)
    print('EQUIV device %s (%s); flags tf32_off %s; probe %s' % (res['env']['gpu'].get('name'), device.type, flags['tf32_off'], 'PASS' if probe['pass'] else 'FAIL'))
    if not probe['pass']:
        res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        res['gate'] = 'STOP: the precision probe failed'
        ec.save_json(path, res)
        return res
    tally = ec.WarnTally()
    with tally:
        for name in nets:
            if name in res['device_test']:
                continue
            r, out = device_test(name, device, width, dev_batch, out_dir, device.type)
            res['device_test'][name] = r
            if cpu_check and device.type != 'cpu':
                cpu = torch.device('cpu')
                rc, outc = device_test(name, cpu, width, dev_batch, out_dir, 'samecpu')
                res['same_machine_cpu'][name] = cc['compare_device_results'](r, rc, out, outc)
                print('EQUIV %s same-machine CPU: pass %s' % (name, res['same_machine_cpu'][name]['pass']))
            ec.save_json(path, res)
            print('EQUIV %s device test loss %.8f' % (name, r['loss']))
        for name in nets:
            if name in res['train200']:
                continue
            r = train200(name, device, width, steps, batch, ckpt_dir, device.type, cc['FINAL_LAST'])
            res['train200'][name] = r
            ec.save_json(path, res)
            print('EQUIV %s %d-step final %.6f in %.1f s' % (name, steps, r['final_mean_last10'], r['seconds']))
    w = tally.summary()
    res['warnings'] = w
    res['nondeterministic_count'] = w['nondeterministic_count']
    res['gate'] = 'STOP the main runs: %d warnings of an operation without a deterministic implementation' % w['nondeterministic_count'] if w['nondeterministic_count'] else 'PASS (none seen)'
    res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    ec.save_json(path, res)
    print('EQUIV done: gate %s' % res['gate'].split(' (')[0])
    return res

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="The equivalence test on the trainer's module")
    p.add_argument('--device', default='auto')
    p.add_argument('--out', required=True)
    p.add_argument('--width', type=float, default=1.0)
    p.add_argument('--ckpt-dir')
    p.add_argument('--no-cpu-check', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--config-dir')
    p.add_argument('--threads', type=int)
    a = p.parse_args(argv)
    if a.config_dir:
        with open(os.path.join(a.config_dir, 'S3_CHANNELS.json'), 'rb') as fh:
            ec.set_input_counts(json.loads(fh.read().decode('ascii')))
    run_equivalence(a.device, a.out, a.width, ckpt_dir=a.ckpt_dir, cpu_check=not a.no_cpu_check, smoke=a.smoke, threads=a.threads)
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
