"""The loop of one training: the 1 - IOA loss with Adam and step decay, batches of 64 tiles, validation every 250 steps on 1,024 fixed tiles with early stopping, the best state saved and reloaded for prediction, resumable checkpoints, and stops on low memory or non-finite values."""
import concurrent.futures
import csv
import json
import math
import os
import time
import numpy as np
from train import cfg, flags

class MemStop(Exception):
    pass

class SimulatedStop(Exception):
    pass

class NonFiniteStop(Exception):

    def __init__(self, msg, record):
        super().__init__(msg)
        self.record = record

def lr_at(step, base=cfg.LR, gamma=cfg.LR_GAMMA, every=cfg.LR_EVERY):
    return base * gamma ** (int(step) // int(every))

class EarlyStop:

    def __init__(self, patience=cfg.PATIENCE):
        self.patience = int(patience)
        self.best = float('inf')
        self.best_step = None
        self.since = 0

    def update(self, loss, step):
        if loss < self.best:
            self.best, self.best_step, self.since = (float(loss), step, 0)
            return (True, False)
        self.since += 1
        return (False, self.since >= self.patience)

    def state(self):
        return {'patience': self.patience, 'best': self.best, 'best_step': self.best_step, 'since': self.since}

    def set_state(self, d):
        self.patience = int(d['patience'])
        self.best = float(d['best'])
        self.best_step = d['best_step']
        self.since = int(d['since'])

class MemGate:

    def __init__(self, read=None, sleep=time.sleep, floor_gb=cfg.MEM_PAUSE_GB, max_s=cfg.MEM_PAUSE_MAX_S, poll_s=cfg.MEM_POLL_S, clock=time.monotonic):
        self.read = read or self._psutil
        self.sleep = sleep
        self.floor_gb = floor_gb
        self.max_s = max_s
        self.poll_s = poll_s
        self.clock = clock

    @staticmethod
    def _psutil():
        import psutil
        return psutil.virtual_memory().available / 1000000000.0

    def check(self, note=None):
        gb = self.read()
        if gb >= self.floor_gb:
            return (gb, 0.0)
        t0 = self.clock()
        if note:
            note('memory %.2f GB under %.1f GB: paused' % (gb, self.floor_gb))
        while True:
            self.sleep(self.poll_s)
            gb = self.read()
            waited = self.clock() - t0
            if gb >= self.floor_gb:
                if note:
                    note('memory %.2f GB: resumed after %.0f s' % (gb, waited))
                return (gb, waited)
            if waited >= self.max_s:
                raise MemStop('free memory %.2f GB under %.1f GB for %.0f s' % (gb, self.floor_gb, waited))

def build_model(net, n_in, width):
    from train import emul_core
    return emul_core.build_net(net, width, n_in)

def fresh_model_from_best(net, n_in, width, best_path, device):
    from train import emul_core
    best = load_ckpt(best_path)
    m = emul_core.build_net(net, width, n_in)
    m.load_state_dict(best['model'])
    m = m.to(device)
    m.eval()
    return (m, best)

def make_optimizer(model, lr=cfg.LR):
    from train import emul_core
    return emul_core.make_adam(model, lr)

def model_info(model, net, n_in, width):
    from train import emul_core
    d = emul_core.macs_detail(model, n_in)
    return {'net': net, 'class': type(model).__name__, 'in_ch': n_in, 'width': width if net != 'pixmlp' else 'fixed', 'parameters': emul_core.count_params(model), 'float_buffer_elements': emul_core.count_buffers(model), 'macs_per_tile_conv': d['conv_and_transposed'], 'macs_per_tile_linear': d['linear'], 'layers': [{k: r.get(k) for k in ('layer', 'kind', 'kernel', 'stride', 'padding', 'padding_mode', 'bias')} for r in d['layers']], 'core_version': emul_core.CORE_VERSION}

def hyperparameters(training, values, steps):
    return {'loss': '1 - IOA over the masked cells, one O-bar per batch', 'optimizer': 'Adam, betas 0.9 and 0.999, eps 1e-8, no weight decay', 'learning_rate': '%g x 0.93^(step // 120)' % float(training.get('lr0', cfg.LR)), 'batch': values['batch'], 'steps': steps, 'validation': 'every %d steps on %d fixed tiles' % (values['eval_every'], values['val_tiles']), 'early_stop': 'after %d evaluations without improvement' % values['patience'], 'kept_state': 'the whole state_dict at each improving validation, .partial then renamed', 'gradient_clipping': 'none', 'non_finite': 'stops the training', 'precision': 'float32, no autocast, no GradScaler', 'initialisation': 'the defaults of the torch that runs, seeded by the configured seed', 'layers': 'the layers (1) to (9) of the study design', 'augmentation': 'none', 'tile_rule': '64 x 64, at least 25 percent masked', 'dropout': 'The advanced network only: torch.nn.Dropout p 0.1 after the 8 x 8 block, training mode only', 'calendar_information': 'only through the emission inputs'}

def _write_json_atomic(d, path, tries=20, gap=0.5):
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(d, f, indent=1, sort_keys=True, default=str)
        f.write('\n')
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k + 1 == tries:
                raise
            time.sleep(gap)

def load_ckpt(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)

class Prefetcher:

    def __init__(self, td, batch, ahead=True):
        self.td = td
        self.batch = int(batch)
        self.ex = concurrent.futures.ThreadPoolExecutor(1) if ahead else None
        self.nxt = None

    def _draw(self):
        draws = self.td.draw_batch(self.batch)
        st = self.td.sampler_state()
        fut = self.ex.submit(self.td.reader.raw, draws, 'train') if self.ex else None
        return (draws, st, fut)

    def get(self):
        if self.nxt is None:
            self.nxt = self._draw()
        draws, st, fut = self.nxt
        raw = fut.result() if fut else self.td.reader.raw(draws, 'train')
        self.nxt = self._draw() if self.ex else None
        return (draws, st, raw)

    def close(self):
        if self.ex:
            if self.nxt is not None and self.nxt[2] is not None:
                try:
                    self.nxt[2].result()
                except Exception:
                    pass
            self.ex.shutdown(wait=True)
            self.ex = None

def statistic(model, td, draws, mode, chunk):
    import torch
    from train import emul_core
    was = model.training
    model.eval()
    outs, ys, ms = ([], [], [])
    try:
        with torch.no_grad():
            for a in range(0, len(draws), chunk):
                x, y, m = td.batch(draws[a:a + chunk], mode)
                outs.append(model(x).to('cpu', torch.float32).numpy())
                ys.append(y.to('cpu', torch.float32).numpy())
                ms.append(m.to('cpu').numpy())
    finally:
        model.train(was)
    return emul_core.ioa_stat(np.concatenate(outs), np.concatenate(ys), np.concatenate(ms))

def validate(model, td, chunk):
    return statistic(model, td, td.val_draws, 'val', chunk)

class TrainLog:
    FIELDS = ('step', 'lr', 'train_loss_mean', 'val_loss', 'train_stat', 'best_val_loss', 'best_step', 'evals_since_best', 'elapsed_s', 'free_memory_gb', 'tiles_drawn')

    def __init__(self, paths, name, head):
        self.paths = paths
        self.name = name
        self.d = dict(head)
        self.d.setdefault('evaluations', [])
        self.d.setdefault('events', [])
        os.makedirs(paths.ckpt(name), exist_ok=True)
        os.makedirs(paths.train_logs, exist_ok=True)
        prev = os.path.join(paths.ckpt(name), 'train_log.json')
        if os.path.isfile(prev):
            try:
                with open(prev, 'rb') as f:
                    old = json.loads(f.read().decode('ascii'))
                self.d['first_started'] = old.get('first_started') or old.get('started')
                self.d['events'] = list(old.get('events', [])) + self.d['events']
                self.d['memory'] = list(old.get('memory', []))
                self.d['warnings_earlier_processes'] = list(old.get('warnings_earlier_processes', [])) + ([old['warnings']] if old.get('warnings') else [])
                self.event('a new process opened this log')
            except Exception:
                self.event('an earlier log could not be read')

    def event(self, text, **kw):
        e = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': text}
        e.update(kw)
        self.d['events'].append(e)

    def evaluation(self, row):
        self.d['evaluations'].append(row)

    def write(self):
        _write_json_atomic(self.d, os.path.join(self.paths.ckpt(self.name), 'train_log.json'))
        _write_json_atomic(self.d, os.path.join(self.paths.train_logs, self.name + '.json'))
        p = os.path.join(self.paths.train_logs, self.name + '.csv')
        tmp = p + '.partial'
        with open(tmp, 'w', encoding='ascii', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(self.FIELDS)
            for r in self.d['evaluations']:
                w.writerow([r.get(k) for k in self.FIELDS])
        os.replace(tmp, p)

def stopped_path(paths, name):
    return os.path.join(paths.ckpt(name), 'STOPPED.json')

def stopped_record(paths, name):
    p = stopped_path(paths, name)
    if not os.path.isfile(p):
        return None
    try:
        return cfg.read_json(p)
    except cfg.ConfigError:
        return {'status': 'stopped (record unreadable)'}

def train(td, model, opt, steps, values, device, paths, log, note=None, memgate=None, stop_after=None, ahead=None, plan_id=None, on_step=None, inject_nan_at=None):
    import torch
    from train import emul_core
    name = td.training['name']
    lr0 = float(td.training.get('lr0', cfg.LR))
    cdir = paths.ckpt(name)
    os.makedirs(cdir, exist_ok=True)
    last_p = os.path.join(cdir, 'last.pt')
    best_p = os.path.join(cdir, 'best.pt')
    batch = values['batch']
    eval_every, ckpt_every = (values['eval_every'], values['ckpt_every'])
    mem_every = values['mem_every']
    early = EarlyStop(values['patience'])
    memgate = memgate or MemGate()
    step = 0
    elapsed0 = 0.0
    loss_sum, n_sum = (0.0, 0)
    stopped_early = False
    last_ckpt_step = None
    if os.path.isfile(last_p):
        ck = load_ckpt(last_p)
        if ck.get('training') != name or ck.get('seed') != td.training['seed']:
            raise RuntimeError('last.pt of another training or seed')
        if plan_id and (ck.get('plan_id') or {}) != dict(plan_id):
            raise RuntimeError('last.pt of another plan: %r, now %r' % (ck.get('plan_id'), dict(plan_id)))
        if ck.get('format') != 'choi-train-ckpt-2':
            raise RuntimeError('last.pt format %r (checkpoints of the earlier version are not resumed by this code)' % ck.get('format'))
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        td.set_sampler_state(ck['sampler'])
        flags.set_rng_states(ck['rng'], device)
        early.set_state({'patience': ck['patience'], 'best': ck['best'], 'best_step': ck['best_step'], 'since': ck['since_best']})
        step = int(ck['step'])
        last_ckpt_step = step
        elapsed0 = float(ck['elapsed_s'])
        loss_sum, n_sum = (float(ck['loss_acc'][0]), int(ck['loss_acc'][1]))
        stopped_early = bool(ck.get('stopped_early', False))
        log.d['evaluations'] = list(ck['evaluations'])
        log.event('resumed from last.pt', step=step, phase=ck.get('phase'))
        if note:
            note('%s resumed at step %d (%s)' % (name, step, ck.get('phase')))
        if ck.get('phase') == 'trained':
            log.d['train_result'] = ck['result']
            log.write()
            return _result(ck['result'], early)
    t0 = time.monotonic()
    pf = Prefetcher(td, batch, ahead=td.reader_kind == 'memmap' if ahead is None else ahead)
    model.train()
    last_state = td.sampler_state()

    def save_last(phase, result=None):
        emul_core.save_atomic(emul_core.resume_state(format='choi-train-ckpt-2', training=name, seed=td.training['seed'], step=step, steps=steps, phase=phase, plan_id=dict(plan_id) if plan_id else None, model=model.state_dict(), optimizer=opt.state_dict(), schedule={'base_lr': lr0, 'gamma': cfg.LR_GAMMA, 'every': cfg.LR_EVERY, 'next_lr': lr_at(step, lr0)}, best=early.best, best_step=early.best_step, since_best=early.since, patience=early.patience, rng=flags.get_rng_states(device), sampler=last_state, elapsed_s=elapsed0 + time.monotonic() - t0, loss_acc=[loss_sum, n_sum], stopped_early=stopped_early, evaluations=log.d['evaluations'], result=result), last_p)

    def nonfinite(kind, value):
        rec = {'schema': 'choi-train-stopped-1', 'training': name, 'status': 'stopped: non-finite %s' % kind, 'step': step, 'value': repr(value), 'last_finite_checkpoint_step': last_ckpt_step, 'rule': 'the training stops at once; listed NOT DONE; not run again with any changed setting without a decision', 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        log.d['stopped'] = rec
        log.event('stopped: non-finite %s' % kind, step=step, value=repr(value), last_finite_checkpoint_step=last_ckpt_step)
        log.write()
        _write_json_atomic(rec, stopped_path(paths, name))
        raise NonFiniteStop('%s: non-finite %s at step %d' % (name, kind, step), rec)
    tally = emul_core.WarnTally()
    try:
        with tally:
            while step < steps and (not stopped_early):
                _, st, raw = pf.get()
                x = td.scaler.inputs(raw[0])
                y = td.scaler.target(raw[1])
                m = raw[2]
                lr = lr_at(step, lr0)
                loss = emul_core.train_step(model, opt, x, y, m, lr)
                step += 1
                last_state = st
                lv = float(loss)
                if inject_nan_at is not None and step == inject_nan_at:
                    lv = float('nan')
                if on_step is not None:
                    on_step(step, lv)
                if not math.isfinite(lv):
                    nonfinite('training loss', lv)
                loss_sum += lv
                n_sum += 1
                final = step == steps
                if step % eval_every == 0 or final:
                    train_mean = loss_sum / max(1, n_sum)
                    loss_sum, n_sum = (0.0, 0)
                    val = validate(model, td, batch)
                    if not math.isfinite(val):
                        nonfinite('validation statistic', val)
                    tstat = statistic(model, td, td.stat_draws, 'train', batch)
                    improved, stop = early.update(val, step)
                    if improved:
                        emul_core.save_atomic({'format': 'choi-train-best-2', 'training': name, 'step': step, 'val_loss': val, 'val_stat': val, 'model': model.state_dict()}, best_p)
                    gb = memgate.read()
                    row = {'step': step, 'lr': lr, 'train_loss_mean': train_mean, 'val_loss': val, 'train_stat': tstat, 'best_val_loss': early.best, 'best_step': early.best_step, 'evals_since_best': early.since, 'elapsed_s': round(elapsed0 + time.monotonic() - t0, 3), 'free_memory_gb': round(gb, 3), 'tiles_drawn': step * batch}
                    log.evaluation(row)
                    if stop:
                        stopped_early = True
                        log.event('early stop', step=step, best_step=early.best_step)
                    log.d['warnings'] = tally.summary()
                    log.write()
                    if note:
                        note('%s step %d val %.6f best %.6f at %s%s' % (name, step, val, early.best, early.best_step, ' EARLY STOP' if stop else ''))
                if step % ckpt_every == 0 or final or stopped_early:
                    save_last('train')
                    last_ckpt_step = step
                if step % mem_every == 0 and (not (final or stopped_early)):
                    gb, waited = memgate.check(note)
                    log.d.setdefault('memory', []).append({'step': step, 'free_memory_gb': round(gb, 3), 'paused_s': round(waited, 1)})
                if stop_after is not None and step == stop_after:
                    raise SimulatedStop('stopped after step %d' % step)
    finally:
        pf.close()
        log.d['warnings'] = tally.summary()
    if not os.path.isfile(best_p):
        raise RuntimeError('no best checkpoint after %d steps' % step)
    res = {'steps_run': step, 'steps_planned': steps, 'early_stopped': stopped_early, 'best_step': early.best_step, 'kept_step': early.best_step, 'last_step': step, 'best_val_loss': early.best, 'tiles_drawn': step * batch, 'lr0': lr0, 'warnings_nondeterministic': tally.summary()['nondeterministic_count'], 'train_wall_s': round(elapsed0 + time.monotonic() - t0, 3)}
    save_last('trained', res)
    log.d['train_result'] = res
    log.write()
    return _result(res, early)

def _result(res, early):
    out = dict(res)
    out.setdefault('best_step', early.best_step)
    out.setdefault('best_val_loss', early.best)
    return out

def state_digest(model, opt=None):
    import hashlib
    import torch
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode('ascii'))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    if opt is not None:
        sd = opt.state_dict()
        for pid in sorted(sd['state']):
            for k in sorted(sd['state'][pid]):
                v = sd['state'][pid][k]
                if isinstance(v, torch.Tensor):
                    h.update(v.detach().cpu().contiguous().numpy().tobytes())
                else:
                    h.update(repr(v).encode('ascii'))
    return h.hexdigest()

def np_digest(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
