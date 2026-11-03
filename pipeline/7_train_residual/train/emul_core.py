"""The shared network and training-step module with a mean squared error loss beside 1 - IOA, gradient-norm clipping, a final layer initialized at zero for the residual networks, and the per-cell network PIX-R."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
import warnings
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CORE_VERSION = 'choi-emul-core-3-p8'
CUBLAS_SETTING = ':4096:8'
CUBLAS_KNOWN = (':4096:8', ':16:8')
BASE_SEED = 20261016
FOLD_YEARS = (2015, 2016, 2017, 2018, 2019)
FOLD_2016 = 1
TILE = 64
BASE_CHANNELS = (64, 128, 256, 512)
SE_REDUCTION = 8
DROPOUT_P = 0.1
LEAKY_SLOPE = 0.2
LR0, LR_DECAY, LR_EVERY = (0.001, 0.93, 120)
BETAS, ADAM_EPS, WEIGHT_DECAY = ((0.9, 0.999), 1e-08, 0.0)
CKPT_EVERY = 250
PROBE_SEED = 20261016
PROBE_TOL = 0.0001
NONDET_TEXT = 'does not have a deterministic implementation'
MODEL_IDS = {'baseline': 0, 'advanced': 1, 'f6_TRUE': 2, 'f6_OTHER': 3, 'f6_OWN': 4, 'gbt': 5, 'pixmlp': 6, 'e1x': 7, 'e2mx': 8, 'e6': 15, 'baseline24': 7, 'advanced18': 8}
N_INPUTS = {'baseline': 17, 'advanced': 23, 'pixmlp': 23, 'e1x': 23, 'e2mx': 17, 'e6': 17, 's18': 18, 'baseline24': 23, 'advanced18': 17, 'pixr': 24}
DATA_SEED = BASE_SEED + 9000000
PROJ_SEED = BASE_SEED + 9100000
N_PROJ = 16
_PROFILE_RE = re.compile('(?i)([A-Z]:[\\\\/]+Users[\\\\/]+|/home/|/Users/)[^\\\\/\\s]+')

def set_input_counts(channels_json):
    b = len(channels_json['baseline_channels'])
    a = len(channels_json['advanced_channels'])
    for k in ('baseline', 'e2mx', 'e6', 'advanced18'):
        N_INPUTS[k] = b
    for k in ('advanced', 'pixmlp', 'e1x', 'baseline24'):
        N_INPUTS[k] = a
    if 'arm18_channels' in channels_json:
        N_INPUTS['s18'] = len(channels_json['arm18_channels'])
    return dict(N_INPUTS)

def training_seed(model, fold_year=2016, offset=0):
    return BASE_SEED + 100 * FOLD_YEARS.index(int(fold_year)) + MODEL_IDS[model] + int(offset)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
    except Exception:
        pass

def as_device(device):
    return device if isinstance(device, torch.device) else torch.device(str(device))

def pick_device(name):
    if name in (None, '', 'auto'):
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')
        return torch.device('cpu')
    return torch.device(name)

def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)

def rule_threads():
    return max(1, physical_cores() - 1)

def _one_line(s, n=200):
    s = ' '.join(str(s).split())
    return s if len(s) <= n else s[:n - 3] + '...'

def _get(path):
    obj = torch
    try:
        for part in path.split('.'):
            if not hasattr(obj, part):
                return 'absent'
            obj = getattr(obj, part)
        if isinstance(obj, (bool, int, float, str)) or obj is None:
            return obj
        return _one_line(repr(obj), 60)
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__
PRECISION_READS = ('backends.cuda.matmul.allow_tf32', 'backends.cudnn.allow_tf32', 'backends.fp32_precision', 'backends.cuda.matmul.fp32_precision', 'backends.cudnn.fp32_precision', 'backends.cudnn.conv.fp32_precision', 'backends.cudnn.rnn.fp32_precision', 'backends.mkldnn.allow_tf32', 'backends.mkldnn.fp32_precision', 'backends.mkldnn.conv.fp32_precision', 'backends.mkldnn.matmul.fp32_precision')

def set_flags(device, threads=None):
    device = as_device(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device.type == 'cpu':
            torch.set_num_threads(int(threads) if threads else rule_threads())
    fl = read_flags()
    fl['set_warnings'] = sorted(set((_one_line(c.message) for c in caught)))
    return fl

def cublas_reading():
    v = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if v is None:
        return 'absent'
    return v if v in CUBLAS_KNOWN else 'another value (not logged)'

def k24_check(reads):
    bad, unknown = ([], [])
    for k, v in reads.items():
        if k.endswith('allow_tf32'):
            if v is True:
                bad.append(k)
            elif v not in (False, 'absent'):
                unknown.append(k)
        elif k.endswith('fp32_precision'):
            if v in ('tf32', 'bf16'):
                bad.append(k)
            elif v not in ('ieee', 'none', 'absent'):
                unknown.append(k)
    return {'result': 'FAIL' if bad else 'PROBE' if unknown else 'PASS', 'fail': bad, 'left_to_probe': unknown}

def read_flags():
    reads = {k: _get(k) for k in PRECISION_READS}
    chk = k24_check(reads)
    flags = {'torch_version': torch.__version__, 'precision_reads': reads, 'k24_check': chk, 'tf32_off': chk['result'] != 'FAIL', 'cudnn.benchmark': bool(torch.backends.cudnn.benchmark), 'cudnn.deterministic': bool(torch.backends.cudnn.deterministic), 'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()), 'deterministic_warn_only': bool(torch.is_deterministic_algorithms_warn_only_enabled()), 'float32_matmul_precision': _safe_call(torch.get_float32_matmul_precision), 'cublas_workspace_config': cublas_reading(), 'cublas_workspace_config_set': CUBLAS_SETTING, 'num_threads': int(torch.get_num_threads()), 'mkldnn.enabled': _get('backends.mkldnn.enabled'), 'dtype': 'float32', 'autocast': False, 'grad_scaler': False, 'fp32_precision_set_by_this_code': False}
    return flags

def _safe_call(fn):
    try:
        v = fn()
        return v if isinstance(v, (bool, int, float, str)) else repr(v)
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__

def precision_probe(device, seed=PROBE_SEED, tol=PROBE_TOL):
    device = as_device(device)
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    a = torch.randn(1024, 1024, generator=g, dtype=torch.float32)
    b = torch.randn(1024, 1024, generator=g, dtype=torch.float32)
    x = torch.randn(8, 64, 64, 64, generator=g, dtype=torch.float32)
    w = torch.randn(64, 64, 3, 3, generator=g, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        mm = (a.to(device) @ b.to(device)).to('cpu', torch.float64)
        cv = F.conv2d(x.to(device), w.to(device), padding=1).to('cpu', torch.float64)
        mm_ref = a.double() @ b.double()
        cv_ref = F.conv2d(x.double(), w.double(), padding=1)
    e_mm = float((mm - mm_ref).abs().max() / mm_ref.abs().mean())
    e_cv = float((cv - cv_ref).abs().max() / cv_ref.abs().mean())
    return {'device': str(device), 'seed': int(seed), 'tol': tol, 'matmul_rel_err': e_mm, 'conv_rel_err': e_cv, 'pass': bool(e_mm <= tol and e_cv <= tol), 'seconds': round(time.perf_counter() - t0, 3)}

class WarnTally:

    def __init__(self):
        self.counts = {}
        self._cm = None

    def _show(self, message, category, filename, lineno, file=None, line=None):
        k = '%s: %s' % (getattr(category, '__name__', 'Warning'), _one_line(message))
        self.counts[k] = self.counts.get(k, 0) + 1

    def __enter__(self):
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter('always')
        warnings.showwarning = self._show
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        return False

    def nondeterministic(self):
        return {k: v for k, v in self.counts.items() if NONDET_TEXT in k}

    def summary(self):
        nd = self.nondeterministic()
        return {'warnings': dict(sorted(self.counts.items())), 'nondeterministic': dict(sorted(nd.items())), 'nondeterministic_count': int(sum(nd.values())), 'nondeterministic_kinds': len(nd)}

def gpu_report(device):
    device = as_device(device)
    rep = {'device': str(device)}
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        rep.update({'name': torch.cuda.get_device_name(0), 'count': torch.cuda.device_count(), 'capability': '%d.%d' % (p.major, p.minor), 'total_memory_bytes': int(p.total_memory)})
        try:
            q = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
            rep['nvidia_smi'] = [x.strip() for x in q.stdout.splitlines() if x.strip()]
        except Exception as exc:
            rep['nvidia_smi'] = ['not run: ' + type(exc).__name__]
    elif device.type == 'xpu':
        rep.update({'name': torch.xpu.get_device_name(0), 'count': torch.xpu.device_count()})
        try:
            pr = torch.xpu.get_device_properties(0)
            for k in ('has_fp64', 'has_fp16', 'has_subgroup_matrix_multiply_accumulate_tensor_float32'):
                rep[k] = bool(getattr(pr, k)) if hasattr(pr, k) else 'absent'
        except Exception as exc:
            rep['properties'] = 'not read: ' + type(exc).__name__
    else:
        rep.update({'name': platform.processor() or platform.machine(), 'physical_cores': physical_cores()})
    return rep

def env_report(device):
    device = as_device(device)
    cudnn = None
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        pass
    return {'python': sys.version.split()[0], 'os': '%s %s %s' % (platform.system(), platform.release(), platform.machine()), 'torch': torch.__version__, 'torch_cuda': getattr(torch.version, 'cuda', None), 'torch_xpu': getattr(torch.version, 'xpu', None), 'cudnn': cudnn, 'numpy': np.__version__, 'gpu': gpu_report(device), 'cpu_logical': os.cpu_count(), 'core_version': CORE_VERSION}

def write_pip_freeze(path):
    try:
        q = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True, timeout=600)
        text = q.stdout if q.returncode == 0 else 'pip freeze failed with code %s\n' % q.returncode
    except Exception as exc:
        text = 'pip freeze not run: %s\n' % type(exc).__name__
    text = _PROFILE_RE.sub(lambda m: m.group(1) + '<PROFILE>', text)
    text = text.replace(chr(64), '<at>').replace('%40', '<at>')
    with open(path, 'w', encoding='ascii', errors='replace', newline='\n') as fh:
        fh.write(text)
    return hashlib.sha256(text.encode('ascii', 'replace')).hexdigest()

def sync(device):
    device = as_device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'xpu':
        torch.xpu.synchronize()

def reset_peak(device):
    device = as_device(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'xpu' and hasattr(torch.xpu, 'reset_peak_memory_stats'):
        torch.xpu.reset_peak_memory_stats()

def peak_memory(device):
    device = as_device(device)
    out = {}
    if device.type == 'cuda':
        out['max_memory_allocated'] = int(torch.cuda.max_memory_allocated())
        out['max_memory_reserved'] = int(torch.cuda.max_memory_reserved())
    elif device.type == 'xpu':
        out['max_memory_allocated'] = int(torch.xpu.max_memory_allocated())
        out['max_memory_reserved'] = int(torch.xpu.max_memory_reserved())
    try:
        import psutil
        mi = psutil.Process().memory_info()
        out['process_rss'] = int(mi.rss)
        if hasattr(mi, 'peak_wset'):
            out['process_peak_wset'] = int(mi.peak_wset)
    except Exception:
        pass
    try:
        import resource
        out['process_peak_rss_kb'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        pass
    return out
_U53 = 1.0 / 9007199254740992.0

def uniform(key, n):
    bg = np.random.PCG64(np.random.SeedSequence(list(key)))
    raw = bg.random_raw(int(n))
    return (raw >> np.uint64(11)).astype(np.float64) * _U53

def signs(key, n):
    bg = np.random.PCG64(np.random.SeedSequence(list(key)))
    raw = bg.random_raw((int(n) + 63) // 64).astype('<u8')
    bits = np.unpackbits(raw.view(np.uint8), bitorder='little')[:int(n)]
    return 1.0 - 2.0 * bits.astype(np.float64)

def box_mean(a, k):
    r = k // 2
    pad = [(0, 0)] * (a.ndim - 2) + [(r + 1, r), (r + 1, r)]
    p = np.pad(a, pad)
    c = p.cumsum(axis=-2).cumsum(axis=-1)
    h, w = (a.shape[-2], a.shape[-1])
    s = c[..., k:k + h, k:k + w] - c[..., 0:h, k:k + w] - c[..., k:k + h, 0:w] + c[..., 0:h, 0:w]
    return s / float(k * k)

def synth_batch(key, batch, n_in, size=TILE):
    n_all = 24
    if int(n_in) > n_all:
        raise ValueError('synth_batch: n_in %d above %d' % (n_in, n_all))
    x = uniform(tuple(key) + (0,), batch * n_all * size * size)
    x = x.reshape(batch, n_all, size, size)
    a = uniform((DATA_SEED, 424242), n_all) * 2.0 - 1.0
    a[18:] *= 0.25
    sm = box_mean(x, 3) - 0.5
    z = np.zeros((batch, size, size), dtype=np.float64)
    for c in range(n_all):
        z = z + a[c] * sm[:, c]
    z = z * 3.0
    y = 0.5 + 0.5 * z / (1.0 + np.abs(z))
    noise = uniform(tuple(key) + (1,), batch * size * size)
    m = box_mean(noise.reshape(batch, size, size), 5) > 0.45
    xs = x[:, :n_in].astype(np.float32)
    ys = y[:, None].astype(np.float32)
    ms = m[:, None].astype(np.float32)
    return (xs, ys, ms)

def sha_arrays(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).astype('<f4').tobytes())
    return h.hexdigest()

def width_channels(c, w):
    x = c * float(w)
    n = int(round(x))
    if n < 1 or abs(x - n) > 1e-09:
        raise ValueError('width %r gives %r channels for %d' % (w, x, c))
    return n

def chan(c, w):
    return width_channels(c, w)

class ConvBNReLU(nn.Sequential):

    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False, padding_mode='zeros'), nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1), nn.ReLU())

class UpBNLReLU(nn.Sequential):

    def __init__(self, cin, cout):
        super().__init__(nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False), nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1), nn.LeakyReLU(LEAKY_SLOPE))

class BaselineUNet(nn.Module):

    def __init__(self, n_in=17, w=1.0):
        super().__init__()
        c64, c128, c256, c512 = (width_channels(c, w) for c in BASE_CHANNELS)
        self.n_in, self.w = (int(n_in), float(w))
        self.e1 = nn.Sequential(ConvBNReLU(n_in, c64), ConvBNReLU(c64, c128))
        self.e2 = nn.Sequential(ConvBNReLU(c128, c256, stride=2), ConvBNReLU(c256, c256))
        self.e3 = nn.Sequential(ConvBNReLU(c256, c512, stride=2), ConvBNReLU(c512, c512))
        self.u1 = UpBNLReLU(c512, c512)
        self.d1 = nn.Sequential(ConvBNReLU(c512 + c256, c512), ConvBNReLU(c512, c256))
        self.u2 = UpBNLReLU(c256, c256)
        self.head = nn.Sequential(nn.Conv2d(c256 + c128, c128, 1, bias=True), nn.ReLU(), nn.Conv2d(c128, c64, 1, bias=True), nn.ReLU())
        self.out = nn.Conv2d(c64, 1, 1, bias=True)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        b = self.e3(s2)
        y = self.d1(torch.cat([self.u1(b), s2], 1))
        y = self.head(torch.cat([self.u2(y), s1], 1))
        return self.out(y)

class SEGate(nn.Module):

    def __init__(self, c, r=SE_REDUCTION):
        super().__init__()
        m = c // r
        if m < 1 or m * r != c:
            raise ValueError('SE gate: %d channels, reduction %d' % (c, r))
        self.f1 = nn.Linear(c, m)
        self.f2 = nn.Linear(m, c)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.f2(F.relu(self.f1(s))))
        return x * s[:, :, None, None]

class ResBlock(nn.Module):

    def __init__(self, cin, cout, gate=False):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1)
        self.sc = None if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1))
        self.gate = SEGate(cout) if gate else None

    def branch(self, x):
        return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

    def forward(self, x):
        b = self.branch(x)
        if self.gate is not None:
            b = self.gate(b)
        s = x if self.sc is None else self.sc(x)
        return F.relu(b + s)

class FixedDropout(nn.Dropout):

    def __init__(self, p=DROPOUT_P):
        super().__init__(p)
        self.mask_provider = None

    def forward(self, x):
        if self.mask_provider is None or not self.training or self.p <= 0:
            return super().forward(x)
        keep = self.mask_provider(tuple(x.shape))
        m = torch.from_numpy(keep).to(device=x.device, dtype=x.dtype)
        return x * m / (1.0 - self.p)

class AdvancedResUNet(nn.Module):

    def __init__(self, n_in=23, w=1.0, p_drop=DROPOUT_P):
        super().__init__()
        c = [width_channels(x, w) for x in BASE_CHANNELS]
        self.n_in, self.w = (int(n_in), float(w))
        self.stem = ConvBNReLU(n_in, c[0])
        self.r1 = ResBlock(c[0], c[0])
        self.d1 = ConvBNReLU(c[0], c[1], stride=2)
        self.r2 = ResBlock(c[1], c[1])
        self.d2 = ConvBNReLU(c[1], c[2], stride=2)
        self.r3 = ResBlock(c[2], c[2])
        self.d3 = ConvBNReLU(c[2], c[3], stride=2)
        self.r4 = ResBlock(c[3], c[3])
        self.drop = FixedDropout(p_drop)
        self.u3 = UpBNLReLU(c[3], c[2])
        self.q3 = ResBlock(2 * c[2], c[2], gate=True)
        self.u2 = UpBNLReLU(c[2], c[1])
        self.q2 = ResBlock(2 * c[1], c[1], gate=True)
        self.u1 = UpBNLReLU(c[1], c[0])
        self.q1 = ResBlock(2 * c[0], c[0], gate=True)
        self.out = nn.Conv2d(c[0], 1, 1, bias=True)

    def forward(self, x):
        s1 = self.r1(self.stem(x))
        s2 = self.r2(self.d1(s1))
        s3 = self.r3(self.d2(s2))
        b = self.drop(self.r4(self.d3(s3)))
        y = self.q3(torch.cat([self.u3(b), s3], 1))
        y = self.q2(torch.cat([self.u2(y), s2], 1))
        y = self.q1(torch.cat([self.u1(y), s1], 1))
        return self.out(y)

class PixMLP(nn.Module):

    def __init__(self, n_in=23, w=1.0):
        super().__init__()
        self.n_in = int(n_in)
        layers, cin = ([], int(n_in))
        for cout in (128, 128, 64):
            layers += [nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1), nn.ReLU()]
            cin = cout
        layers.append(nn.Conv2d(cin, 1, 1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
NET_CLASSES = {'baseline': BaselineUNet, 'e1x': BaselineUNet, 'baseline24': BaselineUNet, 'e6': BaselineUNet, 's18': BaselineUNet, 'advanced': AdvancedResUNet, 'e2mx': AdvancedResUNet, 'advanced18': AdvancedResUNet, 'pixmlp': PixMLP, 'pixr': PixMLP}

def final_conv(model):
    if isinstance(model, AdvancedResUNet):
        return model.out
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 1 and (m.kernel_size == (1, 1)):
            last = m
    if last is None:
        raise ValueError('no final 1 x 1 conv to one channel')
    return last

def zero_final(model):
    m = final_conv(model)
    with torch.no_grad():
        m.weight.zero_()
        if m.bias is not None:
            m.bias.zero_()
    return int(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0))

def build_net(name, w=1.0, n_in=None, zero_head=False):
    if name not in NET_CLASSES:
        raise ValueError('unknown network ' + str(name))
    n = int(n_in) if n_in is not None else N_INPUTS[name]
    net = NET_CLASSES[name](n, w)
    if zero_head:
        zero_final(net)
    return net

def set_dropout(model, p):
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)
            n += 1
    return n

def fill_params(model, key):
    bn_weights = set()
    for mname, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            bn_weights.add((mname + '.weight').lstrip('.'))
            mod.reset_running_stats()
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            n = p.numel()
            u = uniform(tuple(key) + (zlib.crc32(name.encode()),), n)
            if name in bn_weights:
                v = 1.0 + 0.1 * (2.0 * u - 1.0)
            elif p.dim() > 1:
                fan_in = int(np.prod(p.shape[1:]))
                b = 1.0 / np.sqrt(float(fan_in))
                v = (2.0 * u - 1.0) * b
            else:
                v = 0.05 * (2.0 * u - 1.0)
            p.copy_(torch.from_numpy(v.astype(np.float32).reshape(tuple(p.shape))))

def params_sha(model):
    h = hashlib.sha256()
    for _name, p in sorted(model.named_parameters()):
        h.update(p.detach().to('cpu').contiguous().numpy().astype('<f4').tobytes())
    return h.hexdigest()

def state_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode('ascii'))
        h.update(v.detach().to('cpu').contiguous().numpy().tobytes())
    return h.hexdigest()

def count_params(model):
    return int(sum((p.numel() for p in model.parameters())))

def count_buffers(model):
    return int(sum((b.numel() for b in model.buffers() if b.dtype.is_floating_point)))

def macs_detail(model, n_in, device='cpu', height=TILE, width=TILE):
    device = as_device(device)
    rows = []
    hooks = []

    def conv_hook(name):

        def f(mod, inp, out):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            if isinstance(mod, nn.ConvTranspose2d):
                x = inp[0]
                macs = x.shape[2] * x.shape[3] * mod.in_channels * (mod.out_channels // mod.groups) * k
                kind = 'convtranspose2d'
            else:
                macs = out.shape[2] * out.shape[3] * (mod.in_channels // mod.groups) * mod.out_channels * k
                kind = 'conv2d'
            rows.append({'layer': name, 'kind': kind, 'in': int(mod.in_channels), 'out': int(mod.out_channels), 'kernel': '%dx%d' % tuple(mod.kernel_size), 'stride': int(mod.stride[0]), 'padding': list(mod.padding) if isinstance(mod.padding, tuple) else str(mod.padding), 'padding_mode': str(mod.padding_mode), 'bias': mod.bias is not None, 'out_hw': '%dx%d' % tuple(out.shape[2:4]), 'macs': int(macs)})
        return f

    def lin_hook(name):

        def f(mod, inp, out):
            rows.append({'layer': name, 'kind': 'linear', 'in': int(mod.in_features), 'out': int(mod.out_features), 'bias': mod.bias is not None, 'macs': int(mod.in_features * mod.out_features)})
        return f
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(mod.register_forward_hook(conv_hook(name)))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook(name)))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, int(n_in), height, width, device=device))
    finally:
        model.train(was)
        for h in hooks:
            h.remove()
    conv = sum((r['macs'] for r in rows if r['kind'] != 'linear'))
    lin = sum((r['macs'] for r in rows if r['kind'] == 'linear'))
    return {'conv_and_transposed': int(conv), 'linear': int(lin), 'layers': rows}

def count_macs(model, n_in, device='cpu'):
    return macs_detail(model, n_in, device)['conv_and_transposed']

def ioa_loss(pred, target, mask):
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('shapes differ: pred %s, target %s, mask %s' % (tuple(pred.shape), tuple(target.shape), tuple(mask.shape)))
    mb = mask if mask.dtype == torch.bool else mask > 0
    m = mb.to(pred.dtype)
    t = torch.where(mb, target, torch.zeros_like(target))
    n = m.sum()
    obar = (t * m).sum() / n
    num = ((pred - t) ** 2 * m).sum()
    den = (((pred - obar).abs() + (t - obar).abs()) ** 2 * m).sum()
    safe = torch.where(den > 0, den, torch.ones_like(den))
    return num / safe

def mse_loss(pred, target, mask):
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('shapes differ: pred %s, target %s, mask %s' % (tuple(pred.shape), tuple(target.shape), tuple(mask.shape)))
    mb = mask if mask.dtype == torch.bool else mask > 0
    m = mb.to(pred.dtype)
    t = torch.where(mb, target, torch.zeros_like(target))
    n = m.sum()
    num = ((pred - t) ** 2 * m).sum()
    safe = torch.where(n > 0, n, torch.ones_like(n))
    return num / safe
LOSSES = {'ioa': ioa_loss, 'mse': mse_loss}

def grad_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(torch.sum(p.grad.detach().to(torch.float64) ** 2))
    return tot ** 0.5

def lr_at(step, base=LR0):
    return base * LR_DECAY ** (int(step) // LR_EVERY)

def make_adam(model, lr=LR0):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)

def train_step(model, opt, x, y, m, lr, loss='ioa', clip=None, want_norm=False):
    fn = LOSSES.get(loss)
    if fn is None:
        raise ValueError('unknown loss %r' % (loss,))
    for g in opt.param_groups:
        g['lr'] = lr
    opt.zero_grad(set_to_none=True)
    lv = fn(model(x), y, m)
    lv.backward()
    gn = grad_norm(model) if want_norm else None
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    opt.step()
    if want_norm:
        return (lv.detach(), gn)
    return lv.detach()

def ioa_stat(pred, target, mask):
    e = np.asarray(pred, dtype=np.float64)
    o = np.asarray(target, dtype=np.float64)
    mk = np.asarray(mask) > 0
    if e.shape != o.shape or e.shape != mk.shape:
        raise ValueError('shapes differ')
    e, o = (e[mk], o[mk])
    if e.size == 0 or not (np.all(np.isfinite(e)) and np.all(np.isfinite(o))):
        return float('nan')
    ob = o.mean()
    num = np.sum((e - o) ** 2)
    den = np.sum((np.abs(e - ob) + np.abs(o - ob)) ** 2)
    return float(num / den) if den > 0 else 0.0

def mse_stat(pred, target, mask, scale=1.0):
    e = np.asarray(pred, dtype=np.float64)
    o = np.asarray(target, dtype=np.float64)
    mk = np.asarray(mask) > 0
    if e.shape != o.shape or e.shape != mk.shape:
        raise ValueError('shapes differ')
    e, o = (e[mk], o[mk])
    if e.size == 0 or not (np.all(np.isfinite(e)) and np.all(np.isfinite(o))):
        return float('nan')
    return float(np.mean(((e - o) * float(scale)) ** 2))

def grad_summary(model, key):
    out = {}
    for name, p in sorted(model.named_parameters()):
        g = p.grad.detach().to('cpu', torch.float64).numpy().ravel()
        proj = []
        for k in range(N_PROJ):
            s = signs(tuple(key) + (k, zlib.crc32(name.encode())), g.size)
            proj.append(float(np.dot(g, s)))
        out[name] = {'n': int(g.size), 'l2': float(np.sqrt(np.dot(g, g))), 'maxabs': float(np.max(np.abs(g))), 'sum': float(g.sum()), 'proj': proj}
    return out

def save_json(path, obj):
    tmp = path + '.partial'
    with open(tmp, 'w', encoding='ascii', newline='\n') as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write('\n')
    os.replace(tmp, path)

def torch_load(path, device='cpu'):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def save_atomic(obj, path, tries=20, gap=0.5):
    tmp = path + '.partial'
    torch.save(obj, tmp)
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k + 1 == tries:
                raise
            time.sleep(gap)

def save_checkpoint(path, step, model, opt, losses):
    save_atomic({'step': int(step), 'model': model.state_dict(), 'opt': opt.state_dict(), 'losses': [float(x) for x in losses]}, path)

def get_rng_states(device):
    device = as_device(device)
    st = np.random.get_state()
    pyst = random.getstate()
    out = {'python': [pyst[0], list(pyst[1]), pyst[2]], 'numpy_global': [st[0], [int(x) for x in st[1]], int(st[2]), int(st[3]), float(st[4])], 'torch_cpu': torch.get_rng_state(), 'torch_device': None, 'device_type': device.type}
    if device.type == 'cuda':
        out['torch_device'] = torch.cuda.get_rng_state()
    elif device.type == 'xpu':
        out['torch_device'] = torch.xpu.get_rng_state()
    return out

def set_rng_states(states, device):
    device = as_device(device)
    p = states['python']
    random.setstate((p[0], tuple(p[1]), p[2]))
    n = states['numpy_global']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(states['torch_cpu'])
    if states.get('torch_device') is not None and states.get('device_type', device.type) == device.type:
        if device.type == 'cuda':
            torch.cuda.set_rng_state(states['torch_device'])
        elif device.type == 'xpu':
            torch.xpu.set_rng_state(states['torch_device'])
RESUME_FIELDS = ('format', 'training', 'seed', 'step', 'steps', 'phase', 'plan_id', 'model', 'optimizer', 'schedule', 'best', 'best_step', 'since_best', 'patience', 'rng', 'sampler', 'elapsed_s', 'loss_acc', 'stopped_early', 'evaluations', 'result')

def resume_state(**fields):
    missing = [k for k in RESUME_FIELDS if k not in fields]
    if missing:
        raise ValueError('resume checkpoint lacks %s' % missing)
    d = dict(fields)
    d['format'] = fields.get('format') or 'choi-train-ckpt-2'
    return d

def default_out(test):
    if os.path.isdir('/kaggle/working'):
        return os.path.join('/kaggle/working', test)
    return os.path.join(os.getcwd(), test + '_out')
