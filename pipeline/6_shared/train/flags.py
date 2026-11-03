"""Sets the device, precision flags, threads, seeds and random states of the trainer and the test entries from the shared module, and reports them in every training log."""
import os
import platform
import sys
CUBLAS_CFG = ':4096:8'

def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        return int(n) if n else 1
    except Exception:
        return os.cpu_count() or 1

def rule_threads():
    return max(1, physical_cores() - 1)

def prepare_cuda_env():
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS_CFG
        return 'set by this code to %s' % CUBLAS_CFG
    return 'already set (value read by emul_core.cublas_reading)'

def _core():
    from train import emul_core
    return emul_core

def read_flags():
    return _core().read_flags()

def device_name(device):
    import torch
    try:
        if device == 'cuda':
            return torch.cuda.get_device_name(0)
        if device == 'xpu':
            return torch.xpu.get_device_name(0)
    except Exception as e:
        return 'ERROR %s' % type(e).__name__
    return 'cpu: %s; %d physical cores' % (platform.processor() or '?', physical_cores())

def versions():
    import numpy
    import torch
    v = {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': numpy.__version__, 'torch.version.cuda': getattr(torch.version, 'cuda', None), 'torch.version.xpu': getattr(torch.version, 'xpu', None), 'emul_core': _core().CORE_VERSION}
    try:
        v['cudnn'] = torch.backends.cudnn.version()
    except Exception:
        v['cudnn'] = None
    try:
        import psutil
        v['psutil'] = psutil.__version__
    except Exception:
        v['psutil'] = None
    return v

def check_device(device):
    import torch
    if device == 'cpu':
        return True
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('device cuda asked but not available')
        return True
    if device == 'xpu':
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            raise ValueError('device xpu asked but not available')
        return True
    raise ValueError('unknown device %r' % (device,))

def sync(device):
    _core().sync(device)

def setup(device, threads):
    rows = {'cublas_workspace_config': prepare_cuda_env()}
    import torch
    check_device(device)
    ec = _core()
    fl = ec.set_flags(device, threads=threads)
    if device != 'cpu':
        torch.set_num_threads(int(threads))
    rows['set_flags'] = {'asked': ['torch.backends.cudnn.allow_tf32 = False', 'torch.backends.cuda.matmul.allow_tf32 = False', 'torch.backends.cudnn.benchmark = False', 'torch.backends.cudnn.deterministic = True', 'torch.use_deterministic_algorithms(True, warn_only=True)'], 'warnings': fl.get('set_warnings', []), 'k24_check': fl['k24_check']}
    rows['set_num_threads'] = {'asked': int(threads), 'result': int(torch.get_num_threads()), 'rule_physical_minus_1': rule_threads()}
    return rows

def flag_report(device, setup_rows=None):
    return {'device': device, 'device_name': device_name(device), 'flags_read_back': read_flags(), 'versions': versions(), 'setup': setup_rows or {}}

def seed_everything(seed):
    _core().seed_all(seed)

def get_rng_states(device):
    return _core().get_rng_states(device)

def set_rng_states(states, device):
    _core().set_rng_states(states, device)
