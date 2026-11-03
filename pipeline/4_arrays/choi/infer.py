"""Predicts the whole 299 x 459 grid from 64 x 64 windows at stride 32, blended by a two-dimensional Hann weight with a floor of 0.001 on an edge-padded grid, and returns the field in ppb with negative values set to zero and counted."""
import math
import numpy as np
GRID = (299, 459)
TILE = 64
STRIDE = 32
FLOOR = 0.001

def hann_weight(n=TILE, floor=FLOOR):
    k = np.arange(n)
    m = np.minimum(k, n - 1 - k).astype(np.float64)
    w1 = 0.5 - 0.5 * np.cos(2.0 * math.pi * m / (n - 1))
    return np.maximum(np.outer(w1, w1), floor)

def padded_size(n, tile=TILE, stride=STRIDE):
    if n <= tile:
        return tile
    return tile + stride * int(math.ceil((n - tile) / stride))

def window_starts(n_pad, tile=TILE, stride=STRIDE):
    return list(range(0, n_pad - tile + 1, stride))

def blend_weights(shape=GRID, tile=TILE, stride=STRIDE, floor=FLOOR):
    hp, wp = (padded_size(shape[0], tile, stride), padded_size(shape[1], tile, stride))
    w = hann_weight(tile, floor)
    s = np.zeros((hp, wp), dtype=np.float64)
    for i in window_starts(hp, tile, stride):
        for j in window_starts(wp, tile, stride):
            s[i:i + tile, j:j + tile] += w
    return s

def predict_grid(model, x, device='cpu', batch=32, to_ppb=None, clip=True, mask=None, tile=TILE, stride=STRIDE, floor=FLOOR):
    import torch
    if isinstance(x, torch.Tensor):
        xa = x.detach().to('cpu', torch.float32).numpy()
    else:
        xa = np.asarray(x, dtype=np.float32)
    if xa.ndim != 3:
        raise ValueError('x must be (C, H, W), got %s' % (xa.shape,))
    c, h, w = xa.shape
    hp, wp = (padded_size(h, tile, stride), padded_size(w, tile, stride))
    xp = np.pad(xa, ((0, 0), (0, hp - h), (0, wp - w)), mode='edge')
    wt = hann_weight(tile, floor)
    acc = np.zeros((hp, wp), dtype=np.float64)
    wsum = np.zeros((hp, wp), dtype=np.float64)
    starts = [(i, j) for i in window_starts(hp, tile, stride) for j in window_starts(wp, tile, stride)]
    is_module = isinstance(model, torch.nn.Module)
    was_training = model.training if is_module else None
    if is_module:
        model.eval()
    try:
        with torch.no_grad():
            for b0 in range(0, len(starts), batch):
                part = starts[b0:b0 + batch]
                tiles = np.stack([xp[:, i:i + tile, j:j + tile] for i, j in part])
                t = torch.from_numpy(np.ascontiguousarray(tiles)).to(device)
                out = model(t)
                out = out.detach().to('cpu', torch.float64).numpy()
                if out.shape != (len(part), 1, tile, tile):
                    raise ValueError('model output %s, expected %s' % (out.shape, (len(part), 1, tile, tile)))
                for k, (i, j) in enumerate(part):
                    acc[i:i + tile, j:j + tile] += out[k, 0] * wt
                    wsum[i:i + tile, j:j + tile] += wt
    finally:
        if is_module and was_training:
            model.train()
    field = (acc / wsum)[:h, :w]
    if to_ppb is not None:
        field = np.asarray(to_ppb(field), dtype=np.float64)
    neg = field < 0
    n_clipped = int(neg.sum())
    n_clipped_masked = None
    if mask is not None:
        n_clipped_masked = int((neg & np.asarray(mask, dtype=bool)).sum())
    if clip:
        field = np.where(neg, 0.0, field)
    return {'field': field.astype(np.float32), 'n_clipped': n_clipped, 'n_clipped_masked': n_clipped_masked, 'n_windows': len(starts), 'padded_shape': (hp, wp), 'weight_sum_min': float(wsum[:h, :w].min())}
