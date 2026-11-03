"""The training loss, 1 - IOA over the masked cells of a batch, computed as the ratio of the squared error to the potential error, with a guard that gives zero for a zero denominator."""
import torch

def ioa_loss(pred, target, mask):
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('shapes differ: pred %s, target %s, mask %s' % (tuple(pred.shape), tuple(target.shape), tuple(mask.shape)))
    m = mask.to(torch.bool)
    e = pred[m]
    o = target[m]
    if o.numel() == 0:
        raise ValueError('no masked cell in the batch')
    obar = o.mean()
    num = torch.sum((e - o) ** 2)
    den = torch.sum((torch.abs(e - obar) + torch.abs(o - obar)) ** 2)
    safe = torch.where(den > 0, den, torch.ones_like(den))
    return num / safe
