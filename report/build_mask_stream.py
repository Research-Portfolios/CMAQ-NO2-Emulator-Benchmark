"""The land mask of the 12 km grid: the 53,594 land cells of the
contiguous United States (299 rows by 459 columns, row 0 the south
edge), read from results/grid/mask_conus_land.npy and tested against
its sha256 before use."""
import hashlib
import os

import numpy as np

SHA = 'ff3d553eaee09ac62fcddf52ab592055617f6dcb1a292f8fc2f5b23df5e466a1'
CELLS = 53594
SHAPE = (299, 459)
NAME = 'mask_conus_land.npy'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACE = os.path.join('results', 'grid', NAME)


def get_mask():
    """Return (mask, place): the boolean land mask and its path in the
    repository, after testing its sha256, shape and cell count."""
    p = os.path.join(ROOT, PLACE)
    with open(p, 'rb') as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise RuntimeError(NAME + ': sha256 differs from the recorded one')
    m = np.load(p, allow_pickle=False)
    if m.shape != SHAPE or m.dtype != bool or int(m.sum()) != CELLS:
        raise RuntimeError(NAME + ': shape, type or cell count differs')
    return m, PLACE.replace(os.sep, '/')
