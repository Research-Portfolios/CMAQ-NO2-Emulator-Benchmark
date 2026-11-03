"""Checks from the per-day minima and maxima that every stored channel of every day is finite and within float16's finite range (read only)."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
import numpy as np
from s3 import s3common as S

def main():
    lo = {c: np.inf for c in S.CH_NAMES}
    hi = {c: -np.inf for c in S.CH_NAMES}
    bad_total = 0
    for y in S.YEARS:
        z = np.load(os.path.join(S.WORK_S3, 'daystats_%d.npz' % y))
        bad = int((~np.isfinite(z['daymin'])).sum() + (~np.isfinite(z['daymax'])).sum() + (~np.isfinite(z['tmin'])).sum() + (~np.isfinite(z['tmax'])).sum())
        bad_total += bad
        for c in S.CH_NAMES:
            j = S.CH_INDEX[c]
            lo[c] = min(lo[c], float(z['daymin'][:, j].min()))
            hi[c] = max(hi[c], float(z['daymax'][:, j].max()))
        print('%d: days %d, non-finite per-day extremes %d' % (y, z['daymin'].shape[0], bad))
    for c in S.CH_NAMES:
        print('%s stored min %.6g max %.6g' % (c, lo[c], hi[c]))
    print('non-finite in all: %d' % bad_total)
if __name__ == '__main__':
    S.run_main(main)
