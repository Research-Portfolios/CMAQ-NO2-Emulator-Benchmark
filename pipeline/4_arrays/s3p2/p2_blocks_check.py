"""Checks that the month-at-once channel sources of the second pass give the same float64 values as the first pass's per-day sources, for three test months (read only)."""
import sys
sys.path.insert(0, '.\\run\\code')
import numpy as np
from s3 import s3_assemble as A
from s3 import s3_build as B
from s3 import s3common as S
from s3p2 import p2_roundtrip as RT
from s3p2 import p2common as P

def main():
    S.common.set_temp()
    inputs = S.inventory_inputs()
    reader = A.MonthReader(inputs, check_raw=False)
    static = A.load_static(inputs)
    veg, _u = A.load_veg(inputs)
    clim = np.zeros((12, 7, 4) + P.G, np.float32)
    total, bad = (0, 0)
    for y, m in ((2015, 1), (2016, 3), (2019, 12)):
        if m == 1:
            prev = B.prev_day_sources(reader, y)
        else:
            prev = {}
            for c in B.LAGGED:
                v, _dates = reader.read_month(P.MET_SERIES[c], y, m - 1)
                prev[c] = np.asarray(v[-1]).copy()
                del v
        blocks, _nprev, days = RT.month_blocks(y, m, reader, prev, static, veg, None)
        md = {ser: reader.read_month(ser, y, m) for ser in RT.MET}
        pv = dict(prev)
        for k, d in enumerate(days):
            met = A.met_of_day(md, k)
            src = A.day_sources(d, met, pv, static, veg, clim)
            for c in P.CH_NAMES:
                if c in P.EMIS:
                    continue
                total += 1
                a = np.asarray(src[c], np.float64)
                b = np.asarray(blocks[c][k], np.float64)
                if not np.array_equal(a, b):
                    bad += 1
                    print('DIFFERS %s %s' % (P.ymd(d), c))
            pv = {c: np.asarray(met[c]).copy() for c in B.LAGGED}
        print('%d-%02d: %d days compared' % (y, m, len(days)), flush=True)
    print('channel-days compared %d, differing %d: %s' % (total, bad, 'EQUAL' if bad == 0 else 'NOT EQUAL'))
if __name__ == '__main__':
    P.run_main(main)
