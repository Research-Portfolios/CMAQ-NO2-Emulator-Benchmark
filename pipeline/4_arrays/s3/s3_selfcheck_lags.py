"""Checks that each day's lagged channels equal the previous day's stored fields across month and year boundaries, and for 1 January 2015 the lag answers of 31 December 2014 (read only)."""
import json
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
import numpy as np
from s3 import s3_assemble as A
from s3 import s3common as S

def main():
    S.common.set_temp()
    with open(os.path.join(S.CFG, 'S3_CHANNELS.json'), 'r', encoding='ascii') as f:
        ch = {c['name']: c for c in json.load(f)['channels']}
    pairs = [(x, m) for x, m in S.LAG_OF.items()]
    for x, m in pairs:
        if ch[x]['array'] != 'inputs_f16' or ch[m]['array'] != 'inputs_f16':
            raise ValueError('lag pair %s %s not both float16' % (x, m))
    res = {}
    prev_last = None
    for y in S.YEARS:
        a = np.load(os.path.join(S.ARRAYS, 'inputs_f16_%d.npy' % y), mmap_mode='r')
        n = a.shape[0]
        bad_inside = 0
        for x, m in pairs:
            jx, jm = (ch[x]['index'], ch[m]['index'])
            for i in range(1, n):
                if not np.array_equal(a[i, jx], a[i - 1, jm]):
                    bad_inside += 1
        if y == 2015:
            reader = A.MonthReader(S.inventory_inputs())
            lag = A.read_lag_answers(reader)
            first = {m: A.store_fn(ch[m]['storage'])(lag[m]) for _x, m in pairs}
        else:
            first = prev_last
        bad_first = sum((0 if np.array_equal(a[0, ch[x]['index']], first[m]) else 1 for x, m in pairs))
        prev_last = {m: np.array(a[n - 1, ch[m]['index']]) for _x, m in pairs}
        res[str(y)] = {'days': n, 'pairs': len(pairs), 'unequal_day_pairs_inside': bad_inside, 'unequal_first_day_pairs': bad_first}
        del a
        print('%s %d: %s' % (S.now_local(), y, res[str(y)]), flush=True)
    print('all equal: %s' % all((v['unequal_day_pairs_inside'] == 0 and v['unequal_first_day_pairs'] == 0 for v in res.values())))
if __name__ == '__main__':
    S.run_main(main)
